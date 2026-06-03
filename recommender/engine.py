"""
Hibrit kitap öneri motoru.

Bu modül; işbirlikçi filtreleme, içerik tabanlı filtreleme, anlamsal geri çağırma,
popülerlik skoru, tür/niyet eşleşmesi ve açıklama üretimi bileşenlerini birlikte
yönetir. LLM doğrudan öneri üretmez; yalnızca hibrit motor tarafından seçilen
gerçek katalog kitapları için açıklama üretiminde kullanılır.
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Optional

from app.core.config import settings
from app.core.schemas import ProductCard, RecommendedProduct, SimilarProduct
from app.services.data_store import DataStore
from app.services.explainer import LLMExplainer

from recommender.collaborative import CollaborativeFilter
from recommender.content_based import ContentBasedFilter
from recommender.semantic import SemanticRetriever
from recommender.scoring import (
    compute_popularity_scores,
    compute_category_intent_scores,
    rank_candidates,
)
from recommender.text_processing import (
    detect_category_intent,
    expand_query,
    build_product_text,
    format_price_tl,
    clean_product_title,
    clean_display_value,
    readable_genre,
    normalize_text,
)
from recommender.profile import build_user_profile

logger = logging.getLogger(__name__)


class HybridRecommendationEngine:
    """CF, CBF, semantik retrieval ve popülerlik sinyallerini birleştiren kitap öneri motoru."""
    def __init__(
        self,
        data_store: DataStore,
        explainer: LLMExplainer,
    ):
        self.ds = data_store
        self.explainer = explainer

        self.cf = CollaborativeFilter()
        self.cbf = ContentBasedFilter()
        self.sem = SemanticRetriever()

        self.popularity_scores: dict[str, float] = {}
        self._initialized: bool = False

        self.latency_log: dict[str, list[float]] = {
            "recommendation": [],
            "search": [],
        }

    def initialize(self) -> None:
        """Tüm bileşenleri eğitir / yükler."""
        if not self.ds.is_loaded:
            raise RuntimeError("DataStore yüklenmeden engine başlatılamaz.")

        logger.info("Hibrit motor başlatılıyor...")

        self.cf.fit(self.ds.interactions_df)

        products = self.ds.products_df.copy()
        if "product_text" not in products.columns:
            products["product_text"] = products.apply(
                lambda r: build_product_text(r.to_dict()), axis=1
            )
        self.cbf.fit(products)

        if self.ds.embeddings is not None:
            self.sem.load_embeddings(self.ds.embeddings, self.ds.product_ids)
        else:
            logger.warning("Embedding matrisi yok, semantik skor devre dışı.")
            # Embedding matrisi yoksa sorgu kodlama için model yüklenir.
            self.sem.load_model()

        self.popularity_scores = compute_popularity_scores(self.ds.interactions_df)

        self._initialized = True
        logger.info("Hibrit motor hazır.")

    def _filter_candidates_by_intent(
        self,
        candidate_ids: list[str],
        detected_cat: str | None,
        min_matches: int = 5,
    ) -> tuple[list[str], dict[str, float]]:
        """Açık tür niyeti varsa aday havuzunu ilgili kitaplarla sınırlar; yeterli eşleşme yoksa havuzu daraltmaz."""
        cat_scores = compute_category_intent_scores(candidate_ids, detected_cat, self.ds.products_df)
        if not detected_cat:
            return candidate_ids, cat_scores
        matched = [pid for pid in candidate_ids if cat_scores.get(pid, 0.0) >= 0.75]
        if len(matched) >= min_matches:
            return matched, {pid: cat_scores.get(pid, 0.0) for pid in matched}
        return candidate_ids, cat_scores


    def _direct_text_score(self, query: str, row: pd.Series) -> float:
        """Başlık, yazar, tür ve açıklama alanlarında doğrudan metin eşleşmesi skoru hesaplar."""
        q = normalize_text(query)
        if not q:
            return 0.0
        fields = ["title", "brand", "author", "category", "style", "description", "product_text"]
        text = normalize_text(" ".join(str(row.get(f, "")) for f in fields))
        if not text:
            return 0.0

        score = 0.0
        if q in text:
            score += 1.0
        words = [w for w in q.replace("/", " ").split() if len(w) >= 3]
        if words:
            hits = sum(1 for w in words if w in text)
            score += min(0.70, hits / len(words))
            if hits == len(words):
                score += 0.35

        expanded_terms = [t for t in normalize_text(expand_query(query)).split() if len(t) >= 4]
        if expanded_terms:
            term_hits = sum(1 for t in expanded_terms[:18] if t in text)
            score += min(0.45, term_hits * 0.06)
        return min(score, 1.0)

    def _direct_text_candidates(
        self,
        query: str,
        top_k: int = 80,
        excluded_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        excluded_ids = excluded_ids or set()
        scored: list[tuple[str, float]] = []
        for _, row in self.ds.products_df.iterrows():
            pid = str(row.get("product_id", ""))
            if not pid or pid in excluded_ids:
                continue
            score = self._direct_text_score(query, row)
            if score > 0:
                scored.append((pid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _merge_candidate_lists(self, *lists: list[str]) -> list[str]:
        seen = set()
        out = []
        for lst in lists:
            for pid in lst:
                if pid not in seen:
                    seen.add(pid)
                    out.append(pid)
        return out

    def _apply_direct_boost(self, query: str, candidate_ids: list[str], sem_scores: dict[str, float]) -> dict[str, float]:
        if not query:
            return sem_scores
        lookup = self.ds.products_df.set_index("product_id", drop=False)
        boosted = dict(sem_scores)
        for pid in candidate_ids:
            if pid in lookup.index:
                row = lookup.loc[pid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                direct = self._direct_text_score(query, row)
                if direct > 0:
                    boosted[pid] = max(boosted.get(pid, 0.0), direct)
        return boosted

    def recommend_for_user(
        self,
        user_id: str,
        k: int = 8,
        query: str = "",
    ) -> list[RecommendedProduct]:
        """
        Mevcut kullanıcı için kişiselleştirilmiş kitap önerileri üretir.
        Adaylar kullanıcı geçmişi, içerik benzerliği, anlamsal benzerlik,
        popülerlik ve tür/niyet eşleşmesi sinyalleriyle sıralanır.
        """
        start = time.time()
        self._check_init()

        interacted = self.cf.get_user_interacted_products(user_id)
        all_product_ids = self.ds.products_df["product_id"].tolist()

        candidates = [p for p in all_product_ids if p not in interacted]

        if query:
            direct_candidates = [pid for pid, _ in self._direct_text_candidates(query, top_k=120, excluded_ids=interacted)]
            if self.sem.is_ready:
                expanded = expand_query(query)
                sem_candidates = self.sem.get_top_semantic_candidates(
                    expanded, top_k=min(200, len(candidates)), excluded_ids=interacted
                )
                sem_ids = [pid for pid, _ in sem_candidates]
            else:
                sem_ids = []
            candidate_ids = self._merge_candidate_lists(direct_candidates, sem_ids, candidates[:100])
        else:
            candidate_ids = candidates[:300]

        detected_cat = detect_category_intent(query) if query else None
        candidate_ids, cat_scores = self._filter_candidates_by_intent(candidate_ids, detected_cat, min_matches=max(3, k))

        cf_scores  = self.cf.get_cf_scores(user_id, candidate_ids)
        cbf_profile = self.cbf.get_user_profile(user_id, self.ds.interactions_df)
        cbf_scores  = self.cbf.get_content_scores(cbf_profile, candidate_ids) if cbf_profile is not None else {}
        sem_scores  = self.sem.get_semantic_scores(query or "", candidate_ids) if query else {}
        sem_scores  = self._apply_direct_boost(query, candidate_ids, sem_scores) if query else sem_scores
        pop_scores  = {pid: self.popularity_scores.get(pid, 0.0) for pid in candidate_ids}

        ranked = rank_candidates(
            candidate_ids, cf_scores, cbf_scores, sem_scores,
            pop_scores, cat_scores, mode="personalized", top_k=k,
        )

        results = []
        for pid, score, dominant in ranked:
            product = self._build_product_card(pid)
            if product is None:
                continue
            if detected_cat and cat_scores.get(pid, 0.0) >= 0.75:
                product.style = f"{readable_genre(detected_cat)} uyumu"
            explanation = self.explainer.quick_explain(
                product_title=product.title,
                product_description=product.description,
                product_category=product.category,
                user_query=query,
                dominant_signal=dominant,
                mode="personalized",
            )
            results.append(RecommendedProduct(
                product=product,
                explanation=explanation,
                hybrid_score=round(score, 4),
                dominant_signal=dominant,
            ))

        elapsed_ms = (time.time() - start) * 1000
        self.latency_log["recommendation"].append(elapsed_ms)
        logger.debug(f"Öneri üretildi: {len(results)} ürün, {elapsed_ms:.1f}ms")
        return results

    def recommend_cold_start(
        self,
        preferences: str,
        k: int = 8,
    ) -> list[RecommendedProduct]:
        """
        Yeni kullanıcı için öneri üretir.
        Kullanıcının doğal dil tercihi, semantik arama, kategori niyeti ve popülerlik
        sinyalleri kullanılır. CF kullanılmaz çünkü kullanıcı geçmişi yoktur.
        """
        start = time.time()
        self._check_init()

        expanded = expand_query(preferences)
        detected_cat = detect_category_intent(preferences)

        direct_candidates = [pid for pid, _ in self._direct_text_candidates(preferences, top_k=120)]
        if self.sem.is_ready:
            sem_candidates = self.sem.get_top_semantic_candidates(expanded, top_k=150)
            sem_ids = [pid for pid, _ in sem_candidates]
        else:
            sem_ids = self.ds.products_df["product_id"].tolist()[:150]
        candidate_ids = self._merge_candidate_lists(direct_candidates, sem_ids)

        candidate_ids, cat_scores = self._filter_candidates_by_intent(candidate_ids, detected_cat, min_matches=max(3, k))

        sem_scores = self.sem.get_semantic_scores(expanded, candidate_ids)
        sem_scores = self._apply_direct_boost(preferences, candidate_ids, sem_scores)
        pop_scores = {pid: self.popularity_scores.get(pid, 0.0) for pid in candidate_ids}

        ranked = rank_candidates(
            candidate_ids,
            cf_scores={}, cbf_scores={},
            sem_scores=sem_scores,
            pop_scores=pop_scores,
            cat_scores=cat_scores,
            mode="cold_start",
            top_k=k,
        )

        results = []
        for pid, score, dominant in ranked:
            product = self._build_product_card(pid)
            if product is None:
                continue
            if detected_cat and cat_scores.get(pid, 0.0) >= 0.75:
                product.style = f"{readable_genre(detected_cat)} uyumu"
            explanation = self.explainer.quick_explain(
                product_title=product.title,
                product_description=product.description,
                product_category=product.category,
                user_query=preferences,
                dominant_signal=dominant if dominant != "collaborative" else "semantic",
                mode="cold_start",
            )
            results.append(RecommendedProduct(
                product=product,
                explanation=explanation,
                hybrid_score=round(score, 4),
                dominant_signal=dominant,
            ))

        elapsed_ms = (time.time() - start) * 1000
        self.latency_log["recommendation"].append(elapsed_ms)
        return results

    def search(self, query: str, k: int = 8) -> list[RecommendedProduct]:
        """
        Doğal dil ile semantik ürün arama.
        Türkçe sorgular desteklenir. Kategori niyeti algılanarak alakasız kategoriler öne çıkmaz.
        """
        start = time.time()
        self._check_init()

        expanded = expand_query(query)
        detected_cat = detect_category_intent(query)

        direct_candidates = [pid for pid, _ in self._direct_text_candidates(query, top_k=120)]
        if self.sem.is_ready:
            sem_candidates = self.sem.get_top_semantic_candidates(expanded, top_k=120)
            sem_ids = [pid for pid, _ in sem_candidates]
        else:
            # Embedding yoksa CBF fallback
            cbf_vec = self.cbf.query_to_vector(expanded)
            if cbf_vec is not None:
                cbf_scores_all = self.cbf.get_content_scores(cbf_vec, self.ds.products_df["product_id"].tolist())
                sem_ids = sorted(cbf_scores_all, key=cbf_scores_all.get, reverse=True)[:100]
            else:
                sem_ids = self.ds.products_df["product_id"].tolist()[:100]
        candidate_ids = self._merge_candidate_lists(direct_candidates, sem_ids)

        candidate_ids, cat_scores = self._filter_candidates_by_intent(candidate_ids, detected_cat, min_matches=max(3, k))

        sem_scores = self.sem.get_semantic_scores(expanded, candidate_ids)
        sem_scores = self._apply_direct_boost(query, candidate_ids, sem_scores)
        pop_scores = {pid: self.popularity_scores.get(pid, 0.0) for pid in candidate_ids}

        ranked = rank_candidates(
            candidate_ids,
            cf_scores={}, cbf_scores={},
            sem_scores=sem_scores,
            pop_scores=pop_scores,
            cat_scores=cat_scores,
            mode="cold_start",
            top_k=k,
        )

        results = []
        for pid, score, dominant in ranked:
            product = self._build_product_card(pid)
            if product is None:
                continue
            if detected_cat and cat_scores.get(pid, 0.0) >= 0.75:
                product.style = f"{readable_genre(detected_cat)} uyumu"
            explanation = self.explainer.quick_explain(
                product_title=product.title,
                product_description=product.description,
                product_category=product.category,
                user_query=query,
                dominant_signal="semantic",
                mode="cold_start",
            )
            results.append(RecommendedProduct(
                product=product,
                explanation=explanation,
                hybrid_score=round(score, 4),
                dominant_signal="semantic",
            ))

        elapsed_ms = (time.time() - start) * 1000
        self.latency_log["search"].append(elapsed_ms)
        return results

    def get_similar(self, product_id: str, k: int = 6) -> list[SimilarProduct]:
        """  Bir ürüne benzer ürünleri döner. """
        self._check_init()

        sem_similar = self.sem.get_similar_by_product(product_id, top_k=50)
        cbf_similar = self.cbf.get_similar_items(
            product_id, top_k=50,
            same_category=True,
            products_df=self.ds.products_df,
        )

        sem_dict = dict(sem_similar)
        cbf_dict = dict(cbf_similar)
        all_ids = set(sem_dict) | set(cbf_dict)
        combined = {}
        for pid in all_ids:
            if pid == product_id:
                continue
            combined[pid] = 0.6 * sem_dict.get(pid, 0.0) + 0.4 * cbf_dict.get(pid, 0.0)

        sorted_ids = sorted(combined, key=combined.get, reverse=True)[:k]

        results = []
        for pid in sorted_ids:
            card = self._build_product_card(pid)
            if card:
                results.append(SimilarProduct(product=card, similarity_score=round(combined[pid], 4)))
        return results

    def _build_product_card(self, product_id: str) -> Optional[ProductCard]:
        """Ürün ID'sinden ProductCard oluşturur."""
        p = self.ds.get_product(product_id)
        if p is None:
            return None

        price_raw = p.get("price", "")
        price_tl = format_price_tl(price_raw) if price_raw else "Fiyat belirtilmemiş"

        # Görsel: Amazon URL veya placeholder
        image_url = p.get("image_url", "") or p.get("image", "") or ""

        return ProductCard(
            product_id=product_id,
            title=clean_product_title(clean_display_value(p.get("title", product_id), max_chars=220), max_words=12),
            brand=clean_display_value(p.get("brand", ""), max_chars=120),
            category=clean_display_value(p.get("category", ""), max_chars=80),
            price_tl=price_tl,
            color=clean_display_value(p.get("color", ""), max_chars=60),
            material=clean_display_value(p.get("material", ""), max_chars=80),
            style=clean_display_value(p.get("style", ""), max_chars=80),
            usage=clean_display_value(p.get("usage", ""), max_chars=120),
            description=clean_display_value(p.get("description", ""), max_chars=420),
            image_url=image_url,
        )

    def get_user_profile_data(self, user_id: str):
        """Kullanıcı profil analizini döner."""
        return build_user_profile(
            user_id=user_id,
            interactions_df=self.ds.interactions_df,
            products_df=self.ds.products_df,
            users_df=self.ds.users_df,
        )

    def _check_init(self) -> None:
        if not self._initialized:
            raise RuntimeError("Engine.initialize() çağrılmadan öneri üretilemez.")

    @property
    def avg_latency(self) -> dict[str, float]:
        result = {}
        for k, vals in self.latency_log.items():
            result[k] = round(sum(vals) / len(vals), 2) if vals else 0.0
        return result
