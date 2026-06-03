"""
TF-IDF tabanlı içerik filtreleme bileşeni.

Kitap başlığı, açıklaması, kategorisi ve yazar/yayıncı bilgisi birleştirilerek
metinsel kitap temsili oluşturulur. Kullanıcının geçmiş etkileşimlerinden
rating ağırlıklı bir profil vektörü üretilir ve aday kitaplarla kosinüs
benzerliği hesaplanır.
TF-IDF matrisi bellek verimliliği için sparse CSR formatında tutulur.
"""

import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContentBasedFilter:
    """Kitap metadata’sı üzerinden TF-IDF tabanlı içerik benzerliği hesaplar."""
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: csr_matrix | None = None
        self.product_ids: list[str] = []
        self.product_index: dict[str, int] = {}
        self._fitted: bool = False

    def fit(self, products_df: pd.DataFrame) -> None:
        """Kitap kataloğundan TF-IDF matrisi oluşturur."""
        if products_df.empty:
            logger.warning("Kitap verisi boş, CBF eğitilemiyor.")
            return

        self.product_ids = products_df["product_id"].astype(str).tolist()
        self.product_index = {pid: i for i, pid in enumerate(self.product_ids)}

        texts = products_df.get("product_text", None)
        if texts is None or texts.empty:
            texts = self._build_texts(products_df)
        texts = texts.fillna("").astype(str).tolist()

        self.vectorizer = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,          # uzun metin etkisini azaltır
            strip_accents="unicode",
        )
        raw_matrix = self.vectorizer.fit_transform(texts)
        self.tfidf_matrix = raw_matrix.tocsr()

        self._fitted = True
        logger.info(
            f"CBF eğitildi — "
            f"{self.tfidf_matrix.shape[0]} kitap, "
            f"{self.tfidf_matrix.shape[1]} TF-IDF terimi"
        )

    @staticmethod
    def _build_texts(df: pd.DataFrame) -> pd.Series:
        """Kitap metadata alanlarından birleşik metin oluşturur."""
        parts = []
        for col in ["title", "description", "category", "brand", "style", "usage", "material"]:
            if col in df.columns:
                parts.append(df[col].fillna(""))
        if not parts:
            return pd.Series([""] * len(df))
        return pd.DataFrame(parts).T.apply(lambda row: " ".join(row.values), axis=1)

    def get_user_profile(
        self, user_id: str, interactions_df: pd.DataFrame
    ) -> csr_matrix | None:
        """ Kullanıcının geçmiş etkileşimlerinden rating ağırlıklı içerik profil vektörü oluşturur. """
        if not self._fitted or self.tfidf_matrix is None:
            return None

        user_df = interactions_df[interactions_df["user_id"] == user_id]
        if user_df.empty:
            return None

        idxs: list[int] = []
        weights: list[float] = []
        for _, row in user_df.iterrows():
            pid = str(row["product_id"])
            idx = self.product_index.get(pid)
            if idx is not None:
                idxs.append(idx)
                weights.append(float(row.get("rating", 3.0)))

        if not idxs:
            return None

        weights_arr = np.array(weights, dtype=float)
        weight_sum = weights_arr.sum()
        if weight_sum <= 0:
            weights_arr = np.ones_like(weights_arr) / len(weights_arr)
        else:
            weights_arr = weights_arr / weight_sum

        # Sparse ağırlıklı ortalama; sonuç CSR formatına çevrilir.
        profile = self.tfidf_matrix[idxs].multiply(weights_arr[:, None]).sum(axis=0)
        return csr_matrix(profile)

    def get_content_scores(
        self,
        user_profile,
        candidate_ids: list[str],
    ) -> dict[str, float]:
        """Kullanıcı profil vektörü ile aday kitaplar arasındaki içerik benzerliğini hesaplar."""
        if not self._fitted or user_profile is None or self.tfidf_matrix is None:
            return {pid: 0.0 for pid in candidate_ids}

        valid: list[tuple[str, int]] = [
            (pid, self.product_index[pid]) for pid in candidate_ids if pid in self.product_index
        ]
        scores = {pid: 0.0 for pid in candidate_ids}
        if not valid:
            return scores

        pids, idxs = zip(*valid)
        item_matrix = self.tfidf_matrix[list(idxs)]
        profile_vec = user_profile if issparse(user_profile) else csr_matrix(user_profile)
        sims = cosine_similarity(profile_vec, item_matrix)[0]
        for pid, sim in zip(pids, sims):
            scores[pid] = float(sim)
        return scores

    def get_similar_items(
        self, product_id: str, top_k: int = 6, same_category: bool = True,
        products_df: pd.DataFrame | None = None,
    ) -> list[tuple[str, float]]:
        """Kullanıcı profil vektörü ile aday kitaplar arasındaki içerik benzerliğini hesaplar."""
        if not self._fitted or self.tfidf_matrix is None or product_id not in self.product_index:
            return []

        idx = self.product_index[product_id]
        item_vec = self.tfidf_matrix[idx]
        all_sims = cosine_similarity(item_vec, self.tfidf_matrix)[0]

        scored = [
            (self.product_ids[i], float(all_sims[i]))
            for i in range(len(self.product_ids))
            if self.product_ids[i] != product_id
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        if same_category and products_df is not None and not products_df.empty:
            target = products_df[products_df["product_id"] == product_id]
            if not target.empty:
                target_cat = target.iloc[0].get("category", "")
                same_cat = [(p, s) for p, s in scored if _get_category(p, products_df) == target_cat]
                other_cat = [(p, s) for p, s in scored if _get_category(p, products_df) != target_cat]
                scored = same_cat[:top_k] + other_cat[:max(0, top_k - len(same_cat))]

        return scored[:top_k]

    def query_to_vector(self, query: str):
        """Serbest metin sorgusunu TF-IDF vektörüne dönüştürür."""
        if not self._fitted or self.vectorizer is None:
            return None
        try:
            return self.vectorizer.transform([query]).tocsr()
        except Exception:
            return None

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def _get_category(product_id: str, products_df: pd.DataFrame) -> str:
    row = products_df[products_df["product_id"] == product_id]
    if row.empty:
        return ""
    return str(row.iloc[0].get("category", ""))
