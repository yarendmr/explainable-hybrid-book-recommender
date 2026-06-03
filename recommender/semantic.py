"""
Semantik geri çağırma modülü.
Ürün ve sorgu metinlerini embedding uzayında temsil ederek semantik benzerlik
skorları üretir.
"""

import logging
import numpy as np
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Sentence-Transformer tabanlı semantik geri çağırma motoru."""
    def __init__(self):
        self._model = None
        self._model_loaded: bool = False
        self.embeddings: Optional[np.ndarray] = None
        self.product_ids: list[str] = []

    def load_model(self) -> bool:
        """Embedding modelini yükler."""
        if self._model_loaded:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            model_name = settings.embedding_model_name
            logger.info(f"Embedding modeli yükleniyor: {model_name}")
            self._model = SentenceTransformer(model_name)
            self._model_loaded = True
            logger.info("Embedding modeli yüklendi.")
            return True
        except Exception as exc:
            logger.error(f"Embedding modeli yüklenemedi: {exc}")
            return False

    def load_embeddings(self, embeddings: np.ndarray, product_ids: list[str]) -> None:
        """Önceden hesaplanmış embedding matrisini yükler."""
        self.embeddings = embeddings
        self.product_ids = product_ids
        logger.info(f"Embedding matrisi yüklendi: {embeddings.shape}")

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        """ Kullanıcı sorgusunu embedding vektörüne dönüştürür. """
        if not self._model_loaded:
            self.load_model()
        if self._model is None:
            return None
        try:
            vec = self._model.encode([query], normalize_embeddings=True)
            return vec[0]
        except Exception as exc:
            logger.error(f"Sorgu encoding hatası: {exc}")
            return None

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Metin listesini toplu olarak embedding vektörlerine dönüştürür."""
        if not self._model_loaded:
            self.load_model()
        if self._model is None:
            raise RuntimeError("Embedding modeli yüklenemedi.")
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def get_semantic_scores(
        self,
        query: str,
        candidate_ids: list[str],
    ) -> dict[str, float]:
        """Sorgu ile aday ürünler arasındaki semantik benzerliği hesaplar."""
        if self.embeddings is None or not candidate_ids:
            return {pid: 0.0 for pid in candidate_ids}

        query_vec = self.encode_query(query)
        if query_vec is None:
            return {pid: 0.0 for pid in candidate_ids}

        # Aday indekslerini bul
        id_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}
        scores: dict[str, float] = {}

        for pid in candidate_ids:
            if pid not in id_to_idx:
                scores[pid] = 0.0
                continue
            idx = id_to_idx[pid]
            item_vec = self.embeddings[idx]
            sim = float(np.dot(query_vec, item_vec))
            scores[pid] = max(0.0, sim)

        return scores

    def get_top_semantic_candidates(
        self,
        query: str,
        top_k: int = 50,
        excluded_ids: Optional[set] = None,
    ) -> list[tuple[str, float]]:
        """Sorguya semantik olarak en yakın ürünleri döner."""
        if self.embeddings is None:
            return []

        query_vec = self.encode_query(query)
        if query_vec is None:
            return []

        sims = self.embeddings @ query_vec   # (n_products,)

        excluded = excluded_ids or set()

        scored = [
            (self.product_ids[i], float(sims[i]))
            for i in range(len(self.product_ids))
            if self.product_ids[i] not in excluded
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_similar_by_product(
        self,
        product_id: str,
        top_k: int = 6,
        excluded_ids: Optional[set] = None,
    ) -> list[tuple[str, float]]:
        """Bir ürüne semantik olarak en yakın ürünleri döner."""
        if self.embeddings is None or product_id not in self.product_ids:
            return []

        idx = self.product_ids.index(product_id)
        product_vec = self.embeddings[idx]

        excluded = excluded_ids or {product_id}

        sims = self.embeddings @ product_vec

        scored = [
            (self.product_ids[i], float(sims[i]))
            for i in range(len(self.product_ids))
            if self.product_ids[i] not in excluded
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @property
    def is_ready(self) -> bool:
        return self.embeddings is not None and len(self.product_ids) > 0

semantic_retriever = SemanticRetriever()
