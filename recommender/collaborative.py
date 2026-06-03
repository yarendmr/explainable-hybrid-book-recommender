"""
İşbirlikçi Filtreleme Modülü — Item-Based Collaborative Filtering.
Kullanıcı–kitap etkileşimlerinden kitap–kitap benzerliği hesaplanır
ve kullanıcının geçmiş etkileşimlerine benzer aday kitaplara CF skoru atanır.
"""

import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
RATING_WEIGHTS = {
    5: 5.0,
    4: 4.0,
    3: 2.5,
    2: 1.0,
    1: 0.5,
}


class CollaborativeFilter:
    """ Kullanıcı–kitap etkileşimlerinden item-based CF skorları hesaplar. """
    def __init__(self):
        self.item_similarity: np.ndarray | None = None
        self.product_index: dict[str, int] = {}
        self.user_item_matrix: pd.DataFrame | None = None
        self._fitted: bool = False

    def fit(self, interactions_df: pd.DataFrame) -> None:
        """Etkileşim verisinden kullanıcı–kitap matrisi ve kitap–kitap benzerliği üretir."""
        if interactions_df.empty:
            logger.warning("Etkileşim verisi boş, CF eğitilemiyor.")
            return

        # Rating ağırlıklarını uygula
        df = interactions_df.copy()
        df["weight"] = df["rating"].apply(
            lambda r: RATING_WEIGHTS.get(int(round(float(r))), float(r))
        )

        # Kullanıcı–Ürün pivot matrisi
        pivot = df.pivot_table(
            index="user_id",
            columns="product_id",
            values="weight",
            aggfunc="max",
            fill_value=0.0,
        )
        self.user_item_matrix = pivot

        # Ürün indeksi oluştur
        self.product_index = {pid: i for i, pid in enumerate(pivot.columns)}

        # Item–item cosine similarity (seyrek matris üzerinde verimli)
        item_matrix = csr_matrix(pivot.values.T)   # (products × users)
        similarity = cosine_similarity(item_matrix, dense_output=True)
        self.item_similarity = similarity

        self._fitted = True
        logger.info(
            f"CF eğitildi — "
            f"{pivot.shape[0]} kullanıcı, "
            f"{pivot.shape[1]} ürün, "
            f"similarity matrisi: {similarity.shape}"
        )

    def get_cf_scores(self, user_id: str, candidate_ids: list[str]) -> dict[str, float]:
        """
        Verilen kullanıcı için aday kitapların CF skorlarını hesaplar.
        Skorlar, kullanıcının geçmiş etkileşimleri ile aday kitapların
        kitap–kitap benzerlikleri üzerinden vektörize biçimde üretilir.
        """
        if not self._fitted or self.user_item_matrix is None or self.item_similarity is None:
            return {pid: 0.0 for pid in candidate_ids}

        if user_id not in self.user_item_matrix.index:
            return {pid: 0.0 for pid in candidate_ids}

        valid = [(pid, self.product_index[pid]) for pid in candidate_ids if pid in self.product_index]
        scores = {pid: 0.0 for pid in candidate_ids}
        if not valid:
            return scores

        pids, idxs = zip(*valid)
        user_ratings = self.user_item_matrix.loc[user_id].values.astype(float)
        interacted_mask = user_ratings > 0
        if not np.any(interacted_mask):
            return scores

        sim_rows = self.item_similarity[list(idxs)]
        weighted_sums = sim_rows @ user_ratings
        sim_sums = np.abs(sim_rows[:, interacted_mask]).sum(axis=1) + 1e-9
        raw_scores = weighted_sums / sim_sums

        max_score = float(np.max(raw_scores)) if len(raw_scores) else 0.0
        if max_score > 0:
            raw_scores = raw_scores / max_score

        for pid, score in zip(pids, raw_scores):
            scores[pid] = float(score)

        return scores

    def get_user_interacted_products(self, user_id: str) -> set[str]:
        """Kullanıcının etkileştiği kitap ID'lerini döndürür."""
        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
            return set()
        row = self.user_item_matrix.loc[user_id]
        return set(row[row > 0].index.tolist())

    def get_similar_items(self, product_id: str, top_k: int = 6) -> list[tuple[str, float]]:
        """ Bir kitaba CF benzerliğine göre en yakın kitapları döndürür. """
        if not self._fitted or product_id not in self.product_index:
            return []

        idx = self.product_index[product_id]
        sim_row = self.item_similarity[idx]

        products = list(self.product_index.keys())
        scored = [
            (products[i], float(sim_row[i]))
            for i in range(len(products))
            if products[i] != product_id
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @property
    def is_fitted(self) -> bool:
        return self._fitted
