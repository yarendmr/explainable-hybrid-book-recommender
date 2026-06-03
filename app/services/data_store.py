
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataStore:
    """
    Merkezi veri deposu.
    
    Attributes
    ----------
    products_df   : pd.DataFrame  — ürün kataloğu
    interactions_df: pd.DataFrame — kullanıcı–ürün etkileşimleri
    users_df      : pd.DataFrame  — kullanıcı tablosu
    embeddings    : np.ndarray    — ürün embedding matrisi (N x D)
    product_ids   : list[str]     — embeddings satırlarına karşılık gelen ürün ID'leri
    """

    def __init__(self):
        self.products_df: pd.DataFrame = pd.DataFrame()
        self.interactions_df: pd.DataFrame = pd.DataFrame()
        self.users_df: pd.DataFrame = pd.DataFrame()
        self.embeddings: np.ndarray | None = None
        self.product_ids: list[str] = []
        self._loaded: bool = False

    # Yükleme

    def load(self) -> None:
        """İşlenmiş CSV ve embedding dosyalarını diskten yükler."""
        processed = settings.processed_dir
        emb_dir = settings.embeddings_dir

        products_path = processed / "products.csv"
        interactions_path = processed / "interactions.csv"
        users_path = processed / "users.csv"
        embeddings_path = emb_dir / "product_embeddings.npy"
        ids_path = emb_dir / "product_ids.txt"

        if not products_path.exists():
            raise FileNotFoundError(
                f"Ürün verisi bulunamadı: {products_path}\n"
                "Lütfen önce generate_sample_data.py veya download_amazon_subset.py çalıştırın."
            )

        logger.info("Veri yükleniyor...")

        self.products_df = pd.read_csv(products_path, dtype=str, encoding="utf-8-sig").fillna("")
        self.interactions_df = pd.read_csv(interactions_path, dtype=str, encoding="utf-8-sig").fillna("")
        self.interactions_df["rating"] = pd.to_numeric(
            self.interactions_df["rating"], errors="coerce"
        ).fillna(3.0)
        self.interactions_df["timestamp"] = pd.to_numeric(
            self.interactions_df["timestamp"], errors="coerce"
        ).fillna(0)

        if users_path.exists():
            self.users_df = pd.read_csv(users_path, dtype=str, encoding="utf-8-sig").fillna("")
        else:
            # Kullanıcı tablosu yoksa etkileşimlerden türet
            self.users_df = self._derive_users()

        if embeddings_path.exists() and ids_path.exists():
            self.embeddings = np.load(str(embeddings_path))
            with open(ids_path, "r", encoding="utf-8") as f:
                self.product_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Embedding yüklendi: {self.embeddings.shape}")
        else:
            logger.warning(
                "Embedding dosyası bulunamadı. Semantik arama devre dışı. "
                "build_embeddings.py çalıştırın."
            )

        self._loaded = True
        logger.info(
            f"Veri yüklendi — "
            f"Ürün: {len(self.products_df)}, "
            f"Kullanıcı: {len(self.users_df)}, "
            f"Etkileşim: {len(self.interactions_df)}"
        )

    def _derive_users(self) -> pd.DataFrame:
        """Etkileşim verisinden kullanıcı tablosu türetir."""
        if self.interactions_df.empty:
            return pd.DataFrame(columns=["user_id", "display_name"])
        user_ids = self.interactions_df["user_id"].unique()
        return pd.DataFrame({
            "user_id": user_ids,
            "display_name": [f"Kullanıcı {i+1}" for i in range(len(user_ids))]
        })

    # Yardımcı erişim metodları

    def get_product(self, product_id: str) -> dict | None:
        """Ürün ID'sine göre ürün sözlüğü döner."""
        if self.products_df.empty:
            return None
        row = self.products_df[self.products_df["product_id"] == product_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_user_interactions(self, user_id: str) -> pd.DataFrame:
        """Kullanıcının tüm etkileşimlerini döner."""
        if self.interactions_df.empty:
            return pd.DataFrame()
        return self.interactions_df[self.interactions_df["user_id"] == user_id].copy()

    def get_embedding(self, product_id: str) -> np.ndarray | None:
        """Ürün embedding vektörünü döner."""
        if self.embeddings is None or product_id not in self.product_ids:
            return None
        idx = self.product_ids.index(product_id)
        return self.embeddings[idx]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def stats(self) -> dict:
        return {
            "product_count": len(self.products_df),
            "user_count": len(self.users_df),
            "interaction_count": len(self.interactions_df),
            "embeddings_available": self.embeddings is not None,
            "data_mode": settings.data_mode,
        }

data_store = DataStore()
