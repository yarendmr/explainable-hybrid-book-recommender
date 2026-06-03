"""
Kitap Embedding Oluşturucu.
İşlenmiş ürün kataloğunu okur ve her kitap için çok dilli sentence-transformer embedding'i hesaplar.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED = BASE_DIR / "data" / "processed"
EMB_DIR = BASE_DIR / "data" / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)


def build_embeddings(batch_size: int = 32, model_name: str | None = None) -> None:
    products_path = PROCESSED / "products.csv"
    if not products_path.exists():
        raise FileNotFoundError(
            f"Kitap verisi bulunamadı: {products_path}\n"
            "Önce scripts/download_amazon_subset.py çalıştırın."
        )

    products_df = pd.read_csv(products_path, dtype=str, encoding="utf-8-sig").fillna("")
    logger.info(f"Kitap sayısı: {len(products_df)}")

    if "product_text" in products_df.columns and products_df["product_text"].str.len().mean() > 10:
        texts = products_df["product_text"].astype(str).tolist()
    else:
        logger.info("product_text sütunu yok/yetersiz, metin oluşturuluyor...")
        texts = []
        for _, row in products_df.iterrows():
            parts = [
                str(row.get("title", "")),
                str(row.get("description", ""))[:500],
                str(row.get("category", "")),
                str(row.get("brand", "")),
                str(row.get("style", "")),
                str(row.get("usage", "")),
            ]
            texts.append(" | ".join(p for p in parts if p.strip()))

    product_ids = products_df["product_id"].astype(str).tolist()

    model_name = model_name or settings.embedding_model_name

    logger.info(f"Embedding modeli yükleniyor: {model_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    logger.info(f"Embedding hesaplanıyor (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    logger.info(f"Embedding matrisi: {embeddings.shape}")

    np.save(EMB_DIR / "product_embeddings.npy", embeddings)
    with open(EMB_DIR / "product_ids.txt", "w", encoding="utf-8") as f:
        for pid in product_ids:
            f.write(str(pid) + "\n")

    logger.info(f"Kaydedildi: {EMB_DIR / 'product_embeddings.npy'}")
    logger.info(f"Kaydedildi: {EMB_DIR / 'product_ids.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Ürün embedding matrisi oluştur")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    build_embeddings(batch_size=args.batch_size, model_name=args.model)


if __name__ == "__main__":
    main()
