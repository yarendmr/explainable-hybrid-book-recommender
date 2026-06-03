"""
Amazon Reviews 2023 Books alt kümesini indirip işler.

Ürün, etkileşim ve kullanıcı verilerini öneri sistemi için CSV formatında hazırlar.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from recommender.text_processing import build_product_text, infer_book_category

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
PROCESSED = BASE_DIR / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

CATEGORY_ALIAS = {
    "Beauty": "All_Beauty",
    "All_Beauty": "All_Beauty",
    "Books": "Books",
    "Clothing": "Clothing_Shoes_and_Jewelry",
    "Clothing_Shoes_and_Jewelry": "Clothing_Shoes_and_Jewelry",
    "Electronics": "Electronics",
    "Home": "Home_and_Kitchen",
    "Home_and_Kitchen": "Home_and_Kitchen",
}

FIRST_NAMES = [
    "Ayşe", "Mehmet", "Zeynep", "Emre", "Elif", "Can", "Deniz", "Merve",
    "Yaren", "Serhat", "Ece", "Kerem", "Selin", "Burak", "Hatice", "Ali"
]
LASTS = ["A.", "D.", "K.", "Y.", "S.", "E.", "T.", "B.", "M."]


def canonical_category(category: str) -> str:
    return CATEGORY_ALIAS.get(category, category)


def load_stream(config: str):
    from datasets import load_dataset

    token = settings.hf_token.strip() or None
    logger.info(f"Hugging Face streaming yükleniyor: {config}")
    return load_dataset(
        DATASET_NAME,
        config,
        split="full",
        streaming=True,
        trust_remote_code=True,
        token=token,
    )


def safe_text(value: Any, max_chars: int = 1200) -> str:
    """Metadata değerlerini düz ve okunabilir metne dönüştürür."""
    if value is None:
        return ""
    if hasattr(value, "tolist") and not isinstance(value, (str, dict)):
        try:
            return safe_text(value.tolist(), max_chars=max_chars)
        except Exception:
            return ""
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return ""
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "nan", "null", "{}", "[]"}:
            return ""
        return text[:max_chars]
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            t = safe_text(item, max_chars=250)
            if t and not t.startswith("http"):
                parts.append(t)
            if len(" ".join(parts)) >= max_chars:
                break
        return " ".join(parts)[:max_chars]
    if isinstance(value, dict):
        # Yazar/store gibi alanlar için önce en anlamlı isim alanı seçilir.
        for key in ["name", "title", "author", "store", "brand"]:
            if key in value:
                t = safe_text(value.get(key), max_chars=250)
                if t and not t.startswith("http"):
                    return t[:max_chars]
        # Details gibi dict'lerde kısa anahtar-değer çiftleri okunabilir hale getirilir.
        parts = []
        for k, v in value.items():
            if str(k).lower() in {"avatar", "image", "images", "url", "about", "videos"}:
                continue
            t = safe_text(v, max_chars=120)
            if t and not t.startswith("http"):
                parts.append(f"{k}: {t}")
            if len("; ".join(parts)) >= max_chars:
                break
        return "; ".join(parts)[:max_chars]
    return str(value)[:max_chars]


def clean_name_field(value: Any) -> str:
    """Author/store alanı için yalnızca görünen isim döndürür."""
    if isinstance(value, dict):
        for key in ["name", "author", "store", "brand", "title"]:
            if key in value:
                name = safe_text(value.get(key), max_chars=160)
                if name:
                    return name
    if isinstance(value, (list, tuple, set)):
        for item in value:
            name = clean_name_field(item)
            if name:
                return name
    return safe_text(value, max_chars=160)


def parse_price(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        text = str(value).replace("$", "").replace(",", "").strip()
        if text.lower() in {"", "none", "nan", "null"}:
            return 0.0
        return float(text)
    except Exception:
        return 0.0


def extract_image_url(value: Any) -> str:
    """Amazon images alanı string/dict/list/array formatlarında gelebilir."""
    if value is None:
        return ""

    if hasattr(value, "tolist") and not isinstance(value, (str, dict)):
        try:
            return extract_image_url(value.tolist())
        except Exception:
            return ""

    if isinstance(value, str):
        value = value.strip()
        return value if value.startswith(("http://", "https://")) else ""

    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return ""
    except Exception:
        pass

    if isinstance(value, dict):
        for key in ["hi_res", "hiRes", "large", "medium", "thumb", "small", "url"]:
            if key in value:
                url = extract_image_url(value.get(key))
                if url:
                    return url
        for v in value.values():
            url = extract_image_url(v)
            if url:
                return url
        return ""

    if isinstance(value, (list, tuple, set)):
        for item in value:
            url = extract_image_url(item)
            if url:
                return url
        return ""

    return ""


def review_to_row(ex: dict[str, Any]) -> dict[str, Any] | None:
    user_id = ex.get("user_id")
    product_id = ex.get("parent_asin") or ex.get("asin")
    rating = ex.get("rating")
    if not user_id or not product_id or rating is None:
        return None
    try:
        rating = float(rating)
    except Exception:
        rating = 3.0
    return {
        "user_id": str(user_id),
        "product_id": str(product_id),
        "rating": rating,
        "timestamp": int(ex.get("timestamp") or 0),
        "review_title": safe_text(ex.get("title")),
        "review_text": safe_text(ex.get("text"))[:500],
    }


def collect_reviews(category: str, stream_limit: int) -> pd.DataFrame:
    config = f"raw_review_{canonical_category(category)}"
    rows = []
    for i, ex in enumerate(load_stream(config)):
        if i >= stream_limit:
            break
        row = review_to_row(ex)
        if row:
            rows.append(row)
        if (i + 1) % 50000 == 0:
            logger.info(f"Review tarandı: {i+1}, geçerli: {len(rows)}")
    df = pd.DataFrame(rows)
    logger.info(f"Toplanan review etkileşimi: {len(df)}")
    return df


def k_core_filter(df: pd.DataFrame, min_user: int, min_item: int, max_iter: int = 20) -> pd.DataFrame:
    out = df.copy()
    for _ in range(max_iter):
        before = len(out)
        item_counts = out["product_id"].value_counts()
        out = out[out["product_id"].isin(item_counts[item_counts >= min_item].index)]
        user_counts = out["user_id"].value_counts()
        out = out[out["user_id"].isin(user_counts[user_counts >= min_user].index)]
        if len(out) == before or out.empty:
            break
    return out.reset_index(drop=True)


def metadata_to_product(ex: dict[str, Any], interaction_counts: Counter) -> dict[str, Any] | None:
    product_id = ex.get("parent_asin") or ex.get("asin")
    title = safe_text(ex.get("title"))
    if not product_id or not title or len(title) < 2:
        return None

    description = safe_text(ex.get("description"))
    features = safe_text(ex.get("features"))
    categories_text = safe_text(ex.get("categories"))
    details = safe_text(ex.get("details"))
    author = clean_name_field(ex.get("author"))
    store = clean_name_field(ex.get("store"))
    main_category = safe_text(ex.get("main_category")) or "Books"

    full_text = " ".join([title, description, features, categories_text, details, author, store, main_category])
    genre = infer_book_category(full_text)

    product = {
        "product_id": str(product_id),
        "title": title,
        "description": (description or features or details)[:800],
        "category": genre,
        "brand": author or store,
        "author": author,
        "price": parse_price(ex.get("price")),
        "average_rating": ex.get("average_rating") or "",
        "rating_number": ex.get("rating_number") or interaction_counts.get(str(product_id), 0),
        "features": features,
        "details": details,
        "main_category": main_category,
        "color": "",
        "material": "Kitap",
        "style": genre,
        "usage": "Okuma / kitap önerisi",
        "image_url": extract_image_url(ex.get("images")),
    }
    product["product_text"] = build_product_text(product)
    return product


def collect_metadata(category: str, candidate_ids: set[str], max_products: int, meta_limit: int, interaction_counts: Counter) -> pd.DataFrame:
    config = f"raw_meta_{canonical_category(category)}"
    products = []
    found = set()
    for i, ex in enumerate(load_stream(config)):
        if i >= meta_limit:
            break
        pid = str(ex.get("parent_asin") or ex.get("asin") or "")
        if pid not in candidate_ids or pid in found:
            continue
        product = metadata_to_product(ex, interaction_counts)
        if product:
            products.append(product)
            found.add(pid)
        if len(products) >= max_products:
            break
        if (i + 1) % 200000 == 0:
            logger.info(f"Metadata tarandı: {i+1}, bulunan ürün: {len(products)}")
    df = pd.DataFrame(products)
    logger.info(f"Metadata ile eşleşen ürün: {len(df)}")
    return df


def build_users(interactions: pd.DataFrame) -> pd.DataFrame:
    user_ids = sorted(interactions["user_id"].astype(str).unique())
    random.seed(42)
    rows = []
    for uid in user_ids:
        rows.append({
            "user_id": uid,
            "display_name": f"{random.choice(FIRST_NAMES)} {random.choice(LASTS)}"
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Amazon Reviews 2023 Books alt kümesi indir")
    parser.add_argument("--category", default=settings.amazon_category)
    parser.add_argument("--max-products", type=int, default=settings.amazon_max_products)
    parser.add_argument("--max-interactions", type=int, default=settings.amazon_max_interactions)
    parser.add_argument("--min-user-interactions", type=int, default=settings.amazon_min_user_interactions)
    parser.add_argument("--min-item-interactions", type=int, default=settings.amazon_min_item_interactions)
    parser.add_argument("--review-stream-limit", type=int, default=settings.amazon_review_stream_limit)
    parser.add_argument("--meta-stream-limit", type=int, default=settings.amazon_meta_stream_limit)
    args = parser.parse_args()

    logger.info(f"Kategori: {args.category} → {canonical_category(args.category)}")
    logger.info(f"Maks ürün: {args.max_products}, maks etkileşim: {args.max_interactions}")
    logger.info(f"Streaming review limit: {args.review_stream_limit}, metadata limit: {args.meta_stream_limit}")

    reviews = collect_reviews(args.category, args.review_stream_limit)
    if reviews.empty:
        raise ValueError("Review verisi toplanamadı. Kategori/config veya HF erişimini kontrol edin.")

    filtered = k_core_filter(reviews, args.min_user_interactions, args.min_item_interactions)
    if filtered.empty:
        logger.warning("3-core filtre boş kaldı; 2-core deneniyor.")
        filtered = k_core_filter(reviews, 2, 2)
    if filtered.empty:
        logger.warning("2-core filtre boş kaldı; 1-core deneniyor.")
        filtered = k_core_filter(reviews, 1, 1)
    if filtered.empty:
        raise ValueError("Filtrelerden sonra etkileşim kalmadı. Stream limitini artırın.")

    logger.info(f"K-core sonrası etkileşim: {len(filtered)}")
    logger.info(f"K-core sonrası kullanıcı: {filtered['user_id'].nunique()}")
    logger.info(f"K-core sonrası ürün: {filtered['product_id'].nunique()}")

    counts = Counter(filtered["product_id"].astype(str))
    candidate_ids = {pid for pid, _ in counts.most_common(args.max_products * 5)}
    products = collect_metadata(args.category, candidate_ids, args.max_products, args.meta_stream_limit, counts)
    if products.empty:
        raise ValueError("Metadata eşleşmesi bulunamadı. Meta stream limitini artırın.")

    valid_ids = set(products["product_id"].astype(str))
    filtered = filtered[filtered["product_id"].astype(str).isin(valid_ids)].copy()
    filtered = k_core_filter(filtered, max(1, min(args.min_user_interactions, 2)), max(1, min(args.min_item_interactions, 2)))
    if filtered.empty:
        raise ValueError("Metadata eşleşmesi sonrası etkileşim kalmadı. max-products/meta-limit artırın.")

    if len(filtered) > args.max_interactions:
        filtered = filtered.sort_values("timestamp", ascending=False).head(args.max_interactions).copy()

    final_ids = set(filtered["product_id"].astype(str))
    products = products[products["product_id"].astype(str).isin(final_ids)].copy()
    users = build_users(filtered)
    interactions = filtered.merge(users, on="user_id", how="left")
    interactions["event_type"] = "review"
    interactions = interactions[["user_id", "display_name", "product_id", "rating", "event_type", "timestamp", "review_title", "review_text"]]
    interactions = interactions.rename(columns={"display_name": "user_name"})

    PROCESSED.mkdir(parents=True, exist_ok=True)
    products.to_csv(PROCESSED / "products.csv", index=False, encoding="utf-8-sig")
    interactions.to_csv(PROCESSED / "interactions.csv", index=False, encoding="utf-8-sig")
    users.to_csv(PROCESSED / "users.csv", index=False, encoding="utf-8-sig")

    logger.info("Amazon Books alt kümesi kaydedildi:")
    logger.info(f"  Ürün: {len(products)}")
    logger.info(f"  Etkileşim: {len(interactions)}")
    logger.info(f"  Kullanıcı: {len(users)}")
    logger.info("Sonraki adım: python scripts/build_embeddings.py")


if __name__ == "__main__":
    main()
