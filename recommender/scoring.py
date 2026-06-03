"""
Hibrit skorlama modülü.
Aday ürünleri işbirlikçi filtreleme, içerik benzerliği, semantik benzerlik,
popülerlik ve kategori niyeti sinyallerine göre sıralar.
"""

import logging
import numpy as np
import pandas as pd

from app.core.config import settings
from recommender.text_processing import genre_match_score, build_product_text

logger = logging.getLogger(__name__)


def get_personalized_weights() -> dict[str, float]:
    """Kişiselleştirilmiş öneri ağırlıklarını döner."""
    return {
        "collaborative": settings.weight_collaborative,
        "content": settings.weight_content,
        "semantic": settings.weight_semantic,
        "popularity": settings.weight_popularity,
        "category_intent": settings.weight_category_intent,
    }


def get_cold_start_weights() -> dict[str, float]:
    """Cold-start öneri ağırlıklarını döner."""
    return {
        "semantic": settings.weight_cold_semantic,
        "category": settings.weight_cold_category,
        "popularity": settings.weight_cold_popularity,
    }


def compute_popularity_scores(interactions_df: pd.DataFrame) -> dict[str, float]:
    """Etkileşim sayısı ve ortalama puana göre popülerlik skoru hesaplar."""
    if interactions_df.empty:
        return {}

    counts = interactions_df.groupby("product_id").size().rename("count")

    avg_ratings = (
        interactions_df.groupby("product_id")["rating"]
        .mean()
        .rename("avg_rating")
    )

    pop_df = pd.concat([counts, avg_ratings], axis=1).fillna(0)

    max_count = pop_df["count"].max() or 1
    max_rating = pop_df["avg_rating"].max() or 1

    pop_df["score"] = (
        0.70 * (pop_df["count"] / max_count)
        + 0.30 * (pop_df["avg_rating"] / max_rating)
    )

    return pop_df["score"].to_dict()


def compute_category_intent_scores(
    candidate_ids: list[str],
    detected_category: str | None,
    products_df: pd.DataFrame,
) -> dict[str, float]:
    """Aday ürünlerin tespit edilen kategori niyetiyle uyumunu hesaplar."""
    if detected_category is None:
        return {pid: 0.5 for pid in candidate_ids}  # nötr

    lookup = products_df.set_index("product_id", drop=False)
    scores: dict[str, float] = {}
    detected_l = str(detected_category).lower()

    for pid in candidate_ids:
        if pid not in lookup.index:
            scores[pid] = 0.0
            continue

        row = lookup.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        cat = str(row.get("category", "")).strip().lower()
        style = str(row.get("style", "")).strip().lower()

        if detected_l and (detected_l in cat or cat in detected_l or detected_l in style):
            scores[pid] = 1.0
            continue

        text = build_product_text(row.to_dict())
        scores[pid] = genre_match_score(text, detected_category)

    return scores


def hybrid_score(
    product_id: str,
    cf_scores: dict[str, float],
    cbf_scores: dict[str, float],
    sem_scores: dict[str, float],
    pop_scores: dict[str, float],
    cat_scores: dict[str, float],
    mode: str = "personalized",
) -> float:
    """Tek bir ürün için hibrit sıralama skorunu hesaplar."""
    cf = cf_scores.get(product_id, 0.0)
    cbf = cbf_scores.get(product_id, 0.0)
    sem = sem_scores.get(product_id, 0.0)
    pop = pop_scores.get(product_id, 0.0)
    cat = cat_scores.get(product_id, 0.5)

    if mode == "cold_start":
        return (
            settings.weight_cold_semantic * sem
            + settings.weight_cold_category * cat
            + settings.weight_cold_popularity * pop
        )

    return (
        settings.weight_collaborative * cf
        + settings.weight_content * cbf
        + settings.weight_semantic * sem
        + settings.weight_popularity * pop
        + settings.weight_category_intent * cat
    )


def rank_candidates(
    candidate_ids: list[str],
    cf_scores: dict[str, float],
    cbf_scores: dict[str, float],
    sem_scores: dict[str, float],
    pop_scores: dict[str, float],
    cat_scores: dict[str, float],
    mode: str = "personalized",
    top_k: int = 8,
) -> list[tuple[str, float, str]]:
    """Aday ürünleri hibrit skora göre sıralar."""
    results: list[tuple[str, float, str]] = []

    for pid in candidate_ids:
        score = hybrid_score(pid, cf_scores, cbf_scores, sem_scores, pop_scores, cat_scores, mode)
        dominant = _dominant_signal(pid, cf_scores, cbf_scores, sem_scores, pop_scores, cat_scores, mode)
        results.append((pid, score, dominant))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _dominant_signal(
    product_id: str,
    cf_scores: dict,
    cbf_scores: dict,
    sem_scores: dict,
    pop_scores: dict,
    cat_scores: dict,
    mode: str,
) -> str:
    """Ürünü öne çıkaran baskın skorlama sinyalini belirler."""
    if mode == "cold_start":
        components = {
            "semantic": settings.weight_cold_semantic * sem_scores.get(product_id, 0.0),
            "category": settings.weight_cold_category * cat_scores.get(product_id, 0.5),
            "popularity": settings.weight_cold_popularity * pop_scores.get(product_id, 0.0),
        }
    else:
        components = {
            "collaborative": settings.weight_collaborative * cf_scores.get(product_id, 0.0),
            "content": settings.weight_content * cbf_scores.get(product_id, 0.0),
            "semantic": settings.weight_semantic * sem_scores.get(product_id, 0.0),
            "popularity": settings.weight_popularity * pop_scores.get(product_id, 0.0),
            "category": settings.weight_category_intent * cat_scores.get(product_id, 0.5),
        }

    return max(components, key=components.get)
