"""
Top-K değerlendirme metrikleri ve temporal holdout değerlendirme protokolü.

Bu modül Precision@K, Recall@K, NDCG@K ve HitRate@K metriklerini hesaplar.
Değerlendirmede kullanıcıların en güncel 2–3 etkileşimi test kümesine ayrılır
ve öneri motoru yalnızca train etkileşimleriyle yeniden kurulur. Böylece test
öğelerinin CF ve CBF profil hesabına sızması engellenir.
"""

import logging
import math
import pandas as pd

logger = logging.getLogger(__name__)


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Precision@K = (İlk K öneri içindeki gerçek pozitif sayısı) / K
    """
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Recall@K = (İlk K öneri içindeki gerçek pozitif sayısı) / |İlgili öğe sayısı|
    """
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    Sıralama kalitesini ölçer: ilgili öğeler üst sırada yer alırsa
    daha yüksek puan alır.
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    idcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant), k))
    )
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """
    HitRate@K = İlk K'da en az bir gerçek pozitif var mı? (0 veya 1)
    """
    top_k = recommended[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def _temporal_holdout_split(
    interactions_df: pd.DataFrame,
    max_users: int,
    min_interactions: int = 4,
    max_test_items: int = 3,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Temporal train/test split oluşturur.

    Her kullanıcı için son 2–3 etkileşim test olarak ayrılır. Böylece değerlendirme
    sırasında öneri motoru test öğelerini eğitimde görmez.
    """
    train_parts = []
    test_relevant: dict[str, set[str]] = {}

    if interactions_df.empty:
        return interactions_df.copy(), test_relevant

    # Zamanı sayısal ele al; eksik değerler sona/başa kaymasın diye 0 kabul edilir.
    df = interactions_df.copy()
    df["timestamp"] = pd.to_numeric(df.get("timestamp", 0), errors="coerce").fillna(0)

    user_counts = df.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= min_interactions].index.tolist()[:max_users]
    eligible_set = set(eligible_users)

    # Değerlendirilmeyen kullanıcıların tüm etkileşimleri train'de kalır.
    if eligible_set:
        train_parts.append(df[~df["user_id"].isin(eligible_set)].copy())
    else:
        return df.copy(), test_relevant

    for user_id in eligible_users:
        user_df = df[df["user_id"] == user_id].sort_values("timestamp")
        if len(user_df) < min_interactions:
            train_parts.append(user_df)
            continue

        test_size = min(max_test_items, max(2, len(user_df) // 5))
        test_df = user_df.tail(test_size)
        train_df = user_df.iloc[:-test_size]

        if train_df.empty or test_df.empty:
            train_parts.append(user_df)
            continue

        relevant = set(test_df["product_id"].astype(str).tolist())
        if relevant:
            test_relevant[str(user_id)] = relevant
            train_parts.append(train_df)
        else:
            train_parts.append(user_df)

    train_interactions = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    return train_interactions, test_relevant


def _build_train_engine(engine, train_interactions: pd.DataFrame):
    """Mevcut engine ile aynı katalog/embeddingleri kullanarak train-only engine kurar."""
    from app.services.data_store import DataStore

    train_ds = DataStore()
    train_ds.products_df = engine.ds.products_df.copy()
    train_ds.users_df = engine.ds.users_df.copy()
    train_ds.interactions_df = train_interactions.copy()
    train_ds.embeddings = engine.ds.embeddings
    train_ds.product_ids = list(engine.ds.product_ids)
    train_ds._loaded = True

    train_engine = engine.__class__(train_ds, engine.explainer)
    train_engine.initialize()
    return train_engine


def evaluate_recommender(
    engine,
    interactions_df: pd.DataFrame,
    k: int = 10,
    max_users: int = 100,
) -> dict[str, float]:
    """
    Leakage içermeyen temporal holdout protokolüyle öneri motorunu değerlendirir.
    Parameters
    ----------
    engine          : HybridRecommendationEngine
    interactions_df : pd.DataFrame — tüm etkileşimler
    k               : int — değerlendirme K değeri
    max_users       : int — değerlendirilecek maksimum kullanıcı sayısı
    Returns
    -------
    dict : {precision, recall, ndcg, hit_rate}
    """
    if interactions_df.empty:
        logger.warning("Etkileşim verisi boş, değerlendirme yapılamıyor.")
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0, "hit_rate_at_k": 0.0}

    train_df, test_relevant = _temporal_holdout_split(interactions_df, max_users=max_users)
    if not test_relevant:
        logger.warning("Temporal holdout için yeterli etkileşimi olan kullanıcı bulunamadı.")
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0, "hit_rate_at_k": 0.0}

    try:
        eval_engine = _build_train_engine(engine, train_df)
    except Exception as exc:
        logger.warning(f"Train-only değerlendirme motoru oluşturulamadı: {exc}")
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0, "hit_rate_at_k": 0.0}

    precisions, recalls, ndcgs, hit_rates = [], [], [], []

    for user_id, relevant in test_relevant.items():
        try:
            recs = eval_engine.recommend_for_user(user_id, k=k)
            recommended_ids = [r.product.product_id for r in recs]

            precisions.append(precision_at_k(recommended_ids, relevant, k))
            recalls.append(recall_at_k(recommended_ids, relevant, k))
            ndcgs.append(ndcg_at_k(recommended_ids, relevant, k))
            hit_rates.append(hit_rate_at_k(recommended_ids, relevant, k))
        except Exception as exc:
            logger.warning(f"Kullanıcı {user_id} değerlendirme hatası: {exc}")
            continue

    def mean(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "precision_at_k": mean(precisions),
        "recall_at_k":    mean(recalls),
        "ndcg_at_k":      mean(ndcgs),
        "hit_rate_at_k":  mean(hit_rates),
        "evaluated_users": len(precisions),
        "test_items_per_user": "2-3 temporal holdout",
        "k": k,
    }