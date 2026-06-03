

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.schemas import (
    CompareRequest,
    CompareResponse,
    ExplainRequest,
    RecommendationResponse,
    SearchResponse,
    SystemMetrics,
    UserProfile,
    ProductCard,
)
from evaluation.metrics import evaluate_recommender
from recommender.text_processing import expand_query, clean_display_value

logger = logging.getLogger(__name__)
router = APIRouter()
def get_engine(request: Request):
    engine = request.app.state.engine
    if engine is None or not engine._initialized:
        raise HTTPException(
            status_code=503,
            detail="Öneri motoru henüz hazır değil. Lütfen birkaç saniye bekleyip tekrar deneyin.",
        )
    return engine


def get_data_store(request: Request):
    ds = request.app.state.data_store
    if ds is None or not ds.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Veri deposu yüklenmedi.",
        )
    return ds

# Kullanıcılar
@router.get("/users", tags=["kullanıcı"], summary="Tüm kullanıcıları listele")
async def get_users(request: Request):
    try:
        ds = get_data_store(request)
        users = ds.users_df[["user_id", "display_name"]].to_dict(orient="records")
        return {"users": users, "count": len(users)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/users hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Kullanıcı listesi alınamadı: {str(e)}")


@router.get("/users/{user_id}/profile", response_model=UserProfile, tags=["kullanıcı"])
async def get_user_profile(user_id: str, request: Request):
    try:
        engine = get_engine(request)
        profile = engine.get_user_profile_data(user_id)
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/users/{user_id}/profile hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Profil alınamadı: {str(e)}")


# Öneri

@router.get("/recommend", response_model=RecommendationResponse, tags=["öneri"])
async def recommend(
    request: Request,
    user_id: str = Query(..., description="Kullanıcı ID"),
    k: int = Query(default=8, ge=1, le=20, description="Öneri sayısı"),
    preferences: str = Query(default="", description="Opsiyonel doğal dil sorgusu"),
):

    try:
        engine = get_engine(request)
        recs = engine.recommend_for_user(user_id=user_id, k=k, query=preferences)
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recs,
            mode="personalized",
            latency_ms=round(engine.avg_latency.get("recommendation", 0.0), 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/recommend hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Öneri üretilemedi: {str(e)}")


@router.get("/recommend/new-user", response_model=RecommendationResponse, tags=["öneri"])
async def recommend_new_user(
    request: Request,
    name: str = Query(default="Misafir", description="Kullanıcı adı"),
    preferences: str = Query(..., description="Doğal dil tercih ifadesi"),
    k: int = Query(default=8, ge=1, le=20),
):

    try:
        engine = get_engine(request)
        recs = engine.recommend_cold_start(preferences=preferences, k=k)
        return RecommendationResponse(
            user_id=f"new_{name}",
            recommendations=recs,
            mode="cold_start",
            latency_ms=round(engine.avg_latency.get("recommendation", 0.0), 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/recommend/new-user hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cold-start öneri üretilemedi: {str(e)}")

# Semantik Arama

@router.get("/search", response_model=SearchResponse, tags=["arama"])
async def search(
    request: Request,
    q: str = Query(..., description="Doğal dil arama sorgusu"),
    k: int = Query(default=8, ge=1, le=20),
):

    try:
        engine = get_engine(request)
        results = engine.search(query=q, k=k)
        return SearchResponse(
            query=q,
            results=results,
            latency_ms=round(engine.avg_latency.get("search", 0.0), 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/search hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Arama başarısız: {str(e)}")

@router.get("/similar/{product_id}", tags=["ürün"])
async def get_similar(
    product_id: str,
    request: Request,
    k: int = Query(default=6, ge=1, le=12),
):

    try:
        engine = get_engine(request)
        similar = engine.get_similar(product_id=product_id, k=k)
        return {"product_id": product_id, "similar": similar, "count": len(similar)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/similar/{product_id} hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benzer ürünler alınamadı: {str(e)}")


@router.get("/products/{product_id}", response_model=ProductCard, tags=["ürün"])
async def get_product(product_id: str, request: Request):
    """Tek bir ürünün detaylarını döner."""
    try:
        engine = get_engine(request)
        card = engine._build_product_card(product_id)
        if card is None:
            raise HTTPException(status_code=404, detail=f"Ürün bulunamadı: {product_id}")
        return card
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/products", tags=["ürün"])
async def list_products(
    request: Request,
    q: str = Query(default="", description="Başlığa göre filtrele"),
    limit: int = Query(default=20, ge=1, le=100),
):
    try:
        ds = get_data_store(request)
        df = ds.products_df.copy()
        if q:
            q_lower = q.lower().strip()
            expanded_terms = [t.lower() for t in expand_query(q).split() if len(t) >= 3]
            searchable_cols = [c for c in ["title", "category", "brand", "author", "description", "product_text", "style"] if c in df.columns]
            mask = False
            for col in searchable_cols:
                col_text = df[col].astype(str).str.lower()
                current = col_text.str.contains(q_lower, na=False, regex=False)
                for term in expanded_terms[:12]:
                    current = current | col_text.str.contains(term, na=False, regex=False)
                mask = current if isinstance(mask, bool) else (mask | current)
            df = df[mask]

            if df.empty:
                engine = get_engine(request)
                sem_results = engine.search(query=q, k=limit)
                products = [r.product.dict() for r in sem_results]
                return {"products": products, "count": len(products)}

        df = df.head(limit).copy()
        for col in ["title", "category", "brand"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: clean_display_value(v, max_chars=140))
        products = df[["product_id", "title", "category", "brand"]].fillna("").to_dict(orient="records")
        return {"products": products, "count": len(products)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", tags=["açıklama"])
async def explain(payload: ExplainRequest, request: Request):
    try:
        from app.services.explainer import llm_explainer
        explanation = llm_explainer.explain(
            product_title=payload.product_title,
            product_description=payload.product_description,
            product_category=payload.product_category,
            user_query=payload.user_query,
            dominant_signal=payload.dominant_signal,
        )
        return {"product_id": payload.product_id, "explanation": explanation}
    except Exception as e:
        logger.error(f"/explain hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Açıklama üretilemedi: {str(e)}")

@router.post("/compare", response_model=CompareResponse, tags=["karşılaştırma"])
async def compare_products(payload: CompareRequest, request: Request):
    try:
        engine = get_engine(request)
        ds = get_data_store(request)

        # Ürün kartlarını al
        cards = []
        for pid in payload.product_ids:
            card = engine._build_product_card(pid)
            if card is None:
                raise HTTPException(status_code=404, detail=f"Ürün bulunamadı: {pid}")
            cards.append(card)

        # Karşılaştırma tablosu
        fields = ["title", "category", "brand", "price_tl", "color", "material", "style", "usage"]
        field_labels = {
            "title": "Ürün Adı", "category": "Kategori", "brand": "Marka",
            "price_tl": "Fiyat", "color": "Renk", "material": "Materyal",
            "style": "Stil", "usage": "Kullanım Amacı",
        }

        table = []
        for field in fields:
            row = {"özellik": field_labels.get(field, field)}
            for i, card in enumerate(cards):
                row[f"ürün_{i+1}"] = getattr(card, field, "") or "—"
            table.append(row)

        # LLM yorumu
        from app.services.explainer import llm_explainer
        products_for_llm = [c.dict() for c in cards]
        comment = llm_explainer.compare(
            products=products_for_llm,
            usage_context=payload.usage_context,
        )

        return CompareResponse(
            products=cards,
            comparison_table=table,
            llm_comment=comment,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/compare hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Karşılaştırma yapılamadı: {str(e)}")

@router.get("/metrics/system", response_model=SystemMetrics, tags=["metrikler"])
async def system_metrics(request: Request):
    """Sistem ve model bilgilerini döner."""
    try:
        ds = get_data_store(request)
        engine = request.app.state.engine

        weights = {
            "collaborative_filtering": settings.weight_collaborative,
            "content_based": settings.weight_content,
            "semantic_similarity": settings.weight_semantic,
            "popularity": settings.weight_popularity,
            "category_intent": settings.weight_category_intent,
        }

        return SystemMetrics(
            product_count=len(ds.products_df),
            user_count=len(ds.users_df),
            interaction_count=len(ds.interactions_df),
            data_mode=settings.data_mode,
            embedding_model=settings.embedding_model_name,
            llm_model=settings.llm_model_name,
            methods_used=[
                "Item-Based Collaborative Filtering",
                "TF-IDF Content-Based Filtering",
                "Semantic Embedding Retrieval (Sentence-Transformers)",
                "Popularity Scoring",
                "Category Intent Detection (Query Expansion)",
                "RAG-Grounded LLM Explanation",
            ],
            hybrid_weights=weights,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/latency", tags=["metrikler"])
async def latency_metrics(request: Request):
    """Öneri ve arama gecikme istatistiklerini döner."""
    try:
        engine = get_engine(request)
        return {
            "avg_recommendation_latency_ms": engine.avg_latency.get("recommendation", 0.0),
            "avg_search_latency_ms": engine.avg_latency.get("search", 0.0),
            "recommendation_requests": len(engine.latency_log.get("recommendation", [])),
            "search_requests": len(engine.latency_log.get("search", [])),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/evaluation", tags=["metrikler"])
async def evaluation_metrics(
    request: Request,
    k: int = Query(default=10, ge=1, le=20),
    max_users: int = Query(default=50, ge=5, le=200),
):
    try:
        engine = get_engine(request)
        ds = get_data_store(request)

        results = evaluate_recommender(
            engine=engine,
            interactions_df=ds.interactions_df,
            k=k,
            max_users=max_users,
        )
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/metrics/evaluation hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Değerlendirme başarısız: {str(e)}")


@router.get("/model-info", tags=["metrikler"])
async def model_info():
    """Kullanılan modeller hakkında bilgi döner."""
    return {
        "llm_modeli": {
            "ad": settings.llm_model_name,
            "amaç": "RAG-grounded öneri açıklaması üretme",
            "kullanım": "Yalnızca retrieve edilmiş ürünler için doğal Türkçe açıklama",
        },
        "embedding_modeli": {
            "ad": settings.embedding_model_name,
            "amaç": "Semantik ürün arama ve benzerlik hesaplama",
            "dil_desteği": "Çok dilli (Türkçe + İngilizce)",
        },
        "hibrit_formül": {
            "kişiselleştirilmiş": "0.25×CF + 0.20×CBF + 0.35×SEM + 0.10×POP + 0.10×CAT",
            "cold_start": "0.70×SEM + 0.20×CAT + 0.10×POP",
        },
    }
