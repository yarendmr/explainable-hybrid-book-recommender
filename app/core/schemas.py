
from pydantic import BaseModel, Field
from typing import Optional

class ProductCard(BaseModel):
    """Kullanıcıya gösterilecek kitap kartı."""
    product_id: str
    title: str
    brand: str = ""
    category: str = ""
    price_tl: str = ""
    color: str = ""
    material: str = ""
    style: str = ""
    usage: str = ""
    description: str = ""
    image_url: str = ""


class RecommendedProduct(BaseModel):
    """Öneri listesindeki her kitap"""
    product: ProductCard
    explanation: str = ""
    hybrid_score: float = 0.0
    dominant_signal: str = ""


class SimilarProduct(BaseModel):
    """Benzer ürün kartı."""
    product: ProductCard
    similarity_score: float = 0.0


class UserProfile(BaseModel):
    user_id: str
    display_name: str
    top_categories: list[str] = []
    avg_price_interest: str = ""
    recent_interactions: list[str] = []
    total_interactions: int = 0

# İstek Modelleri

class ExplainRequest(BaseModel):
    """LLM açıklama isteği."""
    product_id: str
    product_title: str
    product_description: str
    product_category: str
    user_query: str = ""
    dominant_signal: str = "semantic"


class CompareRequest(BaseModel):
    """ Kitap karşılaştırma isteği."""
    product_ids: list[str] = Field(..., min_length=2, max_length=3)
    usage_context: str = ""

# Yanıt Modelleri

class RecommendationResponse(BaseModel):
    """Öneri endpoint yanıtı."""
    user_id: str
    recommendations: list[RecommendedProduct]
    mode: str = "personalized"
    latency_ms: float = 0.0


class SearchResponse(BaseModel):
    """Semantik arama yanıtı."""
    query: str
    results: list[RecommendedProduct]
    latency_ms: float = 0.0


class CompareResponse(BaseModel):
    """Karşılaştırma yanıtı."""
    products: list[ProductCard]
    comparison_table: list[dict]
    llm_comment: str = ""


class SystemMetrics(BaseModel):
    """Sistem değerlendirme metrikleri."""
    # Veri özeti
    product_count: int = 0
    user_count: int = 0
    interaction_count: int = 0
    data_mode: str = ""

    # Model bilgileri
    embedding_model: str = ""
    llm_model: str = ""

    # Öneri yöntemleri
    methods_used: list[str] = []

    # Hibrit skor ağırlıkları
    hybrid_weights: dict = {}

    # Değerlendirme metrikleri
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    hit_rate_at_k: Optional[float] = None
    evaluation_k: int = 10

    # Performans
    avg_recommendation_latency_ms: Optional[float] = None
    avg_search_latency_ms: Optional[float] = None


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
