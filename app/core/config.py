
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # Model ayarları
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # LLM_PROVIDER: "huggingface" veya "template"
    # Projede hedef kullanım: huggingface.
    # Test/çok düşük donanım durumunda template seçilebilir.
    llm_provider: str = "huggingface"
    hf_token: str = ""
    hf_max_new_tokens: int = 90

    # Veri modu
    data_mode: str = "amazon"  # "demo" veya "amazon"
    amazon_category: str = "Books"
    amazon_max_products: int = 1000
    amazon_max_interactions: int = 10000
    amazon_min_user_interactions: int = 3
    amazon_min_item_interactions: int = 3
    amazon_review_stream_limit: int = 300000
    amazon_meta_stream_limit: int = 1200000

    # Öneri ayarları
    default_k: int = 8
    max_k: int = 20

    # Sunucu
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Veri yolları
    @property
    def data_dir(self) -> Path:
        return BASE_DIR / "data"

    @property
    def processed_dir(self) -> Path:
        return BASE_DIR / "data" / "processed"

    @property
    def embeddings_dir(self) -> Path:
        return BASE_DIR / "data" / "embeddings"

    # Hibrit skor ağırlıkları (raporda gerekçelendirilmiştir)
    weight_collaborative: float = 0.25
    weight_content: float = 0.20
    weight_semantic: float = 0.35
    weight_popularity: float = 0.10
    weight_category_intent: float = 0.10

    # Yeni kullanıcı (cold-start) ağırlıkları
    weight_cold_semantic: float = 0.70
    weight_cold_category: float = 0.20
    weight_cold_popularity: float = 0.10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
