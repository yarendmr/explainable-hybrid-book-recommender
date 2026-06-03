import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.data_store import data_store
from app.services.explainer import llm_explainer
from recommender.engine import HybridRecommendationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

engine: HybridRecommendationEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlangıç ve kapanış yaşam döngüsü."""
    global engine

    logger.info("=" * 60)
    logger.info("Açıklanabilir hibrit kitap öneri sistemi başlatılıyor...")
    logger.info("=" * 60)

    try:
        data_store.load()

        engine = HybridRecommendationEngine(
            data_store=data_store,
            explainer=llm_explainer,
        )
        engine.initialize()

        app.state.engine = engine
        app.state.data_store = data_store

        logger.info("Sistem hazır.")
    except FileNotFoundError as e:
        logger.error(f"Veri dosyası bulunamadı: {e}")
        logger.error(
            "Lütfen önce Amazon Books veri alt kümesini hazırlayın ve embedding dosyalarını oluşturun. "
            "Örnek komutlar: "
            "'python scripts/download_amazon_subset.py --category Books --max-products 1000 "
            "--max-interactions 10000 --min-user-interactions 3 --min-item-interactions 3' "
            "ve ardından 'python scripts/build_embeddings.py'"
        )

        app.state.engine = None
        app.state.data_store = data_store

    yield

    logger.info("Uygulama kapatılıyor.")


app = FastAPI(
    title="Açıklanabilir Hibrit Kitap Öneri Sistemi",
    description=(
        "Collaborative Filtering, Content-Based Filtering, anlamsal geri çağırma, "
        "popülerlik ve tür/niyet eşleşmesi bileşenlerini birleştiren "
        "açıklanabilir hibrit kitap öneri sistemi."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "Açıklanabilir Hibrit Kitap Öneri Sistemi",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["root"])
async def health():
    engine_ready = app.state.engine is not None
    data_ready = app.state.data_store.is_loaded if hasattr(app.state, "data_store") else False
    stats = app.state.data_store.stats if data_ready else {}

    return {
        "status": "ok" if (engine_ready and data_ready) else "degraded",
        "engine_ready": engine_ready,
        "data_ready": data_ready,
        "stats": stats,
    }