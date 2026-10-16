# unified_server.py
# Combined ML Pipeline Manager, Classifier Service, and Greeting Response Service API
# Main entry point that wires together all modules
import warnings
import logging
import sys
import os
import uvicorn
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Load environment variables FIRST
load_dotenv()

# Suppress FutureWarning from transformers library about deprecated PyTorch API
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Locate FAQ search engine project for optional integration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAQ_ROOT = PROJECT_ROOT / "faq_search_engine"
faq_router = None

if FAQ_ROOT.exists():
    if str(FAQ_ROOT) not in sys.path:
        sys.path.append(str(FAQ_ROOT))
else:
    logger.warning("FAQ search engine directory not found; FAQ routes will be unavailable.")

# Locate RAG project for integration
RAG_ROOT = PROJECT_ROOT / "RAG"
if RAG_ROOT.exists():
    if str(RAG_ROOT) not in sys.path:
        sys.path.append(str(RAG_ROOT))
    logger.info(f"RAG directory added to sys.path: {RAG_ROOT}")
else:
    logger.warning("RAG directory not found; RAG integration unavailable.")

# Security Configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://agent.converiqo.ai,http://portal.converiqo.ai,http://localhost:2101,http://localhost:3000").split(",")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1,*.railway.app").split(",")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

logger.info(f"🔒 Security Configuration:")
logger.info(f"  Environment: {ENVIRONMENT}")
logger.info(f"  Allowed Origins: {ALLOWED_ORIGINS}")
logger.info(f"  Allowed Hosts: {ALLOWED_HOSTS}")

# Import services
from ml_utils.helpers import initialize_embedder, discover_domains, load_domain
from ml_services.greeting_service import load_custom_greetings

# ============================================================================
# Lifecycle Events (Lifespan)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the service on startup and cleanup on shutdown"""
    import core.state as state_module
    
    try:
        # === MONGODB INITIALIZATION FOR RAG ===
        logger.info("=" * 60)
        logger.info("INITIALIZING RAG MONGODB CONNECTION")
        logger.info("=" * 60)
        try:
            from RAG.app.core.database import init_db
            init_db()
            logger.info("✅ RAG MongoDB initialized successfully")
        except Exception as mongo_error:
            logger.error(f"❌ Failed to initialize RAG MongoDB: {mongo_error}")
            logger.warning("⚠️  RAG document ingestion will not work without MongoDB")
        logger.info("=" * 60)
        
        # === REDIS CACHE INITIALIZATION ===
        from ml_utils.redis_client import flush_all, is_redis_backed
        logger.info("=" * 60)
        logger.info("INITIALIZING PERFORMANCE CACHE")
        logger.info("=" * 60)
        
        # This will initialize Redis and log connection status
        redis_connected = is_redis_backed()
        
        # Clear all cache on startup
        flush_all()
        logger.info("=" * 60)
       
        # Initialize embedder
        initialize_embedder()
        
        # Load custom greetings
        load_custom_greetings()
        
        # === REDIS CACHE WARMING ===
        logger.info("=" * 60)
        logger.info("WARMING REDIS CACHE")
        logger.info("=" * 60)
        try:
            from ml_utils.redis_cache_handler import RedisCacheHandler
            
            redis_cache = RedisCacheHandler()
            
            if redis_cache.is_available():
                logger.info("🔥 Redis connected - warming cache...")
                
                # Warm configs
                config_results = redis_cache.warm_cache_configs()
                logger.info(f"  ✓ Cached {len(config_results)} domain configs")
                
                # Warm personas
                try:
                    from ml_services.persona_service import PersonaService
                    persona_service = PersonaService.get_instance()
                    persona_service.load_personas()
                    logger.info("  ✓ Cached personas")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to cache personas: {e}")
                
                # Get cache stats
                stats = redis_cache.get_cache_stats()
                logger.info(f"  📊 Cache stats: {stats.get('total_keys', 0)} keys, hit rate: {stats.get('hit_rate', 0)}%")
                
                # === FULL WARM-UP MODE ===
                if os.getenv("FULL_WARMUP", "false").lower() == "true":
                    logger.info("🚀 FULL_WARMUP enabled: Pre-loading heavy indices...")
                    
                    # 1. Warm RAG
                    try:
                        from RAG.app.services.document_query import DocumentQueryService
                        rag_service = DocumentQueryService()
                        rag_service.warm_up()
                    except Exception as e:
                        logger.warning(f"  ⚠️ RAG warm-up failed: {e}")
                        
                    # 2. Warm FAQ
                    try:
                        from ml_services.faq_service import warm_up_faq_index
                        warm_up_faq_index()
                    except Exception as e:
                        logger.warning(f"  ⚠️ FAQ warm-up failed: {e}")
                else:
                    logger.info("ℹ️  Standard warm-up: RAG/FAQ will lazy-load on first request")
                    logger.info("    (Set FULL_WARMUP=true to pre-load everything)")
                
                logger.info("🔥 Cache warming complete!")
            else:
                logger.warning("⚠️ Redis unavailable - cache warming skipped")
                logger.info("📝 Operating in fallback mode (S3 + Local)")
        except Exception as e:
            logger.warning(f"⚠️ Cache warming failed: {e}")
            logger.info("📝 Will proceed with S3 + Local fallback")
        
        logger.info("=" * 60)
        
        # Discover and load domains
        available_domains = discover_domains()
        
        if available_domains:
            logger.info("FOUND DOMAINS: %s", available_domains)
            
            # Default to 'default' domain if it exists
            if "default" in available_domains:
                # Ensure models directory exists for the default domain
                legacy_dir = Path("models/default")
                if not legacy_dir.exists():
                    legacy_dir.mkdir(parents=True, exist_ok=True)
                
                # If we have a root classifier, link it to the default domain
                if not (legacy_dir / "classifier.pkl").exists():
                    import shutil
                    shutil.copy("classifier.pkl", legacy_dir / "classifier.pkl")
                    shutil.copy("model_meta.json", legacy_dir / "model_meta.json")
                    logger.info("Migrated legacy model to models/default/")
                
                load_domain("default")
                state_module.active_domain = "default"
            else:
                # Load first available domain
                first_domain = available_domains[0]
                load_domain(first_domain)
                state_module.active_domain = first_domain
            
            logger.info(f"Active domain set to: {state_module.active_domain}")
        else:
            logger.warning("No model domains found")
        
        # === FAQ INDEX INITIALIZATION ===
        logger.info("=" * 60)
        logger.info("INITIALIZING FAQ VECTOR SEARCH ENGINE")
        logger.info("=" * 60)
        try:
            from ml_services.faq_service import refresh_faq_index, faq_stats, load_faq_domain, clear_faq_cache
            from ml_utils.s3_handler import S3Handler
            import glob
            import tempfile
            
            # 🔴 CRITICAL: Clear temp embedding cache on startup to ensure fresh embeddings
            logger.info("🧹 Clearing temporary embedding cache...")
            try:
                cache_dir = tempfile.gettempdir()
                for pkl_file in glob.glob(os.path.join(cache_dir, "faq_embed_*.pkl")):
                    os.remove(pkl_file)
                    logger.info(f"  Deleted: {os.path.basename(pkl_file)}")
                logger.info("✓ Temp embedding cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear temp cache: {e}")
            
            logger.info("🔄 Scanning for FAQ domains in storage...")
            s3 = S3Handler()
            domains = set(["default"]) # Always try to load default
            
            try:
                # List all objects in faq_artifacts/ to find domains
                # Structure: faq_artifacts/{domain}/...
                artifacts = s3.list_files("faq_artifacts/")
                for key in artifacts:
                    # key example: faq_artifacts/finance/faiss_index.bin
                    parts = key.split('/')
                    if len(parts) > 1:
                        # parts[0] is faq_artifacts, parts[1] is domain
                        domain = parts[1]
                        if domain:
                            domains.add(domain)
            except Exception as e:
                logger.warning(f"Failed to scan storage for domains: {e}")
            
            logger.info(f"Found domains to load: {domains}")
            
            # 🔴 FORCE REBUILD for all domains (gets fresh data from storage)
            logger.info("🔄 Force rebuilding FAQ indexes from storage...")
            for domain in domains:
                if domain == "default" and not (FAQ_ROOT / "data" / "default").exists():
                    continue  # Skip default if no data
                    
                logger.info(f"  Rebuilding domain: {domain}...")
                try:
                    result = refresh_faq_index(domain=domain)
                    if result.get("status") == "success":
                        stats = faq_stats(domain)
                        logger.info(f"  ✅ {domain}: {stats.get('vectors_indexed', 0)} vectors")
                    else:
                        logger.warning(f"  ⚠️ {domain}: {result.get('reason', 'unknown error')}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to rebuild {domain}: {e}")
            
            logger.info("✅ FAQ initialization complete!")

        except Exception as e:
            logger.error(f"❌ FAQ initialization failed with error: {e}")
            logger.warning("⚠️  FAQ functionality may be limited or unavailable")
            import traceback
            logger.debug(traceback.format_exc())
        
        logger.info("=" * 60)
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    yield  # Application runs here
    
    # Shutdown logic (if any) can go here
    logger.info("Shutting down Lead Agent 2.0...")

# Initialize FastAPI app
app = FastAPI(
    title="Lead Agent 2.0 - Enterprise AI Platform",
    description="Unified API for ML Pipeline, RAG, FAQ, and Conversational AI with enterprise security",
    version="3.0.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,  # Disable docs in production
    redoc_url="/redoc" if ENVIRONMENT == "development" else None,
    lifespan=lifespan
)
# app.openapi_version = "3.0.2"  # Removed to allow FastAPI 0.115+ default (OpenAPI 3.1.0) for better Swagger UI support

# ============================================================================
# Observability Configuration
# ============================================================================

# Configure structured logging
from core.logging_config import configure_structured_logging
import os

# JSON logs in production, console logs in development
json_logs = ENVIRONMENT == "production"
configure_structured_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    json_logs=json_logs
)
logger.info(f"✅ Structured logging configured (environment={ENVIRONMENT}, json_logs={json_logs})")

# ============================================================================
# Security Middleware
# ============================================================================

# Import security middleware
from middleware.security_headers import SecurityHeadersMiddleware
from middleware.rate_limit import RateLimitMiddleware
from middleware.correlation_id import CorrelationIdMiddleware
from middleware.metrics import PrometheusMiddleware

# Add correlation ID middleware (must be first for proper tracking)
app.add_middleware(CorrelationIdMiddleware)
logger.info("✅ Correlation ID middleware enabled")

# Add Prometheus metrics middleware
app.add_middleware(PrometheusMiddleware)
logger.info("✅ Prometheus metrics middleware enabled")

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)
logger.info("✅ Security headers middleware enabled")

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)
logger.info("✅ Rate limiting middleware enabled")

# Configure CORS with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # 🟢 Environment-based (was "*")
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Content-Type", "X-Request-ID"],
    max_age=600,  # Cache preflight requests for 10 minutes
)
logger.info("✅ CORS configured with restricted origins")

# Add trusted host middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
logger.info("✅ Trusted host middleware enabled")

# ============================================================================
# Import Routes
# ============================================================================

from api.pipeline_routes import router as pipeline_router
from api.classifier_routes import router as classifier_router
from api.greeting_routes import router as greeting_router
from api.hybrid_routes import router as hybrid_router
from api.conversational_routes import include_legacy_conversation_routes

from api.health_routes import router as health_router  # 🟢 NEW: Health checks
from api.metrics_routes import router as metrics_router  # 🟢 NEW: Metrics
# from api.customer_routes import router as customer_router  # 🟢 NEW: Customer Agent (MISSING)
from ml_services.faq_service import faq_stats, refresh_faq_index

# Import RAG router
from api.rag_routes import router as rag_router

# Import FAQ router
from api.faq_routes import router as faq_router

# ============================================================================
# Register Routers
# ============================================================================

# Monitoring and Health
app.include_router(metrics_router)
logger.info("✅ Metrics endpoint mounted at /metrics")

app.include_router(health_router)
logger.info("✅ Health check endpoints mounted under /health")

# Customer Agent (Disabled due to missing module)
# app.include_router(customer_router)
# logger.info("✅ Customer Agent endpoints mounted under /api/v1/customer")

# Core services
app.include_router(pipeline_router)
app.include_router(classifier_router)
app.include_router(greeting_router)
app.include_router(hybrid_router)
app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(faq_router, prefix="/api/v1/faq", tags=["FAQ"])
include_legacy_conversation_routes(app)

logger.info("✅ RAG endpoints mounted under /api/v1/rag")
logger.info("✅ FAQ file management endpoints mounted under /api/v1/faq")



# FAQ functionality is integrated via hybrid_orchestrator, not as separate routes
# if faq_router:
#     app.include_router(faq_router, prefix="/faq", tags=["FAQ Search"])
#     logger.info("FAQ search endpoints mounted under /faq")


# Import services
from ml_utils.helpers import initialize_embedder, discover_domains, load_domain
from ml_services.greeting_service import load_custom_greetings

# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", response_model=Dict[str, str])
def root():
    """Root endpoint with API information"""
    return {
        "message": "ML Pipeline, Classification & Greeting Response API (Unified Server)",
        "version": "3.1.0",
        "docs": "/docs",
        "pipeline_endpoints": "/pipeline/{domain}/train",
        "greeting_endpoints": "/greetings/{domain}/respond",
        "classifier_health": "/health",
        "pipeline_domains": "/pipeline/domains",
        "greeting_domains": "/greetings/domains"
    }

# ============================================================================
# Persona API
# ============================================================================
# Persona management endpoints are available via api/persona_routes.py router
# Endpoints:
#   - POST /api/v1/domains/{domain}/persona - Create/update persona
#   - GET /api/v1/domains/{domain}/persona - Get persona for domain

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Default port is 8000, but can be overridden
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    logger.info(f"Starting unified server on port {port}")
    logger.info("This server handles Pipeline Manager, Classifier Service, and Greeting Response Service APIs")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
