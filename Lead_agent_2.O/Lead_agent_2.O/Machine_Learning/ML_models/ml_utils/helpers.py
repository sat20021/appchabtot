# utils/helpers.py
# Utility helper functions
import logging
import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from models.schemas import IntentsConfig
from core.state import models, active_domain, embedder
from core.domain_model import DomainModel
from config.settings import EMBED_MODEL
from sentence_transformers import SentenceTransformer
from ml_utils.redis_cache_handler import RedisCacheHandler


logger = logging.getLogger(__name__)

# Resolve the absolute models directory
local_storage = os.getenv("LOCAL_STORAGE_PATH")
if local_storage:
    PROJECT_ROOT = Path(local_storage)
    MODELS_DIR = PROJECT_ROOT / "ml_files"
    CONFIGS_DIR = PROJECT_ROOT / "configs"
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Machine_Learning directory
    MODELS_DIR = PROJECT_ROOT / "models"
    CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

def update_pipeline_state(**kwargs):
    """Update the global pipeline state"""
    from core.state import pipeline_state
    for key, value in kwargs.items():
        if key in pipeline_state:
            pipeline_state[key] = value
    logger.info(f"Pipeline state updated: {kwargs}")

def validate_intent_config(config: IntentsConfig):
    """Validate the intent configuration"""
    if len(config.intents) == 0:
        raise ValueError("At least one intent must be defined")
    
    for intent_name, intent_def in config.intents.items():
        if not intent_def.templates:
            raise ValueError(f"Intent '{intent_name}' has no templates")
        
        # Check for placeholder usage
        templates_str = " ".join(intent_def.templates)
        if "{role}" in templates_str and not config.roles:
            raise ValueError(f"Intent '{intent_name}' uses {{role}} placeholder but no roles provided")
        if "{skill}" in templates_str and not config.skills:
            raise ValueError(f"Intent '{intent_name}' uses {{skill}} placeholder but no skills provided")

def discover_domains(models_dir: Path = MODELS_DIR):
    """Discover all available domains in the models directory"""
    if not models_dir.exists():
        logger.warning(f"Models directory does not exist at {models_dir.resolve()}")
        return []
    
    domains = []
    for domain_path in models_dir.iterdir():
        if domain_path.is_dir():
            meta_path = domain_path / "model_meta.json"
            clf_path = domain_path / "classifier.pkl"
            if meta_path.exists() and clf_path.exists():
                domains.append(domain_path.name)
    
    return domains

def load_domain(domain: str) -> DomainModel:
    """Load a specific domain model"""
    import core.state as state_module
    if domain in state_module.models:
        return state_module.models[domain]
    
    ml_model_path = MODELS_DIR / domain
    domain_model = DomainModel(domain, ml_model_path)
    domain_model.load()
    state_module.models[domain] = domain_model
    
    return domain_model

def initialize_embedder():
    """Initialize the sentence transformer embedder"""
    import core.state as state_module
    if state_module.embedder is None:
        logger.info(f"Loading embedder: {EMBED_MODEL}")
        state_module.embedder = SentenceTransformer(EMBED_MODEL)

def discover_domain_configs(configs_dir: Path = CONFIGS_DIR) -> List[str]:
    """Discover all available domain config files"""
    if not configs_dir.exists():
        logger.warning(f"Configs directory does not exist at {configs_dir.resolve()}")
        return []
    
    config_files = []
    for config_file in configs_dir.glob("*_config.json"):
        # Extract domain name from filename (e.g., tech_company_config.json -> tech_company)
        domain_name = config_file.stem.replace("_config", "")
        config_files.append(domain_name)
    
    return config_files

def load_domain_config(domain: str, configs_dir: Path = CONFIGS_DIR) -> Optional[Dict[str, Any]]:
    """
    Load domain configuration with 3-tier caching: Redis → S3 → Local
    
    Performance:
    - Redis hit: <1ms
    - S3 download: ~50-200ms (cached in Redis)
    - Local file: ~1ms (fallback)
    """
    # 1. Try Redis cache first (fastest)
    redis_cache = RedisCacheHandler()
    cache_key = f"config:{domain}"
    
    if redis_cache.is_available():
        cached_config = redis_cache.get_json(cache_key)
        if cached_config:
            logger.debug(f"Config loaded from Redis: {domain} (<1ms)")
            return cached_config
    
    # 2. Try downloading from S3
    try:
        from ml_utils.s3_handler import S3Handler
        s3 = S3Handler()
        
        if s3.s3_client:
            s3_key = f"configs/{domain}_config.json"
            local_path = configs_dir / f"{domain}_config.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            if s3.download_file(s3_key, str(local_path)):
                logger.info(f"Config downloaded from S3: {domain}")
                
                # Load from downloaded file
                with open(local_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Cache in Redis for future requests (1 hour TTL)
                redis_cache.set_json(cache_key, config_data, ttl=3600)
                logger.debug(f"Config cached in Redis: {domain} (TTL=1h)")
                
                return config_data
    except Exception as e:
        logger.warning(f"Failed to load config from S3: {e}")
    
    # 3. Fallback to local file if exists
    config_path = configs_dir / f"{domain}_config.json"
    
    if not config_path.exists():
        logger.warning(f"Config file not found for domain '{domain}' (checked Redis, S3, Local)")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Cache in Redis even from local fallback
        redis_cache.set_json(cache_key, config_data, ttl=3600)
        logger.info(f"Config loaded from local file: {domain}")
        
        return config_data
    except Exception as e:
        logger.error(f"Error loading config for domain '{domain}': {e}")
        return None

def get_all_domain_configs(configs_dir: Path = CONFIGS_DIR) -> Dict[str, Dict[str, Any]]:
    """
    Load all domain configurations (with Redis caching)
    
    This function will benefit from Redis cache warming on startup.
    """
    all_configs = {}
    
    # First, try to get domain list from S3 (authoritative source)
    try:
        from ml_utils.s3_handler import S3Handler
        s3 = S3Handler()
        
        if s3.s3_client:
            config_files = s3.list_files("configs/")
            s3_domains = set()
            
            for file_key in config_files:
                if file_key.endswith("_config.json"):
                    filename = Path(file_key).name
                    domain = filename.replace("_config.json", "")
                    s3_domains.add(domain)
            
            # Load each config (will use Redis cache if available)
            for domain in s3_domains:
                config = load_domain_config(domain, configs_dir)
                if config:
                    all_configs[domain] = config
            
            if all_configs:
                return all_configs
    except Exception as e:
        logger.warning(f"Failed to list configs from S3: {e}")
    
    # Fallback: scan local configs directory
    domain_names = discover_domain_configs(configs_dir)
    
    for domain in domain_names:
        config = load_domain_config(domain, configs_dir)
        if config:
            all_configs[domain] = config
    
    return all_configs

