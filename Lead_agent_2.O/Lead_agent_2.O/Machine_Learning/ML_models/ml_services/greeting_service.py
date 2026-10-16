# services/greeting_service.py
# Greeting service for generating contextual greeting and gratitude responses
# Refactored to use per-domain config storage
import json
import random
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import os
from core.state import DOMAIN_GREETINGS, DOMAIN_GRATITUDE
from config.settings import (
    GREETINGS_DIR, 
    LEGACY_GREETING_CONFIG_PATH,
    GREETING_INTENT, 
    GRATITUDE_INTENT, 
    CONFIDENCE_THRESHOLD
)
from ml_utils.redis_cache_handler import RedisCacheHandler

logger = logging.getLogger(__name__)

# ============================================================================
# PER-DOMAIN CONFIG FUNCTIONS
# ============================================================================

def _get_greetings_dir() -> Path:
    """Get the greetings config directory path."""
    local_storage = os.getenv("LOCAL_STORAGE_PATH")
    if local_storage:
        return Path(local_storage) / GREETINGS_DIR
    else:
        script_dir = Path(__file__).parent.parent
        return script_dir / GREETINGS_DIR


def _get_domain_config_path(domain: str) -> Path:
    """Get the config file path for a specific domain."""
    return _get_greetings_dir() / f"{domain}.json"


def load_domain_greeting(domain: str) -> Optional[Dict]:
    """
    Load a single domain's greeting config with 3-tier caching: Redis → S3 → Local
    
    Args:
        domain: Domain name to load
        
    Returns:
        Dict with greeting config, or None if not found
    """
    redis_cache = RedisCacheHandler()
    cache_key = f"greetings:{domain}"
    
    # 1. Try Redis cache first (fastest)
    if redis_cache.is_available():
        cached_data = redis_cache.get_json(cache_key)
        if cached_data:
            logger.debug(f"Greeting config loaded from Redis: {domain}")
            return cached_data
    
    # 2. Try downloading from S3
    config_file = _get_domain_config_path(domain)
    try:
        from ml_utils.s3_handler import S3Handler
        s3 = S3Handler()
        
        if s3.s3_client:
            s3_key = f"configs/greetings/{domain}.json"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            if s3.download_file(s3_key, str(config_file)):
                logger.info(f"Greeting config downloaded from S3: {domain}")
    except Exception as e:
        logger.warning(f"Failed to download greeting config from S3 for {domain}: {e}")
    
    # 3. Load from local file if exists
    if not config_file.exists():
        logger.debug(f"No greeting config found for domain: {domain}")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Cache in Redis (1 hour TTL)
        if redis_cache.is_available():
            redis_cache.set_json(cache_key, config_data, ttl=3600)
        
        logger.info(f"Greeting config loaded from file: {domain}")
        return config_data
        
    except Exception as e:
        logger.error(f"Error loading greeting config for {domain}: {e}")
        return None


def save_domain_greeting(domain: str, greetings: Dict) -> bool:
    """
    Save a single domain's greeting config to file, S3, and Redis.
    
    Args:
        domain: Domain name
        greetings: Dict with greeting categories (general, personalized, time_based, gratitude)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config_file = _get_domain_config_path(domain)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Write to local file
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(greetings, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved greetings to local file: {domain}")
        
        # 2. Upload to S3
        try:
            from ml_utils.s3_handler import S3Handler
            s3 = S3Handler()
            if s3.s3_client:
                s3_key = f"configs/greetings/{domain}.json"
                s3.upload_file(str(config_file), s3_key)
                logger.info(f"Uploaded greetings to S3: {domain}")
        except Exception as e:
            logger.warning(f"Failed to upload greeting config to S3 for {domain}: {e}")
        
        # 3. Update Redis cache
        redis_cache = RedisCacheHandler()
        if redis_cache.is_available():
            cache_key = f"greetings:{domain}"
            redis_cache.set_json(cache_key, greetings, ttl=3600)
            logger.debug(f"Greeting config cached in Redis: {domain}")
        
        # 4. Update in-memory state
        DOMAIN_GREETINGS[domain] = greetings
        if "gratitude" in greetings:
            DOMAIN_GRATITUDE[domain] = {"gratitude": greetings["gratitude"]}
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save greeting config for {domain}: {e}")
        return False


def load_all_domain_greetings() -> List[str]:
    """
    Load all domain greeting configs from the greetings directory.
    
    Returns:
        List of loaded domain names
    """
    logger.info("=" * 60)
    logger.info("LOADING GREETING CONFIGURATIONS (Per-Domain)")
    logger.info("=" * 60)
    
    greetings_dir = _get_greetings_dir()
    loaded_domains = []
    
    # First, check S3 for available domains
    try:
        from ml_utils.s3_handler import S3Handler
        s3 = S3Handler()
        
        if s3.s3_client:
            s3_files = s3.list_files("configs/greetings/")
            for s3_key in s3_files:
                if s3_key.endswith(".json"):
                    domain = Path(s3_key).stem
                    if domain and domain not in loaded_domains:
                        config = load_domain_greeting(domain)
                        if config:
                            DOMAIN_GREETINGS[domain] = config
                            if "gratitude" in config:
                                DOMAIN_GRATITUDE[domain] = {"gratitude": config["gratitude"]}
                            loaded_domains.append(domain)
                            logger.info(f"  ✓ Loaded domain from S3: {domain}")
    except Exception as e:
        logger.warning(f"Failed to scan S3 for greeting configs: {e}")
    
    # Then, scan local directory for any additional configs
    if greetings_dir.exists():
        for config_file in greetings_dir.glob("*.json"):
            domain = config_file.stem
            if domain not in loaded_domains:
                config = load_domain_greeting(domain)
                if config:
                    DOMAIN_GREETINGS[domain] = config
                    if "gratitude" in config:
                        DOMAIN_GRATITUDE[domain] = {"gratitude": config["gratitude"]}
                    loaded_domains.append(domain)
                    logger.info(f"  ✓ Loaded domain from local: {domain}")
    
    logger.info(f"✅ Loaded {len(loaded_domains)} greeting domain(s): {', '.join(loaded_domains)}")
    logger.info("=" * 60)
    
    return loaded_domains


def migrate_legacy_greetings() -> bool:
    """
    Migrate legacy single-file greeting config to per-domain files.
    
    Returns:
        True if migration was performed, False if not needed
    """
    local_storage = os.getenv("LOCAL_STORAGE_PATH")
    if local_storage:
        legacy_path = Path(local_storage) / LEGACY_GREETING_CONFIG_PATH
    else:
        script_dir = Path(__file__).parent.parent
        legacy_path = script_dir / LEGACY_GREETING_CONFIG_PATH
    
    # Also check the old configs location
    old_configs_path = script_dir.parent / "configs" / "greeting_config.json" if not local_storage else None
    
    paths_to_check = [legacy_path]
    if old_configs_path:
        paths_to_check.append(old_configs_path)
    
    for config_path in paths_to_check:
        if config_path and config_path.exists():
            logger.info(f"🔄 Found legacy greeting config: {config_path}")
            logger.info("   Migrating to per-domain files...")
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    legacy_data = json.load(f)
                
                migrated_count = 0
                for domain, greetings in legacy_data.items():
                    if isinstance(greetings, dict):
                        save_domain_greeting(domain, greetings)
                        migrated_count += 1
                        logger.info(f"   ✓ Migrated domain: {domain}")
                
                # Backup old file
                backup_path = config_path.with_suffix(".json.bak")
                config_path.rename(backup_path)
                logger.info(f"   ✓ Backed up legacy file to: {backup_path}")
                logger.info(f"✅ Migrated {migrated_count} domain(s) to per-domain files")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Migration failed: {e}")
                return False
    
    logger.debug("No legacy greeting config found - migration not needed")
    return False


# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS (kept for backward compatibility)
# ============================================================================

def load_custom_greetings(config_path: str = None):
    """
    Legacy function - now delegates to per-domain loading.
    Kept for backward compatibility with existing code.
    """
    # First, try to migrate legacy config if it exists
    migrate_legacy_greetings()
    
    # Then load all domain configs
    load_all_domain_greetings()


def save_custom_greetings(config_path: str = None):
    """
    Legacy function - now saves all in-memory domains to per-domain files.
    Kept for backward compatibility with existing code.
    """
    logger.warning("save_custom_greetings() is deprecated. Use save_domain_greeting(domain, greetings) instead.")
    
    for domain, greetings in DOMAIN_GREETINGS.items():
        # Merge gratitude back in
        if domain in DOMAIN_GRATITUDE and "gratitude" in DOMAIN_GRATITUDE[domain]:
            greetings["gratitude"] = DOMAIN_GRATITUDE[domain]["gratitude"]
        save_domain_greeting(domain, greetings)


# ============================================================================
# GREETING SELECTION FUNCTIONS
# ============================================================================

def get_time_of_day():
    """Determine the time of day for time-based greetings"""
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def select_greeting(domain: str, user_name: Optional[str] = None) -> str:
    """
    Select an appropriate greeting message based on domain and context
    """
    # Lazy-load domain if not in memory
    if domain not in DOMAIN_GREETINGS:
        config = load_domain_greeting(domain)
        if config:
            DOMAIN_GREETINGS[domain] = config
            if "gratitude" in config:
                DOMAIN_GRATITUDE[domain] = {"gratitude": config["gratitude"]}
    
    # Check if domain exists in config
    if domain not in DOMAIN_GREETINGS:
        raise ValueError(
            f"Domain '{domain}' not found in greeting configuration. "
            f"Available domains: {list(DOMAIN_GREETINGS.keys())}. "
            f"Please add greetings for this domain using POST /greetings/{domain}"
        )
    
    domain_greetings = DOMAIN_GREETINGS[domain]
    
    # If user name provided, use personalized greeting
    if user_name:
        personalized = domain_greetings.get("personalized", [])
        if personalized:
            message = random.choice(personalized)
            return message.format(name=user_name)
    
    # Try time-based greeting (50% chance)
    if random.random() < 0.5:
        time_of_day = get_time_of_day()
        time_greetings = domain_greetings.get("time_based", {}).get(time_of_day, [])
        if time_greetings:
            return random.choice(time_greetings)
    
    # Fallback to general greeting
    general_greetings = domain_greetings.get("general", [])
    if not general_greetings:
        raise ValueError(
            f"Domain '{domain}' has no greeting messages configured. "
            f"Please add 'general' greetings using POST /greetings/{domain}"
        )
    return random.choice(general_greetings)


def select_gratitude(domain: str) -> str:
    """
    Select an appropriate gratitude response message based on domain
    """
    # Lazy-load domain if not in memory
    if domain not in DOMAIN_GRATITUDE:
        config = load_domain_greeting(domain)
        if config and "gratitude" in config:
            DOMAIN_GRATITUDE[domain] = {"gratitude": config["gratitude"]}
    
    # Check if domain exists in config
    if domain not in DOMAIN_GRATITUDE:
        raise ValueError(
            f"Domain '{domain}' not found in gratitude configuration. "
            f"Available domains: {list(DOMAIN_GRATITUDE.keys())}. "
            f"Please add gratitude messages using POST /greetings/{domain}"
        )
    
    domain_gratitude = DOMAIN_GRATITUDE[domain]
    
    # Get gratitude messages
    gratitude_messages = domain_gratitude.get("gratitude", [])
    if not gratitude_messages:
        raise ValueError(
            f"Domain '{domain}' has no gratitude messages configured. "
            f"Please add 'gratitude' messages using POST /greetings/{domain}"
        )
    
    return random.choice(gratitude_messages)


def generate_greeting_response(query: str, domain: Optional[str] = None, user_name: Optional[str] = None) -> Dict:
    """
    Generate greeting or gratitude response based on query intent
    
    Args:
        query: User's query
        domain: Domain for context-aware greetings
        user_name: Optional user name for personalization
        
    Returns:
        Dictionary with response details
    """
    from ml_services.classifier_service import verify_intent_local
    
    # Step 1: Verify intent using local classifier
    classification = verify_intent_local(query, domain)
    
    intent = classification["intent"]
    confidence = classification["confidence"]
    resolved_domain = classification["domain"]
    
    logger.info(f"Query: '{query}' | Intent: {intent} | Confidence: {confidence:.3f}")
    
    # Step 2: Determine response type
    is_greeting = (intent == GREETING_INTENT and confidence >= CONFIDENCE_THRESHOLD)
    is_gratitude = (intent == GRATITUDE_INTENT and confidence >= CONFIDENCE_THRESHOLD)
    
    greeting_message = None
    gratitude_message = None
    response_message = None
    response_type = None
    personalized = False
    
    if is_greeting:
        greeting_message = select_greeting(resolved_domain, user_name)
        response_message = greeting_message
        response_type = "greeting"
        personalized = user_name is not None
        logger.info(f"Generated greeting: '{greeting_message}' (Personalized: {personalized})")
    elif is_gratitude:
        gratitude_message = select_gratitude(resolved_domain)
        response_message = gratitude_message
        response_type = "gratitude"
        logger.info(f"Generated gratitude response: '{gratitude_message}'")
    else:
        logger.info("No suitable greeting or gratitude response generated")
    
    return {
        "is_greeting": is_greeting,
        "is_gratitude": is_gratitude,
        "response_type": response_type,
        "response_message": response_message,
        "greeting_message": greeting_message,
        "gratitude_message": gratitude_message,
        "intent": intent,
        "confidence": confidence,
        "domain": resolved_domain,
        "user_query": query,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "personalized": personalized
    }


def list_greeting_domains() -> List[str]:
    """List all available greeting domains."""
    domains = set(DOMAIN_GREETINGS.keys())
    
    # Also check filesystem for any not-yet-loaded domains
    greetings_dir = _get_greetings_dir()
    if greetings_dir.exists():
        for config_file in greetings_dir.glob("*.json"):
            domains.add(config_file.stem)
    
    return sorted(list(domains))
