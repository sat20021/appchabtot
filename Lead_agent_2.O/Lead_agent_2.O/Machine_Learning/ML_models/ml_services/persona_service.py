import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel

from ml_utils.s3_handler import S3Handler
from ml_utils.redis_cache_handler import RedisCacheHandler

logger = logging.getLogger(__name__)

class Persona(BaseModel):
    name: str
    description: str
    instructions: str
    conditions: Optional[Dict[str, Any]] = {}
    formatting_style: Optional[Dict[str, Any]] = {}
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Mobi AI",
                "description": "A warm, friendly, and human-like AI assistant for Campous OS.",
                "instructions": "You are Mobi AI, a joyful and human-like assistant for Campous OS.",
                "conditions": {
                    "off_topic_questions": {
                        "rule": "If asked questions completely unrelated to technology, politely redirect.",
                        "response": "That's an interesting question, but I'm here to help with tech solutions!"
                    }
                },
                "formatting_style": {
                    "rule": "Use rich markdown formatting",
                    "guidelines": [
                        "Use **bold text** for emphasis",
                        "Use bullet points for lists"
                       
                    ]
                }
            }
        }

class PersonaService:
    _instance = None
    _personas: Dict[str, Persona] = {}
    _config_path = Path("configs/personas.json")
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls() # Create the instance
            cls._instance.load_personas() # Load personas when the instance is first created
        return cls._instance

    def load_personas(self):
        """
        Load personas from Redis → S3 → Local (3-tier caching)
        
        Performance:
        - Redis hit: <1ms
        - S3 download: ~50-200ms (cached in Redis)
        - Local fallback: ~1ms
        """
        redis_cache = RedisCacheHandler()
        cache_key = "personas:all"
        
        # 1. Try Redis cache first (fastest)
        if redis_cache.is_available():
            cached_personas = redis_cache.get_json(cache_key)
            if cached_personas:
                # Convert dicts back to Persona objects
                self._personas = {k: Persona(**v) for k, v in cached_personas.items()}
                logger.debug("✓ Personas loaded from Redis (<1ms)")
                return self._personas
        
        # 2. Try downloading from S3
        try:
            s3 = S3Handler()
            if s3.s3_client:
                s3_key = "configs/personas.json"
                
                if s3.download_file(s3_key, str(self._config_path)):
                    logger.info("Personas downloaded from S3")
                    
                    with open(self._config_path, "r", encoding='utf-8') as f:
                        data = json.load(f)
                        self._personas = {k: Persona(**v) for k, v in data.items()}
                    
                    # Cache in Redis (1 hour TTL)
                    redis_cache.set_json(cache_key, {k: v.dict() for k, v in self._personas.items()}, ttl=3600)
                    logger.debug("✓ Personas cached in Redis (TTL=1h)")
                    
                    return self._personas
        except Exception as e:
            logger.warning(f"Failed to download personas from S3: {e}")
        
        # 3. Fallback to local file
        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    self._personas = {k: Persona(**v) for k, v in data.items()}
                
                # Cache in Redis even from local
                redis_cache.set_json(cache_key, {k: v.dict() for k, v in self._personas.items()}, ttl=3600)
                logger.info("Personas loaded from local file and cached")
                
                return self._personas
            except Exception as e:
                logger.error(f"Failed to load local personas: {e}")
                self._personas = {} # Ensure _personas is empty on failure
                return {}
        
        logger.warning("⚠️ No personas found (checked Redis, S3, Local)")
        self._personas = {} # Ensure _personas is empty if nothing found
        return {}

    def get_persona(self, domain: str) -> Optional[Persona]:
        """Get persona for a specific domain."""
        return self._personas.get(domain)

    def update_persona(self, domain: str, persona: Persona):
        """Update or create a persona for a domain and sync to S3 + Redis"""
        self._personas[domain] = persona
        self.save_personas()

    def save_personas(self):
        """
        Save personas to S3 (source of truth) + Redis (cache) + Local (fallback)
        
        Flow: S3 → Redis → Local
        """
        try:
            # Ensure directory exists
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert Persona objects to dictionaries for JSON serialization
            data_to_save = {k: v.dict() for k, v in self._personas.items()}

            # 1. Save to local file first
            with open(self._config_path, "w", encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2)
            logger.info("✓ Personas saved to local file")
            
            # 2. Upload to S3 (source of truth)
            try:
                s3 = S3Handler()
                if s3.s3_client:
                    s3.upload_file(str(self._config_path), "configs/personas.json")
                    logger.info("✓ Personas uploaded to S3")
            except Exception as e:
                logger.warning(f"Failed to upload personas to S3: {e}")
            
            # 3. Update Redis cache immediately (for instant availability)
            redis_cache = RedisCacheHandler()
            if redis_cache.is_available():
                redis_cache.set_json("personas:all", data_to_save, ttl=3600)
                logger.debug("✓ Personas cache updated in Redis")
                
        except Exception as e:
            logger.error(f"Failed to save personas: {e}")

def get_persona_service():
    return PersonaService.get_instance()
