# core/domain_model.py
# Domain model class for managing ML models
import json
import joblib
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default domain to use when requested domain is not found
DEFAULT_DOMAIN = "default"

# Default metadata for fallback when no model exists
DEFAULT_METADATA = {
    "idx2label": {
        "0": "greeting",
        "1": "faq",
        "2": "lead_generation",
        "3": "general_info",
        "4": "handoff"
    },
    "labels": ["greeting", "faq", "lead_generation", "general_info", "handoff"],
    "domain": "default"
}


class DomainModel:
    """Container for a domain's model and metadata with fallback support"""
    
    _fallback_mode = False  # Class-level flag for fallback state
    
    def __init__(self, domain: str, ml_model_path: Path, fallback_domain: str = None):
        self.domain = domain
        self.ml_model_path = ml_model_path
        self.fallback_domain = fallback_domain or DEFAULT_DOMAIN
        self.clf = None
        self.meta = None
        self.idx2label_int = {}
        self.loaded = False
        self.using_fallback = False
        
    def load(self):
        """
        Load the model and metadata with 3-tier caching and fallback support:
        - Metadata (JSON): Redis → S3 → Local → Fallback to default
        - Classifier (binary): S3 → Local → Fallback to rule-based
        """
        try:
            from ml_utils.redis_cache_handler import RedisCacheHandler
            from ml_utils.s3_handler import S3Handler
            
            meta_path = self.ml_model_path / "model_meta.json"
            clf_path = self.ml_model_path / "classifier.pkl"
            
            # Create paths if they don't exist
            self.ml_model_path.mkdir(parents=True, exist_ok=True)
            
            redis_cache = RedisCacheHandler()
            s3 = S3Handler()
            
            # ============ LOAD METADATA (with Redis caching) ============
            self.meta = self._load_metadata(redis_cache, s3, meta_path)
            
            if not self.meta:
                # Try fallback domain
                logger.warning(f"Domain '{self.domain}' not found, trying fallback: {self.fallback_domain}")
                self.meta = self._load_fallback_metadata(redis_cache, s3)
                self.using_fallback = True
            
            if not self.meta:
                # Use default metadata as last resort
                logger.warning(f"Using default metadata for domain: {self.domain}")
                self.meta = DEFAULT_METADATA.copy()
                self.meta["domain"] = self.domain
                self.using_fallback = True
                DomainModel._fallback_mode = True
            
            # ============ LOAD CLASSIFIER (S3 → Local, no Redis) ============
            self.clf = self._load_classifier(s3, clf_path, redis_cache)
            
            if self.clf is None and not self.using_fallback:
                # Try fallback classifier
                logger.warning(f"Classifier not found for {self.domain}, trying fallback")
                self.clf = self._load_fallback_classifier(s3, redis_cache)
                self.using_fallback = True
            
            # Convert idx2label keys to int for easier lookup
            if self.meta and "idx2label" in self.meta:
                idx2label = self.meta["idx2label"]
                self.idx2label_int = {int(k): v for k, v in idx2label.items()}
            else:
                self.idx2label_int = {int(k): v for k, v in DEFAULT_METADATA["idx2label"].items()}
            
            self.loaded = True
            logger.info(f"✅ Model loaded for domain: {self.domain} (fallback={self.using_fallback})")
            
        except Exception as e:
            logger.error(f"Error loading model for domain {self.domain}: {e}")
            # Don't raise - allow fallback to work
            self._setup_fallback_mode()

    def _load_metadata(self, redis_cache, s3, meta_path) -> Optional[Dict[str, Any]]:
        """Load metadata from Redis, S3, or local file."""
        meta_cache_key = f"model_meta:{self.domain}"
        meta = None
        
        # 1. Try Redis cache for metadata (fastest)
        if redis_cache.is_available():
            meta = redis_cache.get_json(meta_cache_key)
            if meta:
                logger.debug(f"Model metadata loaded from Redis: {self.domain}")
        
        # 2. If not in Redis, try downloading from S3
        if not meta:
            try:
                if s3.s3_client:
                    if s3.download_file(f"models/{self.domain}/model_meta.json", str(meta_path)):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        redis_cache.set_json(meta_cache_key, meta, ttl=1800)
                        logger.info(f"Model metadata downloaded from S3: {self.domain}")
            except Exception as e:
                logger.warning(f"Failed to download metadata from S3 for {self.domain}: {e}")
        
        # 3. Fallback to local metadata file if exists
        if not meta and meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            redis_cache.set_json(meta_cache_key, meta, ttl=1800)
            logger.info(f"Model metadata loaded from local file: {self.domain}")
            
        return meta
    
    def _load_fallback_metadata(self, redis_cache, s3) -> Optional[Dict[str, Any]]:
        """Load metadata from fallback domain."""
        fallback_meta_path = self.ml_model_path.parent / self.fallback_domain / "model_meta.json"
        meta_cache_key = f"model_meta:{self.fallback_domain}"
        meta = None
        
        if redis_cache.is_available():
            meta = redis_cache.get_json(meta_cache_key)
        
        if not meta and fallback_meta_path.exists():
            with open(fallback_meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                
        return meta
    
    def _load_classifier(self, s3, clf_path, redis_cache):
        """Load classifier from S3 or local file."""
        clf = None
        
        # Try downloading from S3 if not exists locally
        if not clf_path.exists():
            try:
                if s3.s3_client:
                    if s3.download_file(f"models/{self.domain}/classifier.pkl", str(clf_path)):
                        logger.info(f"Classifier downloaded from S3: {self.domain}")
            except Exception as e:
                logger.warning(f"Failed to download classifier from S3 for {self.domain}: {e}")
        
        if clf_path.exists():
            try:
                clf = joblib.load(clf_path)
                logger.info(f"Classifier loaded from local cache: {self.domain}")
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
                
        return clf
    
    def _load_fallback_classifier(self, s3, redis_cache):
        """Load classifier from fallback domain."""
        fallback_clf_path = self.ml_model_path.parent / self.fallback_domain / "classifier.pkl"
        if fallback_clf_path.exists():
            try:
                return joblib.load(fallback_clf_path)
            except Exception as e:
                logger.warning(f"Failed to load fallback classifier: {e}")
        return None
    
    
    def _setup_fallback_mode(self):
        """Setup fallback mode with default metadata."""
        self.meta = DEFAULT_METADATA.copy()
        self.meta["domain"] = self.domain
        self.idx2label_int = {int(k): v for k, v in DEFAULT_METADATA["idx2label"].items()}
        self.clf = None
        self.loaded = True
        self.using_fallback = True
        DomainModel._fallback_mode = True
        logger.info(f"Fallback mode activated for domain: {self.domain}")
    
    @classmethod
    def is_fallback_mode(cls) -> bool:
        """Check if system is running in fallback mode."""
        return cls._fallback_mode

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict intent for text with fallback handling.
        Returns dict with 'intent', 'confidence', and 'fallback' keys.
        """
        if self.clf is not None:
            try:
                # Use actual classifier
                pred = self.clf.predict([text])[0]
                proba = self.clf.predict_proba([text])[0]
                confidence = float(max(proba))
                intent = self.idx2label_int.get(int(pred), "general_info")
                return {"intent": intent, "confidence": confidence, "fallback": False}
            except Exception as e:
                logger.warning(f"Classifier prediction failed: {e}")
        
        # Fallback to rule-based classification
        return self._fallback_predict(text)
    
    def _fallback_predict(self, text: str) -> Dict[str, Any]:
        """Rule-based fallback prediction when no classifier available."""
        text_lower = text.lower()
        
        # Simple rule-based intent detection
        if any(w in text_lower for w in ["hello", "hi", "hey", "good morning", "good evening"]):
            return {"intent": "greeting", "confidence": 0.7, "fallback": True}
        elif any(w in text_lower for w in ["price", "cost", "buy", "order", "product"]):
            return {"intent": "lead_generation", "confidence": 0.6, "fallback": True}
        elif "?" in text:
            return {"intent": "faq", "confidence": 0.5, "fallback": True}
        else:
            return {"intent": "general_info", "confidence": 0.4, "fallback": True}

