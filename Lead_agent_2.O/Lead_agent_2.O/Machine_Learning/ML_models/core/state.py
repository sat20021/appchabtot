# core/state.py
# Global state management
import os
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer
from models.schemas import PipelineStatus
from core.domain_model import DomainModel

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

# Pipeline Manager State
pipeline_state = {
    "status": PipelineStatus.IDLE,
    "domain": None,
    "start_time": None,
    "completion_time": None,
    "current_step": None,
    "error": None,
    "dataset_path": None,
    "ml_model_path": None,
    "metrics": None
}

# Classifier Service State
models: Dict[str, DomainModel] = {}
active_domain: Optional[str] = None
embedder: Optional[SentenceTransformer] = None

# Greeting Service State
# These dictionaries are populated from configs/greeting_config.json on startup
# No hardcoded defaults - all greetings must be provided via config file
DOMAIN_GREETINGS: Dict = {}
DOMAIN_GRATITUDE: Dict = {}

# MongoDB State (for authentication and audit logging)
mongo_client: Optional[MongoClient] = None
mongo_db = None

def init_mongodb():
    """Initialize MongoDB connection"""
    global mongo_client, mongo_db
    
    if MongoClient is None:
        print("⚠️  PyMongo not installed, MongoDB features disabled")
        return
    
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("⚠️  MONGO_URI not set, MongoDB features disabled")
        return
    
    try:
        mongo_client = MongoClient(mongo_uri)
        # Test connection
        mongo_client.admin.command('ping')
        mongo_db = mongo_client.get_database()
        print(f"✅ MongoDB connected: {mongo_db.name}")
    except Exception as e:
        print(f"⚠️  MongoDB connection failed: {e}")
        mongo_client = None
        mongo_db = None

