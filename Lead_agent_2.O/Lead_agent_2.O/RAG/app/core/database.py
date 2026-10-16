import os
import logging
from mongoengine import connect, disconnect
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def get_db_connection_string() -> str:
    """
    Constructs the MongoDB connection string from environment variables.
    """
    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        # Fallback for local development if not specified, though .env should have it
        return "mongodb://localhost:27017/core_rag"
    return MONGODB_URI

def init_db():
    """
    Initializes the MongoDB connection.
    """
    db_name = os.getenv("MONGODB_DB", "core_rag")
    db_url = get_db_connection_string()
    
    try:
        # Disconnect any existing connections to avoid conflicts during reloads
        disconnect()
        
        connect(host=db_url, db=db_name)
        logger.info(f"✅ Connected to MongoDB: {db_name}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        raise
