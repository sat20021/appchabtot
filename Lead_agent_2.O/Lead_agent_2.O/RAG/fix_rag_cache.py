import sys
import os
from pathlib import Path

# Add ML_models to path
# RAG/app/services/document_query.py does this:
# ml_models_path = Path(__file__).resolve().parents[3] / "Machine_Learning" / "ML_models"
# We are in RAG/, so we need parents[1] to get to Lead_agent_2.O, then Machine_Learning/ML_models

ml_models_path = Path(__file__).resolve().parents[1] / "Machine_Learning" / "ML_models"
if str(ml_models_path) not in sys.path:
    sys.path.insert(0, str(ml_models_path))

try:
    from ml_utils.redis_cache_handler import RedisCacheHandler
except ImportError as e:
    print(f"❌ Failed to import RedisCacheHandler: {e}")
    print(f"  Search path: {ml_models_path}")
    sys.exit(1)

def clear_rag_cache(domain):
    print(f"Connecting to Redis to clear cache for domain: {domain}")
    try:
        redis_cache = RedisCacheHandler()
        if not redis_cache.is_available():
            print("❌ Redis is not available.")
            return

        # Keys used in document_query.py
        index_key = f"faiss_index:{domain}"
        chunks_key = f"rag_chunks:{domain}"

        # Delete keys using the underlying redis client
        # The RedisCacheHandler wrapper methods prefix keys with json: or bin:
        # But document_query.py uses _redis_cache.set_binary/set_json which ADDS prefixes
        # Wait, let's check document_query.py again.
        # It calls: _redis_cache.set_binary(f"faiss_index:{domain}", ...)
        # set_binary ADDS "bin:" prefix.
        # So the actual key in Redis is "bin:faiss_index:{domain}"
        
        # We should use the handler's invalidate method if possible, but it takes a key without prefix
        # and tries to delete "json:{key}" and "bin:{key}".
        # So if we pass "faiss_index:{domain}", it will try to delete "bin:faiss_index:{domain}".
        # This matches what we want!
        
        redis_cache.invalidate(f"faiss_index:{domain}")
        redis_cache.invalidate(f"rag_chunks:{domain}")
        
        # Also try raw delete just in case keys were set differently in older versions
        # or if my understanding of prefixes is slightly off
        if redis_cache.redis_client:
             redis_cache.redis_client.delete(f"faiss_index:{domain}")
             redis_cache.redis_client.delete(f"rag_chunks:{domain}")
             # Also try with prefixes manually just to be sure
             redis_cache.redis_client.delete(f"bin:faiss_index:{domain}")
             redis_cache.redis_client.delete(f"json:rag_chunks:{domain}")

        print(f"✅ Successfully deleted Redis keys for domain: {domain}")
        print("\n⚠️  Please restart your RAG server (uvicorn) to ensure the in-memory cache is also cleared.")
        
    except Exception as e:
        print(f"❌ Failed to clear cache: {e}")

if __name__ == "__main__":
    domain = "tech_company"
    clear_rag_cache(domain)
