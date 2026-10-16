import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from faq_app.config import Config
from faq_app.utils.logger import get_logger
from faq_app.services.ingestor import DataIngestor
import glob
import os
import torch
import hashlib
import tempfile
import redis
import threading
from io import BytesIO

logger = get_logger("VectorEngine")

class VectorEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        logger.info(f"Loading Model: {Config.MODEL_NAME}...")
        
        # Detect device (GPU if available, else CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        self.model = SentenceTransformer(Config.MODEL_NAME, device=device)
        
        # Enable multi-processing for CPU (speeds up encoding)
        if device == 'cpu':
            self.model.encode = self._get_optimized_encode()
        
        # Registry for loaded domains: {domain: {'index': index, 'metadata': metadata}}
        self.indexes = {}
        
        # Redis connection
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None

        self._initialized = True
        self._domain_locks = {}  # Lock per domain to prevent race conditions
        self._global_lock = threading.Lock()

    def _get_domain_lock(self, domain: str):
        with self._global_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = threading.Lock()
            return self._domain_locks[domain]
    
    def _get_optimized_encode(self):
        """Returns optimized encode function with multi-processing."""
        original_encode = self.model.encode
        def optimized_encode(*args, **kwargs):
            # Use multi-processing pool for CPU
            kwargs.setdefault('batch_size', 128)
            kwargs.setdefault('show_progress_bar', False)
            kwargs.setdefault('convert_to_numpy', True)
            return original_encode(*args, **kwargs)
        return optimized_encode

    def load_domain(self, domain: str) -> bool:
        """
        Load index for a specific domain.
        Priority:
        1. Memory (already loaded)
        2. Redis (cached binary)
        3. Local Disk
        4. S3 (download to local)
        """
        # Use a lock for this specific domain to prevent race conditions
        with self._get_domain_lock(domain):
            # Double-check if loaded while waiting for lock
            if domain in self.indexes:
                return True

            logger.info(f"Loading domain: {domain}")
            
            # 1. Try Redis
            if self.redis_client:
                try:
                    index_key = f"faq:{domain}:index_bin"
                    meta_key = f"faq:{domain}:metadata"
                    
                    index_bin = self.redis_client.get(index_key)
                    meta_bin = self.redis_client.get(meta_key)
                    
                    if index_bin and meta_bin:
                        logger.info(f"Found domain {domain} in Redis. Loading...")
                        
                        # Deserialize Index
                        index_arr = np.frombuffer(index_bin, dtype='uint8')
                        index = faiss.deserialize_index(index_arr)
                        
                        # Deserialize Metadata
                        metadata = pickle.loads(meta_bin)
                        
                        self.indexes[domain] = {'index': index, 'metadata': metadata}
                        logger.info(f"✅ Loaded {domain} from Redis ({index.ntotal} vectors)")
                        return True
                except Exception as e:
                    logger.warning(f"Failed to load {domain} from Redis: {e}")

            # 2. Try S3 (Download to Memory)
            try:
                index_bytes, meta_bytes = self._download_artifacts_to_memory(domain)
                if index_bytes and meta_bytes:
                    logger.info(f"Downloaded {domain} from S3 to memory. deserializing...")
                    
                    # Deserialize Index
                    index_arr = np.frombuffer(index_bytes, dtype='uint8')
                    index = faiss.deserialize_index(index_arr)
                    
                    # Deserialize Metadata
                    metadata = pickle.loads(meta_bytes)
                    
                    self.indexes[domain] = {'index': index, 'metadata': metadata}
                    logger.info(f"✅ Loaded {domain} from S3 (Memory) ({index.ntotal} vectors)")
                    
                    # Cache to Redis for next time
                    self._cache_to_redis(domain, index, metadata)
                    return True
            except Exception as e:
                logger.warning(f"Failed to load {domain} from S3 (Memory): {e}")

            # 3. Fallback: Load from Disk (Only if S3 failed or not available)
            index_path = Config.get_index_path(domain)
            meta_path = Config.get_metadata_path(domain)
            
            if os.path.exists(index_path) and os.path.exists(meta_path):
                try:
                    logger.info(f"Falling back to local disk for {domain}...")
                    index = faiss.read_index(index_path)
                    with open(meta_path, "rb") as f:
                        metadata = pickle.load(f)
                    
                    self.indexes[domain] = {'index': index, 'metadata': metadata}
                    logger.info(f"Loaded {domain} from disk ({index.ntotal} vectors)")
                    
                    # Cache to Redis
                    self._cache_to_redis(domain, index, metadata)
                    return True
                except Exception as e:
                    logger.error(f"Failed to load {domain} from disk: {e}")
                    return False
            else:
                logger.warning(f"No artifacts found for domain {domain} (Redis, S3, or Disk)")
                return False

    def _download_artifacts_to_memory(self, domain: str) -> Tuple[Optional[bytes], Optional[bytes]]:
        """Download artifacts from S3 directly to memory."""
        try:
            from faq_app.services.s3_handler import S3Handler
            s3 = S3Handler()
            
            if not s3.s3_client:
                return None, None

            index_filename = "faiss_index.bin"
            meta_filename = "metadata.pkl"
            
            # S3 structure: faq_artifacts/{domain}/filename
            index_key = f"faq_artifacts/{domain}/{index_filename}"
            meta_key = f"faq_artifacts/{domain}/{meta_filename}"
            
            logger.info(f"Downloading {domain} artifacts from S3 to memory...")
            
            # Download to BytesIO
            index_stream = BytesIO()
            meta_stream = BytesIO()
            
            s3.s3_client.download_fileobj(s3.bucket_name, index_key, index_stream)
            s3.s3_client.download_fileobj(s3.bucket_name, meta_key, meta_stream)
            
            index_stream.seek(0)
            meta_stream.seek(0)
            
            return index_stream.read(), meta_stream.read()
            
        except ImportError:
            logger.warning("S3Handler not available.")
            return None, None
        except Exception as e:
            logger.warning(f"Failed to download artifacts for {domain} from S3: {e}")
            return None, None

    def _cache_to_redis(self, domain: str, index, metadata):
        """Cache index and metadata to Redis."""
        if not self.redis_client:
            return
            
        try:
            logger.info(f"Caching {domain} to Redis...")
            
            # Serialize Index
            # faiss.serialize_index returns numpy array of uint8
            chunk = faiss.serialize_index(index)
            index_bytes = chunk.tobytes()
            
            # Serialize Metadata
            meta_bytes = pickle.dumps(metadata)
            
            # Store in Redis (1 hour expiry? Or persistent? User said "keep them ready", implies persistence or long cache)
            # Let's set 24h expiry for now
            pipeline = self.redis_client.pipeline()
            pipeline.set(f"faq:{domain}:index_bin", index_bytes, ex=86400)
            pipeline.set(f"faq:{domain}:metadata", meta_bytes, ex=86400)
            pipeline.execute()
            
            logger.info(f"✅ Cached {domain} to Redis")
        except Exception as e:
            logger.error(f"Failed to cache {domain} to Redis: {e}")

    def build_full_index(self, domain: str = "default"):
        """ETL Pipeline: Reads files in data/{domain} -> Creates Index -> Caches."""
        
        # Sync data from S3 (optional, assuming data is already placed in data dir by upload API)
        
        data_dir = Config.get_domain_data_dir(domain)
        files = glob.glob(os.path.join(data_dir, "*.*"))
        all_docs = []
        
        for f in files:
            df = DataIngestor.load_file(f)
            if df is not None and not df.empty:
                records = df.to_dict('records')
                # Add source file info
                for r in records:
                    r['source_file'] = os.path.basename(f)
                    r['domain'] = domain
                all_docs.extend(records)
        
        if not all_docs:
            logger.error(f"No valid data found for domain {domain}.")
            return {"status": "failed", "reason": "No data", "domain": domain}

        logger.info(f"Embedding {len(all_docs)} items for domain {domain}...")
        
        # Build searchable text: use normalized_questions if available, otherwise question
        # Also include aliases if available for better matching
        searchable_texts = []
        for d in all_docs:
            # Prefer normalized_questions if available (new format)
            normalized_q = d.get('normalized_questions', '')
            if normalized_q and str(normalized_q).strip() and str(normalized_q).strip().lower() != 'nan':
                base_text = str(normalized_q).strip()
            else:
                # Fallback to question (backward compatibility)
                base_text = str(d['question']).strip().lower()
            
            # Add aliases if available for enhanced search
            alias_text = ""
            aliases = d.get('aliases', '') or d.get('normalized_allias', '')
            if aliases and str(aliases).strip() and str(aliases).strip().lower() != 'nan':
                alias_text = " " + str(aliases).strip()
            
            searchable_texts.append(base_text + alias_text)
        
        # Compute hash for caching embeddings during dev/testing
        data_hash = hashlib.md5("".join(searchable_texts).encode('utf-8')).hexdigest()
        cache_dir = tempfile.gettempdir()
        cache_file = os.path.join(cache_dir, f"faq_embed_{domain}_{Config.MODEL_NAME}_{data_hash}.pkl")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info("✓ Embeddings loaded from temp cache")
        else:
            embeddings = self.model.encode(
                searchable_texts, 
                batch_size=128,
                show_progress_bar=True, 
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.model.device
            )
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        
        # Build FAISS Index
        index = faiss.IndexFlatIP(Config.EMBEDDING_DIM)
        index.add(np.array(embeddings))

        # Persist to Disk
        index_path = Config.get_index_path(domain)
        meta_path = Config.get_metadata_path(domain)
        
        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(all_docs, f)

        # Upload artifacts to S3
        try:
            from faq_app.services.s3_handler import S3Handler
            s3 = S3Handler()
            index_filename = os.path.basename(index_path)
            meta_filename = os.path.basename(meta_path)
            
            s3.upload_file(index_path, f"faq_artifacts/{domain}/{index_filename}")
            s3.upload_file(meta_path, f"faq_artifacts/{domain}/{meta_filename}")
        except Exception as e:
            logger.error(f"Failed to upload artifacts to S3: {e}")

        # Update Memory & Redis
        self.indexes[domain] = {'index': index, 'metadata': all_docs}
        self._cache_to_redis(domain, index, all_docs)
        
        return {"status": "success", "count": len(all_docs), "domain": domain}

    def search(self, query: str, domain: str = "default", threshold: float = Config.CONFIDENCE_THRESHOLD) -> Dict:
        # Ensure domain is loaded
        if domain not in self.indexes:
            if not self.load_domain(domain):
                return {"answer": None, "reason": f"Domain {domain} not found or empty"}

        index_data = self.indexes[domain]
        index = index_data['index']
        metadata = index_data['metadata']
        
        if index.ntotal == 0:
             return {"answer": None, "reason": "Index empty"}

        # Redis Result Cache (Query Cache)
        # ... (Simplified for now, can add query caching later if needed, focusing on index caching)

        query_vec = self.model.encode([query], normalize_embeddings=True)
        D, I = index.search(query_vec, k=1)
        
        score = float(D[0][0])
        idx = int(I[0][0])

        if score >= threshold:
            matched_doc = metadata[idx]
            result = {
                "answer": matched_doc['answer'],
                "confidence": score,
                "matched_question": matched_doc['question'],
                "source_file": matched_doc.get('source_file'),
                "domain": domain
            }
            # Include additional metadata if available (new format)
            if 'id' in matched_doc:
                result['id'] = matched_doc['id']
            if 'category' in matched_doc:
                result['category'] = matched_doc['category']
            if 'audience' in matched_doc:
                result['audience'] = matched_doc['audience']
        else:
            result = {"answer": None, "confidence": score, "reason": "Low confidence"}
        
        return result
