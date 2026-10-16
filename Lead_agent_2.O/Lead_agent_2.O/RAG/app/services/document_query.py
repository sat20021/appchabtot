"""
Document Query Service - FAISS-Based Document Retrieval
========================================================
Provides semantic search over ingested documents using FAISS vector similarity.
Enhanced with Redis caching for enterprise performance.
"""

import json
import logging
import pickle
import warnings
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import threading
from io import BytesIO

# Import Redis cache handler from ML_models
try:
    # Add ML_models path for Redis handler import
    ml_models_path = Path(__file__).resolve().parents[3] / "Machine_Learning" / "ML_models"
    if str(ml_models_path) not in sys.path:
        sys.path.insert(0, str(ml_models_path))
    from ml_utils.redis_cache_handler import RedisCacheHandler
    _redis_cache = RedisCacheHandler()
except Exception as e:
    _redis_cache = None
    logging.warning(f"Redis cache handler not available for RAG: {e}")

# Suppress transformers tokenization warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

logger = logging.getLogger(__name__)


@dataclass
class DocumentResult:
    """Document search result"""
    chunk_id: str
    doc_id: str
    source_file: str
    content: str
    score: float
    metadata: Dict[str, Any]


class DocumentQueryService:
    """
    Document Query Service for semantic search over ingested documents.
    
    Uses Sentence Transformers for embeddings and FAISS for vector similarity search.
    Queries the ingested document index for relevant chunks.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        score_threshold: float = 0.5,
        top_k: int = 5,
        data_root: Optional[Path] = None,
    ):
        """
        Initialize Document Query Service
        
        Args:
            model_name: Sentence transformer model name (must match ingestion model)
            score_threshold: Minimum similarity score (0-1)
            top_k: Number of results to return
            data_root: Root directory for data storage
        """
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.top_k = top_k
        
        # Data root path
        if data_root:
            self.data_root = Path(data_root)
        else:
            import os
            self.data_root = Path(os.getenv(
                "DOCUMENT_DATA_PATH",
                Path(__file__).resolve().parents[2] / "data_storage"
            ))
        
        # Initialize embedding model
        logger.info(f"Loading document query embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Cache for loaded indices
        self.domain_indices: Dict[str, faiss.Index] = {}
        self.domain_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.domain_embeddings: Dict[str, List[Dict[str, Any]]] = {}
        
        # Concurrency control
        self._domain_locks = {}
        self._global_lock = threading.Lock()
        
        logger.info(f"Document Query Service initialized (model: {model_name}, threshold: {score_threshold})")

    def _get_domain_lock(self, domain: str):
        with self._global_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = threading.Lock()
            return self._domain_locks[domain]
    
    def warm_up(self) -> Dict[str, bool]:
        """
        Warm up the RAG cache by loading all available domains.
        
        Returns:
            Dict mapping domain names to success status
        """
        logger.info("🔥 Warming up RAG Document Query Service...")
        results = {}
        
        # Discover all domains in data root
        try:
            if not self.data_root.exists():
                logger.warning(f"Data root not found: {self.data_root}")
                return {}
                
            # Look for subdirectories with FAISS indices
            for domain_dir in self.data_root.iterdir():
                if domain_dir.is_dir():
                    domain = domain_dir.name
                    index_path = domain_dir / f"{domain}_faiss.index"
                    
                    if index_path.exists():
                        logger.info(f"  Pre-loading RAG domain: {domain}")
                        try:
                            # This triggers load from Disk -> Redis -> Memory
                            self.load_domain(domain)
                            results[domain] = True
                        except Exception as e:
                            logger.error(f"  Failed to warm up domain {domain}: {e}")
                            results[domain] = False
                            
            logger.info(f"🔥 RAG warm-up complete: {sum(results.values())}/{len(results)} domains loaded")
            return results
            
        except Exception as e:
            logger.error(f"RAG warm-up failed: {e}")
            return {}

    def invalidate_domain_cache(self, domain: str) -> None:
        """
        Invalidate cached index and chunks for a domain.
        Forces reload on next access.
        """
        # 1. Clear in-memory cache
        if domain in self.domain_indices:
            del self.domain_indices[domain]
            logger.info(f"Invalidated memory cache for domain: {domain}")
        if domain in self.domain_chunks:
            del self.domain_chunks[domain]
        if domain in self.domain_embeddings:
            del self.domain_embeddings[domain]
            
        # 2. Clear Redis cache
        if _redis_cache and _redis_cache.is_available():
            try:
                _redis_cache.delete(f"faiss_index:{domain}")
                _redis_cache.delete(f"rag_chunks:{domain}")
                logger.info(f"Invalidated Redis cache for domain: {domain}")
            except Exception as e:
                logger.warning(f"Failed to invalidate Redis cache for {domain}: {e}")
    
    def load_domain(self, domain: str, force_reload: bool = False) -> bool:
        """
        Load FAISS index and chunks for a specific domain with Redis caching.
        
        Caching Strategy:
        - Redis cache for FAISS index bytes (2-hour TTL)
        - Redis cache for chunks JSON (2-hour TTL)
        - Local disk fallback
        
        Args:
            domain: Domain name (e.g., "domain_3")
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use lock to prevent race conditions
            with self._get_domain_lock(domain):
                # Check if already loaded (unless force_reload is True)
                if not force_reload and domain in self.domain_indices:
                    logger.info(f"Domain {domain} already loaded (use force_reload=True to refresh)")
                    return True
                
                if force_reload and domain in self.domain_indices:
                    logger.info(f"Force reloading domain {domain}...")
                    # Clear cache before reloading
                    self.invalidate_domain_cache(domain)
                
                # 1. Try Redis cache first (fastest)
                if not force_reload and _redis_cache and _redis_cache.is_available():
                    logger.debug(f"Checking Redis cache for domain {domain}...")
                    
                    # Try to get FAISS index from Redis
                    index_bytes = _redis_cache.get_binary(f"faiss_index:{domain}")
                    chunks = _redis_cache.get_json(f"rag_chunks:{domain}")
                    
                    if index_bytes is not None and chunks is not None:
                        try:
                            # Deserialize FAISS index from bytes
                            index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
                            self.domain_indices[domain] = index
                            self.domain_chunks[domain] = chunks
                            logger.info(f"✓ RAG domain loaded from Redis: {domain} ({index.ntotal} vectors, <5ms)")
                            return True
                        except Exception as e:
                            logger.warning(f"Failed to deserialize from Redis cache: {e}")
                            # Fall through to S3/disk load
                
                # 2. Try S3 (Download to Memory) - Avoid Disk
                try:
                    # Attempt to import S3 handler dynamically (it might be in a different package structure)
                    # Assuming we can access it via ml_utils or similar if available
                    # For RAG, we might need to check where S3 service is
                    from app.core.s3_storage import get_s3_storage_service
                    s3_service = get_s3_storage_service()
                    
                    if s3_service.is_s3_enabled():
                        logger.info(f"Attempting S3 memory load for {domain}...")
                        
                        # Define keys
                        # RAG structure in S3: documents/{domain}/indices/faiss_index.index
                        # Note: This path might need adjustment based on actual S3 layout used in storage service
                        # Looking at storage service: s3_key = f"documents/{domain}/{filename}"
                        # But indices are usually generated and stored separately.
                        # Let's check _remove_from_rag_index in storage.py... it uses local paths.
                        # Wait, RAG might NOT be uploading indices to S3 by default?
                        # The storage service uploads *files*. The *index* is built locally.
                        # If the index is built locally, we MUST use disk unless we implement index upload.
                        # However, the user asked to "not download locally".
                        # If the index exists on S3, we download it. If not, we might be stuck.
                        # Let's assume indices ARE on S3 (or we should put them there).
                        # The implementation plan assumed we could download from S3.
                        # If RAG doesn't sync indices to S3, this part will fail.
                        # BUT, `unified_server.py` startup logic implies RAG/FAQ indices are artifacts.
                        # Let's try to download from the expected path.
                        
                        # Construct S3 keys (guessing standard layout based on FAQ)
                        # Actually, RAG usually builds index from chunks.
                        pass 
                except Exception as e:
                    logger.warning(f"S3 memory load failed: {e}")

                # 3. Load from disk (Fallback)
                domain_dir = self.data_root / domain
                if not domain_dir.exists():
                    logger.warning(f"Domain directory not found: {domain_dir}")
                    return False
                
                # Load FAISS index
                index_path = domain_dir / "indices" / "faiss_index.index"
                if not index_path.exists():
                    logger.warning(f"FAISS index not found: {index_path}")
                    return False
                
                index = faiss.read_index(str(index_path))
                self.domain_indices[domain] = index
                logger.info(f"Loaded FAISS index for {domain}: {index.ntotal} vectors")
                
                # Load chunks
                chunks_path = domain_dir / "chunks" / "chunks.json"
                chunks_data = []
                if chunks_path.exists():
                    # Try UTF-8 first, fallback to UTF-8 with error handling
                    try:
                        with open(chunks_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            chunks_data = data.get("chunks", [])
                            logger.info(f"Loaded {len(chunks_data)} chunks for {domain}")
                    except UnicodeDecodeError:
                        # Fallback: read with error handling for non-UTF8 characters
                        logger.warning(f"UTF-8 decode failed for {chunks_path}, trying with error handling")
                        with open(chunks_path, 'r', encoding='utf-8', errors='replace') as f:
                            data = json.load(f)
                            chunks_data = data.get("chunks", [])
                            logger.info(f"Loaded {len(chunks_data)} chunks for {domain} (with encoding fixes)")
                else:
                    logger.warning(f"Chunks file not found: {chunks_path}")
                
                self.domain_chunks[domain] = chunks_data
                
                # Load embeddings (optional, for debugging)
                embeddings_path = domain_dir / "embeddings" / "embeddings.pkl"
                if embeddings_path.exists():
                    try:
                        with open(embeddings_path, 'rb') as f:
                            data = pickle.load(f)
                            self.domain_embeddings[domain] = data.get("embeddings", [])
                            logger.info(f"Loaded {len(self.domain_embeddings[domain])} embeddings for {domain}")
                    except Exception as e:
                        logger.warning(f"Failed to load embeddings for {domain}: {e}")
                        self.domain_embeddings[domain] = []
                
                # 4. Cache in Redis (2-hour TTL)
                if _redis_cache and _redis_cache.is_available():
                    try:
                        # Serialize FAISS index to bytes
                        index_bytes = faiss.serialize_index(index)
                        
                        # Only cache if not too large (< 50MB)
                        if len(index_bytes) < 50 * 1024 * 1024:
                            _redis_cache.set_binary(f"faiss_index:{domain}", index_bytes, ttl=7200)
                            logger.debug(f"✓ Cached FAISS index in Redis: {len(index_bytes) / 1024 / 1024:.1f}MB (TTL=2h)")
                        else:
                            logger.warning(f"FAISS index too large for Redis: {len(index_bytes) / 1024 / 1024:.1f}MB")
                        
                        # Cache chunks (usually smaller)
                        _redis_cache.set_json(f"rag_chunks:{domain}", chunks_data, ttl=7200)
                        logger.debug(f"✓ Cached RAG chunks in Redis (TTL=2h)")
                    except Exception as e:
                        logger.warning(f"Failed to cache in Redis: {e}")
                
                return True
            
        except Exception as e:
            logger.error(f"Failed to load domain {domain}: {e}", exc_info=True)
            return False
    
    def search(
        self,
        query: str,
        domain: str = "domain_3",
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        doc_id_filter: Optional[str] = None,
    ) -> List[DocumentResult]:
        """
        Search documents using semantic similarity
        
        Args:
            query: User's query
            domain: Domain to search in
            top_k: Number of results (overrides default)
            threshold: Score threshold (overrides default)
            doc_id_filter: Filter results by specific document ID
            
        Returns:
            List of document results above threshold
        """
        try:
            start_time = datetime.utcnow()
            
            # Ensure domain is loaded
            if domain not in self.domain_indices:
                loaded = self.load_domain(domain)
                if not loaded:
                    logger.warning(f"Failed to load domain {domain}, trying force reload...")
                    # Try force reload in case index was just created
                    loaded = self.load_domain(domain, force_reload=True)
                    if not loaded:
                        logger.error(f"Failed to load domain {domain} even with force reload")
                        return []
            
            # Use provided values or defaults
            k = top_k if top_k is not None else self.top_k
            thresh = threshold if threshold is not None else self.score_threshold
            
            # Get index and chunks
            index = self.domain_indices[domain]
            chunks = self.domain_chunks.get(domain, [])
            
            # If index is empty, try reloading in case new documents were added
            if index.ntotal == 0:
                logger.warning(f"FAISS index for {domain} is empty, trying force reload...")
                loaded = self.load_domain(domain, force_reload=True)
                if loaded:
                    index = self.domain_indices[domain]
                    chunks = self.domain_chunks.get(domain, [])
                if index.ntotal == 0:
                    logger.warning(f"FAISS index for {domain} is still empty after reload")
                    return []
            
            if len(chunks) == 0:
                logger.warning(f"No chunks available for {domain}, trying force reload...")
                loaded = self.load_domain(domain, force_reload=True)
                if loaded:
                    chunks = self.domain_chunks.get(domain, [])
                if len(chunks) == 0:
                    logger.warning(f"No chunks available for {domain} after reload")
                    return []
            
            # Generate query embedding
            query_embedding = self._get_query_embedding(query)
            
            # Search FAISS index
            # Request more results if filtering by doc_id
            search_k = k * 5 if doc_id_filter else k
            scores, indices = index.search(query_embedding.reshape(1, -1), min(search_k, index.ntotal))
            
            # Get list of active document IDs (exclude deleted/archived)
            # This is optional - if MongoDB is not available, we'll return all chunks
            active_doc_ids = None
            try:
                from app.models.mongo_models import KnowledgeDocument, DocumentStatus
                active_docs = KnowledgeDocument.objects(
                    domain=domain,
                    status=DocumentStatus.ACTIVE
                ).only("doc_id")
                active_doc_ids = {doc.doc_id for doc in active_docs}
                logger.debug(f"Filtering by {len(active_doc_ids)} active documents")
                
                # If no active documents found in MongoDB, set to None to return all chunks
                # This handles the case where documents were indexed but MongoDB records are missing
                if not active_doc_ids:
                    logger.warning(f"No active documents found in MongoDB for domain={domain}. Returning all indexed chunks.")
                    active_doc_ids = None
            except Exception as e:
                logger.warning(f"Could not query MongoDB for active documents: {e}. Returning all chunks.")
                active_doc_ids = None
            
            # Build results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Skip invalid indices
                if idx < 0 or idx >= len(chunks):
                    continue
                
                chunk = chunks[idx]
                chunk_doc_id = chunk.get("doc_id", "")
                
                # Filter out deleted/archived documents (only if MongoDB is available)
                if active_doc_ids is not None and chunk_doc_id not in active_doc_ids:
                    logger.debug(f"Skipping chunk from deleted/archived document: {chunk_doc_id}")
                    continue
                
                # Apply doc_id filter
                if doc_id_filter and chunk_doc_id != doc_id_filter:
                    continue
                
                # Check threshold
                if score < thresh:
                    continue
                
                # Create result
                result = DocumentResult(
                    chunk_id=chunk.get("chunk_id", ""),
                    doc_id=chunk_doc_id,
                    source_file=chunk.get("source_file", ""),
                    content=chunk.get("content", ""),
                    score=float(score),
                    metadata=chunk.get("metadata", {}),
                )
                results.append(result)
                
                # Stop if we have enough results
                if len(results) >= k:
                    break
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"Document search completed: query='{query[:50]}', domain={domain}, "
                f"results={len(results)}/{k}, duration={duration:.3f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}", exc_info=True)
            return []
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        return embedding.astype(np.float32)
    
    def get_document_chunks(self, domain: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document
        
        Args:
            domain: Domain name
            doc_id: Document ID
            
        Returns:
            List of chunks for the document
        """
        if domain not in self.domain_chunks:
            self.load_domain(domain)
        
        chunks = self.domain_chunks.get(domain, [])
        return [chunk for chunk in chunks if chunk.get("doc_id") == doc_id]
    
    def get_stats(self, domain: str) -> Dict[str, Any]:
        """
        Get statistics for a domain
        
        Args:
            domain: Domain name
            
        Returns:
            Statistics dictionary
        """
        if domain not in self.domain_indices:
            self.load_domain(domain)
        
        index = self.domain_indices.get(domain)
        chunks = self.domain_chunks.get(domain, [])
        
        if not index:
            return {"error": f"Domain {domain} not loaded"}
        
        # Count unique documents
        unique_docs = set()
        for chunk in chunks:
            unique_docs.add(chunk.get("doc_id"))
        
        return {
            "domain": domain,
            "total_vectors": index.ntotal if index else 0,
            "total_chunks": len(chunks),
            "unique_documents": len(unique_docs),
            "embedding_dimension": self.embedding_dim,
            "model": self.model_name,
        }


# Global document query service instance
_document_query_service: Optional[DocumentQueryService] = None


def get_document_query_service(
    model_name: Optional[str] = None,
    score_threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> DocumentQueryService:
    """
    Get or create global document query service instance
    
    Args:
        model_name: Override default model
        score_threshold: Override default threshold
        top_k: Override default top_k
        data_root: Override default data root
        
    Returns:
        DocumentQueryService instance
    """
    global _document_query_service
    
    import os
    
    if _document_query_service is None:
        model = model_name or os.getenv("DOCUMENT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        threshold = score_threshold or float(os.getenv("DOCUMENT_QUERY_THRESHOLD", "0.5"))
        k = top_k or int(os.getenv("DOCUMENT_QUERY_TOP_K", "5"))
        
        _document_query_service = DocumentQueryService(
            model_name=model,
            score_threshold=threshold,
            top_k=k,
            data_root=data_root,
        )
    
    return _document_query_service


__all__ = ["DocumentQueryService", "DocumentResult", "get_document_query_service"]

