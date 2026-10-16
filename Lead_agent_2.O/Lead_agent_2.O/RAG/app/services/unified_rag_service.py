"""
Unified RAG Service - Enterprise-Grade Multi-Source Search
===========================================================
Combines Document RAG and Web Scraper data for unified search.
Includes structured logging, correlation IDs, and source metadata.
"""

import os
import json
import uuid
import time
import hashlib
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import faiss

# Import existing DocumentQueryService
try:
    from app.services.document_query import DocumentQueryService, DocumentResult, get_document_query_service
except ImportError:
    DocumentQueryService = None
    DocumentResult = None
    get_document_query_service = None

logger = logging.getLogger(__name__)

# Thread pool for parallel execution
_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class UnifiedResult:
    """Unified search result with source metadata"""
    chunk_id: str
    content: str
    score: float
    source_type: str  # "document" or "web"
    source_url: Optional[str] = None
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "doc_id": self.doc_id,
            "metadata": self.metadata or {}
        }


class StructuredLogger:
    """Enterprise-grade structured JSON logger"""
    
    def __init__(self, service_name: str = "unified_rag"):
        self.service_name = service_name
        self.logger = logging.getLogger(f"structured.{service_name}")
    
    def _log(self, level: str, event: str, correlation_id: str, **kwargs):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "correlation_id": correlation_id,
            "service": self.service_name,
            "event": event,
            **kwargs
        }
        
        log_msg = json.dumps(log_entry)
        
        if level == "DEBUG":
            self.logger.debug(log_msg)
        elif level == "INFO":
            self.logger.info(log_msg)
        elif level == "WARNING":
            self.logger.warning(log_msg)
        elif level == "ERROR":
            self.logger.error(log_msg)
        else:
            self.logger.info(log_msg)
    
    def search_start(self, correlation_id: str, domain: str, query_hash: str):
        self._log("INFO", "search_start", correlation_id, 
                  domain=domain, query_hash=query_hash)
    
    def doc_rag_search(self, correlation_id: str, latency_ms: float, result_count: int):
        self._log("DEBUG", "doc_rag_search", correlation_id,
                  latency_ms=round(latency_ms, 2), result_count=result_count)
    
    def web_rag_search(self, correlation_id: str, latency_ms: float, result_count: int):
        self._log("DEBUG", "web_rag_search", correlation_id,
                  latency_ms=round(latency_ms, 2), result_count=result_count)
    
    def search_complete(self, correlation_id: str, total_results: int, 
                        latency_ms: float, sources: List[str]):
        self._log("INFO", "search_complete", correlation_id,
                  total_results=total_results, latency_ms=round(latency_ms, 2),
                  sources=sources)
    
    def search_error(self, correlation_id: str, error_type: str, error_msg: str):
        self._log("ERROR", "search_error", correlation_id,
                  error_type=error_type, error_msg=error_msg)


class WebScraperQueryService:
    """
    Query service for web scraper data stored in MongoDB.
    Searches the scraped_chunks collection.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        score_threshold: float = 0.5,
        top_k: int = 5
    ):
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.top_k = top_k
        
        # MongoDB connection
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb+srv://EnterpriseAI:EnterpriseAI@ragapi.59bacaa.mongodb.net")
        mongodb_db = os.getenv("MONGODB_DB", "chatbot-ai")
        
        self.mongo_client = MongoClient(mongodb_uri)
        self.db = self.mongo_client[mongodb_db]
        self.chunks_collection = self.db.scraped_chunks
        
        # Load embedding model
        logger.info(f"Loading web scraper query model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # FAISS index cache
        self._index_cache: Dict[str, faiss.Index] = {}
        self._chunks_cache: Dict[str, List[Dict]] = {}
        
        logger.info("WebScraperQueryService initialized")
    
    def _load_domain_index(self, domain: str) -> bool:
        """Load FAISS index for domain from local storage or S3"""
        if domain in self._index_cache:
            return True
        
        try:
            # Try local FAISS index first
            index_dir = os.path.join("faiss_indices")
            
            # Find latest version
            import glob
            pattern = os.path.join(index_dir, f"{domain}_v*.index")
            files = glob.glob(pattern)
            
            if not files:
                # Try alternate naming
                pattern = os.path.join(index_dir, f"{domain}.index")
                files = glob.glob(pattern)
            
            if files:
                # Use most recent
                latest_file = max(files, key=os.path.getctime)
                index = faiss.read_index(latest_file)
                self._index_cache[domain] = index
                logger.info(f"Loaded FAISS index: {latest_file} ({index.ntotal} vectors)")
                return True
            
            # Fallback: build index from MongoDB chunks
            return self._build_index_from_mongo(domain)
            
        except Exception as e:
            logger.error(f"Failed to load domain index {domain}: {e}")
            return False
    
    def _build_index_from_mongo(self, domain: str) -> bool:
        """Build FAISS index from MongoDB chunks"""
        try:
            chunks = list(self.chunks_collection.find({
                "metadata.tenant_id": domain,
                "is_deleted": False,
                "embedding": {"$exists": True}
            }))
            
            if not chunks:
                logger.warning(f"No chunks found in MongoDB for domain: {domain}")
                return False
            
            embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
            dimension = embeddings.shape[1]
            
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            self._index_cache[domain] = index
            self._chunks_cache[domain] = chunks
            
            logger.info(f"Built FAISS index from MongoDB: {domain} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index from MongoDB: {e}")
            return False
    
    def search(
        self,
        query: str,
        domain: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[UnifiedResult]:
        """Search web scraper data"""
        try:
            if not self._load_domain_index(domain):
                return []
            
            k = top_k or self.top_k
            thresh = threshold or self.score_threshold
            
            index = self._index_cache.get(domain)
            if not index or index.ntotal == 0:
                return []
            
            # Get chunks from cache or MongoDB
            chunks = self._chunks_cache.get(domain)
            if not chunks:
                chunks = list(self.chunks_collection.find({
                    "metadata.tenant_id": domain,
                    "is_deleted": False
                }))
                self._chunks_cache[domain] = chunks
            
            # Generate query embedding
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )[0].astype(np.float32)
            
            # Search
            scores, indices = index.search(
                query_embedding.reshape(1, -1),
                min(k * 2, index.ntotal)
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(chunks):
                    continue
                if score < thresh:
                    continue
                
                chunk = chunks[idx]
                metadata = chunk.get("metadata", {})
                
                result = UnifiedResult(
                    chunk_id=chunk.get("chunk_id", str(chunk.get("_id", ""))),
                    content=chunk.get("text", ""),
                    score=float(score),
                    source_type="web",
                    source_url=metadata.get("source_url", ""),
                    doc_id=metadata.get("source_url", ""),
                    metadata={
                        "source_url": metadata.get("source_url", ""),
                        "source_domain": metadata.get("source_domain", ""),
                        "tenant_id": metadata.get("tenant_id", domain),
                        "crawl_timestamp": str(chunk.get("created_at", ""))
                    }
                )
                results.append(result)
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Web scraper search failed: {e}")
            return []


class UnifiedRAGService:
    """
    Enterprise-grade Unified RAG Service.
    Searches both Document RAG and Web Scraper data.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        score_threshold: float = 0.5,
        top_k: int = 5
    ):
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.top_k = top_k
        
        # Initialize structured logger
        self.slog = StructuredLogger("unified_rag")
        
        # Initialize Document RAG service
        self.doc_service = None
        if get_document_query_service:
            try:
                self.doc_service = get_document_query_service()
                logger.info("✓ Document RAG service initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Document RAG: {e}")
        
        # Initialize Web Scraper service
        self.web_service = WebScraperQueryService(
            model_name=model_name,
            score_threshold=score_threshold,
            top_k=top_k
        )
        logger.info("✓ Web Scraper RAG service initialized")
        
        logger.info("✓ UnifiedRAGService ready")
    
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for request tracing"""
        return f"rag-{uuid.uuid4().hex[:12]}"
    
    def _hash_query(self, query: str) -> str:
        """Hash query for logging (privacy)"""
        return hashlib.md5(query.encode()).hexdigest()[:8]
    
    def search(
        self,
        query: str,
        domain: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        correlation_id: Optional[str] = None
    ) -> List[UnifiedResult]:
        """
        Search both Document RAG and Web Scraper.
        Returns merged, ranked results with source metadata.
        """
        start_time = time.time()
        corr_id = correlation_id or self._generate_correlation_id()
        k = top_k or self.top_k
        
        # Log search start
        self.slog.search_start(corr_id, domain, self._hash_query(query))
        
        try:
            all_results: List[UnifiedResult] = []
            sources_used = []
            
            # 1. Search Document RAG
            doc_results = []
            if self.doc_service:
                doc_start = time.time()
                try:
                    raw_results = self.doc_service.search(
                        query=query,
                        domain=domain,
                        top_k=k
                    )
                    for r in raw_results:
                        doc_results.append(UnifiedResult(
                            chunk_id=r.chunk_id,
                            content=r.content,
                            score=r.score,
                            source_type="document",
                            source_url=r.source_file,
                            doc_id=r.doc_id,
                            metadata=r.metadata
                        ))
                    sources_used.append("document")
                except Exception as e:
                    logger.warning(f"Document RAG search failed: {e}")
                
                doc_latency = (time.time() - doc_start) * 1000
                self.slog.doc_rag_search(corr_id, doc_latency, len(doc_results))
            
            # 2. Search Web Scraper RAG
            web_start = time.time()
            web_results = self.web_service.search(
                query=query,
                domain=domain,
                top_k=k
            )
            if web_results:
                sources_used.append("web")
            
            web_latency = (time.time() - web_start) * 1000
            self.slog.web_rag_search(corr_id, web_latency, len(web_results))
            
            # 3. Merge results
            all_results = doc_results + web_results
            
            # 4. Sort by score (descending) and take top-K
            all_results.sort(key=lambda x: x.score, reverse=True)
            final_results = all_results[:k]
            
            # Log completion
            total_latency = (time.time() - start_time) * 1000
            self.slog.search_complete(corr_id, len(final_results), total_latency, sources_used)
            
            logger.info(
                f"🔍 Unified search: query='{query[:50]}...', domain={domain}, "
                f"doc_results={len(doc_results)}, web_results={len(web_results)}, "
                f"total={len(final_results)}, latency={total_latency:.0f}ms"
            )
            
            return final_results
            
        except Exception as e:
            self.slog.search_error(corr_id, type(e).__name__, str(e))
            logger.error(f"Unified search failed: {e}", exc_info=True)
            return []
    
    async def search_async(
        self,
        query: str,
        domain: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        correlation_id: Optional[str] = None
    ) -> List[UnifiedResult]:
        """Async version using thread pool for parallel execution"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.search(query, domain, top_k, threshold, correlation_id)
        )


# Global instance
_unified_rag_service: Optional[UnifiedRAGService] = None


def get_unified_rag_service() -> UnifiedRAGService:
    """Get or create global UnifiedRAGService instance"""
    global _unified_rag_service
    
    if _unified_rag_service is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))
        top_k = int(os.getenv("RAG_TOP_K", "5"))
        
        _unified_rag_service = UnifiedRAGService(
            model_name=model_name,
            score_threshold=threshold,
            top_k=top_k
        )
    
    return _unified_rag_service


__all__ = [
    "UnifiedRAGService",
    "UnifiedResult",
    "WebScraperQueryService",
    "StructuredLogger",
    "get_unified_rag_service"
]

