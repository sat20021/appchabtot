"""
Document ingestion pipeline.

Wires document uploads into a local RAG-ready dataset by:
- Extracting plain text from the uploaded file
- Chunking with overlap
- Generating sentence-transformer embeddings
- Building / refreshing a FAISS index
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Suppress transformers tokenization warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - optional dependency should exist via requirements
    fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover - optional dependency
    DocxDocument = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

from app.models.mongo_models import KnowledgeDocument
from app.core.observability.document_processing_logger import get_document_processing_logger

logger = logging.getLogger(__name__)
processing_logger = get_document_processing_logger()

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data_storage"


def chunk_text(
    text: str,
    *,
    chunk_size: int = 250,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping sentence-aware chunks.
    
    This ensures chunks only contain complete sentences, preventing
    fragments like "For example: ○" at chunk boundaries.

    Args:
        text: Full document text.
        chunk_size: Target number of words per chunk (approximate).
        overlap: Number of words to carry over between chunks (approximate).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        overlap = max(chunk_size // 5, 1)
    
    if not text.strip():
        return []
    
    # Split text into sentences using regex
    # This handles common sentence terminators while being mindful of abbreviations
    import re
    
    # Split by sentence terminators (., !, ?) followed by space and capital letter or end of string
    # This regex tries to avoid splitting on abbreviations like "Dr." or "U.S."
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])|(?<=\.|\?|\!)$'
    sentences = re.split(sentence_pattern, text)
    
    # Clean up sentences and filter empty ones
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks: List[str] = []
    current_chunk_sentences = []
    current_word_count = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_word_count = len(sentence.split())
        
        # If adding this sentence would exceed chunk_size and we already have content
        if current_word_count + sentence_word_count > chunk_size and current_chunk_sentences:
            # Save current chunk
            chunks.append(" ".join(current_chunk_sentences))
            
            # Start new chunk with overlap
            # Calculate how many sentences to carry over for overlap
            overlap_sentences = []
            overlap_word_count = 0
            
            # Go backwards from current position to find sentences for overlap
            for j in range(len(current_chunk_sentences) - 1, -1, -1):
                sent = current_chunk_sentences[j]
                sent_words = len(sent.split())
                if overlap_word_count + sent_words <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_word_count += sent_words
                else:
                    break
            
            current_chunk_sentences = overlap_sentences
            current_word_count = overlap_word_count
        
        # Add current sentence
        current_chunk_sentences.append(sentence)
        current_word_count += sentence_word_count
        i += 1
    
    # Add final chunk if it has content
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
    
    return chunks


class DocumentIngestionService:
    """Coordinates text extraction, chunking, embedding, and FAISS indexing."""

    def __init__(self):
        # Resolve docs root: priority LOCAL_STORAGE_PATH > DOCUMENT_STORAGE_PATH > fallback
        local_storage = os.getenv("LOCAL_STORAGE_PATH")
        if local_storage:
             self.docs_root = Path(local_storage) / "rag" / "documents"
             self.data_root = Path(local_storage) / "rag" / "data"
        else:
            self.docs_root = Path(os.getenv("DOCUMENT_STORAGE_PATH", Path(__file__).resolve().parents[2] / "docs_storage"))
            self.data_root = Path(os.getenv("DOCUMENT_DATA_PATH", DEFAULT_DATA_ROOT))

        self.chunk_size = int(os.getenv("DOCUMENT_CHUNK_SIZE", "250"))
        self.chunk_overlap = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "50"))
        self.embedding_model_name = os.getenv("DOCUMENT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._model: Optional[SentenceTransformer] = None
        self._model_lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def process_document(self, doc_id: str) -> None:
        document = KnowledgeDocument.objects(doc_id=doc_id).first()
        if not document:
            logger.warning("❌ Document %s not found for ingestion", doc_id)
            processing_logger.log_error("LOOKUP", Exception("Document not found"), doc_id=doc_id)
            return

        # Start detailed processing logging
        processing_logger.start_processing(doc_id, document.original_name, document.domain)
        
        # Determine storage type from storage_path
        storage_type = "local"
        storage_path = document.storage_path or ""
        if storage_path.startswith("s3://"):
            storage_type = "S3"
        
        processing_logger.log_upload_complete(
            document.size_bytes, 
            document.content_type or "unknown",
            storage_type=storage_type,
            storage_path=storage_path
        )
        
        logger.info("🚀 Starting ingestion for document %s (%s)", doc_id, document.original_name)
        document.embedding_status = "processing"
        document.save()

        try:
            # ==================== TEXT EXTRACTION ====================
            logger.info("📄 Extracting text from %s...", document.original_name)
            processing_logger.log_extraction_start(Path(document.storage_path).suffix or "unknown")
            
            text = self._extract_text(document)
            if not text.strip():
                raise ValueError("No extractable text")
            
            word_count = len(text.split())
            logger.info("✅ Extracted %d characters from %s", len(text), document.original_name)
            processing_logger.log_extraction_complete(len(text), word_count)

            # ==================== CHUNKING ====================
            logger.info("✂️ Chunking text (size=%d, overlap=%d)...", self.chunk_size, self.chunk_overlap)
            processing_logger.log_chunking_start(self.chunk_size, self.chunk_overlap)
            
            chunk_payloads = self._build_chunks(document, text)
            if not chunk_payloads:
                raise ValueError("Chunking produced no content")
            
            # Calculate average chunk size
            avg_chunk_size = sum(len(c["content"].split()) for c in chunk_payloads) // len(chunk_payloads)
            
            logger.info("✅ Created %d chunks", len(chunk_payloads))
            processing_logger.log_chunking_complete(len(chunk_payloads), avg_chunk_size)

            # ==================== EMBEDDING GENERATION ====================
            logger.info("🧠 Generating embeddings using %s...", self.embedding_model_name)
            processing_logger.log_embedding_start(self.embedding_model_name, len(chunk_payloads))
            
            vectors = self._embed_chunks([chunk["content"] for chunk in chunk_payloads])
            if vectors.size == 0:
                raise ValueError("Embedding generation returned zero vectors")
            
            logger.info("✅ Generated %d embeddings (dimension=%d)", len(vectors), vectors.shape[1])
            processing_logger.log_embedding_complete(len(vectors), vectors.shape[1])

            # ==================== VECTORIZATION & PERSISTENCE ====================
            logger.info("💾 Persisting to FAISS index for domain %s...", document.domain)
            processing_logger.log_persistence_start()
            
            self._persist_domain_data(document.domain, document.doc_id, chunk_payloads, vectors)
            
            logger.info("✅ Persisted chunks and updated FAISS index")
            processing_logger.log_persistence_complete(
                chunks_saved=True,
                embeddings_saved=True,
                index_saved=True
            )

            # ==================== COMPLETION ====================
            document.chunk_count = len(chunk_payloads)
            document.embedding_status = "completed"
            document.save()
            
            # Invalidate query service cache to ensure new document is searchable
            try:
                from app.services.document_query import get_document_query_service
                query_service = get_document_query_service()
                query_service.invalidate_domain_cache(document.domain)
                logger.info(f"✅ Invalidated query cache for domain: {document.domain}")
            except Exception as e:
                logger.warning(f"Failed to invalidate query cache: {e}")
            
            logger.info("🎉 Completed ingestion for document %s (%d chunks)", doc_id, len(chunk_payloads))
            processing_logger.log_processing_complete(len(chunk_payloads), status="success")
            
        except Exception as exc:  # noqa: BLE001
            logger.error("Document ingestion failed for %s: %s", doc_id, exc, exc_info=True)
            processing_logger.log_error("PROCESSING", exc, doc_id=doc_id, filename=document.original_name)
            processing_logger.log_processing_complete(0, status="failed")
            
            document.embedding_status = "failed"
            metadata = document.metadata or {}
            metadata["ingestion_error"] = str(exc)
            document.metadata = metadata
            document.save()

    # ---------------------------------------------------------------- utilities
    def _extract_text(self, document: KnowledgeDocument) -> str:
        """
        Extract text from document.
        Handles both local storage and S3 (downloads from S3 if needed).
        """
        # Use document storage service to get file path
        # This handles S3 downloads and returns local cached path
        from app.services.document_storage import get_document_storage_service
        storage_service = get_document_storage_service()
        
        try:
            path = storage_service.get_file_path(document)
        except Exception as e:
            logger.error(f"Failed to get file path for document {document.doc_id}: {e}")
            raise FileNotFoundError(f"Document file not accessible: {document.storage_path}") from e
        
        if not path.exists():
            raise FileNotFoundError(f"Document file not found at {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if not fitz:
                raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction")
            with fitz.open(path) as pdf:
                return "\n".join(page.get_text("text") for page in pdf)

        if suffix in {".xls", ".xlsx"}:
            if not pd:
                raise RuntimeError("pandas is required for Excel extraction")
            return self._extract_excel_text(path)

        if suffix == ".docx":
            if not DocxDocument:
                raise RuntimeError("python-docx is required for DOCX extraction")
            doc = DocxDocument(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text)

        # Plain text fallback
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="ignore")

    def _extract_excel_text(self, path: Path) -> str:
        """
        Extract text content from Excel files (.xls, .xlsx).
        Converts each sheet into a human-readable, pipe-delimited text block.
        """
        try:
            excel_file = pd.ExcelFile(path, engine="openpyxl")
        except Exception as exc:
            raise RuntimeError(f"Failed to open Excel file {path.name}: {exc}") from exc

        lines: List[str] = []
        for sheet_name in excel_file.sheet_names:
            try:
                df = excel_file.parse(sheet_name)
            except Exception as exc:
                logger.warning("Failed to parse sheet %s in %s: %s", sheet_name, path.name, exc)
                continue

            if df.empty:
                continue

            df = df.fillna("")
            lines.append(f"### Sheet: {sheet_name}")

            headers = [str(col).strip() for col in df.columns]
            if any(headers):
                lines.append(" | ".join(headers))

            for _, row in df.iterrows():
                values = [str(value).strip() for value in row.tolist()]
                if not any(values):
                    continue
                lines.append(" | ".join(values))

            lines.append("")  # Blank line between sheets

        return "\n".join(lines).strip()

    def _build_chunks(self, document: KnowledgeDocument, text: str) -> List[Dict[str, str]]:
        chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        payloads = []
        for idx, chunk in enumerate(chunks):
            payloads.append(
                {
                    "chunk_id": f"{document.doc_id}_{idx}",
                    "doc_id": document.doc_id,
                    "domain": document.domain,
                    "source_file": document.original_name,
                    "content": chunk,
                    "metadata": {
                        "tags": document.tags or [],
                        "description": document.description,
                        "uploaded_by": document.uploaded_by,
                    },
                }
            )
        return payloads

    def _load_model(self) -> SentenceTransformer:
        if self._model:
            return self._model
        with self._model_lock:
            if self._model is None:
                logger.info("Loading document embedding model: %s", self.embedding_model_name)
                self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _embed_chunks(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    # -------------------------------------------------------------- persistence
    def _persist_domain_data(
        self,
        domain: str,
        doc_id: str,
        chunks: List[Dict[str, str]],
        vectors: np.ndarray,
    ) -> None:
        domain_dir = self.data_root / domain
        chunks_dir = domain_dir / "chunks"
        embeddings_dir = domain_dir / "embeddings"
        indices_dir = domain_dir / "indices"
        for directory in (chunks_dir, embeddings_dir, indices_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # Persist chunks json
        chunks_payload, chunk_stats = self._load_existing_chunks(chunks_dir)
        chunks_payload = [chunk for chunk in chunks_payload if chunk.get("doc_id") != doc_id]
        chunks_payload.extend(chunks)
        chunk_stats = {
            "total_chunks": len(chunks_payload),
            "updated_at": datetime.utcnow().isoformat(),
        }
        chunks_file = chunks_dir / "chunks.json"
        chunks_file.write_text(
            json.dumps({"chunks": chunks_payload, "statistics": chunk_stats}, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

        # Persist embeddings pickle
        embeddings_payload = self._load_existing_embeddings(embeddings_dir)
        embeddings_payload = [record for record in embeddings_payload if record.get("doc_id") != doc_id]
        for chunk, vector in zip(chunks, vectors):
            record = dict(chunk)
            record["embedding"] = vector.tolist()
            embeddings_payload.append(record)

        embeddings_stats = {
            "total_embeddings": len(embeddings_payload),
            "embedding_dimension": vectors.shape[1],
            "updated_at": datetime.utcnow().isoformat(),
        }
        embeddings_file = embeddings_dir / "embeddings.pkl"
        with embeddings_file.open("wb") as fh:
            pickle.dump({"embeddings": embeddings_payload, "statistics": embeddings_stats}, fh)

        # Build FAISS index from all embeddings
        self._build_faiss_index(embeddings_payload, vectors.shape[1], indices_dir)

    def _load_existing_chunks(self, chunks_dir: Path) -> tuple[List[Dict[str, str]], Dict[str, str]]:
        chunks_file = chunks_dir / "chunks.json"
        if chunks_file.exists():
            try:
                data = json.loads(chunks_file.read_text(encoding='utf-8'))
                return data.get("chunks", []), data.get("statistics", {})
            except json.JSONDecodeError:
                logger.warning("Failed to parse existing chunks file at %s", chunks_file)
            except UnicodeDecodeError:
                logger.warning("UTF-8 decode failed for chunks file at %s", chunks_file)
        return [], {}

    def _load_existing_embeddings(self, embeddings_dir: Path) -> List[Dict[str, str]]:
        embeddings_file = embeddings_dir / "embeddings.pkl"
        if embeddings_file.exists():
            try:
                with embeddings_file.open("rb") as fh:
                    data = pickle.load(fh)
                    return data.get("embeddings", [])
            except Exception:  # noqa: BLE001
                logger.warning("Failed to read embeddings pickle at %s", embeddings_file, exc_info=True)
        return []

    def _build_faiss_index(self, embeddings_payload: List[Dict[str, str]], dimension: int, indices_dir: Path) -> None:
        if not embeddings_payload:
            logger.warning("No embeddings available to build index")
            processing_logger.log_warning("VECTORIZATION", "No embeddings available for index creation")
            return

        # Log existing vectors count
        existing_count = len(embeddings_payload)
        processing_logger.log_vectorization_start(indices_dir.parent.name, existing_count)

        # Convert to numpy array
        processing_logger.log_vectorization_progress("array_conversion", f"Converting {existing_count} embeddings to numpy array")
        vectors = np.array([record["embedding"] for record in embeddings_payload], dtype=np.float32)
        if vectors.size == 0:
            logger.warning("Embedding payload produced empty vector set")
            processing_logger.log_warning("VECTORIZATION", "Embedding payload produced empty vector set")
            return

        # Normalize vectors
        processing_logger.log_vectorization_progress("normalization", "Normalizing L2 vectors")
        faiss.normalize_L2(vectors)

        # Create FAISS index
        processing_logger.log_vectorization_progress("index_creation", f"Creating FAISS IndexFlatIP with dimension={dimension}")
        index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to index
        processing_logger.log_vectorization_progress("index_population", f"Adding {len(vectors)} vectors to index")
        index.add(vectors)

        # Write index to disk
        index_path = indices_dir / "faiss_index.index"
        processing_logger.log_vectorization_progress("index_serialization", f"Writing index to {index_path.name}")
        faiss.write_index(index, str(index_path))
        
        # Get index file size
        index_size_mb = index_path.stat().st_size / (1024 * 1024) if index_path.exists() else 0

        # Create and save manifest
        processing_logger.log_vectorization_progress("manifest_creation", f"Creating manifest with {len(embeddings_payload)} entries")
        manifest = {
            "domain": indices_dir.parent.name,
            "total_vectors": len(embeddings_payload),
            "embedding_dimension": dimension,
            "embedding_model": self.embedding_model_name,
            "updated_at": datetime.utcnow().isoformat(),
            "metadata_mapping": [
                {
                    "chunk_id": record["chunk_id"],
                    "doc_id": record["doc_id"],
                    "source_file": record["source_file"],
                }
                for record in embeddings_payload
            ],
        }
        manifest_path = indices_dir / "index_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        
        processing_logger.log_vectorization_complete(len(embeddings_payload), index_size_mb)


_document_ingestion_service: Optional[DocumentIngestionService] = None


def get_document_ingestion_service() -> DocumentIngestionService:
    global _document_ingestion_service
    if _document_ingestion_service is None:
        _document_ingestion_service = DocumentIngestionService()
    return _document_ingestion_service


__all__ = ["DocumentIngestionService", "get_document_ingestion_service", "chunk_text"]

