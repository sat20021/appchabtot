"""
Document storage service for RAG knowledge base assets.
Handles file persistence plus Mongo metadata for uploaded documents.
Supports both local storage and AWS S3.
"""
from __future__ import annotations

import json
import hashlib
import mimetypes
import os
import re
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from io import BytesIO

from fastapi import UploadFile

from app.models.mongo_models import KnowledgeDocument, DocumentStatus
from app.core.s3_storage import get_s3_storage_service

DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MB

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe filename."""
    if not filename:
        return "document"
    cleaned = filename.strip().replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    return cleaned or "document"


def parse_tag_input(raw_tags: Optional[str]) -> List[str]:
    """Parse comma-separated or JSON list strings into a tag list."""
    if raw_tags is None:
        return []
    raw_tags = raw_tags.strip()
    if not raw_tags:
        return []
    # Try JSON list
    try:
        parsed = json.loads(raw_tags)
        if isinstance(parsed, list):
            return [str(tag).strip() for tag in parsed if str(tag).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback to comma-separated string
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def parse_metadata_input(raw_metadata: Optional[str]) -> Dict[str, Any]:
    """Parse metadata JSON string into a dictionary."""
    if not raw_metadata:
        return {}
    try:
        parsed = json.loads(raw_metadata)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


class DocumentStorageService:
    """
    Service object that coordinates filesystem/S3 + Mongo metadata.
    Automatically uses S3 if configured, falls back to local storage.
    """

    def __init__(self, base_path: Optional[str] = None, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.base_path = self._resolve_base_path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        
        # Initialize S3 service
        self.s3_service = get_s3_storage_service()
        self.use_s3 = self.s3_service.is_s3_enabled()
        
        if self.use_s3:
            logger.info(f"📦 Document storage MODE: S3 enabled | Bucket: {self.s3_service.bucket_name}")
        else:
            logger.info(f"📁 Document storage MODE: Local storage | Path: {self.base_path}")

    def _resolve_base_path(self, base_path: Optional[str]) -> Path:
        if base_path:
            return Path(base_path)
        
        # Use centralized LOCAL_STORAGE_PATH if available
        local_storage = os.getenv("LOCAL_STORAGE_PATH")
        if local_storage:
             # Structure: ./storage/rag/documents
            return (Path(local_storage) / "rag" / "documents").resolve()

        env_path = os.getenv("DOCUMENT_STORAGE_PATH")
        if env_path:
            return Path(env_path).resolve()
        project_root = Path(__file__).resolve().parents[2]
        return (project_root / "docs_storage").resolve()

    def _ensure_domain_dir(self, domain: str) -> Path:
        domain_path = self.base_path / domain
        domain_path.mkdir(parents=True, exist_ok=True)
        return domain_path

    def _absolute_path(self, stored_path: str) -> Path:
        # Handle legacy local:// prefix
        if stored_path.startswith("local://"):
            stored_path = stored_path.replace("local://", "", 1)
        
        path = Path(stored_path)
        if not path.is_absolute():
            return self.base_path / path
        return path
    
    def _get_s3_key(self, domain: str, filename: str) -> str:
        """Generate S3 key for document"""
        return f"documents/{domain}/{filename}"

    async def save_upload(
        self,
        file: UploadFile,
        domain: str,
        *,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        uploaded_by: Optional[str] = None,
    ) -> KnowledgeDocument:
        """
        Persist uploaded file and return Mongo document record.
        Automatically uploads to S3 if configured, otherwise uses local storage.
        """
        domain = domain or "domain_3"
        tags = tags or []
        metadata = metadata or {}

        doc_id = uuid.uuid4().hex
        safe_original = sanitize_filename(file.filename or "document")
        stored_name = f"{doc_id}_{safe_original}"
        
        # Determine MIME type
        mime_type = (
            file.content_type
            or mimetypes.guess_type(file.filename or "")[0]
            or "application/octet-stream"
        )
        
        # Calculate checksum and size while reading file
        checksum = hashlib.sha256()
        size = 0
        file_buffer = BytesIO()
        
        while True:
            chunk = await file.read(self.chunk_size)
            if not chunk:
                break
            size += len(chunk)
            checksum.update(chunk)
            file_buffer.write(chunk)
        
        file_buffer.seek(0)  # Reset to beginning
        
        # Determine storage location
        storage_path = None
        s3_key = None
        
        if self.use_s3:
            # Upload to S3
            s3_key = self._get_s3_key(domain, stored_name)
            logger.info(f"📦 STORAGE MODE: S3 upload | domain={domain}, file={safe_original}")
            success = self.s3_service.upload_file(file_buffer, s3_key, mime_type)
            
            if success:
                storage_path = f"s3://{self.s3_service.bucket_name}/{s3_key}"
                logger.info(f"✅ File stored in S3: {storage_path}")
            else:
                # Fall back to local storage on S3 failure
                logger.warning("⚠️ S3 upload failed, falling back to LOCAL storage")
                self.use_s3 = False  # Temporarily disable S3 for this upload
        
        if not self.use_s3:
            # Use local storage
            domain_dir = self._ensure_domain_dir(domain)
            target_path = domain_dir / stored_name
            logger.info(f"📁 STORAGE MODE: Local upload | domain={domain}, file={safe_original}")
            
            with target_path.open("wb") as buffer:
                file_buffer.seek(0)
                buffer.write(file_buffer.read())
            
            storage_path = str(target_path.resolve())
            logger.info(f"✅ File stored locally: {storage_path}")

        # Try to reset file pointer
        try:
            await file.seek(0)
        except Exception:
            pass

        # Create MongoDB document
        document = KnowledgeDocument(
            doc_id=doc_id,
            domain=domain,
            file_name=stored_name,
            original_name=file.filename or safe_original,
            content_type=mime_type,
            storage_path=storage_path,
            size_bytes=size,
            checksum=checksum.hexdigest(),
            tags=tags,
            description=description,
            metadata=metadata or {},
            uploaded_by=uploaded_by,
            status=DocumentStatus.ACTIVE,
            uploaded_at=datetime.utcnow(),
        )
        document.save()
        return document

    def list_documents(
        self,
        *,
        domain: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        limit: int = 50,
    ) -> List[KnowledgeDocument]:
        """Return documents filtered by domain/status."""
        queryset = KnowledgeDocument.objects
        if domain:
            queryset = queryset.filter(domain=domain)
        if status:
            queryset = queryset.filter(status=status)
        else:
            queryset = queryset.filter(status__ne=DocumentStatus.DELETED)
        return list(queryset.order_by("-uploaded_at")[:limit])

    def get_document(
        self,
        doc_id: str,
        *,
        include_deleted: bool = False,
    ) -> Optional[KnowledgeDocument]:
        """Fetch a document by ID."""
        queryset = KnowledgeDocument.objects(doc_id=doc_id)
        if not include_deleted:
            queryset = queryset.filter(status__ne=DocumentStatus.DELETED)
        return queryset.first()

    def get_file_path(self, document: KnowledgeDocument) -> Path:
        """
        Resolve absolute file path for document.
        If stored in S3, downloads to local cache first.
        """
        storage_path = document.storage_path
        
        # Check if stored in S3
        if storage_path.startswith("s3://"):
            # Extract S3 key from storage path
            # Format: s3://bucket-name/documents/domain/filename
            parts = storage_path.replace("s3://", "").split("/", 1)
            if len(parts) == 2:
                s3_key = parts[1]
                
                # Create local cache path
                cache_dir = self.base_path / "s3_cache" / document.domain
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / document.file_name
                
                # Download from S3 if not already cached
                if not cache_path.exists():
                    logger.info(f"🔄 Cache MISS - downloading from S3 | key={s3_key}")
                    success = self.s3_service.download_to_file(s3_key, cache_path)
                    if not success:
                        raise FileNotFoundError(f"Failed to download from S3: {s3_key}")
                else:
                    # File already in cache - log cache hit
                    from app.core.observability.document_processing_logger import get_document_processing_logger
                    processing_logger = get_document_processing_logger()
                    processing_logger.log_s3_cache_hit(str(cache_path))
                    processing_logger.log_s3_cache_hit(str(cache_path))
                    logger.info(f"✅ Cache HIT - using cached file | path={cache_path}")
                
                return cache_path
        
        # Local storage
        return self._absolute_path(storage_path)

    def delete_document(
        self,
        doc_id: str,
        *,
        hard_delete: bool = False,
    ) -> Optional[KnowledgeDocument]:
        """
        Archive or permanently delete a document.
        Also removes chunks/embeddings from RAG index and rebuilds index.
        """
        document = self.get_document(doc_id, include_deleted=True)
        if not document:
            return None

        # Remove from RAG index (chunks, embeddings, and rebuild FAISS)
        logger.info(f"🗑️ Removing document {doc_id} from RAG index...")
        
        # Log deletion start
        try:
            from app.core.observability.document_processing_logger import get_document_processing_logger
            processing_logger = get_document_processing_logger()
            processing_logger.set_document_context(doc_id)
            processing_logger.start_stage("DELETION")
            processing_logger._log('info', 'DELETION', 
                                  f"Removing document from RAG index | Domain: {document.domain} | Hard delete: {hard_delete}")
        except Exception:
            pass  # Logging is optional
        
        self._remove_from_rag_index(document.domain, doc_id)

        if hard_delete:
            self._remove_file(document)
            document.status = DocumentStatus.DELETED
            document.deleted_at = datetime.utcnow()
        else:
            document.status = DocumentStatus.ARCHIVED
        document.save()
        
        # Invalidate query service cache to force reload of updated index
        try:
            from app.services.document_query import get_document_query_service
            query_service = get_document_query_service()
            query_service.invalidate_domain_cache(document.domain)
            logger.info(f"✅ Invalidated query cache for domain: {document.domain}")
        except Exception as e:
            logger.warning(f"Failed to invalidate query cache: {e}")
        
        logger.info(f"✅ Document {doc_id} {'hard deleted' if hard_delete else 'archived'}")
        
        # Log deletion complete
        try:
            from app.core.observability.document_processing_logger import get_document_processing_logger
            processing_logger = get_document_processing_logger()
            processing_logger.end_stage("DELETION", 
                                       action="hard_delete" if hard_delete else "archive",
                                       domain=document.domain)
            processing_logger.clear_document_context()
        except Exception:
            pass  # Logging is optional
        
        return document
    
    def _remove_from_rag_index(self, domain: str, doc_id: str) -> None:
        """
        Remove document's chunks and embeddings from RAG index and rebuild FAISS index.
        """
        try:
            import json
            import pickle
            import numpy as np
            import faiss
            from pathlib import Path
            import os
            
            # Get data root path
            local_storage = os.getenv("LOCAL_STORAGE_PATH")
            if local_storage:
                data_root = Path(local_storage) / "rag" / "data"
            else:
                 data_root = Path(os.getenv("DOCUMENT_DATA_PATH", Path(__file__).resolve().parents[2] / "data_storage"))
            
            domain_dir = data_root / domain
            
            if not domain_dir.exists():
                logger.warning(f"Domain directory not found: {domain_dir}")
                return
            
            chunks_dir = domain_dir / "chunks"
            embeddings_dir = domain_dir / "embeddings"
            indices_dir = domain_dir / "indices"
            
            # Remove chunks
            chunks_file = chunks_dir / "chunks.json"
            if chunks_file.exists():
                try:
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    original_count = len(data.get("chunks", []))
                    # Filter out chunks for this document
                    data["chunks"] = [chunk for chunk in data.get("chunks", []) if chunk.get("doc_id") != doc_id]
                    new_count = len(data["chunks"])
                    removed_count = original_count - new_count
                    
                    if removed_count > 0:
                        data["statistics"] = {
                            "total_chunks": new_count,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                        with open(chunks_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        logger.info(f"✅ Removed {removed_count} chunks for document {doc_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to remove chunks: {e}")
            
            # Remove embeddings
            embeddings_file = embeddings_dir / "embeddings.pkl"
            if embeddings_file.exists():
                try:
                    with open(embeddings_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    original_count = len(data.get("embeddings", []))
                    # Filter out embeddings for this document
                    data["embeddings"] = [emb for emb in data.get("embeddings", []) if emb.get("doc_id") != doc_id]
                    new_count = len(data["embeddings"])
                    removed_count = original_count - new_count
                    
                    if removed_count > 0:
                        data["statistics"] = {
                            "total_embeddings": new_count,
                            "embedding_dimension": data.get("statistics", {}).get("embedding_dimension", 384),
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                        with open(embeddings_file, 'wb') as f:
                            pickle.dump(data, f)
                        logger.info(f"✅ Removed {removed_count} embeddings for document {doc_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to remove embeddings: {e}")
            
            # Rebuild FAISS index
            embeddings_file_path = embeddings_dir / "embeddings.pkl"
            self._rebuild_faiss_index(domain, embeddings_file_path if embeddings_file_path.exists() else None)
            
        except Exception as e:
            logger.error(f"❌ Failed to remove document from RAG index: {e}", exc_info=True)
    
    async def sync_database_with_storage(self) -> Dict[str, int]:
        """
        Sync MongoDB records with local file storage.
        Scans self.base_path (e.g. storage/rag/documents) for files and updates DB.
        """
        if self.use_s3:
            logger.info("Skipping local storage sync (S3 enabled)")
            return {"added": 0, "updated": 0, "deleted": 0}

        stats = {"added": 0, "updated": 0, "deleted": 0}
        found_doc_ids = set()
        
        logger.info(f"🔄 Starting storage sync scan at: {self.base_path}")
        
        # 1. Scan and Upsert
        if self.base_path.exists():
            for domain_path in self.base_path.iterdir():
                if domain_path.is_dir():
                    domain = domain_path.name
                    for file_path in domain_path.glob("*"):
                        if file_path.is_file():
                            filename = file_path.name
                            
                            # Parse doc_id and original_name from {doc_id}_{original_name}
                            doc_id = None
                            original_name = filename
                            if "_" in filename:
                                parts = filename.split("_", 1)
                                if len(parts) == 2 and len(parts[0]) == 32: # Basic uuid check
                                    doc_id = parts[0]
                                    original_name = parts[1]
                            
                            if not doc_id:
                                logger.debug(f"Skipping non-standard file: {filename}")
                                continue
                                
                            found_doc_ids.add(doc_id)
                            
                            # Upsert
                            existing = KnowledgeDocument.objects(doc_id=doc_id).first()
                            abs_path = str(file_path.resolve())
                            
                            if not existing:
                                logger.info(f"➕ Found new file: {filename}")
                                try:
                                    doc = KnowledgeDocument(
                                        doc_id=doc_id,
                                        original_name=original_name,
                                        file_name=filename,
                                        domain=domain,
                                        storage_path=abs_path,
                                        status=DocumentStatus.ACTIVE,
                                        uploaded_at=datetime.utcnow(),
                                        size_bytes=file_path.stat().st_size,
                                        content_type=mimetypes.guess_type(filename)[0] or "application/octet-stream"
                                    )
                                    doc.save()
                                    stats["added"] += 1
                                except Exception as e:
                                    logger.error(f"Failed to create doc {doc_id}: {e}")
                            else:
                                # Update path if changed
                                if existing.storage_path != abs_path:
                                    existing.storage_path = abs_path
                                    existing.status = DocumentStatus.ACTIVE # Revive if was deleted
                                    existing.save()
                                    stats["updated"] += 1
        
        # 2. Cleanup Orphans
        # Only check docs that claim to be local (not starting with s3://)
        all_docs = KnowledgeDocument.objects()
        for doc in all_docs:
            if doc.storage_path and doc.storage_path.startswith("s3://"):
                continue
            
            if doc.doc_id not in found_doc_ids:
                # Double check existence before deleting
                try:
                    path = Path(doc.storage_path) if doc.storage_path else None
                    if path and path.exists():
                        # It exists but we missed it? Maybe mismatched ID parsing?
                        # Keep it.
                        continue
                except Exception:
                    pass
                
                logger.info(f"🗑️ Removing orphan record: {doc.doc_id}")
                doc.delete()
                stats["deleted"] += 1
                
        logger.info(f"✅ Sync complete: {stats}")
        return stats

    def _remove_file(self, document: KnowledgeDocument) -> None:
        """Remove the physical file if it exists (from S3 or local storage)."""
        storage_path = document.storage_path
        
        # Check if stored in S3
        if storage_path.startswith("s3://"):
            # Extract S3 key
            parts = storage_path.replace("s3://", "").split("/", 1)
            if len(parts) == 2:
                s3_key = parts[1]
                logger.info(f"🗑️ Deleting from S3: {s3_key}")
                self.s3_service.delete_file(s3_key)
                
                # Also remove from local cache if exists
                cache_path = self.base_path / "s3_cache" / document.domain / document.file_name
                if cache_path.exists():
                    cache_path.unlink()
                    logger.debug(f"🗑️ Removed from cache: {cache_path}")
        else:
            # Local storage
            if path.exists():
                path.unlink()
                logger.info(f"🗑️ Deleted locally: {path}")

    def get_available_domains(self) -> List[str]:
        """List all available domains from storage (Local or S3)."""
        domains = set()
        
        # 1. Check Local
        if self.base_path.exists():
            for d in self.base_path.iterdir():
                if d.is_dir() and not d.name.startswith('.'):
                    domains.add(d.name)
        
        # 2. Check S3
        if self.use_s3 and self.s3_service.s3_client:
            try:
                # List objects in bucket with prefix 'documents/'
                # Use Delimiter='/' to get "subdirectories" (CommonPrefixes)
                paginator = self.s3_service.s3_client.get_paginator('list_objects_v2')
                # We want to list what's under 'documents/'
                # Output CommonPrefixes will be like 'documents/domain1/', 'documents/domain2/'
                for page in paginator.paginate(Bucket=self.s3_service.bucket_name, Prefix='documents/', Delimiter='/'):
                    if 'CommonPrefixes' in page:
                        for prefix in page['CommonPrefixes']:
                            # prefix['Prefix'] is 'documents/domain_name/'
                            parts = prefix['Prefix'].strip('/').split('/')
                            if len(parts) >= 2:
                                domain = parts[1]
                                if domain:
                                    domains.add(domain)
            except Exception as e:
                logger.warning(f"Failed to list S3 domains: {e}")
                
        return list(domains)



_document_storage_service: Optional[DocumentStorageService] = None


def get_document_storage_service() -> DocumentStorageService:
    """Singleton accessor for DocumentStorageService."""
    global _document_storage_service
    if _document_storage_service is None:
        _document_storage_service = DocumentStorageService()
    return _document_storage_service


__all__ = [
    "DocumentStorageService",
    "DocumentStatus",
    "get_document_storage_service",
    "sanitize_filename",
    "parse_tag_input",
    "parse_metadata_input",
]

def get_file_stream(self, doc_id: str) -> Optional[tuple[BytesIO, str, str]]:
    """
    Retrieve file content as a stream.
    Returns: (file_stream, filename, content_type)
    """
    doc = self.get_document(doc_id)
    if not doc:
        return None
        
    try:
        file_stream = BytesIO()
        
        if self.use_s3 and doc.storage_path.startswith("s3://"):
            # Extract key from s3://bucket/key
            key = doc.storage_path.replace(f"s3://{self.s3_service.bucket_name}/", "")
            
            # We need to use the S3 client directly to get the object
            client = self.s3_service.s3_client
            if not client:
                return None
                
            response = client.get_object(Bucket=self.s3_service.bucket_name, Key=key)
            file_stream.write(response['Body'].read())
            file_stream.seek(0)
            
        else:
            # Local file
            path = Path(doc.storage_path)
            if not path.exists():
                return None
            with open(path, "rb") as f:
                file_stream.write(f.read())
            file_stream.seek(0)
            
        return file_stream, doc.original_name, doc.content_type
        
    except Exception as e:
        logger.error(f"Failed to retrieve file stream for {doc_id}: {e}")
        return None
