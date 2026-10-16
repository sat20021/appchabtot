"""
RAG API routes for unified server.
These endpoints expose RAG functionality in the unified server's Swagger docs.
"""
from functools import lru_cache
import importlib
import logging
import os
from pathlib import Path
import sys
import time
from typing import List, Optional, Annotated

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Internal Helpers -------------------------------------------------------

@lru_cache()
def _discover_rag_path() -> Optional[str]:
    """
    Attempt to locate the RAG project directory relative to this file.

    The unified server is nested inside `Lead_agent_2.O/Machine_Learning/...`
    so we search upward for a directory that contains a `RAG` sibling. If an
    explicit `RAG_ROOT` environment variable is provided, we honor that first.
    """
    env_rag_root = os.getenv("RAG_ROOT")
    if env_rag_root:
        candidate = Path(env_rag_root).expanduser().resolve()
        if candidate.exists():
            return str(candidate)

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        candidate = parent / "RAG"
        if candidate.exists():
            return str(candidate)
    return None



def _ensure_rag_path() -> None:
    rag_path = _discover_rag_path()
    if not rag_path:
        raise RuntimeError(
            "Unable to locate the RAG project. Set the RAG_ROOT environment "
            "variable or ensure the repository layout contains a 'RAG' "
            "directory alongside this module."
        )
    # Don't add to sys.path to avoid namespace collision with local 'app' module
    return rag_path


def _import_rag_module_from_file(module_name: str, rag_path: str):
    """
    Import a module directly from file path to avoid namespace collision.
    
    The RAG project has an 'app' package that conflicts with the main project's
    'app' package. Using spec_from_file_location allows us to import the RAG
    modules without polluting sys.path.
    """
    import importlib.util
    
    # Convert module name to file path
    # e.g., "app.services.document_query" -> "app/services/document_query.py"
    module_path = Path(rag_path) / module_name.replace(".", "/")
    
    # Try .py file first
    if module_path.with_suffix(".py").exists():
        file_path = module_path.with_suffix(".py")
    elif (module_path / "__init__.py").exists():
        file_path = module_path / "__init__.py"
    else:
        raise ModuleNotFoundError(f"Cannot find module file for {module_name} in {rag_path}")
    
    # Create a unique module name to avoid cache conflicts
    unique_name = f"rag_{module_name}"
    
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Cannot create spec for {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    
    # Ensure parent modules are also available
    # This is needed for relative imports within the module
    parent_parts = module_name.split(".")[:-1]
    for i in range(len(parent_parts)):
        parent_name = ".".join(parent_parts[:i+1])
        rag_parent_name = f"rag_{parent_name}"
        if rag_parent_name not in sys.modules:
            parent_path = Path(rag_path) / "/".join(parent_parts[:i+1])
            parent_init = parent_path / "__init__.py"
            if parent_init.exists():
                parent_spec = importlib.util.spec_from_file_location(rag_parent_name, parent_init)
                if parent_spec and parent_spec.loader:
                    parent_module = importlib.util.module_from_spec(parent_spec)
                    sys.modules[rag_parent_name] = parent_module
    
    spec.loader.exec_module(module)
    return module


@lru_cache()
def _import_rag_module(module_name: str):
    rag_path = _ensure_rag_path()
    
    # First, try adding to sys.path temporarily
    # This is needed because the RAG modules may have internal imports
    original_path = sys.path.copy()
    try:
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)
        
        # Check if 'app' is already imported (namespace conflict)
        if 'app' in sys.modules:
            # Use direct file import to avoid conflict
            return _import_rag_module_from_file(module_name, rag_path)
        
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Try direct file import as fallback
        try:
            return _import_rag_module_from_file(module_name, rag_path)
        except Exception:
            pass
        raise RuntimeError(
            f"Unable to import '{module_name}'. Ensure the RAG project directory "
            "is available and added to PYTHONPATH."
        ) from exc
    finally:
        # Restore original path
        sys.path = original_path


def _get_ingestion_service():
    module = _import_rag_module("app.services.document_ingestion")
    return module.get_document_ingestion_service()


def _get_storage_service():
    module = _import_rag_module("app.services.document_storage")
    return module.get_document_storage_service()


def _get_query_service():
    module = _import_rag_module("app.services.document_query")
    return module.get_document_query_service()


def _get_knowledge_document_model():
    module = _import_rag_module("app.models.mongo_models")
    return module.KnowledgeDocument

# --- Pydantic Models ---

class IngestResponse(BaseModel):
    message: str
    doc_ids: List[str]

class QueryRequest(BaseModel):
    query: str
    domain: str = "tech_domain"
    top_k: int = 5
    threshold: float = 0.3

class DocumentResult(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    content: str
    score: float
    metadata: dict

class QueryResponse(BaseModel):
    results: List[DocumentResult]
    processing_time_ms: float


# --- Background Tasks ---

def process_document_background(doc_id: str):
    """
    Background task wrapper for document ingestion.
    Uses subprocess to avoid namespace conflicts with RAG's app module.
    """
    import subprocess
    import os
    
    try:
        logger.info(f"⏳ [Background] Starting processing for {doc_id}")
        
        # Get the RAG project directory
        rag_path = _discover_rag_path()
        if not rag_path:
            logger.error(f"❌ [Background] RAG path not found for {doc_id}")
            return
        
        # Create a Python script that initializes DB and processes document
        # Make sure RAG path is in sys.path for imports like app.core.s3_storage
        process_script = f'''\nimport sys
import os

# Add RAG directory to sys.path for module imports
rag_path = r"{rag_path}"
if rag_path not in sys.path:
    sys.path.insert(0, rag_path)

# Initialize MongoDB connection first
from app.core.database import init_db
init_db()

# Now process the document
from app.services.document_ingestion import get_document_ingestion_service
service = get_document_ingestion_service()
service.process_document("{doc_id}")
print("Processing complete")
'''
        
        # Prepare environment variables with absolute paths
        # The subprocess runs with cwd=rag_path, so relative paths would be wrong
        env = os.environ.copy()
        
        # Convert relative paths to absolute paths
        if "LOCAL_STORAGE_PATH" in env and env["LOCAL_STORAGE_PATH"].startswith("./"):
            # Get the main project root (parent of Lead_agent_2.O)
            project_root = Path(rag_path).parent.parent
            abs_storage_path = (project_root / env["LOCAL_STORAGE_PATH"].lstrip("./")).resolve()
            env["LOCAL_STORAGE_PATH"] = str(abs_storage_path)
            logger.debug(f"Converted LOCAL_STORAGE_PATH to absolute: {env['LOCAL_STORAGE_PATH']}")
        
        if "DOCUMENT_DATA_PATH" in env and env["DOCUMENT_DATA_PATH"].startswith("./"):
            project_root = Path(rag_path).parent.parent
            abs_data_path = (project_root / env["DOCUMENT_DATA_PATH"].lstrip("./")).resolve()
            env["DOCUMENT_DATA_PATH"] = str(abs_data_path)
            logger.debug(f"Converted DOCUMENT_DATA_PATH to absolute: {env['DOCUMENT_DATA_PATH']}")
        
        # Run in subprocess with RAG as the working directory
        # Pass environment variables with absolute paths
        result = subprocess.run(
            [sys.executable, "-c", process_script],
            cwd=rag_path,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env

        )
        
        if result.returncode == 0:
            logger.info(f"✅ [Background] Finished processing for {doc_id}")
            if result.stdout:
                logger.debug(f"Processing output: {result.stdout}")
        else:
            logger.error(f"❌ [Background] Failed processing for {doc_id}: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ [Background] Timeout processing {doc_id}")
    except Exception as e:
        logger.error(f"❌ [Background] Failed processing for {doc_id}: {e}")


# --- Endpoints ---

@router.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest_documents(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="File to upload for RAG indexing")],
    domain: Annotated[str, Form(description="Target domain")],
    tags: Annotated[Optional[str], Form(description="Optional comma-separated tags")] = None,
):
    """
    Upload a single file to RAG system for indexing.
    
    The file is uploaded to storage, and then processed in the background
    to create embeddings and FAISS index for semantic search.
    
    **Supported formats**: PDF, DOCX, TXT
    """
    import uuid
    import hashlib
    import mimetypes
    import re
    from datetime import datetime
    from io import BytesIO
    from ml_utils.s3_handler import S3Handler
    
    def sanitize_filename(filename: str) -> str:
        if not filename:
            return "document"
        cleaned = filename.strip().replace(" ", "_")
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
        return cleaned or "document"
    
    try:
        KnowledgeDocument = _get_knowledge_document_model()
        s3_handler = S3Handler()
        doc_ids = []
        
        start_time = time.time()
        
        # Parse tags
        tag_list = [t.strip() for t in tags.split(",")] if tags else []

        # Process the single file
        try:
            # Generate doc_id and filenames
            doc_id = uuid.uuid4().hex
            safe_original = sanitize_filename(file.filename or "document")
            stored_name = f"{doc_id}_{safe_original}"
            
            # Determine MIME type
            mime_type = (
                file.content_type
                or mimetypes.guess_type(file.filename or "")[0]
                or "application/octet-stream"
            )
            
            # Read file content
            file_content = await file.read()
            file_buffer = BytesIO(file_content)
            size = len(file_content)
            checksum = hashlib.sha256(file_content).hexdigest()
            
            # Upload to storage
            s3_key = f"documents/{domain}/{stored_name}"
            storage_path = None
            
            # Use S3Handler (handles both S3 and local storage)
            success = s3_handler.upload_file(file_content, s3_key) if hasattr(s3_handler, 'upload_binary') else False
            # Fallback if upload_binary doesn't exist
            if not success:
                if s3_handler.use_s3:
                    if s3_handler.s3_client:
                        try:
                            s3_handler.s3_client.put_object(
                                Bucket=s3_handler.bucket_name,
                                Key=s3_key,
                                Body=file_content,
                                ContentType=mime_type
                            )
                            storage_path = f"s3://{s3_handler.bucket_name}/{s3_key}"
                            logger.info(f"✅ Uploaded to S3: {storage_path}")
                        except Exception as upload_err:
                            logger.error(f"❌ S3 upload failed: {upload_err}")
                            raise HTTPException(status_code=500, detail="Failed to upload to S3")
                    else:
                        raise HTTPException(status_code=500, detail="S3 client not initialized")
                else:
                    try:
                        local_path = s3_handler._get_local_storage_path(s3_key)
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_bytes(file_content)
                        storage_path = str(local_path.resolve())
                        logger.info(f"✅ Saved to local storage: {storage_path}")
                    except Exception as upload_err:
                        logger.error(f"❌ Local storage save failed: {upload_err}")
                        raise HTTPException(status_code=500, detail="Failed to save file locally")
            
            # Create MongoDB document
            document = KnowledgeDocument(
                doc_id=doc_id,
                domain=domain,
                file_name=stored_name,
                original_name=file.filename or safe_original,
                content_type=mime_type,
                storage_path=storage_path,
                size_bytes=size,
                checksum=checksum,
                tags=tag_list,
                uploaded_by="api_user",
                status="active",
                uploaded_at=datetime.utcnow(),
            )
            document.save()
            doc_ids.append(doc_id)
            
            # Queue for processing
            background_tasks.add_task(process_document_background, doc_id)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

        duration = time.time() - start_time
        logger.info(f"⚡ Uploaded file in {duration:.2f}s")

        return IngestResponse(
            message="Successfully uploaded file. Processing started in background.",
            doc_ids=doc_ids
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG service error: {str(e)}")


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(request: QueryRequest):
    """
    Semantic search over ingested documents.
    
    Uses FAISS vector similarity search to find relevant document chunks
    based on the input query.
    
    **Returns**: Top-k most similar document chunks with similarity scores.
    """
    try:
        start_time = time.time()
        
        query_service = _get_query_service()
        
        results = query_service.search(
            query=request.query,
            domain=request.domain,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Convert dataclass to dict for Pydantic
        results_dict = [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "source_file": r.source_file,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in results
        ]
        
        return QueryResponse(
            results=results_dict,
            processing_time_ms=duration_ms
        )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"RAG service error: {str(e)}")


@router.get("/documents", response_model=List[dict], tags=["RAG"])
async def list_documents(
    domain: str = "tech_domain",
    status: str = "active",
    limit: int = 100
):
    """
    List all documents in the RAG system.
    
    **Filters**: domain, status (active/archived), limit
    """
    try:
        KnowledgeDocument = _get_knowledge_document_model()
        docs = KnowledgeDocument.objects(domain=domain, status=status).limit(limit)
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        # Return empty list if MongoDB is not available
        return []


@router.get("/documents/{doc_id}", response_model=dict, tags=["RAG"])
async def get_document_details(doc_id: str):
    """
    Get details for a specific document.
    """
    try:
        KnowledgeDocument = _get_knowledge_document_model()
        doc = KnowledgeDocument.objects(doc_id=doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}/download", tags=["RAG"])
async def download_document(doc_id: str):
    """
    Downloads the original document file.
    
    Uses Redis cache for performance, fetches from storage on cache miss.
    Returns the file as a streaming attachment.
    """
    from fastapi.responses import StreamingResponse
    from ml_utils.redis_cache_handler import RedisCacheHandler
    from ml_utils.s3_handler import S3Handler
    import io
    
    try:
        # First get document info from MongoDB
        KnowledgeDocument = _get_knowledge_document_model()
        doc = KnowledgeDocument.objects(doc_id=doc_id).first()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        filename = doc.original_name
        # storage_path format: s3://bucket/key or /local/path
        storage_path = doc.storage_path
        
        # Extract S3 key from storage_path (format: s3://bucket/key)
        if storage_path.startswith("s3://"):
            # Remove s3://bucket/ prefix to get the key
            parts = storage_path.replace("s3://", "").split("/", 1)
            s3_key = parts[1] if len(parts) > 1 else ""
        else:
            # For local paths, use as-is
            s3_key = storage_path
        
        cache_key = f"rag_file:{doc_id}"
        redis_cache = RedisCacheHandler()
        
        # Try to get from Redis cache first
        if redis_cache.is_available():
            cached_data = redis_cache.get_binary(cache_key)
            if cached_data:
                logger.info(f"Serving RAG document from Redis cache: {cache_key}")
                
                # Determine content type
                content_type = _get_content_type(filename)
                
                return StreamingResponse(
                    io.BytesIO(cached_data),
                    media_type=content_type,
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
        
        # Cache miss - fetch from storage
        s3_handler = S3Handler()
        
        # Handle local storage path
        if storage_path.startswith("local://"):
            # Extract local file path from local:// prefix
            local_file_path = storage_path.replace("local://", "")
            from pathlib import Path
            local_path = Path(local_file_path)
            
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found in local storage: {local_file_path}")
            
            file_content = local_path.read_bytes()
            logger.info(f"Serving RAG document from local storage: {local_file_path}")
        elif storage_path.startswith("s3://"):
            # S3 mode
            if not s3_handler.s3_client:
                raise HTTPException(status_code=503, detail="S3 client not initialized")
            
            if not s3_handler.file_exists(s3_key):
                raise HTTPException(status_code=404, detail=f"File not found in S3: {s3_key}")
            
            # Get file content from S3
            response = s3_handler.s3_client.get_object(
                Bucket=s3_handler.bucket_name,
                Key=s3_key
            )
            file_content = response['Body'].read()
            logger.info(f"Serving RAG document from S3: {s3_key}")
        else:
            # Fallback - try local storage using s3_key
            if not s3_handler.file_exists(s3_key):
                raise HTTPException(status_code=404, detail=f"File not found: {s3_key}")
            
            local_path = s3_handler._get_local_storage_path(s3_key)
            file_content = local_path.read_bytes()
            logger.info(f"Serving RAG document from local storage: {s3_key}")
        
        # Cache the content in Redis (TTL 1 hour)
        if redis_cache.is_available():
            redis_cache.set_binary(cache_key, file_content, ttl=3600)
            logger.info(f"Cached RAG document in Redis: {cache_key}")
        
        # Determine content type
        content_type = _get_content_type(filename)
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download: {str(e)}")


def _get_content_type(filename: str) -> str:
    """Determine content type based on file extension."""
    if filename.endswith('.pdf'):
        return "application/pdf"
    elif filename.endswith('.docx'):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif filename.endswith('.doc'):
        return "application/msword"
    elif filename.endswith('.txt'):
        return "text/plain"
    elif filename.endswith('.csv'):
        return "text/csv"
    elif filename.endswith(('.xlsx', '.xls')):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        return "application/octet-stream"


@router.delete("/documents/{doc_id}", tags=["RAG"])
async def delete_document(
    doc_id: str,
    hard_delete: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Delete a document from the RAG system.
    
    - **hard_delete=False** (default): Archives the document (soft delete)
    - **hard_delete=True**: Permanently removes the document and file
    
    Both options remove the document from the RAG index and rebuild FAISS.
    """
    from datetime import datetime
    from ml_utils.s3_handler import S3Handler
    
    try:
        # Get document from MongoDB
        KnowledgeDocument = _get_knowledge_document_model()
        doc = KnowledgeDocument.objects(doc_id=doc_id).first()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        domain = doc.domain
        storage_path = doc.storage_path
        
        # Handle hard delete - remove file from storage
        if hard_delete:
            if storage_path.startswith("s3://"):
                # Extract S3 key from storage path
                parts = storage_path.replace("s3://", "").split("/", 1)
                if len(parts) == 2:
                    s3_key = parts[1]
                try:
                    s3_handler = S3Handler()
                    s3_handler.delete_file(s3_key)
                    logger.info(f"✅ Deleted file from storage: {s3_key}")
                except Exception as e:
                    logger.warning(f"Failed to delete file from storage: {e}")
            
            # Update status to deleted
            doc.status = "deleted"
            doc.deleted_at = datetime.utcnow()
        else:
            # Soft delete - archive
            doc.status = "archived"
        
        doc.save()
        logger.info(f"✅ Document {doc_id} {'hard deleted' if hard_delete else 'archived'}")
        
        # Invalidate query cache for the domain
        try:
            query_service = _get_query_service()
            query_service.invalidate_domain_cache(domain)
            logger.info(f"✅ Invalidated query cache for domain: {domain}")
        except Exception as e:
            logger.warning(f"Failed to invalidate query cache: {e}")
        
        # Invalidate Redis file cache for this document
        try:
            from ml_utils.redis_cache_handler import RedisCacheHandler
            redis_cache = RedisCacheHandler()
            if redis_cache.is_available():
                cache_key = f"rag_file:{doc_id}"
                redis_cache.invalidate(cache_key)
                logger.info(f"✅ Invalidated file cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to invalidate file cache: {e}")
        
        return {
            "message": "Document deleted successfully" if hard_delete else "Document archived successfully",
            "doc_id": doc_id,
            "hard_delete": hard_delete,
            "domain": domain
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains", tags=["RAG"])
def list_rag_domains():
    """List available RAG domains."""
    try:
        storage_service = _get_storage_service()
        domains = storage_service.get_available_domains()
        return {"domains": domains, "count": len(domains)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list domains: {str(e)}")


@router.get("/health", tags=["RAG"])
def rag_health_check():
    """Check if RAG service is operational."""
    try:
        query_service = _get_query_service()
        return {
            "status": "ok",
            "service": "RAG",
            "model": query_service.model_name,
            "threshold": query_service.score_threshold
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "RAG",
            "error": str(e)
        }
@router.post("/sync", tags=["RAG"])
async def sync_storage():
    """
    Force synchronization between physical storage and database.
    Runs in a subprocess to avoid namespace collisions.
    """
    import subprocess
    import sys
    
    try:
        # Assuming running from project root
        script_path = Path("scripts/sync_db_with_storage.py").resolve()
        if not script_path.exists():
            # Try finding it relative to this file if not at root
            # This file is in Lead_agent_2.O/Machine_Learning/ML_models/api
            # scripts is in root/scripts
            root = Path(__file__).resolve().parents[4]
            script_path = (root / "scripts" / "sync_db_with_storage.py").resolve()
            
        if not script_path.exists():
             raise FileNotFoundError(f"Sync script not found at {script_path}")

        logger.info(f"🔄 Executing sync script: {script_path}")
        
        # Run script
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✅ Sync script output: {proc.stdout}")
        return {"message": "Synchronization complete", "details": proc.stdout}
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Sync script failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Sync execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync execution failed: {str(e)}")

async def perform_rag_sync():
    """
    Public helper to trigger RAG storage sync on startup.
    """
    import subprocess
    import sys
    
    try:
        logger.info("🔄 Auto-syncing RAG storage (Subprocess)...")
        script_path = Path("scripts/sync_db_with_storage.py").resolve()
        
        if not script_path.exists():
             root = Path(__file__).resolve().parents[4]
             script_path = (root / "scripts" / "sync_db_with_storage.py").resolve()
        
        if script_path.exists():
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True
            )
            if proc.returncode == 0:
                logger.info("✅ RAG Sync complete on startup")
            else:
                logger.warning(f"⚠️ RAG Sync script failed: {proc.stderr}")
        else:
            logger.warning("⚠️ RAG Sync script not found, skipping auto-sync")
            
    except Exception as e:
        logger.error(f"⚠️ Failed to auto-sync RAG storage: {e}")
