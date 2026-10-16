import logging
import time
import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.database import init_db
from app.services.document_storage import get_document_storage_service
from app.services.document_ingestion import get_document_ingestion_service
from app.services.document_query import get_document_query_service, DocumentResult
from app.models.mongo_models import KnowledgeDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Enterprise RAG API...")
    init_db()
    
    # Auto-sync local storage
    try:
        storage_service = get_document_storage_service()
        await storage_service.sync_database_with_storage()
    except Exception as e:
        logger.error(f"⚠️ Failed to auto-sync storage on startup: {e}")
        
    yield
    # Shutdown
    logger.info("🛑 Shutting down Enterprise RAG API...")

app = FastAPI(
    title="Enterprise RAG API",
    description="High-performance RAG API with S3 storage and MongoDB metadata.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class IngestResponse(BaseModel):
    message: str
    doc_ids: List[str]

class QueryRequest(BaseModel):
    query: str
    domain: str = "domain_3"
    top_k: int = 5
    threshold: float = 0.0  # Default to 0 to return all matches, let client filter

class QueryResponse(BaseModel):
    results: List[DocumentResult]
    processing_time_ms: float

# --- Background Tasks ---

def process_document_background(doc_id: str):
    """
    Background task wrapper for document ingestion.
    """
    try:
        logger.info(f"⏳ [Background] Starting processing for {doc_id}")
        service = get_document_ingestion_service()
        service.process_document(doc_id)
        logger.info(f"✅ [Background] Finished processing for {doc_id}")
    except Exception as e:
        logger.error(f"❌ [Background] Failed processing for {doc_id}: {e}")

# --- Endpoints ---

@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    domain: str = Form("domain_3"),
    tags: Optional[str] = Form(None),
):
    """
    Upload multiple files to S3 and trigger background ingestion.
    """
    storage_service = get_document_storage_service()
    doc_ids = []
    
    start_time = time.time()
    
    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    for file in files:
        try:
            # Save to S3/Local and create Mongo record
            doc = await storage_service.save_upload(
                file=file,
                domain=domain,
                tags=tag_list,
                uploaded_by="api_user"
            )
            doc_ids.append(doc.doc_id)
            
            # Queue for processing
            background_tasks.add_task(process_document_background, doc.doc_id)
            
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            # Continue with other files or raise? 
            # For multi-upload, we usually want to process as many as possible.
            continue

    duration = time.time() - start_time
    logger.info(f"⚡ Uploaded {len(doc_ids)} files in {duration:.2f}s")

    return IngestResponse(
        message=f"Successfully uploaded {len(doc_ids)} files. Processing started in background.",
        doc_ids=doc_ids
    )

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Semantic search over ingested documents.
    """
    start_time = time.time()
    
    query_service = get_document_query_service()
    
    results = query_service.search(
        query=request.query,
        domain=request.domain,
        top_k=request.top_k,
        threshold=request.threshold
    )
    
    duration_ms = (time.time() - start_time) * 1000
    
    return QueryResponse(
        results=results,
        processing_time_ms=duration_ms
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- Document Management Endpoints ---

@app.get("/api/v1/documents", response_model=List[dict])
async def list_documents(
    domain: str = "domain_3",
    status: str = "active",
    limit: int = 100
):
    """
    List documents with filtering.
    """
    docs = KnowledgeDocument.objects(domain=domain, status=status).limit(limit)
    return [doc.to_dict() for doc in docs]

@app.get("/api/v1/documents/{doc_id}", response_model=dict)
async def get_document_details(doc_id: str):
    """
    Get details for a specific document.
    """
    doc = KnowledgeDocument.objects(doc_id=doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.to_dict()

@app.get("/api/v1/documents/{doc_id}/download")
async def download_document(doc_id: str):
    """
    Download the original file content.
    """
    storage_service = get_document_storage_service()
    result = storage_service.get_file_stream(doc_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="File not found or inaccessible")
        
    file_stream, filename, content_type = result
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        file_stream, 
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.delete("/api/v1/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    hard_delete: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    Delete a document.
    
    - If hard_delete=False (default): Archives the document (soft delete)
    - If hard_delete=True: Permanently removes the document and file
    
    Both options remove the document from the RAG index and rebuild FAISS.
    """
    storage_service = get_document_storage_service()
    
    # Perform deletion
    deleted_doc = storage_service.delete_document(doc_id, hard_delete=hard_delete)
    
    if not deleted_doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Invalidate query cache for the domain
    try:
        query_service = get_document_query_service()
        query_service.invalidate_domain_cache(deleted_doc.domain)
        logger.info(f"✅ Invalidated query cache for domain: {deleted_doc.domain}")
    except Exception as e:
        logger.warning(f"Failed to invalidate query cache: {e}")
    
    return {
        "message": "Document deleted successfully" if hard_delete else "Document archived successfully",
        "doc_id": doc_id,
        "hard_delete": hard_delete,
        "domain": deleted_doc.domain
    }

@app.get("/api/v1/domains", response_model=dict, tags=["Documents"])
async def list_domains():
    """List available RAG domains (folders)."""
    try:
        storage_service = get_document_storage_service()
        domains = storage_service.get_available_domains()
        return {"domains": domains, "count": len(domains)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list domains: {str(e)}")

@app.post("/api/v1/rag/sync", tags=["Maintenance"])
async def sync_storage():
    """
    Force synchronization between physical storage and database.
    Useful if files are manually added/removed from the storage directory.
    """
    try:
        storage_service = get_document_storage_service()
        stats = await storage_service.sync_database_with_storage()
        return {"message": "Synchronization complete", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@app.post("/api/v1/rag/sync", tags=["Maintenance"])
async def sync_storage():
    """
    Force synchronization between physical storage and database.
    Useful if files are manually added/removed from the storage directory.
    """
    try:
        storage_service = get_document_storage_service()
        stats = await storage_service.sync_database_with_storage()
        return {"message": "Synchronization complete", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

