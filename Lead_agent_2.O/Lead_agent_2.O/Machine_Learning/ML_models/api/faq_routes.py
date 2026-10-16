from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Annotated
import os
import shutil
from pathlib import Path
from pydantic import BaseModel

from ml_utils.s3_handler import S3Handler
from ml_utils.redis_cache_handler import RedisCacheHandler
import logging
from ml_services.faq_service import refresh_faq_index, faq_stats, ask_faq, load_faq_domain

# Import Config and Schemas
try:
    from faq_app.config import Config
    from faq_app.api.schemas import AskRequest, AskResponse
except ImportError:
    Config = None
    AskRequest = None
    AskResponse = None

logger = logging.getLogger("FAQ_Routes")

router = APIRouter()

# ...

@router.post("/ask", response_model=AskResponse)
async def ask_faq_endpoint(request: AskRequest):
    """
    Ask a question to the FAQ engine for a specific domain.
    """
    if not AskRequest:
        raise HTTPException(status_code=503, detail="FAQ Schemas not loaded")
        
    result = ask_faq(request.question, domain=request.domain, threshold=request.threshold)
    
    if result.get("answer"):
        return AskResponse(
            answer=result["answer"],
            confidence=result.get("confidence") or 0.0,
            source=result.get("source", "vector_db"),
            matched_question=result.get("matched_question"),
            domain=result.get("domain")
        )
    else:
        return AskResponse(
            answer="I'm sorry, I don't have enough information to answer that.",
            confidence=result.get("confidence") or 0.0,
            source="fallback",
            matched_question=None,
            domain=request.domain
        )

# S3 Prefix for FAQ data
S3_FAQ_PREFIX = "faq_data/"

class UploadResponse(BaseModel):
    message: str
    uploaded_files: List[str]
    domain: str
    rebuild_triggered: bool

class DeleteResponse(BaseModel):
    message: str
    deleted_from_s3: bool
    deleted_from_local: bool
    domain: str

class RebuildResponse(BaseModel):
    message: str
    status: str
    vectors_indexed: int
    domain: str

@router.get("/domains")
def list_domains():
    """List available FAQ domains."""
    try:
        s3 = S3Handler()
        domains = set()
        
        # 1. Check storage (FAQ artifacts are stored in 'faq_artifacts/')
        artifacts = s3.list_files("faq_artifacts/")
        for key in artifacts:
            parts = key.split('/')
            if len(parts) > 1:
                domains.add(parts[1])
        
        # 2. Check Local/Loaded (via Config if available)
        if Config:
            # Check data dir
            if os.path.exists(Config.DATA_DIR):
                for d in os.listdir(Config.DATA_DIR):
                    if os.path.isdir(os.path.join(Config.DATA_DIR, d)):
                        domains.add(d)

        return {"domains": list(domains), "count": len(domains)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list domains: {str(e)}")

@router.get("/health")
async def get_faq_health(domain: str = "default"):
    """
    Checks the health of the FAQ engine for a specific domain.
    """
    stats = faq_stats(domain)
    return stats

@router.get("/files")
async def list_faq_files(domain: str = "default"):
    """
    Lists all FAQ data files present in S3 or local storage for a domain with full metadata.
    Returns file objects with filename, key, size_bytes, and last_modified.
    """
    s3_handler = S3Handler()
    s3_prefix = f"{S3_FAQ_PREFIX}{domain}/"
    
    files = []
    
    # Check storage
    try:
        # Use S3Handler to list files (handles both S3 and local storage)
        s3_files = s3_handler.list_files(s3_prefix)
        
        for key in s3_files:
            filename = os.path.basename(key)
            
            # Skip empty filenames (prefix itself)
            if not filename:
                continue
            
            # For local storage, we might not have all S3 metadata, so we use defaults or simple stat
            size_bytes = 0
            last_modified = None
            
            if s3_handler.use_s3 and s3_handler.s3_client:
                # Get full metadata from S3 if possible
                try:
                    head = s3_handler.s3_client.head_object(Bucket=s3_handler.bucket_name, Key=key)
                    size_bytes = head.get('ContentLength', 0)
                    last_modified = head.get('LastModified', '').isoformat() if head.get('LastModified') else None
                except:
                    pass
            else:
                # Local stat
                try:
                    local_path = s3_handler._get_local_storage_path(key)
                    if local_path.exists():
                        stat_info = local_path.stat()
                        size_bytes = stat_info.st_size
                        last_modified = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                except:
                    pass

            files.append({
                "filename": filename,
                "key": key,
                "size_bytes": size_bytes,
                "last_modified": last_modified
            })
            
        logger.info(f"Listed {len(files)} FAQ files from storage for domain {domain}")
        
    except Exception as e:
        logger.error(f"Failed to list storage files: {e}")
    else:
        # Fallback to local storage
        if Config:
            local_dir = Config.get_domain_data_dir(domain)
            if os.path.exists(local_dir):
                from datetime import datetime
                for filename in os.listdir(local_dir):
                    file_path = os.path.join(local_dir, filename)
                    if os.path.isfile(file_path) and filename.endswith(('.csv', '.xlsx', '.xls')):
                        stat_info = os.stat(file_path)
                        files.append({
                            "filename": filename,
                            "key": f"local://{file_path}",
                            "size_bytes": stat_info.st_size,
                            "last_modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                        })
                logger.info(f"Listed {len(files)} FAQ files from local storage for domain {domain}")
    
    return {
        "domain": domain,
        "files": files,
        "count": len(files)
    }

@router.get("/files/{filename}/download")
async def download_faq_file(
    filename: str,
    domain: str = Query("default")
):
    """
    Downloads a FAQ data file for a domain.
    Uses Redis cache for performance, fetches from storage on cache miss.
    Returns the file as a streaming attachment.
    """
    from fastapi.responses import StreamingResponse
    import io
    
    # Security check for path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    cache_key = f"faq_file:{domain}:{filename}"
    redis_cache = RedisCacheHandler()
    
    # Try to get from Redis cache first
    if redis_cache.is_available():
        cached_data = redis_cache.get_binary(cache_key)
        if cached_data:
            logger.info(f"Serving FAQ file from Redis cache: {cache_key}")
            
            # Determine content type
            content_type = "application/octet-stream"
            if filename.endswith('.csv'):
                content_type = "text/csv"
            elif filename.endswith(('.xlsx', '.xls')):
                content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
            return StreamingResponse(
                io.BytesIO(cached_data),
                media_type=content_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
    
    # Cache miss - fetch from S3
    s3_handler = S3Handler()
    s3_key = f"{S3_FAQ_PREFIX}{domain}/{filename}"
    
    if not s3_handler.s3_client:
        raise HTTPException(status_code=503, detail="S3 client not initialized")
    
    if not s3_handler.file_exists(s3_key):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in domain '{domain}'")
    
    try:
        # Get file content from S3
        response = s3_handler.s3_client.get_object(
            Bucket=s3_handler.bucket_name,
            Key=s3_key
        )
        file_content = response['Body'].read()
        
        # Cache the content in Redis (TTL 1 hour)
        if redis_cache.is_available():
            redis_cache.set_binary(cache_key, file_content, ttl=3600)
            logger.info(f"Cached FAQ file in Redis: {cache_key}")
        
        # Determine content type
        content_type = "application/octet-stream"
        if filename.endswith('.csv'):
            content_type = "text/csv"
        elif filename.endswith(('.xlsx', '.xls')):
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        logger.info(f"Serving FAQ file from S3: {s3_key}")
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Failed to download file from S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")




@router.post("/files/upload", response_model=UploadResponse)
async def upload_faq_files(
    file: Annotated[UploadFile, File(description="FAQ file to upload (CSV/Excel)")],
    domain: Annotated[str, Form(description="Target domain")] = "default",
    rebuild_index: Annotated[bool, Form(description="Whether to trigger an index rebuild")] = True
):
    """
    Uploads a FAQ data file (CSV/Excel) to S3 and local storage for a domain.
    Optionally triggers an index rebuild.
    """
    if not Config:
        raise HTTPException(status_code=503, detail="FAQ Configuration not loaded")

    uploaded_filenames = []
    s3_handler = S3Handler()
    local_dir = Config.get_domain_data_dir(domain)

    filename = file.filename
    if not filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and Excel files are allowed.")
        
    local_path = os.path.join(local_dir, filename)
    
    # Save locally
    try:
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved {filename} locally to {local_path}")
    except Exception as e:
        logger.error(f"Failed to save {filename} locally: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save {filename} locally: {e}")

    # Upload to S3
    s3_key = f"{S3_FAQ_PREFIX}{domain}/{filename}"
    if s3_handler.s3_client:
        s3_handler.upload_file(local_path, s3_key)
    
    uploaded_filenames.append(filename)

    if not uploaded_filenames:
        raise HTTPException(status_code=400, detail="No valid files uploaded")

    # Invalidate Redis cache for uploaded files
    redis_cache = RedisCacheHandler()
    if redis_cache.is_available():
        for filename in uploaded_filenames:
            cache_key = f"faq_file:{domain}:{filename}"
            redis_cache.invalidate(cache_key)
            logger.info(f"Invalidated FAQ cache: {cache_key}")

    # Trigger Rebuild
    if rebuild_index:
        refresh_faq_index(domain=domain)
        
    return UploadResponse(
        message=f"Successfully uploaded {len(uploaded_filenames)} files",
        uploaded_files=uploaded_filenames,
        domain=domain,
        rebuild_triggered=rebuild_index
    )

@router.delete("/files/{filename}", response_model=DeleteResponse)
async def delete_faq_file(
    filename: str,
    domain: str = Query("default"),
    rebuild_index: bool = True
):
    """
    Deletes a FAQ data file from S3 and local storage for a domain.
    """
    if not Config:
        raise HTTPException(status_code=503, detail="FAQ Configuration not loaded")
        
    # Security check for path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    local_dir = Config.get_domain_data_dir(domain)
    local_path = os.path.join(local_dir, filename)
    deleted_locally = False
    
    if os.path.exists(local_path):
        try:
            os.remove(local_path)
            deleted_locally = True
            logger.info(f"Deleted local file: {local_path}")
        except Exception as e:
            logger.error(f"Error deleting local file {local_path}: {e}")
    
    s3_handler = S3Handler()
    s3_key = f"{S3_FAQ_PREFIX}{domain}/{filename}"
    deleted_from_s3 = False
    
    if s3_handler.s3_client:
        if s3_handler.file_exists(s3_key):
            deleted_from_s3 = s3_handler.delete_file(s3_key)
    
    if not deleted_locally and not deleted_from_s3:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in S3 or local storage for domain {domain}")
    
    # Invalidate Redis cache for deleted file
    redis_cache = RedisCacheHandler()
    if redis_cache.is_available():
        cache_key = f"faq_file:{domain}:{filename}"
        redis_cache.invalidate(cache_key)
        logger.info(f"Invalidated FAQ cache: {cache_key}")
        
    if rebuild_index:
        refresh_faq_index(domain=domain)
        
    return DeleteResponse(
        message=f"File '{filename}' deleted",
        deleted_from_s3=deleted_from_s3,
        deleted_from_local=deleted_locally,
        domain=domain
    )

@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_faq_index_endpoint(
    domain: str = Query("default"),
    background_tasks: BackgroundTasks = None
):
    """
    Manually triggers a rebuild of the FAQ vector index for a domain.
    """
    logger.info(f"Manual FAQ index rebuild triggered for domain: {domain}")
    result = refresh_faq_index(domain=domain)
    stats = faq_stats(domain)
    
    return RebuildResponse(
        message=f"FAQ index rebuild completed for {domain}",
        status=result.get("status", "unknown"),
        vectors_indexed=stats.get("vectors_indexed", 0),
        domain=domain
    )
