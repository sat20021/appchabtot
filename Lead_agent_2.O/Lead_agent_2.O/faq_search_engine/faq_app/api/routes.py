from fastapi import APIRouter, HTTPException, BackgroundTasks
from faq_app.api.schemas import AskRequest, AskResponse
from faq_app.services.engine import VectorEngine
from faq_app.config import Config

router = APIRouter()
engine = VectorEngine() # Gets the singleton instance

@router.get("/health")
def health():
    return {"status": "online", "vectors_indexed": engine.index.ntotal if engine.index else 0}

@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    result = engine.search(payload.question, payload.threshold or Config.CONFIDENCE_THRESHOLD)
    
    if result["answer"]:
        return AskResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            source="vector_db",
            matched_question=result["matched_question"]
        )
    else:
        return AskResponse(
            answer="I'm sorry, I don't have enough information to answer that.",
            confidence=result.get("confidence", 0.0),
            source="fallback",
            matched_question=None
        )

@router.post("/admin/reindex")
def trigger_reindex(background_tasks: BackgroundTasks):
    """Non-blocking ingestion trigger."""
    background_tasks.add_task(engine.build_full_index)
    return {"message": "Re-indexing started in background."}

@router.get("/domains")
def list_domains():
    """List available FAQ domains."""
    try:
        from faq_app.services.s3_handler import S3Handler
        s3 = S3Handler()
        domains = set()
        
        # 1. Check S3 (Authoritative)
        if s3.s3_client:
            artifacts = s3.list_files("faq_artifacts/")
            for key in artifacts:
                parts = key.split('/')
                if len(parts) > 1:
                    domains.add(parts[1])
        
        # 2. Check Loaded Domains
        domains.update(engine.indexes.keys())
        
        return {"domains": list(domains), "count": len(domains)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list domains: {str(e)}")

