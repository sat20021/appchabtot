"""
Question Registry Module
------------------------
Authoritative source of truth for question status in the conversation.
Persists status to Redis via session_manager to prevent repeated questions.
"""
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging

# We use local imports for session_manager to avoid circular dependencies
# if session_manager needs to import types from here in the future.

logger = logging.getLogger(__name__)

class QuestionStatus(Enum):
    PENDING = "pending"           # Not yet asked
    ASKED = "asked"               # Asked, awaiting response
    ANSWERED = "answered"         # User provided valid answer
    DECLINED = "declined"         # User refused to answer
    SKIPPED = "skipped"           # System skipped (e.g., pre-extracted)
    PRE_EXTRACTED = "pre_extracted"  # Auto-filled from context
    EXHAUSTED = "exhausted"       # Max retries reached

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_question_registry(session_id: str) -> Dict[str, Dict]:
    """
    Retrieve the full question registry for a session.
    Returns a dict of {question_key: registry_entry_dict}
    """
    from hybrid_orchestrator.session_manager import get_session
    session = get_session(session_id)
    if not session:
        return {}
    return session.get("question_registry", {})

def get_question_status(session_id: str, question_key: str) -> QuestionStatus:
    """Get the status of a specific question."""
    registry = get_question_registry(session_id)
    entry = registry.get(question_key)
    if not entry:
        return QuestionStatus.PENDING
    
    try:
        return QuestionStatus(entry.get("status", "pending"))
    except ValueError:
        logger.warning(f"Invalid status '{entry.get('status')}' for question '{question_key}'")
        return QuestionStatus.PENDING

def update_question_status(
    session_id: str, 
    question_key: str, 
    status: QuestionStatus, 
    value: Optional[str] = None,
    reason: Optional[str] = None,
    source: str = "system"
) -> Dict:
    """
    Update the status of a question in the registry.
    This is the PRIMARY method for changing question state.
    """
    from hybrid_orchestrator.session_manager import upsert_session, get_session
    
    # Load existing to preserve other fields if needed (though we mostly overwrite)
    session = get_session(session_id) or {"session_id": session_id}
    registry = session.get("question_registry", {})
    
    current_entry = registry.get(question_key, {})
    attempts = current_entry.get("attempts", 0)
    
    if status == QuestionStatus.ASKED:
        attempts += 1
        
    new_entry = {
        "question_key": question_key,
        "status": status.value,
        "value": value,
        "attempts": attempts,
        "declined_reason": reason,
        "timestamp": _now_iso(),
        "source": source
    }
    
    registry[question_key] = new_entry
    
    # Also update completed_questions list for fast lookup
    completed_questions = session.get("completed_questions", [])
    is_complete = status in [
        QuestionStatus.ANSWERED, 
        QuestionStatus.DECLINED, 
        QuestionStatus.SKIPPED, 
        QuestionStatus.PRE_EXTRACTED,
        QuestionStatus.EXHAUSTED
    ]
    
    if is_complete and question_key not in completed_questions:
        completed_questions.append(question_key)
    elif not is_complete and question_key in completed_questions:
        # If status reverted (unlikely but possible), remove from completed
        completed_questions.remove(question_key)
        
    updates = {
        "question_registry": registry,
        "completed_questions": completed_questions
    }
    
    return upsert_session(session_id, updates)

def bulk_update_registry(session_id: str, updates: Dict[str, Dict]):
    """
    Update multiple registry entries at once.
    updates: {question_key: {status: ..., value: ...}}
    """
    from hybrid_orchestrator.session_manager import upsert_session, get_session
    
    session = get_session(session_id) or {"session_id": session_id}
    registry = session.get("question_registry", {})
    completed_questions = session.get("completed_questions", [])
    
    for key, data in updates.items():
        # Merge with existing
        entry = registry.get(key, {})
        entry.update(data)
        entry["timestamp"] = _now_iso()
        registry[key] = entry
        
        status_val = entry.get("status")
        is_complete = status_val in [
            s.value for s in [
                QuestionStatus.ANSWERED, 
                QuestionStatus.DECLINED, 
                QuestionStatus.SKIPPED, 
                QuestionStatus.PRE_EXTRACTED,
                QuestionStatus.EXHAUSTED
            ]
        ]
        
        if is_complete and key not in completed_questions:
            completed_questions.append(key)
            
    return upsert_session(session_id, {
        "question_registry": registry,
        "completed_questions": completed_questions
    })

def is_question_complete(session_id: str, question_key: str) -> bool:
    """Check if a question is definitively complete."""
    status = get_question_status(session_id, question_key)
    return status in [
        QuestionStatus.ANSWERED, 
        QuestionStatus.DECLINED, 
        QuestionStatus.SKIPPED, 
        QuestionStatus.PRE_EXTRACTED,
        QuestionStatus.EXHAUSTED
    ]

def get_pending_questions(session_id: str, all_questions: List[str]) -> List[str]:
    """Return list of questions that still need to be asked."""
    registry = get_question_registry(session_id)
    pending = []
    for q in all_questions:
        status_str = registry.get(q, {}).get("status")
        # If not in registry, it's pending. If in registry but status is pending/asked, it's pending.
        if not status_str or status_str in [QuestionStatus.PENDING.value, QuestionStatus.ASKED.value]:
            pending.append(q)
    return pending
