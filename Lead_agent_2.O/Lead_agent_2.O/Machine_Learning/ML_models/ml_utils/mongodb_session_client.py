"""
MongoDB Session Client for Lead_agent_2.O session persistence.

This module provides MongoDB persistence for hybrid sessions,
enabling long-term storage and analytics while Redis serves as cache.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "enterprise_sessions")
SESSION_COLLECTION = "session_histories"

# Global client (lazy initialization)
_mongo_client: Optional[MongoClient] = None
_session_collection = None


def _get_collection():
    """
    Get or initialize the MongoDB session collection.
    Uses lazy initialization to avoid connection at import time.
    """
    global _mongo_client, _session_collection
    
    if _session_collection is not None:
        return _session_collection
    
    try:
        _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = _mongo_client[MONGODB_DB]
        _session_collection = db[SESSION_COLLECTION]
        
        # Create indexes for efficient querying (non-unique to avoid conflicts)
        try:
            _session_collection.create_index("session_id")  # Non-unique for compatibility
            _session_collection.create_index("user_id")
            _session_collection.create_index("domain")
            _session_collection.create_index("created_at")
            _session_collection.create_index([("forms_used.form_type", ASCENDING)])
        except PyMongoError as idx_err:
            logger.warning(f"⚠️ Index creation skipped (may already exist): {idx_err}")
        
        logger.info(f"✅ MongoDB session client connected to {MONGODB_URI}/{MONGODB_DB}")
        return _session_collection
    except PyMongoError as e:
        logger.error(f"❌ MongoDB session client connection failed: {e}")
        return None


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def save_session_to_mongo(session_data: Dict) -> bool:
    """
    Save or update session in MongoDB.
    
    Args:
        session_data: Session dictionary containing session_id and other fields
        
    Returns:
        True if successful, False otherwise
    """
    collection = _get_collection()
    if collection is None:
        logger.warning("MongoDB not available, skipping session persistence")
        return False
    
    try:
        session_id = session_data.get("session_id")
        if not session_id:
            logger.error("Cannot save session without session_id")
            return False
        
        # Add timestamps
        now = _now_iso()
        session_data["updated_at"] = now
        if "created_at" not in session_data:
            session_data["created_at"] = now
        
        # Add title and domain for compatibility with existing unique index (title, domain)
        # Use session_id as title to ensure uniqueness
        if "title" not in session_data or session_data.get("title") is None:
            session_data["title"] = session_id
        if "domain" not in session_data or session_data.get("domain") is None:
            session_data["domain"] = "domain_3"  # Default domain for Lead Agent sessions
        
        # Upsert the session using title + domain (existing unique index)
        collection.update_one(
            {"title": session_data["title"], "domain": session_data["domain"]},
            {"$set": session_data},
            upsert=True
        )
        
        logger.debug(f"📦 Session {session_id} saved to MongoDB")
        return True
        
    except PyMongoError as e:
        logger.error(f"❌ Failed to save session to MongoDB: {e}")
        return False


def append_to_history(
    session_id: str, 
    query: str, 
    response: str, 
    domain: str = "domain_3",
    session_data: Optional[Dict] = None
) -> bool:
    """
    Append a query/response pair to the session history and update session state.
    
    This matches the existing session_histories format with:
    - history: [{query, response, timestamp}, ...]
    - Plus form state fields (form_data, completed_forms, etc.)
    
    Args:
        session_id: The session UUID (used as title)
        query: User's query/message
        response: Assistant's response
        domain: Domain identifier (default: domain_3)
        session_data: Optional session state to include (form_data, completed_forms, etc.)
        
    Returns:
        True if successful, False otherwise
    """
    collection = _get_collection()
    if collection is None:
        return False
    
    try:
        now = _now_iso()
        history_entry = {
            "query": query,
            "response": response,
            "timestamp": now
        }
        
        # Use session_id as title (unique key with domain)
        title = session_id
        
        # Build update set with session state fields
        update_set = {"updated_at": now}
        
        # Include session state fields if provided
        if session_data:
            state_fields = [
                "form_data", "completed_forms", "completed_questions",
                "active_flow", "graph_state_id", "form_status",
                "form_started_ts", "last_activity_ts", "forms_used",
                "question_registry"
            ]
            for field in state_fields:
                if field in session_data:
                    update_set[field] = session_data[field]
        
        # Update or create document keyed by title+domain (existing unique index)
        # Also include session_id to satisfy session_id_1 unique index
        result = collection.update_one(
            {"title": title, "domain": domain},
            {
                "$push": {"history": history_entry},
                "$set": update_set,
                "$setOnInsert": {
                    "session_id": session_id,  # Include to satisfy unique index
                    "title": title,
                    "domain": domain,
                    "user_id": None,
                    "created_at": now
                }
            },
            upsert=True
        )
        
        logger.debug(f"💬 History appended for session {session_id}")
        return True
        
    except PyMongoError as e:
        logger.error(f"❌ Failed to append history: {e}")
        return False


def get_session_from_mongo(session_id: str) -> Optional[Dict]:
    """
    Retrieve session from MongoDB.
    
    Args:
        session_id: The session UUID
        
    Returns:
        Session dictionary or None if not found
    """
    collection = _get_collection()
    if collection is None:
        return None
    
    try:
        session = collection.find_one({"session_id": session_id})
        if session:
            # Remove MongoDB _id field
            session.pop("_id", None)
            logger.debug(f"📥 Session {session_id} loaded from MongoDB")
            return session
        return None
        
    except PyMongoError as e:
        logger.error(f"❌ Failed to get session from MongoDB: {e}")
        return None


def delete_session_from_mongo(session_id: str) -> bool:
    """
    Delete session from MongoDB.
    
    Note: By default, we don't delete sessions to preserve analytics data.
    This function is provided for explicit cleanup when needed.
    
    Args:
        session_id: The session UUID
        
    Returns:
        True if deleted, False otherwise
    """
    collection = _get_collection()
    if collection is None:
        return False
    
    try:
        result = collection.delete_one({"session_id": session_id})
        if result.deleted_count > 0:
            logger.info(f"🗑️ Session {session_id} deleted from MongoDB")
            return True
        return False
        
    except PyMongoError as e:
        logger.error(f"❌ Failed to delete session from MongoDB: {e}")
        return False


def track_form_usage(
    session_id: str, 
    form_type: str, 
    status: str = "in_progress",
    completed_at: Optional[str] = None
) -> bool:
    """
    Track form usage in a session.
    
    Args:
        session_id: The session UUID
        form_type: Type of form (e.g., "lead_questions")
        status: Form status ("in_progress", "completed", "expired", "abandoned")
        completed_at: ISO timestamp when form was completed (optional)
        
    Returns:
        True if successful, False otherwise
    """
    collection = _get_collection()
    if collection is None:
        return False
    
    try:
        now = _now_iso()
        
        # Check if this form already exists in forms_used
        existing = collection.find_one({
            "session_id": session_id,
            "forms_used.form_type": form_type
        })
        
        if existing:
            # Update existing form entry
            collection.update_one(
                {"session_id": session_id, "forms_used.form_type": form_type},
                {
                    "$set": {
                        "forms_used.$.status": status,
                        "forms_used.$.completed_at": completed_at or (now if status == "completed" else None),
                        "updated_at": now
                    }
                }
            )
        else:
            # Add new form entry
            form_entry = {
                "form_type": form_type,
                "started_at": now,
                "completed_at": completed_at if status == "completed" else None,
                "status": status
            }
            collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"forms_used": form_entry},
                    "$set": {"updated_at": now}
                },
                upsert=True
            )
        
        logger.info(f"📊 Form '{form_type}' tracked for session {session_id}: {status}")
        return True
        
    except PyMongoError as e:
        logger.error(f"❌ Failed to track form usage: {e}")
        return False


def get_user_form_history(user_id: str) -> List[Dict]:
    """
    Get all form usage history for a user.
    
    Args:
        user_id: The user identifier
        
    Returns:
        List of session documents with form usage
    """
    collection = _get_collection()
    if collection is None:
        return []
    
    try:
        sessions = list(collection.find(
            {"user_id": user_id},
            {"session_id": 1, "domain": 1, "forms_used": 1, "created_at": 1, "_id": 0}
        ).sort("created_at", DESCENDING))
        
        return sessions
        
    except PyMongoError as e:
        logger.error(f"❌ Failed to get user form history: {e}")
        return []


def get_sessions_by_form_type(form_type: str, status: Optional[str] = None) -> List[Dict]:
    """
    Get all sessions that used a specific form type.
    
    Args:
        form_type: Type of form (e.g., "lead_questions")
        status: Optional filter by status
        
    Returns:
        List of session documents
    """
    collection = _get_collection()
    if collection is None:
        return []
    
    try:
        query = {"forms_used.form_type": form_type}
        if status:
            query["forms_used.status"] = status
        
        sessions = list(collection.find(
            query,
            {"session_id": 1, "user_id": 1, "domain": 1, "forms_used": 1, "form_data": 1, "_id": 0}
        ))
        
        return sessions
        
    except PyMongoError as e:
        return []
        
    except PyMongoError as e:
        logger.error(f"❌ Failed to get sessions by form type: {e}")
        return []

# ---------------------------------------------------------------------------
# Lead Management Helpers
# ---------------------------------------------------------------------------

def _extract_lead_sequence(raw_value) -> int:
    """Extract the numeric portion from an LD- styled identifier."""
    if not raw_value:
        return 0

    if isinstance(raw_value, str):
        value = raw_value.strip()
        if value.upper().startswith("LD-"):
            numeric_part = value[3:]
            # Remove leading zeros but maintain zero if the remainder is empty
            numeric_part = numeric_part.lstrip('0') or '0'
            if numeric_part.isdigit():
                try:
                    return int(numeric_part)
                except ValueError:
                    return 0
    return 0


def _calculate_highest_lead_sequence(leads_collection) -> int:
    """Scan existing leads to determine the highest LD-### sequence."""
    highest = 0
    try:
        cursor = leads_collection.find({}, {"id": 1, "_id": 1})
        for doc in cursor:
            highest = max(
                highest,
                _extract_lead_sequence(doc.get("id")),
                _extract_lead_sequence(doc.get("_id")),
            )
    except Exception as exc:
        logger.warning(f"Unable to calculate highest lead sequence: {exc}")
    return highest


def generate_lead_id(mongo_db=None):
    """Generate sequential lead ID with LD- prefix, with reset awareness."""
    # If mongo_db is not provided, try to get it from the client
    if mongo_db is None:
        client = _mongo_client
        if client is None:
            # Try to initialize
            _get_collection()
            client = _mongo_client
        
        if client:
            mongo_db = client[MONGODB_DB]
        else:
            # Fallback if DB not available
            return f"LD-{int(datetime.now().timestamp())}"

    leads_collection = mongo_db.leads
    counters_collection = mongo_db.system_counters
    counter_key = "lead_id_sequence"

    # Check if collection is empty - if so, reset counter to 0
    try:
        lead_count = leads_collection.count_documents({})
    except Exception as exc:
        logger.error(f"Failed to count leads: {exc}")
        lead_count = 0

    from pymongo import ReturnDocument

    # CRITICAL: If collection is empty, reset counter to 0 to start from LD-1
    if lead_count == 0:
        # Delete the counter document to ensure a clean reset
        counters_collection.delete_one({"_id": counter_key})
        # Initialize counter to 0
        counters_collection.update_one(
            {"_id": counter_key},
            {"$set": {"value": 0}},
            upsert=True
        )
        logger.info("Leads collection is empty - resetting lead ID counter to 0")
    else:
        # Collection has leads - ensure counter is at least as high as the highest existing lead sequence
        counter_doc = counters_collection.find_one({"_id": counter_key})
        current_counter = int(counter_doc.get("value", 0)) if counter_doc else 0
        
        highest_sequence = _calculate_highest_lead_sequence(leads_collection)
        if highest_sequence > current_counter:
            # Update counter to match the highest existing lead ID
            counters_collection.update_one(
                {"_id": counter_key},
                {"$set": {"value": highest_sequence}},
                upsert=True
            )
            logger.info(f"Updated lead ID counter to {highest_sequence} based on existing leads")

    # Atomically increment and get the next ID
    updated_counter = counters_collection.find_one_and_update(
        {"_id": counter_key},
        {"$inc": {"value": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )

    next_id = int(updated_counter.get("value", 1))

    if next_id < 10:
        return f"LD-{next_id}"
    if next_id < 100:
        return f"LD-{next_id:02d}"
    return f"LD-{next_id:03d}"


def save_lead_to_mongo(lead_data: Dict) -> Optional[str]:
    """
    Save a new lead to the leads collection.
    
    Args:
        lead_data: Dictionary containing lead details
        
    Returns:
        The generated Lead ID if successful, None otherwise
    """
    collection = _get_collection()
    if collection is None:
        return None
        
    try:
        # Get DB reference
        db = _mongo_client[MONGODB_DB]
        
        # Generate ID
        lead_id = generate_lead_id(db)
        
        # Prepare document
        doc = lead_data.copy()
        doc["_id"] = lead_id
        doc["id"] = lead_id
        doc["created_at"] = _now_iso()
        doc["updated_at"] = _now_iso()
        doc["status"] = doc.get("status", "new")
        doc["stage"] = doc.get("stage", "Lead Received")
        
        # Insert
        db.leads.insert_one(doc)
        logger.info(f"✅ Lead saved to MongoDB: {lead_id}")
        
        return lead_id
        
    except PyMongoError as e:
        logger.error(f"❌ Failed to save lead to MongoDB: {e}")
        return None

