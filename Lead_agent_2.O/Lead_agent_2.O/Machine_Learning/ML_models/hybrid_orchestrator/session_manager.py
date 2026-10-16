"""
Hybrid session manager leveraging Redis or in-memory store with MongoDB persistence.

This module is the single source of truth for hybrid session structure.
It now also tracks timing and status for long-running form flows so that
we can safely auto-expire stale lead forms while preserving partial data.

Architecture:
- Redis: Fast cache layer (5 min TTL)
- MongoDB: Persistent storage for analytics and session recovery
"""
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ml_utils.redis_client import delete_key, get_json, set_json
from ml_utils.mongodb_session_client import (
    save_session_to_mongo,
    get_session_from_mongo,
    track_form_usage,
    append_to_history
)

# 🔴 Email Integration
import sys
from pathlib import Path

# Add Conversational_AI utils to path for centralized_email
conv_ai_path = Path(__file__).resolve().parents[3] / "Conversational_AI"
if str(conv_ai_path) not in sys.path:
    sys.path.insert(0, str(conv_ai_path))

try:
    from utils.centralized_email import send_leads_email as get_email_sender
except ImportError:
    logging.warning("⚠️ Could not import utils.centralized_email - email notifications disabled")
    get_email_sender = None


logger = logging.getLogger(__name__)

SESSION_PREFIX = "hybrid_session:"

# 15 minutes, expressed in seconds
LEAD_FORM_TIMEOUT_SECONDS = 15 * 60


def _key(session_id: str) -> str:
    return f"{SESSION_PREFIX}{session_id}"


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def create_session(seed: Optional[Dict] = None) -> Dict:
    """
    Create a new hybrid session with sane defaults.

    Notes:
        - Form-related fields are initialized here so that all callers
          can rely on their presence.
        - Sessions are persisted to both Redis (cache) and MongoDB (persistent)
    """
    seed = seed or {}
    session_id = seed.get("session_id") or str(uuid.uuid4())
    now = _now_iso()
    data = {
        "session_id": session_id,
        "user_id": seed.get("user_id"),  # User identifier from metadata
        "domain": seed.get("domain"),  # Domain/tenant context
        "active_flow": None,
        "graph_state_id": None,
        "last_three_turns": [],
        "form_data": {},  # User info collected from forms (name, email, phone, etc.)
        # Form timing / status tracking
        "form_started_ts": None,  # ISO timestamp when a form flow started
        "last_activity_ts": None,  # ISO timestamp of last user activity in a form
        "form_status": None,  # "active" | "completed" | "expired" | None
        # Question Registry for deterministic state
        "question_registry": {},
        "completed_questions": [], 
        "completed_forms": [],
        "forms_used": [],  # Track all forms used with timestamps
        # 🔒 GOVERNANCE: Consent state for Anti-Gravity architecture
        # Each scope is independent: pii, marketing, data_retention
        "consent_state": {
            "pii": False,
            "marketing": False,
            "data_retention": False,
        },
        "consent_history": [],  # Audit trail of consent changes
        "session_state": "free_chat",  # State machine: free_chat, consent_requested, form_active, etc.
        # 🔒 ANTI-GRAVITY POLICY ROUTER: Conversation mode state
        # conversation_mode gates routing decisions at the policy level
        "conversation_mode": "DEFAULT",  # DEFAULT | QA_CONTINUATION | FORM_ACTIVE | HANDOFF_PENDING | HANDOFF_ACTIVE
        "last_route_family": None,  # FAQ | ML_ACTION | RAG_QA | LLM | CLARIFY | HANDOFF
        "qa_continuation_turns_left": 0,  # Bounded QA continuity (0-2 turns)
        "created_at": now,
        "updated_at": now,
        **(seed or {}),
    }
    # Save to Redis (cache) - MongoDB persistence happens via append_conversation
    set_json(_key(session_id), data)
    logger.debug(f"✅ Session {session_id} created in Redis")
    return data


def get_session(session_id: str) -> Optional[Dict]:
    """
    Get session from Redis, with MongoDB fallback for cache misses.
    """
    # Try Redis first (fast path)
    data = get_json(_key(session_id))
    if data:
        return data
    
    # Fallback to MongoDB (cache miss recovery)
    data = get_session_from_mongo(session_id)
    if data:
        logger.info(f"♻️ Session {session_id} recovered from MongoDB")
        # Re-populate Redis cache
        set_json(_key(session_id), data)
        return data
    
    return None


def upsert_session(session_id: str, updates: Dict) -> Dict:
    """
    Merge updates into an existing session, creating it if missing.
    MongoDB persistence happens via append_conversation.
    """
    data = get_session(session_id) or create_session({"session_id": session_id})
    data.update(updates)
    data["updated_at"] = _now_iso()
    # Save to Redis - MongoDB persistence happens via append_conversation
    set_json(_key(session_id), data)
    return data


def append_turn(session_id: str, role: str, message: str):
    """
    Append a conversation turn to the rolling history (Redis only).
    MongoDB persistence happens via append_conversation.
    """
    from datetime import datetime
    data = get_session(session_id) or create_session({"session_id": session_id})
    turns: List[Dict[str, str]] = data.get("last_three_turns", [])
    turns.append({
        "role": role, 
        "message": message,
        "timestamp": _now_iso()
    })
    data["last_three_turns"] = turns[-3:]  # Keep last 3 turns
    data["updated_at"] = _now_iso()
    # Save to Redis only - MongoDB uses append_conversation with query/response format
    set_json(_key(session_id), data)
    logger.debug(f"💬 Turn appended for session {session_id}: {role}")
    return data


def append_conversation(session_id: str, query: str, response: str, domain: str = "domain_3"):
    """
    Append a complete query/response conversation pair to MongoDB history.
    
    This stores in the format: history: [{query, response, timestamp}, ...]
    matching the existing session_histories schema.
    Also includes session state (form_data, completed_forms, etc.)
    
    Args:
        session_id: The session UUID
        query: User's query/message
        response: Assistant's response
        domain: Domain identifier (default: domain_3)
    """
    # Get current session data to include form state
    session_data = get_session(session_id)
    
    # Append to MongoDB in the correct format with session state
    append_to_history(session_id, query, response, domain, session_data)
    logger.info(f"📝 Conversation saved for session {session_id}")


def clear_session(session_id: str):
    delete_key(_key(session_id))


# ---------------------------------------------------------------------------
# Email Notification Helper
# ---------------------------------------------------------------------------

def _send_lead_email_notification(session_data: Dict, event_type: str = "completion"):
    """
    Send professional email notification to the user using the centralized email system.
    Matches the format defined in backend_routes/leads.py.
    """
    if not get_email_sender:
        return

    form_data = session_data.get("form_data", {})
    if not form_data:
        return

    # Try multiple keys for email
    email = form_data.get("email") or form_data.get("email_address")
    if not email or "@" not in email:
        return

    # Extract display data
    name = form_data.get("name", "Valued Customer")
    phone = form_data.get("phone", "Not provided")
    interest = form_data.get("interest", "General Inquiry")
    source = form_data.get("source", "Website")
    message_content = form_data.get("message") or form_data.get("additional_message", "")
    lead_id = session_data.get("lead_id", "Pending")
    # Generate a temporary Lead ID for display if pending, to look professional
    if lead_id == "Pending":
        import time
        lead_id = f"LD-{int(time.time())}"
        
    created_at = session_data.get("created_at") or _now_iso()

    # Determine Subject and Intro based on event
    if event_type == "completion":
        subject = "Thank you for your inquiry"
        intro_text = "Thank you for your inquiry! We have received your request and our team is reviewing it."
    else:
        subject = "We received your inquiry"
        intro_text = "We noticed your session ended, but we have saved your inquiry details. Our team will contact you shortly."

    # Style Configuration from image
    # Header: Blue (#3B82F6), Footer: Dark (#111827), Box: Light Green (#ECFDF5)
    
    # Precompute HTML-safe message strings
    lead_message_html = (message_content or "").replace('\n', '<br>')
    
    body_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{subject}</title>
    </head>
    <body style="margin: 0; padding: 0; background-color: #f3f4f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        
        <!-- Main Container -->
        <!-- Main Container -->
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" style="max_width: 500px; margin: 0 auto; background-color: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            
            <!-- Header -->
            <tr>
                <td style="background-color: #3B82F6; padding: 40px 30px; text-align: center;">
                    <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 24px; font-weight: 700;">{subject}</h1>
                    <p style="color: #e0e7ff; margin: 0; font-size: 16px;">{intro_text}</p>
                </td>
            </tr>
            
            <!-- Body Content -->
            <tr>
                <td style="padding: 40px 30px; background-color: #ffffff;">
                    <h2 style="color: #111827; font-size: 20px; margin: 0 0 20px 0;">Hi {name},</h2>
                    <p style="color: #4b5563; font-size: 16px; line-height: 1.6; margin-bottom: 24px;">
                        Thank you for reaching out to us. We have successfully received your inquiry details. Our team will review your request and get back to you shortly.
                    </p>
                    
                    <!-- Details Box (Green Style) -->
                    <div style="background-color: #ECFDF5; border: 1px solid #D1FAE5; border-radius: 12px; padding: 24px;">
                        
                        <!-- Row 1: Lead ID & Email -->
                        <table role="presentation" width="100%" border="0" style="margin-bottom: 16px;">
                            <tr>
                                <td width="50%" valign="top" style="padding-bottom: 16px;">
                                    <p style="margin: 0 0 4px 0; font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Reference ID</p>
                                    <p style="margin: 0; font-size: 16px; font-weight: 600; color: #10B981;">{lead_id}</p>
                                </td>
                            </tr>
                            <tr>
                                <td width="100%" valign="top" style="padding-bottom: 16px;">
                                    <p style="margin: 0 0 4px 0; font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Email Address</p>
                                    <a href="mailto:{email}" style="margin: 0; font-size: 16px; font-weight: 600; color: #2563EB; text-decoration: underline;">{email}</a>
                                </td>
                            </tr>
                        </table>

                        <!-- Row 2: Phone & Interest -->
                        <table role="presentation" width="100%" border="0">
                            <tr>
                                <td width="50%" valign="top" style="padding-bottom: 16px;">
                                    <p style="margin: 0 0 4px 0; font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Phone Number</p>
                                    <p style="margin: 0; font-size: 16px; font-weight: 600; color: #8B5CF6;">{phone}</p> 
                                </td>
                            </tr>
                            <tr>
                                <td width="100%" valign="top">
                                    <p style="margin: 0 0 4px 0; font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Interest</p>
                                    <p style="margin: 0; font-size: 16px; font-weight: 600; color: #EF4444;">{interest}</p>
                                </td>
                            </tr>
                        </table>
                        
                        <!-- Optional Message Row -->
                        <div style="margin-top: 16px; border-top: 1px dashed #A7F3D0; padding-top: 16px;">
                           <p style="margin: 0 0 4px 0; font-size: 13px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.5px;">Your Message</p>
                           <p style="margin: 0; font-size: 15px; color: #374151; font-style: italic;">"{lead_message_html}"</p>
                        </div>
                    </div>
                </td>
            </tr>
            
            <!-- Footer -->
            <tr>
                <td style="background-color: #111827; padding: 30px; text-align: center;">
                    <p style="color: #9CA3AF; font-size: 12px; margin: 0 0 8px 0;">Powered by <a href="#" style="color: #3B82F6; text-decoration: none;">Converiqo.ai</a></p>
                    <p style="color: #6B7280; font-size: 12px; margin: 0 0 16px 0;">Empowering businesses with intelligent automation</p>
                    
                    <div style="margin-bottom: 16px;">
                        <a href="#" style="color: #9CA3AF; text-decoration: underline; font-size: 12px; margin: 0 8px;">Privacy Policy</a>
                        <a href="#" style="color: #9CA3AF; text-decoration: underline; font-size: 12px; margin: 0 8px;">Unsubscribe</a>
                    </div>
                    
                    <p style="color: #4B5563; font-size: 12px; margin: 0;">&copy; 2025 Converiqo.ai Technologies. All rights reserved.</p>
                </td>
            </tr>
            
        </table>
        
        <!-- Spacing for verification -->
        <div style="height: 40px;"></div>
    </body>
    </html>
    """

    try:
        sender = get_email_sender("leads")
        sender.send_email(
            to_email=email,
            subject=subject,
            body=body_html,
            module="leads",
            is_html=True,
        )
        logger.info(f"📧 Sent {event_type} email to {email}")
    except Exception as e:
        logger.error(f"❌ Failed to send {event_type} email to {email}: {e}")


# ---------------------------------------------------------------------------
# Form timeout helpers
# ---------------------------------------------------------------------------

def mark_lead_form_started(session_id: str, form_type: str = "lead_questions"):
    """
    Initialize timing fields for a new lead form flow.

    We keep form timing in the central session store so that all
    orchestrator entry-points can make consistent timeout decisions.
    Also tracks form usage in MongoDB for analytics.
    """
    now = _now_iso()
    upsert_session(
        session_id,
        {
            "form_started_ts": now,
            "last_activity_ts": now,
            "form_status": "active",
        },
    )
    # Track form usage in MongoDB
    track_form_usage(session_id, form_type, status="in_progress")


def touch_lead_form_activity(session_id: str):
    """
    Update last_activity_ts for an active lead form.

    This should be called on every user turn that is routed through
    a lead form flow (start or continuation) as long as the form
    has not already expired or completed.
    """
    session = get_session(session_id)
    if not session:
        return

    # Only touch when we're still in an active flow
    active_flow = session.get("active_flow")
    if not active_flow or active_flow != "lead_questions":
        return
    if session.get("form_status") in {"completed", "expired"}:
        return

    upsert_session(session_id, {"last_activity_ts": _now_iso()})


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # datetime.fromisoformat handles timezone-aware strings out of the box
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def is_lead_form_timed_out(session: Dict, timeout_seconds: int = LEAD_FORM_TIMEOUT_SECONDS) -> bool:
    """
    Determine whether the current lead form has exceeded the inactivity timeout.

    Args:
        session: Hybrid session dictionary from Redis.
        timeout_seconds: Timeout threshold in seconds (default: 15 minutes).
    """
    active_flow = session.get("active_flow")
    if not active_flow or active_flow != "lead_questions":
        return False

    # If the form has already been marked as expired or completed, do not re-timeout.
    if session.get("form_status") in {"completed", "expired"}:
        return False

    # Prefer last_activity_ts, fall back to form_started_ts
    ts = _parse_iso(session.get("last_activity_ts")) or _parse_iso(session.get("form_started_ts"))
    if not ts:
        return False

    now = datetime.now(timezone.utc)
    # Ensure both are timezone-aware before subtraction
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    delta = now - ts
    return delta.total_seconds() >= timeout_seconds


def expire_form_session(session_id: str, reason: str = "timeout") -> Optional[Dict]:
    """
    Mark the current form (if any) as expired and clear routing pointers.

    This does NOT delete the session; it only resets form-related fields so
    that subsequent messages are treated as fresh interactions.
    Also updates form status in MongoDB for analytics.
    """
    session = get_session(session_id)
    if not session:
        return None

    # Get the form type before clearing
    form_type = session.get("active_flow")
    
    updates = {
        "active_flow": None,
        "graph_state_id": None,
        "form_status": "expired",
        "form_expired_reason": reason,
    }
    result = upsert_session(session_id, updates)
    
    # Update form tracking in MongoDB
    if form_type:
        track_form_usage(session_id, form_type, status="expired")
    
    # 🔴 Trigger Expiry Email
    # Only if we have captured enough data (at least an email)
    if session and session.get("form_data"):
        _send_lead_email_notification(session, event_type="expiry")

    return result


def mark_form_completed(session_id: str, form_type: str, lead_id: Optional[str] = None):
    """
    Mark a form as completed in the session and MongoDB.
    
    Args:
        session_id: The session UUID
        form_type: Type of form completed (e.g., "lead_questions")
        lead_id: Optional Lead ID (e.g., LD-123) if generated by the form flow
    """
    session = get_session(session_id)
    if not session:
        return
    
    completed_forms = session.get("completed_forms", [])
    if form_type not in completed_forms:
        completed_forms.append(form_type)
    
    updates = {
        "completed_forms": completed_forms,
        "form_status": "completed",
        "active_flow": None,
    }
    
    # Store Lead ID if provided
    if lead_id:
        updates["lead_id"] = lead_id
        session["lead_id"] = lead_id # Update local session obj for email
    
    upsert_session(session_id, updates)
    
    # Track completion in MongoDB
    track_form_usage(session_id, form_type, status="completed")
    
    # 🔴 Trigger Completion Email
    _send_lead_email_notification(session, event_type="completion")
    
    logger.info(f"✅ Form '{form_type}' marked complete for session {session_id}")


# ---------------------------------------------------------------------------
# 🔒 GOVERNANCE: Consent Management (Anti-Gravity Architecture)
# ---------------------------------------------------------------------------

def grant_consent(session_id: str, scope: str, source: str = "user_explicit") -> Dict:
    """
    Grant consent for a specific scope.
    
    Args:
        session_id: The session UUID
        scope: Consent scope ('pii', 'marketing', 'data_retention')
        source: How consent was obtained (e.g., 'user_explicit', 'implicit')
    
    Returns:
        Updated session dict
    """
    session = get_session(session_id) or create_session({"session_id": session_id})
    
    consent_state = session.get("consent_state", {})
    consent_state[scope] = True
    
    # Add to audit trail
    consent_history = session.get("consent_history", [])
    consent_history.append({
        "scope": scope,
        "action": "grant",
        "source": source,
        "timestamp": _now_iso()
    })
    
    updates = {
        "consent_state": consent_state,
        "consent_history": consent_history,
    }
    
    result = upsert_session(session_id, updates)
    logger.info(f"🔒 Consent GRANTED for '{scope}' on session {session_id}")
    return result


def revoke_consent(session_id: str, scope: str, source: str = "user_explicit") -> Dict:
    """
    Revoke consent for a specific scope.
    
    Note: Revocation is ALWAYS allowed, no questions asked.
    
    Args:
        session_id: The session UUID
        scope: Consent scope to revoke
        source: How revocation was triggered
    
    Returns:
        Updated session dict
    """
    session = get_session(session_id)
    if not session:
        return None
    
    consent_state = session.get("consent_state", {})
    consent_state[scope] = False
    
    # Add to audit trail
    consent_history = session.get("consent_history", [])
    consent_history.append({
        "scope": scope,
        "action": "revoke",
        "source": source,
        "timestamp": _now_iso()
    })
    
    updates = {
        "consent_state": consent_state,
        "consent_history": consent_history,
    }
    
    result = upsert_session(session_id, updates)
    logger.warning(f"🔒 Consent REVOKED for '{scope}' on session {session_id}")
    return result


def get_consent_state(session_id: str) -> Dict[str, bool]:
    """
    Get current consent state for a session.
    
    Returns:
        Dict of {scope: True/False}
    """
    session = get_session(session_id)
    if not session:
        return {}
    return session.get("consent_state", {})


def has_consent(session_id: str, scope: str) -> bool:
    """
    Check if a specific consent is granted.
    
    Args:
        session_id: The session UUID
        scope: The consent scope to check
    
    Returns:
        True if consent is granted, False otherwise
    """
    consent_state = get_consent_state(session_id)
    return consent_state.get(scope, False)


def update_session_state(session_id: str, new_state: str, trigger: str = "system") -> Optional[Dict]:
    """
    Update the session state machine state.
    
    Valid states: free_chat, consent_requested, pii_form_active, human_handoff, completed
    
    Args:
        session_id: The session UUID
        new_state: The new state to transition to
        trigger: What triggered the transition
    
    Returns:
        Updated session or None if transition invalid
    """
    session = get_session(session_id)
    if not session:
        return None
    
    old_state = session.get("session_state", "free_chat")
    
    # Log state transition
    logger.info(f"🔄 Session {session_id}: {old_state} → {new_state} (trigger: {trigger})")
    
    return upsert_session(session_id, {"session_state": new_state})


# ---------------------------------------------------------------------------
# 🔒 ANTI-GRAVITY POLICY ROUTER: Conversation Mode Management
# ---------------------------------------------------------------------------

# Valid conversation modes
CONVERSATION_MODES = {"DEFAULT", "QA_CONTINUATION", "FORM_ACTIVE", "HANDOFF_PENDING", "HANDOFF_ACTIVE", "FORM_CONFIRMATION"}

# Valid route families  
ROUTE_FAMILIES = {"FAQ", "ML_ACTION", "RAG_QA", "LLM", "CLARIFY", "HANDOFF"}


def get_mode(session: Dict) -> str:
    """
    Get conversation mode from session dict.
    
    Args:
        session: Session dictionary
    
    Returns:
        Conversation mode string (defaults to "DEFAULT" if missing)
    """
    return session.get("conversation_mode", "DEFAULT") if session else "DEFAULT"


def set_mode(session_id: str, mode: str, trigger: str = "system") -> Optional[Dict]:
    """
    Set conversation mode for a session.
    
    Args:
        session_id: The session UUID
        mode: New mode (DEFAULT | QA_CONTINUATION | FORM_ACTIVE | HANDOFF_PENDING | HANDOFF_ACTIVE)
        trigger: What triggered the mode change
    
    Returns:
        Updated session or None if invalid
    """
    if mode not in CONVERSATION_MODES:
        logger.warning(f"⚠️ Invalid conversation mode: {mode}")
        return None
    
    session = get_session(session_id)
    if not session:
        return None
    
    old_mode = get_mode(session)
    
    # Automatically reset qa_continuation_turns_left when exiting QA_CONTINUATION
    updates = {"conversation_mode": mode}
    if mode != "QA_CONTINUATION" and old_mode == "QA_CONTINUATION":
        updates["qa_continuation_turns_left"] = 0
    
    # Set initial QA turns when entering QA_CONTINUATION
    if mode == "QA_CONTINUATION" and old_mode != "QA_CONTINUATION":
        updates["qa_continuation_turns_left"] = 2  # Default: 2 follow-up turns allowed
    
    logger.info(f"🔄 CONV_MODE {session_id}: {old_mode} → {mode} (trigger: {trigger})")
    
    return upsert_session(session_id, updates)


def get_last_route_family(session: Dict) -> Optional[str]:
    """
    Get last route family from session dict.
    
    Args:
        session: Session dictionary
    
    Returns:
        Last route family or None if not set
    """
    return session.get("last_route_family") if session else None


def set_last_route_family(session_id: str, family: str) -> Optional[Dict]:
    """
    Set last route family for a session.
    
    Args:
        session_id: The session UUID
        family: Route family (FAQ | ML_ACTION | RAG_QA | LLM | CLARIFY | HANDOFF)
    
    Returns:
        Updated session or None if invalid
    """
    if family and family not in ROUTE_FAMILIES:
        logger.warning(f"⚠️ Invalid route family: {family}")
        return None
    
    return upsert_session(session_id, {"last_route_family": family})


def set_qa_turns(session_id: str, turns: int) -> Optional[Dict]:
    """
    Set QA continuation turns remaining.
    
    Args:
        session_id: The session UUID
        turns: Number of turns remaining (0-5)
    
    Returns:
        Updated session
    """
    turns = max(0, min(5, turns))  # Clamp to 0-5
    return upsert_session(session_id, {"qa_continuation_turns_left": turns})


def dec_qa_turns(session_id: str) -> Optional[Dict]:
    """
    Decrement QA continuation turns and auto-transition to DEFAULT if exhausted.
    
    Args:
        session_id: The session UUID
    
    Returns:
        Updated session
    """
    session = get_session(session_id)
    if not session:
        return None
    
    current_turns = session.get("qa_continuation_turns_left", 0)
    new_turns = max(0, current_turns - 1)
    
    updates = {"qa_continuation_turns_left": new_turns}
    
    # Auto-transition to DEFAULT when turns exhausted
    if new_turns == 0 and get_mode(session) == "QA_CONTINUATION":
        updates["conversation_mode"] = "DEFAULT"
        logger.info(f"🔄 CONV_MODE {session_id}: QA_CONTINUATION → DEFAULT (turns exhausted)")
    
    return upsert_session(session_id, updates)
