"""
Utilities for persisting conversational form submissions into the leads collection.
"""
from datetime import datetime
from typing import Any, Dict, Optional

from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

try:
    # Ensure project root is in path to find backend_routes
    import sys
    from pathlib import Path
    
    # Add project root (Lead_agent_2.O's parent) to path
    # Current file: Lead_agent_2.O/Conversational_AI/database/lead_writer.py
    # Root: Lead_agent_2.O/..
    current_path = Path(__file__).resolve()
    # Go up 3 levels to Lead_agent_2.O, then one more to root
    project_root = current_path.parents[3]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Reuse the same ID generation helpers used by the FastAPI leads endpoints
    from backend_routes.leads import generate_lead_id  # type: ignore
except ImportError as exc:  # pragma: no cover - surfaced to caller
    generate_lead_id = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class LeadWriter:
    """Helper for creating lead documents from collected responses."""

    def __init__(self):
        mongo_uri = os.getenv("MONGODB_URI")
        mongo_db_name = os.getenv("MONGODB_DB")
        if not mongo_uri or not mongo_db_name:
            raise RuntimeError("MongoDB configuration missing (MONGODB_URI / MONGODB_DB).")

        self.client = MongoClient(mongo_uri)
        self.db = self.client[mongo_db_name]

        if generate_lead_id is None:
            raise RuntimeError(
                f"Unable to import lead helpers from backend_routes.leads: {_IMPORT_ERROR}"
            )
    
    def _trigger_scoring(self, lead_id: str):
        """Trigger async scoring for the lead."""
        try:
            from backend_routes.leads_integration import process_lead_automatically_async
            import asyncio
            
            # Check if there's a running loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(process_lead_automatically_async(lead_id, self.db))
                print(f"[lead-scoring] 🚀 Triggered async scoring for {lead_id}")
            except RuntimeError:
                print(f"[lead-scoring] ⚠️ No running event loop, skipping async scoring for {lead_id}")
                
        except ImportError as e:
            print(f"[lead-scoring] ❌ Could not import scoring function: {e}")
        except Exception as e:
            print(f"[lead-scoring] ❌ Error triggering scoring: {e}")

    def generate_new_lead_id(self) -> str:
        """Generate a new sequential lead ID."""
        return generate_lead_id(self.db)  # type: ignore[arg-type]

    def initialize_lead_shell(
        self,
        lead_id: str,
        session_id: Optional[str],
        question_set_title: str,
        domain: Optional[str],
    ) -> None:
        """Ensure the baseline document exists for incremental updates."""
        now_iso = datetime.utcnow().isoformat()
        metadata = {
            "question_set": question_set_title,
            "domain": domain,
            "raw_responses": {},
        }
        base_doc = {
            "_id": lead_id,
            "id": lead_id,
            "session_id": session_id or lead_id,
            "status": "in_progress",
            "created_at": now_iso,
            "updated_at": now_iso,
            "lead_metadata": metadata,
        }
        self.db.leads.update_one(
            {"_id": lead_id},
            {"$setOnInsert": base_doc},
            upsert=True,
        )

    def _prepare_update_payload(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Derive dependent fields when incremental updates arrive."""
        derived: Dict[str, Any] = {}
        if "additional_message" in updates and "message" not in updates:
            derived["message"] = updates.get("additional_message")
        return derived

    def upsert_partial_lead(
        self,
        lead_id: str,
        session_id: Optional[str],
        updates: Dict[str, Any],
    ) -> None:
        """Incrementally update lead fields as soon as they are collected."""
        if not updates:
            return

        now_iso = datetime.utcnow().isoformat()
        set_fields = {
            "updated_at": now_iso,
            "session_id": session_id or lead_id,
        }
        set_fields.update(updates)
        set_fields.update(self._prepare_update_payload(updates))

        metadata_updates = {
            f"lead_metadata.raw_responses.{key}": value
            for key, value in updates.items()
        }
        
        # Flatten updates into top-level metadata for scoring
        for key, value in updates.items():
            metadata_updates[f"lead_metadata.{key}"] = value

        update_doc = {"$set": set_fields}
        if metadata_updates:
            update_doc["$set"].update(metadata_updates)

        self.db.leads.update_one(
            {"_id": lead_id},
            update_doc,
            upsert=True,
        )
        
        # ========== SALES NOTIFICATION: Trigger when email is collected ==========
        # IMPORTANT: This is CODE-BASED (not LLM) to avoid hallucinations
        if "email" in updates:
            try:
                from .lead_email_notification import (
                    should_send_notification,
                    send_lead_email_notification
                )
                
                # Only send if not already sent (duplicate prevention)
                if should_send_notification(lead_id, self.db):
                    lead_email = updates["email"]
                    success = send_lead_email_notification(lead_email, lead_id, self.db)
                    
                    if success:
                        print(f"[lead-notification] ✅ Sent sales notification for {lead_email} (Lead ID: {lead_id})")
                    else:
                        print(f"[lead-notification] ⚠️ Failed to send notification for {lead_email}")
                else:
                    print(f"[lead-notification] ℹ️ Notification already sent for Lead ID: {lead_id}, skipping")
                    
            except Exception as e:
                # Log error but DON'T crash the form flow
                print(f"[lead-notification] ❌ Error sending notification: {e}")
                # Continue execution - notification failure shouldn't block lead submission
        # ========== END SALES NOTIFICATION ==========

    def finalize_lead_document(
        self,
        lead_id: str,
        session_id: Optional[str],
        responses: Dict[str, Any],
        question_set_title: str,
        domain: Optional[str],
    ) -> Dict[str, Any]:
        """Mark a lead as completed when the form finishes."""
        self.initialize_lead_shell(lead_id, session_id, question_set_title, domain)
        if responses:
            self.upsert_partial_lead(lead_id, session_id, responses)

        now_iso = datetime.utcnow().isoformat()
        completion_fields = {
            "status": "completed",
            "completed_at": now_iso,
            "updated_at": now_iso,
            "lead_metadata.question_set": question_set_title,
            "lead_metadata.domain": domain,
        }
        self.db.leads.update_one(
            {"_id": lead_id},
            {"$set": completion_fields},
            upsert=True,
        )
        
        # Trigger scoring
        self._trigger_scoring(lead_id)
        
        return self.db.leads.find_one({"_id": lead_id}) or {}

    def close(self):
        """Close the Mongo connection."""
        self.client.close()

    def create_lead_from_responses(
        self,
        responses: Dict[str, Any],
        *,
        session_id: Optional[str],
        question_set_title: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Persist a lead built from collected responses.

        Args:
            responses: Extracted responses keyed by question identifier.
            session_id: Conversation/session identifier (falls back to lead_id).
            question_set_title: Question set used to collect the responses.
            domain: Optional tenant/persona identifier.

        Returns:
            Inserted lead document (including generated lead_id).
        """
        mongo_leads = self.db.leads
        lead_id = generate_lead_id(self.db)  # type: ignore[arg-type]
        now_iso = datetime.utcnow().isoformat()

        lead_document: Dict[str, Any] = {
            "_id": lead_id,
            "id": lead_id,
            "name": responses.get("name"),
            "email": responses.get("email"),
            "phone": responses.get("phone"),
            "source": responses.get("source"),
            "interest": responses.get("interest"),
            "message": responses.get("message") or responses.get("additional_message"),
            "lead_metadata": responses.get("lead_metadata") or {},
            "created_at": now_iso,
            "status": "new",
            "session_id": session_id or lead_id,
        }

        assigned_to = responses.get("assigned_to")
        if assigned_to:
            lead_document["assigned_to"] = assigned_to

        # Preserve raw questionnaire context for the ops team.
        # AND flatten responses for scoring compatibility
        lead_metadata = (lead_document.get("lead_metadata") or {}).copy()
        # Flatten responses into metadata, excluding lead_metadata itself to avoid cycles
        responses_flat = {k: v for k, v in responses.items() if k != "lead_metadata"}
        lead_metadata.update(responses_flat)
        lead_metadata.update({
            "question_set": question_set_title,
            "domain": domain,
            "raw_responses": responses,
        })
        lead_document["lead_metadata"] = lead_metadata

        mongo_leads.insert_one(lead_document)
        
        # Trigger scoring
        self._trigger_scoring(lead_id)
        
        return lead_document


def create_lead(responses: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Convenience wrapper that manages LeadWriter lifecycle for single use.
    """
    writer = LeadWriter()
    try:
        return writer.create_lead_from_responses(responses, **kwargs)
    finally:
        writer.close()


def create_realtime_lead_id(
    session_id: str,
    question_set_title: str,
    domain: Optional[str] = None,
) -> str:
    """Generate and persist a lead ID before the form starts."""
    writer = LeadWriter()
    try:
        lead_id = writer.generate_new_lead_id()
        writer.initialize_lead_shell(lead_id, session_id, question_set_title, domain)
        return lead_id
    finally:
        writer.close()


def upsert_realtime_lead(
    *,
    lead_id: str,
    session_id: str,
    updates: Dict[str, Any],
    question_set_title: str,
    domain: Optional[str] = None,
) -> None:
    """Save incremental answers for an in-progress conversational lead."""
    writer = LeadWriter()
    try:
        writer.initialize_lead_shell(lead_id, session_id, question_set_title, domain)
        writer.upsert_partial_lead(lead_id, session_id, updates)
    finally:
        writer.close()


def finalize_realtime_lead(
    *,
    lead_id: str,
    responses: Dict[str, Any],
    session_id: str,
    question_set_title: str,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Finalize the conversational lead, keeping the existing ID."""
    writer = LeadWriter()
    try:
        document = writer.finalize_lead_document(
            lead_id=lead_id,
            session_id=session_id,
            responses=responses,
            question_set_title=question_set_title,
            domain=domain,
        )
        return {
            "status": "success",
            "lead_id": lead_id,
            "lead_document": document,
        }
    finally:
        writer.close()


