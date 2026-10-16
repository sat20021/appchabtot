"""
State Synchronization Module
Ensures consistency between:
1. Redis Session State (Long-term memory)
2. LangGraph State (Short-term execution state)
3. Question Registry (Authoritative question status)
"""

import logging
from typing import Dict, List, Optional
from hybrid_orchestrator.session_manager import get_session, upsert_session
from hybrid_orchestrator.question_registry import (
    get_question_registry,
    bulk_update_registry,
    QuestionStatus
)

logger = logging.getLogger(__name__)

class StateSynchronizer:
    
    @staticmethod
    def sync_on_completion(
        session_id: str, 
        collected_data: Dict, 
        completed_form_name: str
    ) -> bool:
        """
        Final synchronization when a form is completed.
        1. Updates Redis session with final collected data.
        2. Marks form as completed in Redis.
        3. Clears active flow.
        4. Updates Question Registry to mark all relevant questions as ANSWERED/EXHAUSTED.
        """
        try:
            session = get_session(session_id)
            if not session:
                logger.error(f"❌ Sync failed: Session {session_id} not found")
                return False
                
            # 1. Update Session Data
            # Merge new data with existing form_data
            existing_data = session.get("form_data", {})
            merged_data = {**existing_data, **collected_data}
            
            # 2. Update Completed Forms
            completed_forms = session.get("completed_forms", [])
            if completed_form_name not in completed_forms:
                completed_forms.append(completed_form_name)
                
            updates = {
                "form_data": merged_data,
                "completed_forms": completed_forms,
                "active_flow": None, # Clear active flow
                "graph_state_id": None # Reset graph state
            }
            
            upsert_session(session_id, updates)
            logger.info(f"✅ Synced session {session_id}: Form '{completed_form_name}' marked complete.")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in sync_on_completion: {e}")
            return False

    @staticmethod
    def sync_registry_to_graph(session_id: str, graph_state: Dict) -> Dict:
        """
        Pull latest registry status into graph state.
        Useful when resuming a conversation.
        """
        registry = get_question_registry(session_id)
        
        # Calculate definitively completed keys
        completed_keys = [
            k for k, v in registry.items() 
            if v["status"] in [
                QuestionStatus.ANSWERED.value, 
                QuestionStatus.DECLINED.value,
                QuestionStatus.SKIPPED.value,
                QuestionStatus.PRE_EXTRACTED.value,
                QuestionStatus.EXHAUSTED.value
            ]
        ]
        
        graph_state["question_status_registry"] = registry
        graph_state["definitively_completed_keys"] = completed_keys
        
        return graph_state
