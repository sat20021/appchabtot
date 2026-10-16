"""
LLM Context Awareness Module: Enterprise-grade structured context for LLM decisions.

This module ensures the LLM always has proper understanding of:
- Form completion status  
- User information collected
- Conversation history
- Valid actions based on state

Created: 2025-12-04
Purpose: Prevent duplicate leads by making context impossible to miss
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class FormState(Enum):
    """Form completion states."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class LLMConversationContext:
    """
    Structured conversation context for LLM.
    
    This ensures LLM receives context in a format that's
    impossible to misunderstand.
    """
    # State information
    form_state: FormState
    completed_forms: List[str]
    collected_data: Dict[str, str]
    applicant_id: Optional[str] = None  # NEW: Application ID if available
    
    # Conversation history
    recent_messages: List[Dict]
    
    # Constraints (what LLM CAN'T do)
    forbidden_actions: List[str]
    
    # Available actions (what LLM CAN do)
    allowed_actions: List[str]
    
    # Reasoning hints
    context_hints: List[str]
    
    def to_llm_context(self) -> str:
        """
        Convert to LLM-friendly format with EXPLICIT instructions.
        
        Returns:
            str: Formatted context string for LLM prompt
        """
        lines = []
        
        # 1. CRITICAL STATE (can't be missed)
        lines.append("=" * 70)
        lines.append("CRITICAL CONTEXT - READ THIS BEFORE MAKING ANY DECISION")
        lines.append("=" * 70)
        lines.append("")
        
        # 2. Form completion status (EXPLICIT)
        if self.form_state == FormState.COMPLETED:
            lines.append("🔴 FORM STATUS: *** COMPLETED ***")
            lines.append(f"   Completed forms: {', '.join(self.completed_forms)}")
            lines.append("")
            lines.append("   ⚠️  CRITICAL: DO NOT RESTART THESE FORMS!")
            lines.append("   ⚠️  User has already provided their information!")
            lines.append("")
        elif self.form_state == FormState.IN_PROGRESS:
            lines.append("🟡 FORM STATUS: IN PROGRESS")
            lines.append("   Continue the current form, do NOT start a new one")
            lines.append("")
        else:
            lines.append("🟢 FORM STATUS: NOT STARTED")
            lines.append("   You may start a form if user wants to provide info")
            lines.append("")
        
        # 3. Collected data (EXPLICIT)
        if self.collected_data:
            lines.append("📋 USER INFORMATION ALREADY ON FILE:")
            lines.append("")
            for key, value in self.collected_data.items():
                lines.append(f"   ✓ {key.capitalize()}: {value}")
            lines.append("")
            lines.append("   ⚠️  IMPORTANT: Since you have this information,")
            lines.append("       DO NOT ask for it again or restart the form!")
            lines.append("")
        
        # 4. Applicant ID (if available)
        if self.applicant_id:
            lines.append("🎫 APPLICATION ID:")
            lines.append(f"   Application ID: {self.applicant_id}")
            lines.append("")
            lines.append("   ⚠️  IMPORTANT: Always include this Application ID in your")
            lines.append("       response when confirming the application was received!")
            lines.append("")
        
        # 4. What LLM CANNOT do (EXPLICIT CONSTRAINTS)
        if self.forbidden_actions:
            lines.append("🚫 FORBIDDEN ACTIONS - YOU MUST NOT USE THESE:")
            lines.append("")
            for action in self.forbidden_actions:
                lines.append(f"   ✗ {action}")
            lines.append("")
        
        # 5. What LLM CAN do (ALLOWED ACTIONS)
        lines.append("✅ ALLOWED ACTIONS - YOU MAY ONLY USE THESE:")
        lines.append("")
        for action in self.allowed_actions:
            lines.append(f"   ✓ {action}")
        lines.append("")
        
        # 6. Reasoning hints
        if self.context_hints:
            lines.append("💡 CONTEXT HINTS TO GUIDE YOUR DECISION:")
            lines.append("")
            for hint in self.context_hints:
                lines.append(f"   • {hint}")
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("")
        
        return "\n".join(lines)


class LLMContextBuilder:
    """
    Builds structured context for LLM based on session state.
    
    This is the single source of truth for what context the LLM receives.
    """
    
    @staticmethod
    def build(
        session_data: Dict,
        form_data: Dict,
        completed_forms: List[str],
        history: List[Dict],
        applicant_id: Optional[str] = None  # NEW: Pass applicant_id explicitly
    ) -> LLMConversationContext:
        """
        Build structured context from session state.
        
        Args:
            session_data: Full session data from Redis
            form_data: User information collected from forms
            completed_forms: List of completed form names
            history: Conversation history
            applicant_id: Application ID if available
            
        Returns:
            LLMConversationContext: Structured context for LLM
        """
        
        # Try to get applicant_id from session if not provided
        if not applicant_id:
            applicant_id = (
                session_data.get("applicant_id") or
                form_data.get("applicant_id") or
                session_data.get("form_data", {}).get("applicant_id")
            )
        
        # Determine form state
        if completed_forms:
            form_state = FormState.COMPLETED
            logger.info(f"📊 LLM Context: Forms COMPLETED - {completed_forms}")
        elif session_data.get("active_flow"):
            form_state = FormState.IN_PROGRESS
            logger.info(f"📊 LLM Context: Form IN PROGRESS - {session_data.get('active_flow')}")
        else:
            form_state = FormState.NOT_STARTED
            logger.info("📊 LLM Context: No forms started")
        
        # Build forbidden actions based on state
        forbidden_actions = []
        if form_state == FormState.COMPLETED:
            # Explicitly forbid restarting completed forms
            for form in completed_forms:
                if "lead" in form.lower():
                    forbidden_actions.append(
                        "START_LEAD_FORM - Lead form already completed, DO NOT restart"
                    )
        
        # Build allowed actions
        allowed_actions = ["RESPOND_MESSAGE - Answer the user's question directly"]
        
        if form_state == FormState.NOT_STARTED:
            allowed_actions.extend([
                "START_LEAD_FORM - Only if user wants to provide contact info",
            ])
        elif form_state == FormState.IN_PROGRESS:
            allowed_actions.append("Continue the current form - Do not start a new one")

        
        # Build context hints
        context_hints = []
        
        if form_data:
            data_count = len(form_data)
            context_hints.append(
                f"User has already provided {data_count} piece(s) of information. "
                "Use this data to answer their questions."
            )
        
        if form_state == FormState.COMPLETED:
            context_hints.extend([
                "If user asks about their info (name, email, phone, etc.), "
                "answer using the collected data shown above.",
                "DO NOT restart the form - they already completed it!",
                "If they want to UPDATE their info, acknowledge what you have "
                "and ask what they'd like to change."
            ])
        
        # Add applicant_id context hint
        if applicant_id:
            context_hints.append(
                f"The user's application has been submitted. Their Application ID is {applicant_id}. "
                "ALWAYS mention this ID when confirming their application was received."
            )
        
        # Build recent messages context
        recent_messages = history[-5:] if history else []
        
        logger.info(
            f"📊 LLM Context built: state={form_state.value}, "
            f"data_fields={len(form_data)}, "
            f"forbidden_actions={len(forbidden_actions)}"
        )
        
        return LLMConversationContext(
            form_state=form_state,
            completed_forms=completed_forms,
            collected_data=form_data or {},
            applicant_id=applicant_id,  # NEW: Include applicant_id
            recent_messages=recent_messages,
            forbidden_actions=forbidden_actions,
            allowed_actions=allowed_actions,
            context_hints=context_hints
        )
