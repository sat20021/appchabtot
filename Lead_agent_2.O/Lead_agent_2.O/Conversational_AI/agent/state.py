"""
State schema for the conversational agent
"""
from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add


def dedupe_add(left: List[str], right: List[str]) -> List[str]:
    """
    Custom reducer that adds items but deduplicates.
    This prevents the massive list growth bug.
    """
    if left is None:
        left = []
    if right is None:
        right = []
    # Use dict.fromkeys to preserve order while deduplicating
    combined = list(dict.fromkeys(left + right))
    return combined


class AgentState(TypedDict):
    """State of the conversational agent"""
    
    # Question set information
    question_set_title: str
    question_set_data: Dict
    
    # Current progress
    current_question_key: Optional[str]
    questions_queue: List[str]  # List of question keys to ask
    questions_asked: Annotated[List[str], dedupe_add]  # 🔴 FIX: Deduplicating reducer
    
    # Collected data
    collected_responses: Dict  # {question_key: answer}
    lead_id: Optional[str]
    declined_questions: Annotated[List[str], dedupe_add]  # 🔴 FIX: Deduplicating reducer
    
    # Conversation tracking
    messages: Annotated[List[Dict], add]  # Conversation history
    user_message: str  # Latest user message
    agent_message: str  # Latest agent message
    
    # Flags
    is_first_message: bool
    needs_clarification: bool
    off_topic_handled: bool
    all_questions_completed: bool
    
    # Clarification retry tracking
    clarification_retry_count: Dict[str, int]  # {question_key: retry_count}
    
    # 🆕 Validation error message for clarification prompts
    validation_error: Optional[str]  # Human-friendly error when input validation fails
    
    # LLM processing results (temporary, used between nodes)
    llm_result: Optional[Dict]
    nlu_data: Optional[Dict]  # Structured NLU output {intent, key, value}
    intermediate_steps: Optional[List]
    
    # Session info
    session_id: str
    domain: str  # Domain/tenant for persona and RAG context

    # Job application context (optional fields, mainly for JD-based flows)
    job_description_selected: Optional[bool]
    selected_job_id: Optional[str]
    selected_job_details: Optional[Dict]
    applicant_id: Optional[str]
    
    # Resume processing
    resume_file_path: Optional[str]
    resume_parsing_status: Optional[str]  # 'pending', 'processing', 'completed', 'failed'
    resume_analysis_result: Optional[Dict]
    
    # 🔴 NEW: Persisted user context from previous form sessions
    session_form_data: Optional[Dict]  # User data from completed forms in this session

