"""
Tools that the LLM can use to extract information and handle exceptions.

This module also aggregates additional tool sets (e.g., MongoDB tools) that can
be exposed to the agent depending on the active question set.
"""
from langchain_core.tools import tool
from typing import Dict, Optional, List, Union


@tool
def extract_answer(question_key: str, answer: str, confidence: str = "high") -> Dict:
    """
    Extract and store an answer for a specific question.
    
    Args:
        question_key: The key of the question being answered
        answer: The extracted answer from user's response
        confidence: Confidence level (high/medium/low)
    
    Returns:
        Dictionary with extraction status
    """
    return {
        "tool": "extract_answer",
        "question_key": question_key,
        "answer": answer,
        "confidence": confidence,
        "status": "success"
    }


@tool
def extract_multiple_answers(answers: Union[Dict[str, str], str]) -> Dict:
    """
    Extract multiple answers from a single user response.
    
    Args:
        answers: Dictionary mapping question_key to extracted answer.
                Can also be a JSON string representation of the dictionary.
        Example: {"name": "John Doe", "email": "john@example.com"}
    
    Returns:
        Dictionary with extraction status
    """
    # Handle string input (common LLM behavior)
    if isinstance(answers, str):
        try:
            import json
            # Clean up potential markdown code blocks if present
            cleaned_answers = answers.strip()
            if cleaned_answers.startswith("```json"):
                cleaned_answers = cleaned_answers.replace("```json", "").replace("```", "")
            elif cleaned_answers.startswith("```"):
                cleaned_answers = cleaned_answers.replace("```", "")
            
            answers = json.loads(cleaned_answers)
        except Exception as e:
            return {
                "tool": "extract_multiple_answers",
                "error": f"Failed to parse answers JSON: {str(e)}",
                "status": "error"
            }

    return {
        "tool": "extract_multiple_answers",
        "answers": answers,
        "count": len(answers),
        "status": "success"
    }


@tool
def mark_question_declined(question_key: str, reason: str = "User preferred not to answer") -> Dict:
    """
    Mark a question as declined by the user.
    
    Args:
        question_key: The key of the question user declined to answer
        reason: Reason for declining
    
    Returns:
        Dictionary with decline status
    """
    return {
        "tool": "mark_question_declined",
        "question_key": question_key,
        "reason": reason,
        "status": "declined"
    }


@tool
def request_clarification(question_key: str, reason: str) -> Dict:
    """
    Request clarification from user because answer was unclear or invalid.
    
    Args:
        question_key: The question that needs clarification
        reason: Why clarification is needed
    
    Returns:
        Dictionary with clarification request
    """
    return {
        "tool": "request_clarification",
        "question_key": question_key,
        "reason": reason,
        "status": "needs_clarification"
    }




def update_lead_data(key: str, value: str, state: Dict) -> Dict:
    """
    Validate and update lead data.
    
    Args:
        key: The field to update
        value: The new value
        state: The current agent state (for validation rules)
        
    Returns:
        Dict with 'success', 'message', 'error'
    """
    # Validation Logic
    if key in state.get("question_set_data", {}):
        question_data = state["question_set_data"][key]
        q_type = question_data.get("type", "text")
        
        # Handle Options
        if q_type == "options":
            options = [opt.lower() for opt in question_data.get("options", [])]
            if value.lower() not in options:
                return {
                    "success": False,
                    "error": f"Invalid option. Available: {', '.join(question_data.get('options', []))}"
                }
        
        # Handle Email
        elif "email" in key.lower() or q_type == "email":
            import re
            email_regex = r"[^@]+@[^@]+\.[^@]+"
            if not re.match(email_regex, value):
                return {
                    "success": False,
                    "error": "Invalid email format."
                }
        
        # Handle Phone
        elif "phone" in key.lower() or "mobile" in key.lower() or q_type == "phone":
            import re
            phone_clean = re.sub(r"[^0-9+]", "", value)
            if len(phone_clean) < 7:
                return {
                    "success": False,
                    "error": "Invalid phone number format."
                }

    return {
        "success": True,
        "message": f"Updated {key} to {value}"
    }
def update_response(question_key: str, new_value: str, reason: str = "User correction") -> Dict:
    """
    Update a previously collected answer with a new value.
    Use this when the user explicitly corrects information (e.g., "Change my email to...").
    
    Args:
        question_key: The key of the question to update (e.g., 'email', 'name')
        new_value: The new value provided by the user
        reason: Why the update is happening
    
    Returns:
        Dictionary with update status
    """
    return {
        "tool": "update_response",
        "question_key": question_key,
        "new_value": new_value,
        "reason": reason,
        "status": "update_requested"
    }


@tool
def provide_final_summary(collected_data: Dict, declined_data: List[str], status: str = "completed") -> Dict:
    """
    Provide a final summary of the conversation's collected and declined data.
    Call this when the conversation ends, is stopped by the user, or all questions are answered/declined.
    
    Args:
        collected_data: Dictionary of all information collected so far
        declined_data: List of question keys that were declined by the user
        status: Status of the conversation (e.g., "completed", "stopped_by_user")
    
    Returns:
        Dictionary with the summary
    """
    return {
        "tool": "provide_final_summary",
        "collected_data": collected_data,
        "declined_data": declined_data,
        "status": status
    }


@tool
def handle_off_topic(user_query: str, appropriate_response: str) -> Dict:
    """
    Handle off-topic questions from user and provide appropriate response.
    
    Args:
        user_query: The off-topic question asked by user
        appropriate_response: A brief, helpful response to the off-topic query
    
    Returns:
        Dictionary with off-topic handling
    """
    return {
        "tool": "handle_off_topic",
        "user_query": user_query,
        "response": appropriate_response,
        "status": "off_topic"
    }


@tool
def query_knowledge_base(question: str, domain: str = "default") -> Dict:
    """
    Query the RAG knowledge base for company-specific information.
    
    Use this tool when you need to look up information about the company,
    products, services, policies, or any domain-specific knowledge that
    might help answer the user's questions or provide better context.
    
    Args:
        question: The specific question to search for in the knowledge base
        domain: The domain/tenant ID to search in. Defaults to "default".
    
    Returns:
        Dictionary with search results and context
    
    Examples:
        - query_knowledge_base("What services does the company offer?", domain="tech_corp")
        - query_knowledge_base("What is the return policy?", domain="retail_store")
    """
    try:
        # Lazy import to avoid circular dependencies and startup overhead
        import sys
        from pathlib import Path
        
        # Add RAG directory to path if needed
        repo_root = Path(__file__).resolve().parents[3]
        rag_root = repo_root / "RAG"
        if str(rag_root) not in sys.path:
            sys.path.insert(0, str(rag_root))
        
        from app.services.document_query import get_document_query_service
        
        # Get RAG service instance
        query_service = get_document_query_service()
        
        # Search with lower threshold for better recall
        results = query_service.search(
            query=question,
            domain=domain,
            top_k=3,
            threshold=0.3
        )
        
        if not results:
            return {
                "tool": "query_knowledge_base",
                "question": question,
                "domain": domain,
                "found": False,
                "answer": "No relevant information found in the knowledge base.",
                "status": "no_results"
            }
        
        # Format results for LLM
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "source": doc.source_file,
                "content": doc.content,
                "relevance_score": doc.score
            })
        
        # Combine top results into context
        context = "\n\n".join([
            f"Source: {doc.source_file}\n{doc.content}"
            for doc in results[:2]  # Top 2 results
        ])
        
        return {
            "tool": "query_knowledge_base",
            "question": question,
            "domain": domain,
            "found": True,
            "answer": context,
            "results_count": len(results),
            "sources": [doc.source_file for doc in results],
            "status": "success"
        }
    
    except ImportError as e:
        return {
            "tool": "query_knowledge_base",
            "question": question,
            "domain": domain,
            "found": False,
            "answer": "Knowledge base is not available at this time.",
            "error": f"Import error: {str(e)}",
            "status": "error"
        }
    except Exception as e:
        return {
            "tool": "query_knowledge_base",
            "question": question,
            "domain": domain,
            "found": False,
            "answer": f"Error searching knowledge base: {str(e)}",
            "error": str(e),
            "status": "error"
        }


def _create_lead_record_payload(
    responses: Dict,
    session_id: str,
    question_set_title: str,
    domain: str = "default",
) -> Dict:
    try:
        from database.lead_writer import create_lead

        lead_doc = create_lead(
            responses,
            session_id=session_id,
            question_set_title=question_set_title,
            domain=domain,
        )
        return {
            "tool": "create_lead_record_tool",
            "status": "success",
            "lead_id": lead_doc["_id"],
            "lead_document": lead_doc,
        }
    except Exception as exc:  # pragma: no cover - surfaced to agent logic
        return {
            "tool": "create_lead_record_tool",
            "status": "error",
            "error": str(exc),
        }


@tool
def create_lead_record_tool(
    responses: Union[Dict, str],
    session_id: str,
    question_set_title: str,
    domain: str = "default",
) -> Dict:
    """
    Persist collected responses into the leads collection and return the new lead ID.
    """
    # Handle string input for responses
    if isinstance(responses, str):
        try:
            import json
            cleaned_responses = responses.strip()
            if cleaned_responses.startswith("```json"):
                cleaned_responses = cleaned_responses.replace("```json", "").replace("```", "")
            elif cleaned_responses.startswith("```"):
                cleaned_responses = cleaned_responses.replace("```", "")
            responses = json.loads(cleaned_responses)
        except Exception as e:
            return {
                "tool": "create_lead_record_tool",
                "status": "error",
                "error": f"Failed to parse responses JSON: {str(e)}"
            }

    return _create_lead_record_payload(
        responses=responses,
        session_id=session_id,
        question_set_title=question_set_title,
        domain=domain,
    )


def create_lead_record_action(
    responses: Dict,
    session_id: str,
    question_set_title: str,
    domain: str = "default",
) -> Dict:
    """
    Direct callable variant for non-LLM flows (e.g., deterministic nodes).
    """
    return _create_lead_record_payload(
        responses=responses,
        session_id=session_id,
        question_set_title=question_set_title,
        domain=domain,
    )


def _load_mongodb_tools() -> List:
    """
    Lazy‑load MongoDB tools so this module has no hard dependency on them.

    This allows the conversational agent to run in environments where the
    MongoDB engine is not configured, while still enabling job‑search flows
    when it is available.
    """
    try:
        from mongodb_engine.tools import MONGODB_TOOLS

        return list(MONGODB_TOOLS)
    except Exception as exc:  # pragma: no cover - defensive, surfaced via logs
        print(f"[agent.tools] MongoDB tools not available: {exc}")
        return []


# Base tools that are always available
BASE_AGENT_TOOLS = [
    extract_answer,
    extract_multiple_answers,
    update_response,
    provide_final_summary,
    mark_question_declined,
    request_clarification,
    handle_off_topic,
    query_knowledge_base,
    create_lead_record_tool,
]


def get_agent_tools(include_mongo: bool = False) -> list:
    """
    Return the tool set for the agent.

    Args:
        include_mongo: When True, also include MongoDB job‑search tools.
    """

    tools: List = list(BASE_AGENT_TOOLS)
    if include_mongo:
        tools.extend(_load_mongodb_tools())
    return tools


# Backwards‑compatible export used by existing nodes
AGENT_TOOLS = get_agent_tools(include_mongo=True)

