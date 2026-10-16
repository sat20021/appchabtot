"""
MongoDB engine nodes for LangGraph workflow.

This module provides custom nodes for the conversation graph that interact
with MongoDB, such as job description selection.
"""
from typing import Dict, Any
from agent.state import AgentState
from mongodb_engine.tools import search_job_listings_tool, get_job_details_tool
import logging

logger = logging.getLogger(__name__)


def job_description_selection_node(state: AgentState) -> AgentState:
    """
    Node for handling job description selection when user is applying for a job.
    
    This node is used in job application flows where the user needs to select
    a specific job from available listings before proceeding with the application.
    
    Args:
        state: Current agent state
    
    Returns:
        Updated agent state with selected job information
    """
    user_message = state.get("user_message", "").strip()
    
    if not user_message:
        # No user input yet - ask them to select a job
        state["agent_message"] = (
            "To start your job application, please let me know which position "
            "you're interested in, or I can show you our available openings."
        )
        return state
    
    # Try to search for jobs based on user's message
    try:
        # Extract potential role keywords from message
        # Use .invoke() for LangChain tools (deprecated __call__ syntax)
        search_result = search_job_listings_tool.invoke({
            "role_keywords": user_message,
            "limit": 10
        })
        
        job_listings = search_result.get("results", [])
        job_count = search_result.get("count", 0)
        
        if job_count == 0:
            state["agent_message"] = (
                "I couldn't find any openings matching your search. "
                "Would you like me to show you all available positions?"
            )
            return state
        
        if job_count == 1:
            # Exactly one match - auto-select it
            selected_job = job_listings[0]
            state["selected_job_id"] = selected_job.get("job_id")
            state["selected_job_title"] = selected_job.get("title")
            state["agent_message"] = (
                f"Great! I found the {selected_job.get('title')} position. "
                f"Let's start your application!"
            )
            logger.info(f"Auto-selected job: {selected_job.get('title')} (job_id: {selected_job.get('job_id')})")
            return state
        
        # Multiple matches - ask user to select
        formatted_jobs = []
        for idx, job in enumerate(job_listings, 1):
            title = job.get("title", "Unknown Position")
            location = job.get("location", "")
            job_type = job.get("job_type", "")
            
            job_summary = f"{idx}. {title}"
            if location:
                job_summary += f" - {location}"
            if job_type:
                job_summary += f" ({job_type})"
            
            formatted_jobs.append(job_summary)
        
        jobs_display = "\n".join(formatted_jobs)
        state["agent_message"] = (
            f"I found {job_count} positions matching your search:\n\n"
            f"{jobs_display}\n\n"
            f"Which position would you like to apply for? Please specify the number or position name."
        )
        
        # Store job listings in state for next interaction
        state["pending_job_selection"] = job_listings
        
        return state
        
    except Exception as e:
        logger.error(f"Error in job_description_selection_node: {e}", exc_info=True)
        state["agent_message"] = (
            "I'm having trouble fetching job listings right now. "
            "Please try again in a moment."
        )
        return state
