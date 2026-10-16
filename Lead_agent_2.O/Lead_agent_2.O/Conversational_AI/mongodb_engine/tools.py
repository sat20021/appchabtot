"""
LangChain tools for querying MongoDB job listings.

These tools are designed for ReAct-style use by the conversational agent:
- `search_job_listings_tool` to find relevant job descriptions
- `get_job_details_tool` to fetch a single job by `job_id`
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pymongo import ASCENDING, DESCENDING

from mongodb_engine.client import get_job_listings_collection


def _build_job_search_query(
    role_keywords: Optional[str] = None,
    location: Optional[str] = None,
    experience_level: Optional[str] = None,
    job_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a MongoDB query for job listings based on simple filters.

    We intentionally keep this conservative so it works even without full‑text
    indexes – using case‑insensitive regex on title/description when needed.
    
    Filters applied:
    - deleted: Must not be True
    - status: Must not be "Deleted", "Closed", or "Archived"
    - is_active: Must be True (strict check)
    - closing_date: Must be in the future or not set
    """
    
    # Get current datetime for closing_date comparison
    now_iso = datetime.utcnow().isoformat()

    query: Dict[str, Any] = {
        "deleted": {"$ne": True},
        "status": {"$nin": ["Deleted", "Closed", "Archived"]},
        "is_active": True,  # Strict check - only truly active jobs
        # Closing date filter: either not set, null, or in the future
        "$and": [
            {
                "$or": [
                    {"closing_date": {"$exists": False}},
                    {"closing_date": None},
                    {"closing_date": ""},
                    {"closing_date": {"$gte": now_iso}},
                ]
            }
        ]
    }

    if role_keywords:
        # Match against title and description using a simple regex OR clause.
        # We need to use $and to combine with existing conditions
        query["$and"].append({
            "$or": [
                {"title": {"$regex": role_keywords, "$options": "i"}},
                {"description": {"$regex": role_keywords, "$options": "i"}},
            ]
        })

    if location:
        query["location"] = {"$regex": location, "$options": "i"}

    if experience_level:
        query["experience_level"] = {"$regex": experience_level, "$options": "i"}

    if job_type:
        query["job_type"] = {"$regex": job_type, "$options": "i"}

    return query


@tool
def search_job_listings_tool(
    role_keywords: Optional[str] = None,
    location: Optional[str] = None,
    experience_level: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Search the MongoDB `job_listings` collection for matching job descriptions.

    Args:
        role_keywords: Free‑text role or skill description (e.g., "python backend").
        location: Preferred location string (e.g., "Delhi", "Remote").
        experience_level: Experience level hint (e.g., "entry", "mid", "senior").
        job_type: Job type hint (e.g., "full-time", "internship").
        limit: Maximum number of jobs to return.

    Returns:
        Dict with `status`, `results` (a list of compact job summaries),
        and `query_used` for transparency.
    """

    try:
        collection = get_job_listings_collection()
        
        # Use the same robust query building logic as job_selection.py
        # Base filter: Status must be Active (case-insensitive) or Open
        status_filter = {
            "$or": [
                {"status": {"$regex": "^active$", "$options": "i"}},
                {"status": {"$regex": "^open$", "$options": "i"}}
            ]
        }
        
        # Start with the status filter
        filter_doc = status_filter.copy()
        
        search_criteria = []
        
        if role_keywords:
            search_criteria.append({
                '$or': [
                    {'title': {'$regex': role_keywords, '$options': 'i'}},
                    {'skills': {'$regex': role_keywords, '$options': 'i'}},
                    {'description': {'$regex': role_keywords, '$options': 'i'}},
                    {'job_function': {'$regex': role_keywords, '$options': 'i'}}
                ]
            })
            
        if location and location.lower() != "none":
            search_criteria.append({
                'location': {'$regex': location, '$options': 'i'}
            })
            
        if experience_level:
            search_criteria.append({
                'experience_level': {'$regex': experience_level, '$options': 'i'}
            })
            
        if job_type:
            search_criteria.append({
                'job_type': {'$regex': job_type, '$options': 'i'}
            })
            
        if search_criteria:
            filter_doc['$and'] = search_criteria

        cursor = (
            collection.find(filter_doc)
            .sort([("is_active", DESCENDING), ("created_at", DESCENDING)])
            .limit(max(limit, 1))
        )

        results: List[Dict[str, Any]] = []
        for doc in cursor:
            results.append(
                {
                    "job_id": doc.get("job_id"),
                    "title": doc.get("title"),
                    "location": doc.get("location"),
                    "job_type": doc.get("job_type"),
                    "experience_level": doc.get("experience_level"),
                    "salary_range": doc.get("salary_range"),
                    "status": doc.get("status"),
                    "is_active": doc.get("is_active"),
                    "short_description": (doc.get("description") or "")[:280],
                    "key_skills": doc.get("key_skills", [])[:10],
                    "source_file_url": doc.get("source_file_url"),
                    "closing_date": doc.get("closing_date")
                }
            )

        return {
            "tool": "search_job_listings_tool",
            "status": "success",
            "count": len(results),
            "results": results,
            "query_used": filter_doc,
        }
    except Exception as exc:  # pragma: no cover - surfaced to agent logic
        return {
            "tool": "search_job_listings_tool",
            "status": "error",
            "error": str(exc),
        }


@tool
def get_job_details_tool(job_id: str) -> Dict[str, Any]:
    """
    Fetch a single job description document from MongoDB by `job_id`.

    This is used after the user selects a specific job from search results.
    """

    if not job_id:
        return {
            "tool": "get_job_details_tool",
            "status": "error",
            "error": "job_id is required",
        }

    try:
        collection = get_job_listings_collection()
        # Only return job details if the job is Active or Open
        doc = collection.find_one(
            {
                "job_id": job_id,
                "deleted": {"$ne": True},
                "$or": [
                    {"status": {"$regex": "^active$", "$options": "i"}},
                    {"status": {"$regex": "^open$", "$options": "i"}}
                ]
            }
        )

        if not doc:
            return {
                "tool": "get_job_details_tool",
                "status": "not_found",
                "job_id": job_id,
                "message": "Job not found or no longer active"
            }

        # Only return fields that are relevant for the form and UX
        job_details = {
            "job_id": doc.get("job_id"),
            "title": doc.get("title"),
            "job_function": doc.get("job_function"),
            "location": doc.get("location"),
            "description": doc.get("description"),
            "key_skills": doc.get("key_skills", []),
            "education_requirements": doc.get("education_requirements"),
            "experience_level": doc.get("experience_level"),
            "job_type": doc.get("job_type"),
            "salary_range": doc.get("salary_range"),
            "closing_date": doc.get("closing_date"),
            "status": doc.get("status"),
            "is_active": doc.get("is_active"),
        }

        return {
            "tool": "get_job_details_tool",
            "status": "success",
            "job": job_details,
        }
    except Exception as exc:  # pragma: no cover - surfaced to agent logic
        return {
            "tool": "get_job_details_tool",
            "status": "error",
            "error": str(exc),
        }


# Convenience export for registration alongside existing tools
MONGODB_TOOLS = [
    search_job_listings_tool,
    get_job_details_tool,
]




