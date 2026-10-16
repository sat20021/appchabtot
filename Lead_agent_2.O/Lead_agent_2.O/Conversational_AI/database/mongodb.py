"""
MongoDB operations for loading question sets and saving responses
"""
from pymongo import MongoClient
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

try:
    from database.lead_writer import create_lead
except ImportError:
    from Conversational_AI.database.lead_writer import create_lead

load_dotenv()


class MongoDBHandler:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client[os.getenv("MONGODB_DB")]
        self.collection = self.db["llm_questions"]
    
    def load_question_set(self, title: str, jd_selected: bool = None) -> Optional[Dict]:
        """
        Load a specific question set by title.
        
        Args:
            title: The question set title (e.g., 'job_application_questions')
            jd_selected: Optional. If True, load the variant with job_description_selected='true'
                        (for when user selected a specific job). If False, load the variant
                        with job_description_selected='false' (when no specific job selected).
                        If None, load the first matching question set.
        """
        query = {"title": title}
        
        # If jd_selected is specified, filter by that field
        if jd_selected is not None:
            jd_value = "true" if jd_selected else "false"
            query["job_description_selected"] = jd_value
        
        question_set = self.collection.find_one(query)
        if question_set:
            # Remove MongoDB _id from the dict
            question_set.pop('_id', None)
        return question_set
    
    def get_all_question_sets(self) -> List[Dict]:
        """Get all available question sets"""
        return list(self.collection.find({}, {"title": 1, "_id": 0}))
    
    def save_response(
        self,
        session_id: str,
        question_set_title: str,
        responses: Dict,
        *,
        domain: Optional[str] = None,
    ) -> Dict:
        """
        Persist collected responses into the unified leads collection.

        Returns:
            The inserted lead document (including generated lead_id / _id).
        """
        if not responses:
            raise ValueError("Cannot save empty responses.")

        return create_lead(
            responses,
            session_id=session_id,
            question_set_title=question_set_title,
            domain=domain,
        )
    
    def upsert_question_set(self, question_set_data: Dict):
        """Update or insert a question set"""
        title = question_set_data.get("title")
        if not title:
            return False
        
        # Use replace_one with upsert=True to overwrite the existing document
        self.collection.replace_one(
            {"title": title},
            question_set_data,
            upsert=True
        )
        return True
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()

