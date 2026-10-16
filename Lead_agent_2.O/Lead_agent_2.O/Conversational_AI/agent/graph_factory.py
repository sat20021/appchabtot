"""
Factory to create a dedicated graph/node for each question set
Ensures each question set has its own graph instance
"""
from agent.graph import create_conversation_graph
from typing import Dict
from database.mongodb import MongoDBHandler


class QuestionSetGraphFactory:
    """Factory to create and manage graphs for each question set"""
    
    def __init__(self):
        self.graphs: Dict[str, any] = {}  # question_set_title -> graph
        self.db = MongoDBHandler()
    
    def get_or_create_graph(self, question_set_title: str):
        """
        Get existing graph or create new one for a question set
        
        This ensures each question set has its own dedicated graph instance
        
        Args:
            question_set_title: Title of the question set
            
        Returns:
            Compiled LangGraph instance for this question set
        """
        if question_set_title not in self.graphs:
            # Load question set to validate it exists
            question_set = self.db.load_question_set(question_set_title)
            if not question_set:
                raise ValueError(f"Question set '{question_set_title}' not found")
            
            # Create dedicated graph for this question set
            graph = create_conversation_graph()
            self.graphs[question_set_title] = graph
        
        return self.graphs[question_set_title]
    
    def create_agent_for_question_set(self, question_set_title: str):
        """
        Create a ConversationalAgent instance for a specific question set
        Each question set gets its own graph/node
        
        Args:
            question_set_title: Title of the question set
            
        Returns:
            ConversationalAgent instance
        """
        from conv_main import ConversationalAgent
        return ConversationalAgent(question_set_title)
    
    def clear_cache(self):
        """Clear the graph cache (useful for testing or reloading)"""
        self.graphs.clear()
    
    def get_cached_graphs(self) -> list:
        """Get list of question set titles with cached graphs"""
        return list(self.graphs.keys())


# Global factory instance
graph_factory = QuestionSetGraphFactory()

