"""
Main entry point for the conversational agent
"""
from agent.graph import create_conversation_graph
from agent.state import AgentState
from database.mongodb import MongoDBHandler
from dotenv import load_dotenv
import os

load_dotenv()


class ConversationalAgent:
    """Wrapper class for the conversational agent"""
    
    def __init__(self, question_set_title: str, domain: str = "default"):
        """
        Initialize the conversational agent with a specific question set
        
        Args:
            question_set_title: Title of the question set from MongoDB
            domain: Domain/tenant identifier for persona and RAG context
        """
        self.question_set_title = question_set_title
        self.domain = domain
        self.db = MongoDBHandler()
        self.graph = create_conversation_graph()
        self.state = None
        
        # Load question set
        question_set = self.db.load_question_set(question_set_title)
        if not question_set:
            raise ValueError(f"Question set '{question_set_title}' not found in database")
        
        # Initialize state with all required fields
        # Fields annotated with 'add' must be initialized as empty lists, not None
        self.state = {
            "question_set_title": question_set_title,
            "question_set_data": question_set,
            "session_id": "",
            "domain": domain,  # ← NEW: Domain for persona and RAG
            "user_message": "",
            "agent_message": "",
            "current_question_key": None,
            "questions_queue": [],
            "questions_asked": [],  # Annotated with 'add' - must be list
            "collected_responses": {},
            "lead_id": None,
            "declined_questions": [],  # Annotated with 'add' - must be list
            "messages": [],  # Annotated with 'add' - must be list
            "is_first_message": True,
            "needs_clarification": False,
            "off_topic_handled": False,
            "all_questions_completed": False,
            "clarification_retry_count": {},
            "llm_result": None,
            "intermediate_steps": None,
            "session_form_data": {},  # 🔴 NEW: Initialize for persisted context
            "validation_error": None,  # 🆕 For clarification prompts when input validation fails
        }
    
    
    def start(self, initial_context: str = None):
        """
        Start the conversation
        
        Args:
            initial_context: Optional previous user message to extract info from
        """
        # Run initialization first
        result = self.graph.invoke(self.state)
        self.state.update(result)
        
        # If we have initial context (previous message), process it to extract info
        if initial_context and initial_context.strip():
            print(f"🔍 Processing initial context: '{initial_context}'")
            
            # Temporarily set user message to process the context
            self.state["user_message"] = initial_context
            
            # Process through LLM to extract any information
            result = self.graph.invoke(self.state)
            self.state.update(result)
            
            # Clear user_message so next invoke asks the question
            self.state["user_message"] = ""
            
            # Check if we extracted any info
            extracted = self.state.get("collected_responses", {})
            if extracted:
                print(f"✅ Pre-extracted from context: {list(extracted.keys())}")
                
                # CRITICAL FIX: Advance current_question_key to skip already-answered questions
                # Otherwise ask_question_node will ask for already-collected info
                current_key = self.state.get("current_question_key")
                questions_queue = list(self.state.get("questions_queue", []))
                questions_asked = list(self.state.get("questions_asked", []))
                
                # Skip over all collected questions to find next unanswered one
                while current_key and current_key in extracted:
                    if current_key in questions_queue:
                        questions_queue.remove(current_key)
                    if current_key not in questions_asked:
                        questions_asked.append(current_key)
                    # Get next question from queue
                    current_key = questions_queue[0] if questions_queue else None
                
                
                print(f"🔄 Advanced to next question: {current_key}")
                
                # CRITICAL: Update state with advanced question tracking BEFORE re-invoking
                # This ensures ask_question_node sees the correct current_question_key
                self.state["current_question_key"] = current_key
                self.state["questions_queue"] = questions_queue
                self.state["questions_asked"] = questions_asked
                
                # CRITICAL: Clear old results to force fresh question generation
                self.state["llm_result"] = None
                self.state["user_message"] = ""
                
                print(f"✅ State updated BEFORE re-invoke: current_question_key={self.state['current_question_key']}")
                
                # Re-invoke to get the next question message
                result = self.graph.invoke(self.state)
                
                # Only update agent_message and messages, preserve our question tracking
                self.state["agent_message"] = result.get("agent_message", "")
                if "messages" in result:
                    self.state["messages"] = result["messages"]
                
                print(f"✅ Final current_question_key: {self.state.get('current_question_key')}")
                print(f"✅ Final agent_message (first 100 chars): {self.state.get('agent_message', '')[:100]}")
        
        return self.state.get("agent_message", "")
    
    def send_message(self, user_message: str):
        """
        Send a user message and get agent response
        
        Args:
            user_message: The user's response
            
        Returns:
            Agent's response message
        """
        # Update state with user message
        self.state["user_message"] = user_message
        
        # Process through graph - the router will route to llm_processing
        # since user_message now exists
        result = self.graph.invoke(self.state)
        self.state.update(result)
        
        return self.state.get("agent_message", "")
    
    def get_collected_data(self):
        """Get all collected responses"""
        return self.state.get("collected_responses", {})
    
    def get_state(self) -> dict:
        """Return the current internal state for persistence."""
        return self.state

    def load_state(self, state_data: dict):
        """Restore internal state from persistence."""
        if state_data:
            # Merge restored state, ensuring critical keys exist
            self.state.update(state_data)
            # Ensure questions_queue is a list
            if not isinstance(self.state.get("questions_queue"), list):
                # If queue is corrupted/missing, rebuild it from question_set_data
                keys = [k for k in self.state.get("question_set_data", {}).keys() 
                       if k not in ["_id", "title"]]
                self.state["questions_queue"] = keys
    
    def close(self):
        """Close database connection"""
        self.db.close()


def main():
    """Main function to run the agent"""
    
    print("Conversational Agent POC")
    print("=" * 50)
    
    # List available question sets
    db = MongoDBHandler()
    question_sets = db.get_all_question_sets()
    
    print("\nAvailable Question Sets:")
    for idx, qs in enumerate(question_sets, 1):
        print(f"{idx}. {qs['title']}")
    
    # Select question set
    choice = input("\nSelect question set number: ")
    selected_title = question_sets[int(choice) - 1]['title']
    
    # Create agent
    agent = ConversationalAgent(selected_title)
    
    # Start conversation
    print("\n" + "=" * 50)
    first_message = agent.start()
    print(f"Agent: {first_message}")
    
    # Conversation loop
    while not agent.state.get("all_questions_completed"):
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'stop']:
            print("Agent: Thank you for your time! Goodbye.")
            break
        
        response = agent.send_message(user_input)
        print(f"Agent: {response}")
    
    # Show collected data
    print("\n" + "=" * 50)
    print("Collected Data:")
    print(agent.get_collected_data())
    
    agent.close()


if __name__ == "__main__":
    main()

