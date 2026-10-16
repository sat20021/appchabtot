"""
FastAPI REST API for the conversational agent
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union, Literal
from conv_main import ConversationalAgent
from database.mongodb import MongoDBHandler
from dotenv import load_dotenv
import uuid
from enum import Enum
import json

load_dotenv()

app = FastAPI(
    title="Conversational Agent API",
    description="REST API for conversational information collection agent",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis or database in production)
active_sessions: Dict[str, ConversationalAgent] = {}


# ==================== Request/Response Models ====================

class QuestionType(str, Enum):
    """Question type enumeration"""
    TEXT = "text"
    OPTIONS = "options"
    FILE = "file"


class TextQuestionResponse(BaseModel):
    """Response model for text-type questions"""
    session_id: str
    question_key: str
    question_text: str
    question_type: Literal["text"] = "text"
    agent_message: str
    is_complete: bool = False
    collected_data: Dict = {}


class OptionsQuestionResponse(BaseModel):
    """Response model for options-type questions"""
    session_id: str
    question_key: str
    question_text: str
    question_type: Literal["options"] = "options"
    options: List[str]
    agent_message: str
    is_complete: bool = False
    collected_data: Dict = {}


class FileQuestionResponse(BaseModel):
    """Response model for file-type questions"""
    session_id: str
    question_key: str
    question_text: str
    question_type: Literal["file"] = "file"
    agent_message: str
    accepted_formats: Optional[List[str]] = None  # e.g., ["pdf", "doc", "docx"]
    max_file_size: Optional[str] = None  # e.g., "10MB"
    is_complete: bool = False
    collected_data: Dict = {}


class StartConversationRequest(BaseModel):
    """Request model to start a conversation"""
    question_set_title: str = Field(..., description="Title of the question set to use")


class StartConversationResponse(BaseModel):
    """Response model for starting conversation"""
    session_id: str
    message: str


class SendMessageRequest(BaseModel):
    """Request model for sending a message"""
    session_id: str = Field(..., description="Session ID from start conversation")
    message: str = Field(..., description="User's response message")


class SendMessageResponse(BaseModel):
    """Response model for sending message (union of all question types)"""
    session_id: str
    question_key: Optional[str] = None
    question_text: str
    question_type: QuestionType
    agent_message: str
    is_complete: bool = False
    collected_data: Dict = {}
    lead_id: Optional[str] = None
    # Fields specific to question types
    options: Optional[List[str]] = None  # For options type
    accepted_formats: Optional[List[str]] = None  # For file type
    max_file_size: Optional[str] = None  # For file type


class UploadFileRequest(BaseModel):
    """Request model for file upload"""
    session_id: str = Field(..., description="Session ID")
    question_key: str = Field(..., description="Question key for file upload")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None


# ==================== Helper Functions ====================

def get_question_text(question_data: Dict) -> str:
    """Extract question text from question data (handles both dict and string formats)"""
    if isinstance(question_data.get("name"), dict):
        return question_data.get("name", {}).get("name", "")
    else:
        return question_data.get("name", "")


def get_question_response(
    agent: ConversationalAgent,
    session_id: str
) -> Union[TextQuestionResponse, OptionsQuestionResponse, FileQuestionResponse]:
    """
    Get appropriate response model based on current question type
    
    Args:
        agent: ConversationalAgent instance
        session_id: Session ID
        
    Returns:
        Appropriate response model based on question type
    """
    state = agent.state
    current_key = state.get("current_question_key")
    
    if not current_key:
        # No more questions
        return TextQuestionResponse(
            session_id=session_id,
            question_key="",
            question_text="",
            question_type="text",
            agent_message=state.get("agent_message", ""),
            is_complete=True,
            collected_data=state.get("collected_responses", {})
        )
    
    question_data = state["question_set_data"].get(current_key, {})
    question_type = question_data.get("type", "text")
    question_text = get_question_text(question_data)
    agent_message = state.get("agent_message", "")
    collected_data = state.get("collected_responses", {})
    is_complete = state.get("all_questions_completed", False)
    
    if question_type == "options":
        options = question_data.get("options", [])
        return OptionsQuestionResponse(
            session_id=session_id,
            question_key=current_key,
            question_text=question_text,
            question_type="options",
            options=options,
            agent_message=agent_message,
            is_complete=is_complete,
            collected_data=collected_data
        )
    
    elif question_type == "file":
        # You can configure accepted formats and max size per question
        accepted_formats = question_data.get("accepted_formats", ["pdf", "doc", "docx", "txt"])
        max_file_size = question_data.get("max_file_size", "10MB")
        
        return FileQuestionResponse(
            session_id=session_id,
            question_key=current_key,
            question_text=question_text,
            question_type="file",
            agent_message=agent_message,
            accepted_formats=accepted_formats,
            max_file_size=max_file_size,
            is_complete=is_complete,
            collected_data=collected_data
        )
    
    else:  # text type
        return TextQuestionResponse(
            session_id=session_id,
            question_key=current_key,
            question_text=question_text,
            question_type="text",
            agent_message=agent_message,
            is_complete=is_complete,
            collected_data=collected_data
        )


def get_unified_response(
    agent: ConversationalAgent,
    session_id: str
) -> SendMessageResponse:
    """
    Get unified response model that works for all question types
    
    Args:
        agent: ConversationalAgent instance
        session_id: Session ID
        
    Returns:
        Unified SendMessageResponse
    """
    state = agent.state
    current_key = state.get("current_question_key")
    
    lead_id = state.get("lead_id")

    if not current_key:
        return SendMessageResponse(
            session_id=session_id,
            question_key=None,
            question_text="",
            question_type=QuestionType.TEXT,
            agent_message=state.get("agent_message", ""),
            is_complete=True,
            collected_data=state.get("collected_responses", {}),
            lead_id=lead_id,
        )
    
    question_data = state["question_set_data"].get(current_key, {})
    question_type = QuestionType(question_data.get("type", "text"))
    question_text = get_question_text(question_data)
    agent_message = state.get("agent_message", "")
    collected_data = state.get("collected_responses", {})
    is_complete = state.get("all_questions_completed", False)
    
    response = SendMessageResponse(
        session_id=session_id,
        question_key=current_key,
        question_text=question_text,
        question_type=question_type,
        agent_message=agent_message,
        is_complete=is_complete,
        collected_data=collected_data,
        lead_id=lead_id,
    )
    
    # Add type-specific fields
    if question_type == QuestionType.OPTIONS:
        response.options = question_data.get("options", [])
    elif question_type == QuestionType.FILE:
        response.accepted_formats = question_data.get("accepted_formats", ["pdf", "doc", "docx", "txt"])
        response.max_file_size = question_data.get("max_file_size", "10MB")
    
    return response


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Conversational Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/question-sets", response_model=List[Dict])
async def get_question_sets():
    """
    Get all available question sets
    
    Returns:
        List of available question sets with their titles
    """
    try:
        db = MongoDBHandler()
        question_sets = db.get_all_question_sets()
        db.close()
        return question_sets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching question sets: {str(e)}")


@app.post("/conversation/start", response_model=Union[TextQuestionResponse, OptionsQuestionResponse, FileQuestionResponse])
async def start_conversation(request: StartConversationRequest):
    """
    Start a new conversation session
    
    Args:
        request: StartConversationRequest with question_set_title
        
    Returns:
        First question response (type varies based on question)
    """
    try:
        # Create agent
        agent = ConversationalAgent(request.question_set_title)
        
        # Start conversation
        agent.start()
        
        # Generate session ID
        session_id = agent.state.get("session_id") or str(uuid.uuid4())
        agent.state["session_id"] = session_id
        
        # Store session
        active_sessions[session_id] = agent
        
        # Get appropriate response based on question type
        response = get_question_response(agent, session_id)
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting conversation: {str(e)}")


@app.post("/conversation/message", response_model=SendMessageResponse)
async def send_message(request: SendMessageRequest):
    """
    Send a message in an active conversation
    
    Args:
        request: SendMessageRequest with session_id and message
        
    Returns:
        Agent response with next question (type varies based on question)
    """
    # Get agent from session
    agent = active_sessions.get(request.session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new conversation.")
    
    try:
        # Process user message
        agent.send_message(request.message)
        
        # Get unified response
        response = get_unified_response(agent, request.session_id)
        
        # Clean up if conversation is complete
        if response.is_complete:
            agent.close()
            active_sessions.pop(request.session_id, None)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.post("/conversation/upload-file")
async def upload_file(
    file: UploadFile = File(...)
):
    """
    Upload a JSON file containing question sets and update the collection
    
    Args:
        file: Uploaded JSON file
        
    Returns:
        Status message
    """
    if not file.filename.endswith('.json'):
         raise HTTPException(status_code=400, detail="File must be a JSON file")
         
    try:
        content = await file.read()
        data = json.loads(content)
        
        if isinstance(data, dict):
            data = [data]
            
        if not isinstance(data, list):
             raise HTTPException(status_code=400, detail="JSON content must be a list of question sets or a single question set object")
             
        db = MongoDBHandler()
        count = 0
        for item in data:
            if db.upsert_question_set(item):
                count += 1
        db.close()
        
        return {"message": f"Successfully updated {count} question sets"}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/conversation/{session_id}/status")
async def get_conversation_status(session_id: str):
    """
    Get current status of a conversation session
    
    Args:
        session_id: Session ID
        
    Returns:
        Current conversation status
    """
    agent = active_sessions.get(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = agent.state
    return {
        "session_id": session_id,
        "question_set_title": state.get("question_set_title"),
        "current_question_key": state.get("current_question_key"),
        "questions_asked": state.get("questions_asked", []),
        "questions_remaining": len(state.get("questions_queue", [])),
        "collected_data": state.get("collected_responses", {}),
        "is_complete": state.get("all_questions_completed", False),
        "lead_id": state.get("lead_id"),
    }


@app.get("/conversation/{session_id}/data")
async def get_collected_data(session_id: str):
    """
    Get all collected data for a session
    
    Args:
        session_id: Session ID
        
    Returns:
        All collected responses
    """
    agent = active_sessions.get(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "collected_data": agent.get_collected_data(),
        "lead_id": agent.state.get("lead_id"),
    }


@app.delete("/conversation/{session_id}")
async def end_conversation(session_id: str):
    """
    End a conversation session
    
    Args:
        session_id: Session ID
        
    Returns:
        Success message
    """
    agent = active_sessions.get(session_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        agent.close()
        active_sessions.pop(session_id, None)
        return {"message": "Conversation ended successfully", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending conversation: {str(e)}")


# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

