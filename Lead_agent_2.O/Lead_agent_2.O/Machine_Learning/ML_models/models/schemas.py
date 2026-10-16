# models/schemas.py
# Pydantic models for API requests and responses
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from enum import Enum

# ============================================================================
# Pipeline Manager Models
# ============================================================================

class PipelineStatus(str, Enum):
    IDLE = "idle"
    GENERATING_DATA = "generating_data"
    TRAINING_MODEL = "training_model"
    COMPLETED = "completed"
    FAILED = "failed"

class IntentDefinition(BaseModel):
    templates: List[str] = Field(..., description="List of template strings for the intent")
    
class IntentsConfig(BaseModel):
    domain: str = Field(..., description="Domain name (e.g., 'tech_company', 'healthcare')")
    intents: Dict[str, IntentDefinition] = Field(..., description="Intent definitions with templates")
    roles: Optional[List[str]] = Field(None, description="List of roles for job-related templates")
    skills: Optional[List[str]] = Field(None, description="List of skills for job-related templates")
    samples_per_intent: int = Field(400, description="Number of samples to generate per intent", ge=100, le=2000)
    train_split: float = Field(0.8, description="Train/validation split ratio", ge=0.5, le=0.95)
    add_noise: bool = Field(True, description="Add realistic noise like typos")

class PipelineStatusResponse(BaseModel):
    status: PipelineStatus
    domain: Optional[str] = None
    start_time: Optional[str] = None
    completion_time: Optional[str] = None
    current_step: Optional[str] = None
    error: Optional[str] = None
    dataset_path: Optional[str] = None
    ml_model_path: Optional[str] = None
    metrics: Optional[Dict] = None

# ============================================================================
# Classifier Service Models
# ============================================================================

class PredictRequest(BaseModel):
    query: str = Field(..., description="User query text to classify", min_length=1)
    domain: Optional[str] = Field(None, description="Domain to use for classification (defaults to active domain)")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    return_all_probs: bool = Field(False, description="Return probabilities for all intents")

class IntentProbability(BaseModel):
    intent: str
    probability: float

class PredictResponse(BaseModel):
    intent: str
    confidence: float
    domain: str
    all_probabilities: Optional[List[IntentProbability]] = None
    query: str
    timestamp: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    active_domain: Optional[str]
    available_domains: List[str]
    num_intents: int

class ModelInfo(BaseModel):
    domain: str
    embed_model: str
    intents: List[str]
    num_intents: int
    ml_model_type: str
    accuracy: Optional[float] = None
    training_date: Optional[str] = None

# ============================================================================
# Greeting Service Models
# ============================================================================

class GreetingRequest(BaseModel):
    query: str = Field(..., description="User's greeting query", min_length=1)
    domain: Optional[str] = Field(None, description="Domain for context-aware greetings")
    user_name: Optional[str] = Field(None, description="User's name for personalization")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")

class GreetingResponse(BaseModel):
    is_greeting: bool
    is_gratitude: bool
    response_type: Optional[str]
    response_message: Optional[str]
    greeting_message: Optional[str]
    gratitude_message: Optional[str]
    intent: str
    confidence: float
    domain: str
    user_query: str
    timestamp: str
    personalized: bool

