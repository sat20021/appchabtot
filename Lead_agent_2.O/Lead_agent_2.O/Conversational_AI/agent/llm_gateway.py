import os
import logging
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from Lead_agent_2.O .env file
_lead_agent_env = Path(__file__).resolve().parents[3] / ".env"
if _lead_agent_env.exists():
    load_dotenv(_lead_agent_env)

# Also try loading from Backend .env as fallback
_backend_env = Path(__file__).resolve().parents[5] / ".env"
if _backend_env.exists():
    load_dotenv(_backend_env, override=False)

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Supported Gemini models (updated 2024)
VALID_GEMINI_MODELS = [
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-pro-latest",
    "models/gemini-flash-latest",
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-flash-latest",
]

# Non-retryable error patterns (fail-fast)
NON_RETRYABLE_ERRORS = [
    "404",
    "NOT_FOUND",
    "INVALID_ARGUMENT",
    "MODEL_NOT_FOUND",
    "PERMISSION_DENIED",
    "API_KEY_INVALID",
    "RESOURCE_EXHAUSTED",
]


class LLMGateway:
    """
    Gateway to handle multiple LLM providers (OpenAI, Gemini) with fallback support.
    Allows switching providers via environment variables.
    Includes fail-fast mechanism for non-retryable errors.
    """
    
    _last_error: Optional[Exception] = None
    _failed_models: List[str] = []
    
    @classmethod
    def get_llm(cls, model_name: Optional[str] = None, temperature: float = 0.7):
        """
        Get an LLM instance based on the configured provider.
        
        Args:
            model_name: Model to use (optional). If not provided, reads from environment.
            temperature: Temperature for response generation
        
        Returns:
            LLM instance
        """
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        if provider == "gemini":
            return cls._get_gemini_llm(model_name, temperature)
        elif provider == "openai":
            return cls._get_openai_llm(model_name, temperature)
        elif provider == "openrouter":
            return cls._get_openrouter_llm(model_name, temperature)
        else:
            # Default to Gemini
            return cls._get_gemini_llm(model_name, temperature)
    
    
    @classmethod
    def _get_gemini_llm(cls, model_name: Optional[str], temperature: float):
        """Get Gemini LLM with fallback support."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Primary model from env or default
        primary_model = model_name or os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash")
        
        # Fallback model
        fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "models/gemini-pro-latest")
        
        # Validate model name format
        if not cls._is_valid_gemini_model(primary_model):
            logger.warning(f"Model '{primary_model}' may not be valid. Using fallback: {fallback_model}")
            primary_model = fallback_model
        
        return ChatGoogleGenerativeAI(
            model=primary_model,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
    
    @classmethod
    def _get_openai_llm(cls, model_name: Optional[str], temperature: float):
        """Get OpenAI LLM."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY not found or not configured")
        
        model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature
        )

    @classmethod
    def _get_openrouter_llm(cls, model_name: Optional[str], temperature: float):
        """Get OpenRouter LLM (via OpenAI compatibility layer)."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        model = model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature
        )
    
    
    @staticmethod
    def _is_valid_gemini_model(model_name: str) -> bool:
        """Check if model name matches known valid Gemini models."""
        # Check exact match or starts with valid prefix
        for valid in VALID_GEMINI_MODELS:
            if model_name == valid or model_name.startswith(valid.split("-latest")[0]):
                return True
        return False
    
    
    @classmethod
    def is_retryable_error(cls, error: Exception) -> bool:
        """
        Check if error is retryable or should fail-fast.
        
        Args:
            error: Exception that occurred
        
        Returns:
            True if retryable, False if should fail-fast
        """
        error_str = str(error).upper()
        for pattern in NON_RETRYABLE_ERRORS:
            if pattern.upper() in error_str:
                return False
        return True
    
    
    @classmethod
    def get_fallback_llm(cls, temperature: float = 0.7):
        """
        Get fallback LLM when primary fails.
        
        Args:
            temperature: Temperature for response generation
        
        Returns:
            Fallback LLM instance or None
        """
        fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "models/gemini-pro-latest")
        
        # Avoid infinite loop - don't return fallback if it already failed
        if fallback_model in cls._failed_models:
            logger.error("Fallback model also failed, no more options")
            return None
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        
        logger.info(f"Using fallback model: {fallback_model}")
        return ChatGoogleGenerativeAI(
            model=fallback_model,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
    
    @classmethod
    def mark_model_failed(cls, model_name: str):
        """Mark a model as failed to prevent retry loops."""
        if model_name not in cls._failed_models:
            cls._failed_models.append(model_name)
            logger.warning(f"Model '{model_name}' marked as failed")
    
    @classmethod
    def reset_failed_models(cls):
        """Reset the failed models list (for testing or recovery)."""
        cls._failed_models = []
        cls._last_error = None
