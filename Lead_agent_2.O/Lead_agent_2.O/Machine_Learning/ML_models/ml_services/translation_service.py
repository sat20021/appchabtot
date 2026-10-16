"""
Translation Service for Enterprise AI Agent.

This module provides translation capabilities using the LLM Gateway.
It handles translating agent responses back to the user's preferred locale.
"""
import logging
import json
from pathlib import Path
import importlib.util
from typing import Optional

# Import LLMGateway using absolute file path to avoid module path issues
# Similar to lcel_router.py
try:
    llm_gateway_path = Path(__file__).parent.parent.parent.parent / "Conversational_AI" / "agent" / "llm_gateway.py"
    spec = importlib.util.spec_from_file_location("llm_gateway", llm_gateway_path)
    llm_gateway_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_gateway_module)
    LLMGateway = llm_gateway_module.LLMGateway
except Exception as e:
    logging.warning(f"Failed to import LLMGateway for translation service: {e}")
    LLMGateway = None

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from ml_utils.redis_cache_handler import RedisCacheHandler

logger = logging.getLogger(__name__)

# Cache TTL for translations (24 hours)
TRANSLATION_CACHE_TTL = 86400

_TRANSLATION_CHAIN = None

def _get_translation_chain():
    """Lazy-load the translation chain."""
    global _TRANSLATION_CHAIN
    if _TRANSLATION_CHAIN is None and LLMGateway:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a professional translator. Translate the following text to {target_locale}. "
                    "Preserve the tone, formatting (markdown), and any special tokens like URLs or email addresses. "
                    "Return ONLY the translated text, no explanations."
                ),
                ("human", "{text}"),
            ]
        )
        # Use a low temperature for deterministic translations
        model = LLMGateway.get_llm(temperature=0.1)
        _TRANSLATION_CHAIN = prompt | model | RunnableLambda(lambda msg: getattr(msg, "content", str(msg)))
    
    return _TRANSLATION_CHAIN

def translate_text(text: str, target_locale: str) -> str:
    """
    Translate text to the target locale.
    
    Args:
        text: The text to translate.
        target_locale: The target locale code (e.g., 'fr', 'es', 'hi').
        
    Returns:
        The translated text, or the original text if translation fails or is unnecessary.
    """
    if not text or not target_locale:
        return text
        
    # Normalize locale
    target_locale = target_locale.lower().split('-')[0]
    
    # Skip if English (assuming base language is English)
    if target_locale == 'en':
        return text
        
    # Check cache first
    redis = RedisCacheHandler()
    cache_key = f"trans:{target_locale}:{hash(text)}"
    
    if redis.is_available():
        cached = redis.get_str(cache_key)
        if cached:
            logger.debug(f"Translation cache hit for locale {target_locale}")
            return cached
            
    # Perform translation
    try:
        chain = _get_translation_chain()
        if not chain:
            logger.warning("Translation chain not available (LLMGateway import failed?)")
            return text
            
        logger.info(f"Translating response to {target_locale}...")
        translated_text = chain.invoke({"text": text, "target_locale": target_locale})
        translated_text = str(translated_text).strip()
        
        # Cache the result
        if redis.is_available():
            redis.set_str(cache_key, translated_text, ttl=TRANSLATION_CACHE_TTL)
            
        return translated_text
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text
