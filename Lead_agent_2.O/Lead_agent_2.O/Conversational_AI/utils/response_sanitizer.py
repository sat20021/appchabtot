"""
Response Sanitizer - Post-processing for human-like responses

This module removes robotic phrases and applies human-like replacements
to LLM responses. It's rule-based with ZERO hallucination risk.
"""

import re
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# BANNED PHRASES - These make responses sound robotic
# ============================================================================
BANNED_PHRASES = [
    "to help me assist you",
    "to help you better",
    "to better assist you",
    "in order to",
    "please provide",
    "could you please share",
    "could you please provide",
    "may i have your",
    "i understand that",
    "thank you for providing",
    "to better serve you",
    "for your convenience",
    "i would be happy to",
    "i am here to help",
    "how may i assist you",
    "is there anything else i can help you with",
    "i appreciate your",
    "kindly provide",
    "kindly share",
]


# ============================================================================
# PHRASE REPLACEMENTS - Make language more natural
# ============================================================================
PHRASE_REPLACEMENTS = {
    "what is your email": "what's a good email",
    "what is your phone": "what number works for you",
    "what is your name": "what should i call you",
    "may i know your": "what's your",
    "could you tell me your": "what's your",
    "please share your": "what's your",
    "would you like to share": "want to share",
    "i would like to know": "i'm curious about",
    "can you provide": "what's",
    "we need your": "what's your",
}


# ============================================================================
# SAFE ACKNOWLEDGMENTS - Hardcoded, never LLM-generated
# ============================================================================
ACKNOWLEDGMENTS = [
    "Got it!", "Perfect!", "Great!", "Awesome!", "Noted!",
    "Thanks!", "Cool!", "Sweet!", "Nice!", "Lovely!"
]

TRANSITIONS = [
    "And", "Now,", "Quick one -", "Just need", "One more -",
    "By the way,", "Oh and", "Last thing -"
]


# ============================================================================
# SANITIZATION FUNCTIONS
# ============================================================================
def remove_banned_phrases(text: str) -> str:
    """Remove robotic phrases from response."""
    result = text
    text_lower = text.lower()
    
    for phrase in BANNED_PHRASES:
        # Find and remove the phrase (case-insensitive)
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        result = pattern.sub("", result)
    
    # Clean up extra spaces and punctuation
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*,\s*,', ',', result)
    result = re.sub(r'\s*\.\s*\.', '.', result)
    result = result.strip()
    
    # Fix leading punctuation
    result = re.sub(r'^[,\.\s]+', '', result)
    
    return result


def apply_replacements(text: str) -> str:
    """Replace formal phrases with casual ones."""
    result = text
    
    for old, new in PHRASE_REPLACEMENTS.items():
        pattern = re.compile(re.escape(old), re.IGNORECASE)
        result = pattern.sub(new, result)
    
    return result


def fix_capitalization(text: str) -> str:
    """Ensure proper sentence capitalization."""
    if not text:
        return text
    
    # Capitalize first letter
    result = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Capitalize after periods
    result = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), result)
    
    return result


def get_random_acknowledgment() -> str:
    """Get a random acknowledgment phrase."""
    return random.choice(ACKNOWLEDGMENTS)


def get_random_transition() -> str:
    """Get a random transition phrase."""
    return random.choice(TRANSITIONS)


def add_acknowledgment_prefix(text: str, has_previous_answer: bool = True) -> str:
    """
    Add a friendly acknowledgment before the question.
    
    Args:
        text: The question/response text
        has_previous_answer: Whether user just answered something
    
    Returns:
        Text with acknowledgment prefix
    """
    if not has_previous_answer:
        return text
    
    # Don't add if already starts with an acknowledgment
    text_lower = text.lower().strip()
    for ack in ACKNOWLEDGMENTS:
        if text_lower.startswith(ack.lower().rstrip('!')):
            return text
    
    ack = get_random_acknowledgment()
    transition = get_random_transition()
    
    # Combine naturally
    return f"{ack} {transition} {text[0].lower()}{text[1:]}" if text else text


def sanitize_response(
    response: str, 
    add_prefix: bool = False,
    has_previous_answer: bool = False
) -> str:
    """
    Full sanitization pipeline for LLM responses.
    
    Args:
        response: Raw LLM response
        add_prefix: Whether to add acknowledgment prefix
        has_previous_answer: Whether user just provided an answer
    
    Returns:
        Sanitized, human-like response
    """
    if not response:
        return response
    
    # Step 1: Remove banned phrases
    result = remove_banned_phrases(response)
    
    # Step 2: Apply phrase replacements
    result = apply_replacements(result)
    
    # Step 3: Fix capitalization
    result = fix_capitalization(result)
    
    # Step 4: Optionally add acknowledgment prefix
    if add_prefix and has_previous_answer:
        result = add_acknowledgment_prefix(result, has_previous_answer)
    
    logger.debug(f"Sanitized response: '{response[:50]}...' -> '{result[:50]}...'")
    
    return result


# ============================================================================
# QUESTION VALIDATION - Anti-hallucination
# ============================================================================
QUESTION_VALIDATORS = {
    "email": ["email", "@", "reach you", "contact", "inbox"],
    "phone": ["phone", "number", "call", "reach", "mobile", "cell"],
    "name": ["name", "call you", "who", "yourself"],
    "company": ["company", "business", "organization", "work", "firm"],
    "company_name": ["company", "business", "organization", "work", "firm"],
    "source": ["find us", "hear about", "discover", "learn about"],
    "interest": ["interested", "looking", "build", "project", "need"],
    "additional_message": ["else", "more", "additional", "share", "anything"],
    "resume": ["resume", "cv", "upload", "document"],
    "experience": ["experience", "years", "level", "background"],
    "preferred_role": ["role", "position", "job", "looking for"],
    "user_persona": ["student", "parent", "are you"],
    "program_interest": ["program", "course", "interested", "study"],
    "program_type": ["level", "ug", "pg", "diploma", "phd", "undergraduate", "postgraduate"],
    "highest_education": ["education", "qualification", "completed", "study", "degree"],
    "entrance_exam": ["exam", "entrance", "test", "taken"],
    "exam_name": ["which", "exam", "name"],
    "exam_score": ["score", "rank", "percentile", "marks"],
    "financial_aid": ["scholarship", "financial", "aid", "funding"],
}

FALLBACK_QUESTIONS = {
    "email": "What's a good email to reach you?",
    "phone": "What number works best for you?",
    "name": "And you are...?",
    "company": "What company are you with?",
    "company_name": "What company are you with?",
    "source": "How'd you find us?",
    "interest": "What are you looking to build?",
    "additional_message": "Anything else you'd like to share?",
    "resume": "Want to share your resume?",
    "experience_level": "How would you describe your experience level?",
    "preferred_role": "What kind of role are you looking for?",
}


def validate_question_response(response: str, question_key: str) -> tuple:
    """
    Validate that LLM response actually asks for the intended field.
    
    Args:
        response: LLM-generated response
        question_key: The field we're supposed to ask about
    
    Returns:
        (is_valid, final_response) - uses fallback if invalid
    """
    validators = QUESTION_VALIDATORS.get(question_key, [question_key])
    response_lower = response.lower()
    
    # Check if any validator keyword is present
    if any(v in response_lower for v in validators):
        return True, response
    
    # Hallucination detected - use fallback
    logger.warning(
        f"🚨 Response validation FAILED for '{question_key}'. "
        f"Response: '{response[:80]}...' Missing keywords: {validators}"
    )
    
    fallback = FALLBACK_QUESTIONS.get(question_key, f"Could you share your {question_key}?")
    return False, fallback


# ============================================================================
# TESTING
# ============================================================================
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "To help me assist you further, could you please share your phone number?",
        "In order to better serve you, may I have your email address?",
        "I understand that you need help. What is your name?",
        "Thank you for providing that information. Could you please provide your company name?",
        "Great! What is your email?",
    ]
    
    print("=" * 60)
    print("RESPONSE SANITIZER TESTS")
    print("=" * 60)
    
    for orig in test_cases:
        sanitized = sanitize_response(orig, add_prefix=True, has_previous_answer=True)
        print(f"\nORIGINAL:   {orig}")
        print(f"SANITIZED:  {sanitized}")
    
    print("\n" + "=" * 60)
    print("VALIDATION TESTS")
    print("=" * 60)
    
    validation_tests = [
        ("What's your email?", "email"),
        ("Tell me about your background", "email"),  # Should fail
        ("What number can I reach you at?", "phone"),
    ]
    
    for response, key in validation_tests:
        is_valid, final = validate_question_response(response, key)
        print(f"\nResponse: '{response}' | Key: {key}")
        print(f"Valid: {is_valid} | Final: '{final}'")
