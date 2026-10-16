"""
Utility helper functions
"""
import re
from typing import Dict, Any, Optional


EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_CLEANUP_REGEX = re.compile(r"[^\d]")


def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email:
        return False
    return EMAIL_REGEX.fullmatch(email.strip()) is not None


def validate_phone(phone: str) -> bool:
    """Validate phone number (basic validation)."""
    if not phone:
        return False
    cleaned = PHONE_CLEANUP_REGEX.sub("", phone)
    return 10 <= len(cleaned) <= 15 and cleaned.isdigit()


def extract_first_email(value: str) -> Optional[str]:
    """Extract the first email-like token from arbitrary text."""
    if not value:
        return None
    match = EMAIL_REGEX.search(value)
    return match.group(0) if match else None


def normalize_email_answer(value: str) -> Optional[str]:
    """Normalize conversational email answer into a clean address."""
    email_candidate = (value or "").strip()
    if validate_email(email_candidate):
        return email_candidate
    return extract_first_email(email_candidate)


def normalize_phone_answer(value: str) -> Optional[str]:
    """Normalize conversational phone answer into digits-only string."""
    if not value:
        return None
    cleaned = PHONE_CLEANUP_REGEX.sub("", value)
    digits = "".join(filter(str.isdigit, cleaned))
    if 10 <= len(digits) <= 15:
        return digits
    return None


def extract_question_keys(question_set: Dict) -> list:
    """Extract question keys from question set, excluding metadata"""
    excluded_keys = ['_id', 'title', 'job_description_selected']
    return [key for key in question_set.keys() if key not in excluded_keys]


def format_question_for_display(question_data: Dict) -> str:
    """Format question with options if type is 'options'"""
    question_text = question_data.get('name', '')
    
    if question_data.get('type') == 'options':
        options = question_data.get('options', [])
        options_text = ', '.join(options)
        return f"{question_text} (Options: {options_text})"
    
    return question_text


def get_next_question_key(questions_queue: list) -> str:
    """Get the next question key from queue"""
    if questions_queue:
        return questions_queue[0]
    return None

