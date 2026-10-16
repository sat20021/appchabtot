# utils/synthetic_data_utils.py
# Utility functions for generating synthetic training data
import random
from faker import Faker

fake = Faker()

POLITENESS_PREFIXES = ["please ", "could you ", "hey, ", "hi, ", "excuse me, ", ""]
PUNCTUATION = ["", ".", "?", "!!", "!", " :)", " thanks"]

def random_email():
    """Generate a random email address"""
    return fake.email()

def random_phone():
    """Generate a random phone number"""
    return fake.msisdn()[0:10]

def random_name():
    """Generate a random name"""
    return fake.name()

def add_typos(text, typo_rate=0.05):
    """Introduce realistic typos"""
    if random.random() > typo_rate:
        return text
    
    words = text.split()
    if not words:
        return text
    
    typo_patterns = [
        lambda w: w[:-1] if len(w) > 3 else w,  # Missing last char
        lambda w: w + w[-1] if len(w) > 2 else w,  # Double last char
        lambda w: w.replace('e', '3') if 'e' in w else w,  # Leetspeak
        lambda w: w.replace('a', '@') if 'a' in w else w,
    ]
    
    idx = random.randint(0, len(words) - 1)
    words[idx] = random.choice(typo_patterns)(words[idx])
    return ' '.join(words)

