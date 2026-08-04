import re
from typing import List

def clean_text(text: str) -> str:
    """Basic clean up of white spaces, tabs, and duplicate lines."""
    if not text:
        return ""
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace multiple newlines with at most double newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def extract_emails(text: str) -> List[str]:
    """Extracts email addresses from raw text using regex."""
    pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    return re.findall(pattern, text)

def extract_phones(text: str) -> List[str]:
    """Extracts phone numbers from raw text using regex."""
    # Match various Indian and US formats (e.g. +91 9999999999, (123) 456-7890, 123-456-7890, etc.)
    pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{1,4}[-.\s]?\d{10}'
    matches = re.findall(pattern, text)
    # Strip spaces/dashes and keep unique ones
    return list(set([m.strip() for m in matches]))
