import re
from typing import List


def clean_text(text: str) -> str:
    """Clean up whitespace, stray control characters, and repeated blank lines."""
    if not text:
        return ""
    text = text.replace("\x0c", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def normalize_resume_text(text: str) -> str:
    """Normalize raw PDF/OCR text into a more extraction-friendly layout."""
    if not text:
        return ""

    text = clean_text(text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-•*]+\s*", "", line)
        line = re.sub(r"\s{2,}", " ", line)
        lines.append(line)

    normalized_lines = []
    for line in lines:
        lower = line.lower()
        if re.match(r"^(skills|experience|education|projects|certifications|awards|publications|research|volunteer|leadership|hackathons|languages|summary|objective|contact|technical skills|soft skills)\b", lower):
            normalized_lines.append(line)
            continue
        if re.match(r"^[A-Z][A-Za-z .,'()&/-]+$", line) and len(line.split()) <= 8 and not re.search(r"\d", line):
            normalized_lines.append(line)
            continue
        normalized_lines.append(line)

    return "\n".join(normalized_lines).strip()


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from raw text using regex."""
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    return re.findall(pattern, text)


def extract_phones(text: str) -> List[str]:
    """Extract phone numbers from raw text using regex."""
    pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{1,4}[-.\s]?\d{10}"
    matches = re.findall(pattern, text)
    return list(dict.fromkeys(m.strip() for m in matches))
