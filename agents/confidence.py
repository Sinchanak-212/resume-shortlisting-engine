import logging
from models.schemas import ParsedResume
from config import logger

class ConfidenceAgent:
    def __init__(self):
        pass

    def evaluate_quality_and_confidence(self, parsed_resume: ParsedResume, parse_status: str, parse_reason: str) -> tuple[str, str]:
        """Assess parse quality and assign a practical confidence level."""
        logger.info("Evaluating parse quality and confidence...")

        if parse_status == "Failed" or (not parsed_resume.name and not parsed_resume.email):
            logger.warning("Crucial identifier fields (name, email) are missing. Flagging as Failed parse.")
            return "Failed", "Low"

        is_ocr = "OCR" in parse_reason or "pdfplumber" in parse_reason
        missing_fields = []
        if not parsed_resume.email:
            missing_fields.append("email")
        if not parsed_resume.skills:
            missing_fields.append("skills")
        if not parsed_resume.projects and not parsed_resume.experience and not parsed_resume.internships:
            missing_fields.append("practical_experience")

        if is_ocr or len(missing_fields) >= 2:
            quality = "Partial"
        else:
            quality = "Clean"

        if quality == "Partial":
            confidence = "Low" if ("skills" in missing_fields or is_ocr) else "Medium"
        else:
            confidence = "High" if len(missing_fields) == 0 else "Medium"

        logger.info("Evaluated Quality: %s, Confidence: %s. Reason: %s", quality, confidence, parse_reason)
        return quality, confidence
