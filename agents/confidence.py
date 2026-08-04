import logging
from models.schemas import ParsedResume
from config import logger

class ConfidenceAgent:
    def __init__(self):
        pass

    def evaluate_quality_and_confidence(self, parsed_resume: ParsedResume, parse_status: str, parse_reason: str) -> tuple[str, str]:
        """
        Assesses the quality of the extraction and assigns an overall confidence rating.
        Returns:
            Tuple[str, str]: (parse_quality, confidence_level)
            parse_quality: 'Clean', 'Partial', 'Failed'
            confidence_level: 'High', 'Medium', 'Low'
        """
        logger.info("Evaluating parse quality and confidence...")

        # 1. Check if the parser failed, or if BOTH crucial identifier fields are missing.
        # BUGFIX: previously this triggered "Failed" if name alone was missing (even with a
        # valid email present), and both branches of the inner if/else returned the same
        # result anyway - so resumes with a perfectly good email but a name the LLM didn't
        # confidently label were being discarded and scored 0. Only fail when we truly have
        # no way to identify the candidate.
        if parse_status == "Failed" or (not parsed_resume.name and not parsed_resume.email):
            logger.warning("Crucial identifier fields (name, email) are missing. Flagging as Failed parse.")
            return "Failed", "Low"

        # 2. Check if parser indicated OCR was used
        is_ocr = "OCR" in parse_reason or "pdfplumber" in parse_reason
        
        # 3. Check for missing vital sections
        missing_fields = []
        if not parsed_resume.email:
            missing_fields.append("email")
        if not parsed_resume.skills:
            missing_fields.append("skills")
        if not parsed_resume.projects and not parsed_resume.experience and not parsed_resume.internships:
            missing_fields.append("practical_experience")

        # 4. Determine Parse Quality
        if is_ocr:
            quality = "Partial"
        elif len(missing_fields) >= 2:
            quality = "Partial"
        else:
            quality = "Clean"

        # 5. Determine Confidence Level
        if quality == "Failed":
            confidence = "Low"
        elif quality == "Partial":
            # Partial parse must produce Medium or Low confidence at most (Tricky Part 4)
            if "skills" in missing_fields or is_ocr:
                confidence = "Low"
            else:
                confidence = "Medium"
        else:
            # Clean parse: Can be High or Medium
            if len(missing_fields) == 1:
                confidence = "Medium"
            else:
                confidence = "High"

        logger.info(f"Evaluated Quality: {quality}, Confidence: {confidence}. Reason: {parse_reason}")
        return quality, confidence
