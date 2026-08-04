import re
import logging
from typing import Tuple
from models.schemas import ParsedResume
from config import logger

class GradeNormalizerAgent:
    def __init__(self):
        pass

    def normalize(self, parsed_resume: ParsedResume, raw_text: str) -> Tuple[float, str]:
        """
        Normalizes percentage, 4-point GPA, or 10-point CGPA to a unified 10-point scale.
        Returns:
            Tuple[float, str]: (normalized_cgpa, normalization_reason)
        """
        cgpa = parsed_resume.cgpa
        percentage = parsed_resume.percentage
        gpa = parsed_resume.gpa

        logger.info(f"Normalizing grades: CGPA={cgpa}, Percentage={percentage}, GPA={gpa}")

        # Case 1: Explicit 10-point CGPA is present
        if cgpa is not None:
            # Check if it looks like a 10-point scale (usually between 0.0 and 10.0)
            if 0.0 <= cgpa <= 10.0:
                return round(cgpa, 2), f"Direct CGPA match: {cgpa}/10"
            elif cgpa > 10.0:
                # Might actually be a percentage entered in the cgpa field
                normalized = cgpa / 9.5
                return min(round(normalized, 2), 10.0), f"CGPA value {cgpa} > 10 treated as percentage (divided by 9.5)"

        # Case 2: Percentage is present
        if percentage is not None:
            # Percentage can be written as 79 or 0.79
            pct_val = percentage
            if 0.0 <= pct_val <= 1.0:
                pct_val = pct_val * 100.0
            
            if 30.0 <= pct_val <= 100.0:
                normalized = pct_val / 9.5
                return min(round(normalized, 2), 10.0), f"Percentage {pct_val}% normalized by dividing by 9.5"

        # Case 3: 4-point GPA is present
        if gpa is not None:
            if 0.0 <= gpa <= 4.0:
                normalized = gpa * 2.5
                return min(round(normalized, 2), 10.0), f"GPA {gpa}/4.0 normalized by multiplying by 2.5"
            elif 4.0 < gpa <= 5.0:
                # 5-point GPA scale (Singapore, etc.) -> multiply by 2.0
                normalized = gpa * 2.0
                return min(round(normalized, 2), 10.0), f"GPA {gpa}/5.0 normalized by multiplying by 2.0"

        # Case 4: Ambiguous context - try to search the raw text around "CGPA", "GPA", "percentage", "aggregate", "marks", "scores"
        # We search for numbers like X.XX or XX.XX% or X.X/10 or X.X/4
        logger.info("Grade fields empty or ambiguous. Inspecting raw text...")
        
        # Regex to find grade patterns
        # 1. 10-point scale match: e.g. "8.5/10", "cgpa: 9.2", "gpa of 8.9"
        cgpa_matches = re.findall(r'(?:cgpa|gpa|pointer|sgpa)\s*(?:of|is|:)?\s*([0-9]\.[0-9]{1,2})(?:\s*/\s*10)?', raw_text, re.IGNORECASE)
        if cgpa_matches:
            val = float(cgpa_matches[0])
            if 4.0 < val <= 10.0:
                return round(val, 2), f"Inferred CGPA {val}/10 from text patterns"

        # 2. Percentage match: e.g. "85%", "78.4 %", "percentage: 90"
        pct_matches = re.findall(r'(\d{2}(?:\.\d{1,2})?)\s*%', raw_text)
        if not pct_matches:
            pct_matches = re.findall(r'(?:percentage|aggregate|marks|score)\s*(?:of|is|:)?\s*(\d{2}(?:\.\d{1,2})?)', raw_text, re.IGNORECASE)
        if pct_matches:
            val = float(pct_matches[0])
            if 35.0 <= val <= 100.0:
                normalized = val / 9.5
                return min(round(normalized, 2), 10.0), f"Inferred Percentage {val}% from text patterns (divided by 9.5)"

        # 3. 4-point scale match: e.g. "3.5/4", "gpa: 3.2"
        gpa_4_matches = re.findall(r'([0-3]\.[0-9]{1,2})\s*/\s*4', raw_text)
        if gpa_4_matches:
            val = float(gpa_4_matches[0])
            normalized = val * 2.5
            return min(round(normalized, 2), 10.0), f"Inferred GPA {val}/4 from text patterns (multiplied by 2.5)"

        # Default fallback: return a passing grade but flag as low confidence
        logger.warning("No academic grades found. Using default fallback of 6.5.")
        return 6.5, "No grade info found; using fallback of 6.5 CGPA with low confidence"
