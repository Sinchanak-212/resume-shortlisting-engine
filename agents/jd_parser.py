import logging
from utils.llm import LLMClient
from models.schemas import ParsedJD
from config import logger

class JDParserAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def parse_jd(self, raw_jd_text: str) -> ParsedJD:
        """
        Parses a raw job description string and extracts structured fields using LLM.
        """
        logger.info("Parsing raw Job Description text...")
        system_instruction = (
            "You are an expert HR agent specialized in parsing Job Descriptions. "
            "Your job is to take raw, unstructured Job Description text (like a LinkedIn post or a document) "
            "and extract structured requirements: role name, required skills (must-have), preferred skills (nice-to-have), "
            "minimum CGPA requirement, and the number of available slots. "
            "Convert any percentage or GPA requirements to a standard 10-point scale CGPA (e.g., 60% -> 6.3, 3.0/4 -> 7.5)."
        )
        
        prompt = (
            f"Please parse the following Job Description text and extract details into JSON.\n\n"
            f"--- JOB DESCRIPTION TEXT ---\n"
            f"{raw_jd_text}\n"
            f"--- END OF JOB DESCRIPTION TEXT ---\n"
        )
        
        try:
            parsed_jd = self.llm_client.query_json(
                prompt=prompt,
                schema=ParsedJD,
                system_instruction=system_instruction
            )
            logger.info(f"Successfully parsed Job Description for role: {parsed_jd.role_name}")
            return parsed_jd
        except Exception as e:
            logger.error(f"Error in JDParserAgent: {e}")
            # Mock or return empty JD if LLM fails
            return ParsedJD(
                role_name="Software Developer",
                required_skills=[],
                preferred_skills=[],
                min_cgpa=6.0,
                slots=5
            )
