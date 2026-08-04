import logging
from utils.llm import LLMClient
from models.schemas import ParsedResume
from config import logger

class ResumeExtractorAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract_resume_info(self, raw_text: str) -> ParsedResume:
        """
        Calls the LLM to extract structured fields from raw resume text.
        """
        logger.info("Extracting structured info from resume text...")
        system_instruction = (
            "You are an expert AI Resume Parser. Your job is to read unstructured resume text "
            "and extract information into a structured JSON format. "
            "Extract as much detail as possible. Do not guess; if a field is not present, leave it as null or empty."
        )
        
        prompt = (
            f"Here is the raw text extracted from a candidate's resume. "
            f"Please parse it and populate the JSON fields.\n\n"
            f"--- RAW RESUME TEXT ---\n"
            f"{raw_text}\n"
            f"--- END OF RAW RESUME TEXT ---\n"
        )
        
        try:
            parsed_resume = self.llm_client.query_json(
                prompt=prompt,
                schema=ParsedResume,
                system_instruction=system_instruction
            )
            logger.info(f"Successfully extracted resume details for: {parsed_resume.name or 'Unknown'}")
            return parsed_resume
        except Exception as e:
            logger.error(f"Error in ResumeExtractorAgent: {e}")
            # Return empty schema if LLM fails
            return ParsedResume()
