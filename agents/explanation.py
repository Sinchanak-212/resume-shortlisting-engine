import logging
from typing import List
from utils.llm import LLMClient
from models.schemas import ParsedResume, ParsedJD, SkillMatchDetail
from config import logger

class ExplanationAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_explanation(
        self, 
        parsed_resume: ParsedResume, 
        parsed_jd: ParsedJD, 
        skill_matches: List[SkillMatchDetail],
        score: float,
        normalized_cgpa: float,
        confidence: str
    ) -> List[str]:
        """
        Generates exactly three bullet points explaining the ranking justification, strengths, weaknesses, and missing skills.
        """
        logger.info(f"Generating candidate matches explanation using LLM...")
        
        # Prepare summaries for the LLM
        matched_skills = [m.skill for m in skill_matches if m.match_type != "none"]
        missing_skills = [m.skill for m in skill_matches if m.match_type == "none"]
        
        req_missing = [m.skill for m in skill_matches if m.match_type == "none" and m.skill in parsed_jd.required_skills]
        pref_missing = [m.skill for m in skill_matches if m.match_type == "none" and m.skill in parsed_jd.preferred_skills]

        projects_summary = ", ".join([p.title for p in parsed_resume.projects]) if parsed_resume.projects else "None"
        exp_summary = f"{len(parsed_resume.experience)} experience entries, {len(parsed_resume.internships)} internships"

        system_instruction = (
            "You are a Staff Recruiter and AI Talent Analyst. "
            "Your task is to review a candidate's evaluation against a Job Description and output EXACTLY THREE bullet points. "
            "The bullet points must represent: "
            "1. Core Strengths (why the candidate ranked high/medium, education, projects, or experience) "
            "2. Missing Skills & Alignment (explicitly mention required or preferred skills from the JD that are missing) "
            "3. Overall Fit / Weaknesses (CGPA requirements check, confidence rating, or other observations). "
            "Ensure the output is exactly a list of 3 strings. Do not include markdown headers, just the three bullet points."
        )

        prompt = (
            f"Candidate: {parsed_resume.name or 'Unknown'}\n"
            f"Target Role: {parsed_jd.role_name}\n"
            f"Overall Match Score: {score}/100\n"
            f"Parse Confidence: {confidence}\n"
            f"Candidate CGPA: {normalized_cgpa:.2f} (Required Min: {parsed_jd.min_cgpa:.2f})\n"
            f"Skills Matched: {', '.join(matched_skills) if matched_skills else 'None'}\n"
            f"Required Skills Missing: {', '.join(req_missing) if req_missing else 'None'}\n"
            f"Preferred Skills Missing: {', '.join(pref_missing) if pref_missing else 'None'}\n"
            f"Projects Listed: {projects_summary}\n"
            f"Work Experience Summary: {exp_summary}\n\n"
            f"Please generate exactly three clear, actionable bullet points explaining these evaluation results."
        )

        try:
            raw_output = self.llm_client.query(prompt, system_instruction=system_instruction)
            # Parse output into lines
            lines = [line.strip().lstrip("*-• ").strip() for line in raw_output.splitlines() if line.strip()]
            
            # Clean up and ensure exactly three lines
            bullets = [l for l in lines if l]
            if len(bullets) < 3:
                # Pad if LLM under-generated
                while len(bullets) < 3:
                    bullets.append("Evaluation data incomplete or default metric met.")
            elif len(bullets) > 3:
                # Trim if LLM over-generated
                bullets = bullets[:3]
                
            logger.info("Successfully generated explanation bullets.")
            return bullets
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Fallback bullets
            return [
                f"Candidate achieved matching score of {score}/100 based on exact and semantic skill alignment.",
                f"Missing required skills: {', '.join(req_missing[:3]) if req_missing else 'None'}.",
                f"CGPA is {normalized_cgpa:.2f} compared to the requirement of {parsed_jd.min_cgpa:.2f}."
            ]
class ExplanationAgentMock:
    """Mock agent that generates explanation bullets deterministically without calling LLM."""
    def generate_explanation(self, parsed_resume, parsed_jd, skill_matches, score, normalized_cgpa, confidence) -> List[str]:
        req_missing = [m.skill for m in skill_matches if m.match_type == "none" and m.skill in parsed_jd.required_skills]
        matched_req = [m.skill for m in skill_matches if m.match_type != "none" and m.skill in parsed_jd.required_skills]
        
        bullet1 = f"Strong fit with {len(matched_req)} required skills matched, including {', '.join(matched_req[:2]) if matched_req else 'general skills'}."
        bullet2 = f"Missing required skills: {', '.join(req_missing) if req_missing else 'None'}."
        bullet3 = f"Education profile: CGPA of {normalized_cgpa:.2f} relative to the target min of {parsed_jd.min_cgpa:.2f}."
        return [bullet1, bullet2, bullet3]
