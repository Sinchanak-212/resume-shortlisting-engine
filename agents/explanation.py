import logging
from typing import List
from models.schemas import ParsedResume, ParsedJD, SkillMatchDetail
from config import logger

class ExplanationAgent:
    def __init__(self):
        pass

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
        Generates deterministic explanation bullets using resume and JD matching details.
        This avoids slow external LLM calls for every candidate.
        """
        logger.info("Generating candidate matches explanation locally.")

        matched_required = [m.skill for m in skill_matches if m.match_type != "none" and m.skill in parsed_jd.required_skills]
        matched_preferred = [m.skill for m in skill_matches if m.match_type != "none" and m.skill in parsed_jd.preferred_skills]
        missing_required = [m.skill for m in skill_matches if m.match_type == "none" and m.skill in parsed_jd.required_skills]
        missing_preferred = [m.skill for m in skill_matches if m.match_type == "none" and m.skill in parsed_jd.preferred_skills]

        contact_info = parsed_resume.email or parsed_resume.phone or "contact details missing"
        education_ok = normalized_cgpa >= parsed_jd.min_cgpa if normalized_cgpa is not None else False

        strength_parts = []
        if matched_required:
            strength_parts.append(f"Matched {len(matched_required)} required skills: {', '.join(matched_required[:3])}")
        if matched_preferred:
            strength_parts.append(f"Also aligned with preferred skills: {', '.join(matched_preferred[:3])}")
        if parsed_resume.projects:
            strength_parts.append(f"Has {len(parsed_resume.projects)} project entries")
        if parsed_resume.experience or parsed_resume.internships:
            strength_parts.append(f"Includes work/internship experience")
        if not strength_parts:
            strength_parts.append("Limited explicit skill or experience information found")

        bullet1 = ". ".join(strength_parts) + "."
        bullet2 = (
            f"Missing required skills: {', '.join(missing_required) if missing_required else 'None'}; "
            f"preferred skills missing: {', '.join(missing_preferred) if missing_preferred else 'None'}."
        )
        bullet3 = (
            f"Score {score:.0f}/100 with CGPA {'meets' if education_ok else 'below'} the requirement ({normalized_cgpa:.2f} vs {parsed_jd.min_cgpa:.2f}) and confidence set to {confidence}."
        )

        return [bullet1, bullet2, bullet3]
