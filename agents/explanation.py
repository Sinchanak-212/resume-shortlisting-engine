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
        confidence: str,
    ) -> List[str]:
        """Create deterministic recruiter-facing explanation bullets."""
        logger.info("Generating candidate matches explanation locally.")

        matched_required = [m.skill for m in skill_matches if m.match_type != "none" and m.skill in parsed_jd.required_skills]
        matched_preferred = [m.skill for m in skill_matches if m.match_type != "none" and m.skill in parsed_jd.preferred_skills]
        missing_required = [m.skill for m in skill_matches if m.match_type == "none" and m.skill in parsed_jd.required_skills]
        missing_preferred = [m.skill for m in skill_matches if m.match_type == "none" and m.skill in parsed_jd.preferred_skills]
        education_ok = normalized_cgpa >= parsed_jd.min_cgpa if normalized_cgpa is not None else False

        strengths = []
        if matched_required:
            strengths.append(f"Matched {len(matched_required)} required skills: {', '.join(matched_required[:3])}")
        if matched_preferred:
            strengths.append(f"Aligned with preferred skills: {', '.join(matched_preferred[:3])}")
        if parsed_resume.projects:
            strengths.append(f"Has {len(parsed_resume.projects)} project entry/entries")
        if parsed_resume.experience or parsed_resume.internships:
            strengths.append("Includes work or internship experience")
        if parsed_resume.certifications:
            strengths.append("Shows professional certifications")
        if not strengths:
            strengths.append("Limited explicit experience or skill evidence was found")

        weaknesses = []
        if missing_required:
            weaknesses.append(f"Missing required skills: {', '.join(missing_required[:3])}")
        if missing_preferred:
            weaknesses.append(f"Missing preferred skills: {', '.join(missing_preferred[:3])}")
        if not education_ok:
            weaknesses.append("CGPA is below the stated requirement")
        if not weaknesses:
            weaknesses.append("No major gaps identified")

        improvement_suggestions = []
        if missing_required:
            improvement_suggestions.append("Add evidence for the missing required skills in future resumes")
        if parsed_resume.github or parsed_resume.linkedin or parsed_resume.portfolio:
            improvement_suggestions.append("Keep professional profiles up to date and link them clearly")
        else:
            improvement_suggestions.append("Add GitHub, LinkedIn, or portfolio links to strengthen profile credibility")
        if not parsed_resume.projects:
            improvement_suggestions.append("Include measurable project outcomes to strengthen the profile")

        recruiter_summary = (
            f"Overall fit is {score:.0f}/100 with {confidence.lower()} confidence. "
            f"The candidate shows a solid match in {'; '.join(strengths[:2]) if strengths else 'core competencies'}."
        )
        interview_recommendation = "Proceed to interview" if score >= 70 else ("Hold for secondary review" if score >= 50 else "Decline or revisit")

        return [
            ". ".join(strengths) + ".",
            ". ".join(weaknesses) + ".",
            f"Score {score:.0f}/100 with CGPA {'meets' if education_ok else 'below'} the requirement ({normalized_cgpa:.2f} vs {parsed_jd.min_cgpa:.2f}); recruiter summary: {recruiter_summary}; interview recommendation: {interview_recommendation}.",
        ]
