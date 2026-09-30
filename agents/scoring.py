import re
import logging
from typing import List, Dict, Any, Tuple
from models.schemas import ParsedResume, ParsedJD, SkillMatchDetail
from config import logger, SCORE_WEIGHTS, ENABLE_COLLEGE_BONUS

# Word-boundary patterns. Plain substring checks (e.g. "it" in "architecture",
# "cs" in "physics", "nit" in "institute") produced false CS / tier-1 matches.
_CS_PATTERNS = [
    r"\bcomputer", r"\bcs\b", r"\bcse\b", r"\bit\b", r"information technology",
    r"\bsoftware", r"data science", r"\bai\b", r"artificial intelligence",
]
_TIER1_PATTERNS = [r"\biit\b", r"\bnit\b", r"\bbits\b", r"\biiit\b", r"\brvce\b", r"\bbms\b", r"\bpes\b"]


def _matches_any(text: str, patterns) -> bool:
    return any(re.search(p, text) for p in patterns)


class ScoringAgent:
    def __init__(self):
        pass

    def calculate_score(
        self, 
        parsed_resume: ParsedResume, 
        normalized_cgpa: float, 
        skill_matches: List[SkillMatchDetail], 
        parsed_jd: ParsedJD
    ) -> Tuple[float, str]:
        """
        Computes a weighted scoring out of 100 for the candidate.
        Returns:
            Tuple[float, str]: (final_score, score_breakdown_formula_string)
        """
        logger.info(f"Calculating score for candidate...")
        
        # 1. Required Skills Score (Max 45)
        req_matches = [m for m in skill_matches if m.skill in parsed_jd.required_skills]
        req_score = 0.0
        num_req = len(parsed_jd.required_skills)
        
        # Determine weight adjustments if preferred skills are empty
        num_pref = len(parsed_jd.preferred_skills)
        max_req_weight = SCORE_WEIGHTS["required_skills"]
        max_pref_weight = SCORE_WEIGHTS["preferred_skills"]
        
        if num_pref == 0:
            max_req_weight += max_pref_weight
            max_pref_weight = 0
            
        if num_req > 0:
            # Sum of scores for required skills multiplied by (max weight / count)
            req_score = sum([m.score for m in req_matches]) * (max_req_weight / num_req)
        else:
            # If JD has no required skills, award full marks for this section
            req_score = float(max_req_weight)

        # 2. Preferred Skills Score (Max 15 or 0)
        pref_score = 0.0
        if num_pref > 0:
            pref_matches = [m for m in skill_matches if m.skill in parsed_jd.preferred_skills]
            pref_score = sum([m.score for m in pref_matches]) * (max_pref_weight / num_pref)
        else:
            pref_score = float(max_pref_weight)

        # 3. Projects Score (Max 10)
        proj_count = len(parsed_resume.projects)
        proj_score = 0.0
        if proj_count == 1:
            proj_score = 6.0
        elif proj_count >= 2:
            proj_score = 10.0

        # 4. Experience/Internships Score (Max 10)
        exp_count = len(parsed_resume.experience) + len(parsed_resume.internships)
        exp_score = 0.0
        if exp_count == 1:
            exp_score = 6.0
        elif exp_count >= 2:
            exp_score = 10.0

        # 5. CGPA Score (Max 10)
        cgpa_score = 0.0
        min_cgpa = parsed_jd.min_cgpa
        if normalized_cgpa >= min_cgpa:
            cgpa_score = 10.0
        else:
            # Scaled down ratio if below minimum
            if min_cgpa > 0:
                cgpa_score = max(0.0, (normalized_cgpa / min_cgpa) * 10.0)
            else:
                cgpa_score = 10.0

        # 6. Certifications Score (Max 5)
        cert_count = len(parsed_resume.certifications)
        cert_score = 0.0
        if cert_count == 1:
            cert_score = 3.0
        elif cert_count >= 2:
            cert_score = 5.0

        # 7. Education Score (Max 5)
        # Check if degree/branch is in CS/IT fields
        edu_score = 3.0
        branch = (parsed_resume.branch or "").lower()
        degree = (parsed_resume.degree or "").lower()
        college = (parsed_resume.college or "").lower()
        
        is_cs = _matches_any(branch, _CS_PATTERNS) or _matches_any(degree, _CS_PATTERNS)
        is_tier1 = ENABLE_COLLEGE_BONUS and _matches_any(college, _TIER1_PATTERNS)

        if is_cs or is_tier1:
            edu_score = 5.0

        # Compute Final Score
        final_score = req_score + pref_score + proj_score + exp_score + cgpa_score + cert_score + edu_score
        final_score = min(round(final_score, 2), 100.0)

        # Document Formula
        breakdown = (
            f"Required Skills Match: {req_score:.2f}/{max_req_weight} | "
            f"Preferred Skills Match: {pref_score:.2f}/{max_pref_weight} | "
            f"Projects: {proj_score:.2f}/10 | "
            f"Experience/Internships: {exp_score:.2f}/10 | "
            f"CGPA Alignment: {cgpa_score:.2f}/10 | "
            f"Certifications: {cert_score:.2f}/5 | "
            f"Education Context: {edu_score:.2f}/5"
        )
        
        logger.info(f"Final Score: {final_score}/100. Breakdown: {breakdown}")
        return final_score, breakdown
