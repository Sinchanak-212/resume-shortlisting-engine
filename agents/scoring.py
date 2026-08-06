import logging
import re
from typing import List, Dict, Any, Tuple
from models.schemas import ParsedResume, ParsedJD, SkillMatchDetail
from config import logger, SCORE_WEIGHTS

class ScoringAgent:
    def __init__(self):
        pass

    def calculate_score(
        self,
        parsed_resume: ParsedResume,
        normalized_cgpa: float,
        skill_matches: List[SkillMatchDetail],
        parsed_jd: ParsedJD,
    ) -> Tuple[float, str]:
        """Compute a weighted enterprise-style score with an explainable breakdown."""
        logger.info("Calculating candidate score...")

        req_matches = [m for m in skill_matches if m.skill in parsed_jd.required_skills]
        pref_matches = [m for m in skill_matches if m.skill in parsed_jd.preferred_skills]
        num_req = len(parsed_jd.required_skills)
        num_pref = len(parsed_jd.preferred_skills)

        max_req_weight = SCORE_WEIGHTS["required_skills"]
        max_pref_weight = SCORE_WEIGHTS["preferred_skills"]
        if num_pref == 0:
            max_req_weight += max_pref_weight
            max_pref_weight = 0

        required_component = 0.0
        if num_req > 0:
            required_component = sum(m.score for m in req_matches) * (max_req_weight / max(num_req, 1))
        else:
            required_component = float(max_req_weight)

        preferred_component = 0.0
        if num_pref > 0:
            preferred_component = sum(m.score for m in pref_matches) * (max_pref_weight / max(num_pref, 1))
        else:
            preferred_component = float(max_pref_weight)

        proj_count = len(parsed_resume.projects)
        proj_component = 10.0 if proj_count >= 2 else (6.0 if proj_count == 1 else 0.0)

        exp_count = len(parsed_resume.experience) + len(parsed_resume.internships)
        exp_component = 10.0 if exp_count >= 2 else (6.0 if exp_count == 1 else 0.0)

        cgpa_score = 10.0 if normalized_cgpa and normalized_cgpa >= parsed_jd.min_cgpa else (max(0.0, (normalized_cgpa / parsed_jd.min_cgpa) * 10.0) if parsed_jd.min_cgpa and normalized_cgpa else 0.0)

        cert_component = 5.0 if len(parsed_resume.certifications) >= 2 else (3.0 if len(parsed_resume.certifications) == 1 else 0.0)

        branch = (parsed_resume.branch or "").lower()
        degree = (parsed_resume.degree or "").lower()
        college = (parsed_resume.college or "").lower()
        cs_keywords = ["computer", "cs", "it", "information technology", "software", "data science", "ai", "artificial intelligence"]
        tier1_keywords = ["iit", "nit", "bits", "iiit", "rvce", "bms", "pes"]
        is_cs = any(k in branch or k in degree for k in cs_keywords)
        is_tier1 = any(k in college for k in tier1_keywords)
        edu_component = 5.0 if is_cs or is_tier1 else 3.0

        achievements_component = 2.5 if parsed_resume.achievements else 0.0
        open_source_component = 2.5 if parsed_resume.open_source_contributions else 0.0
        leadership_component = 2.5 if parsed_resume.leadership else 0.0
        hackathon_component = 2.5 if parsed_resume.hackathons else 0.0
        resume_quality_component = 5.0 if parsed_resume.skills and parsed_resume.projects else 2.5
        ats_component = 5.0 if parsed_resume.github or parsed_resume.linkedin or parsed_resume.portfolio else 2.5

        final_score = required_component + preferred_component + proj_component + exp_component + cgpa_score + cert_component + edu_component + achievements_component + open_source_component + leadership_component + hackathon_component + resume_quality_component + ats_component
        final_score = round(min(final_score, 100.0), 2)

        breakdown = [
            f"Required skills: {required_component:.2f}/{max_req_weight}",
            f"Preferred skills: {preferred_component:.2f}/{max_pref_weight}",
            f"Projects: {proj_component:.2f}/10",
            f"Experience: {exp_component:.2f}/10",
            f"CGPA: {cgpa_score:.2f}/10",
            f"Certifications: {cert_component:.2f}/5",
            f"Education: {edu_component:.2f}/5",
            f"Achievements: {achievements_component:.2f}/2.5",
            f"Open source: {open_source_component:.2f}/2.5",
            f"Leadership: {leadership_component:.2f}/2.5",
            f"Hackathons: {hackathon_component:.2f}/2.5",
            f"Resume quality: {resume_quality_component:.2f}/5",
            f"Profiles: {ats_component:.2f}/5",
        ]
        logger.info("Final score: %.2f/100. Breakdown: %s", final_score, breakdown)
        return final_score, " | ".join(breakdown)

    def analyze_ats_compliance(self, parsed_resume: ParsedResume, raw_text: str) -> Tuple[float, List[str]]:
        """Provide a lightweight ATS compliance score and issues list."""
        score = 100.0
        issues = []
        if not raw_text:
            return 0.0, ["No text available for ATS analysis"]

        if re.search(r"\|", raw_text):
            issues.append("Uses vertical separators that can hurt ATS parsing")
            score -= 5
        if re.search(r"\b(table|tables)\b", raw_text.lower()):
            issues.append("Contains table-like structure that may reduce ATS readability")
            score -= 5
        if len(raw_text.split()) > 1200:
            issues.append("Resume appears overly long for ATS readability")
            score -= 5
        if not parsed_resume.name or not parsed_resume.email:
            issues.append("Missing core contact details")
            score -= 10
        if not parsed_resume.skills:
            issues.append("Skills section is missing or hard to detect")
            score -= 10
        if not parsed_resume.projects and not parsed_resume.experience:
            issues.append("Experience or projects section is missing")
            score -= 10
        if not parsed_resume.education_history:
            issues.append("Education section is missing")
            score -= 5

        return round(max(0.0, score), 2), issues

    def detect_duplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group near-duplicate candidates based on identifiers and similarity."""
        groups = []
        for candidate in candidates:
            key = candidate.get("email") or candidate.get("phone") or candidate.get("github") or candidate.get("linkedin")
            if not key:
                continue
            found = False
            for group in groups:
                if group.get("key") == key:
                    group["members"].append(candidate)
                    found = True
                    break
            if not found:
                groups.append({"key": key, "members": [candidate]})
        return groups
