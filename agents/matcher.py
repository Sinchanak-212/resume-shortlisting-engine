import logging
from typing import List, Dict, Any
from models.schemas import ParsedResume, ParsedJD, SkillMatchDetail
from config import logger
from agents.skill_extractor import SkillExtractionAgent
from rapidfuzz import fuzz

class MatchingAgent:
    def __init__(self, skill_extractor: SkillExtractionAgent):
        self.skill_extractor = skill_extractor

    def match_candidate_to_jd(self, candidate_skills: List[Dict[str, Any]], parsed_jd: ParsedJD) -> List[SkillMatchDetail]:
        """
        Compares candidate's enriched skills against JD required and preferred skills.
        Performs exact, synonym, partial, and implicit matching.
        """
        match_details = []
        
        # Combine required and preferred skills to evaluate
        jd_skills = [(skill, "required") for skill in parsed_jd.required_skills] + \
                    [(skill, "preferred") for skill in parsed_jd.preferred_skills]
                    
        for jd_skill, category in jd_skills:
            jd_skill_norm = self.skill_extractor.normalize_skill(jd_skill)
            best_match_type = "none"
            best_score = 0.0
            best_matched_term = None
            best_reason = f"No match found for '{jd_skill}'"

            for cand_skill in candidate_skills:
                raw_cand = cand_skill["skill"]
                norm_cand = cand_skill["normalized"]
                confidence = cand_skill["confidence"]
                if not raw_cand:
                    continue

                if jd_skill.lower() == raw_cand.lower():
                    best_match_type = "exact"
                    best_score = 1.0 * confidence
                    best_matched_term = raw_cand
                    best_reason = f"Exact match with '{raw_cand}'"
                    break

                if jd_skill_norm.lower() == norm_cand.lower():
                    best_match_type = "synonym"
                    best_score = 1.0 * confidence
                    best_matched_term = raw_cand
                    best_reason = f"Synonym match: both '{jd_skill}' and '{raw_cand}' normalize to '{jd_skill_norm}'"
                    break

                ratio = fuzz.ratio(jd_skill.lower(), raw_cand.lower())
                partial_ratio = fuzz.partial_ratio(jd_skill.lower(), raw_cand.lower())
                if ratio >= 65:
                    max_ratio = max(ratio, partial_ratio)
                    if max_ratio >= 80:
                        score = (max_ratio / 100.0) * confidence
                        if score > best_score:
                            best_match_type = "partial"
                            best_score = score
                            best_matched_term = raw_cand
                            best_reason = f"Partial/fuzzy match with '{raw_cand}' (similarity: {max_ratio}%)"

                sem_similarity = self.skill_extractor.get_semantic_similarity(jd_skill, raw_cand)
                if sem_similarity >= 0.70:
                    score = sem_similarity * confidence
                    if score > best_score:
                        best_match_type = "implicit"
                        best_score = score
                        best_matched_term = raw_cand
                        best_reason = f"Implicit/semantic match with '{raw_cand}' (cosine similarity: {sem_similarity:.2f})"

            match_details.append(SkillMatchDetail(
                skill=jd_skill,
                match_type=best_match_type if best_score > 0 else "none",
                score=round(best_score, 2),
                matched_term=best_matched_term,
                reason=best_reason,
            ))

        return match_details
