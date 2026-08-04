import re
import logging
import csv
import json
from typing import List
from models.schemas import MatchResult, ParsedJD
from config import logger, REPORTS_DIR


def _sanitize_slug(text: str) -> str:
    """Converts arbitrary role_name text into a filesystem-safe slug.
    BUGFIX: role_name can originate from an LLM parsing free-text JD input, so it is not
    guaranteed to be filename-safe (could contain '/', ':', quotes, etc.)."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "role"

class RankingAgent:
    def __init__(self):
        pass

    def rank_and_allocate_slots(self, candidates: List[MatchResult], parsed_jd: ParsedJD) -> List[MatchResult]:
        """
        Sorts candidates by score, handles slot allocation, and divides them into Shortlisted vs Reserve lists.
        Failed parses are pushed to the bottom and never receive slots.
        """
        logger.info(f"Ranking candidates for role: {parsed_jd.role_name} (Slots: {parsed_jd.slots})")

        # Separate failed parses from valid ones
        valid_candidates = [c for c in candidates if c.parse_quality != "Failed"]
        failed_candidates = [c for c in candidates if c.parse_quality == "Failed"]

        # Sort valid candidates by score descending
        valid_candidates.sort(key=lambda c: c.score, reverse=True)

        # Allocate slots
        # BUGFIX: parsed_jd.min_cgpa was defined in the schema but never actually enforced
        # here - a candidate could be ranked #1 by score and still get auto-shortlisted
        # despite their own CGPA falling below the JD's stated minimum (their explanation
        # bullet would even say so, while their status said "Shortlisted"). Now a candidate
        # must both have score > 0 AND meet the CGPA floor to consume a slot; anyone who
        # fails the floor drops to Reserve and the next qualifying candidate backfills the
        # slot, since this is a straightforward sequential scan over score-sorted candidates.
        shortlisted_count = 0
        for i, cand in enumerate(valid_candidates):
            meets_cgpa_bar = cand.normalized_cgpa >= parsed_jd.min_cgpa
            if shortlisted_count < parsed_jd.slots and cand.score > 0 and meets_cgpa_bar:
                cand.is_shortlisted = True
                cand.is_reserve = False
                shortlisted_count += 1
            else:
                cand.is_shortlisted = False
                cand.is_reserve = True

        # Failed parses have neither shortlist nor reserve status, they are just flagged as Failed
        for cand in failed_candidates:
            cand.is_shortlisted = False
            cand.is_reserve = False
            cand.score = 0.0 # Force zero score

        # Combine back
        ranked_list = valid_candidates + failed_candidates
        logger.info(f"Ranking complete: {shortlisted_count} shortlisted, {len(valid_candidates) - shortlisted_count} on reserve, {len(failed_candidates)} failed.")
        return ranked_list

    def export_reports(self, ranked_candidates: List[MatchResult], parsed_jd: ParsedJD, prefix: str = "") -> dict:
        """
        Generates CSV, JSON, Markdown Report, and Parse Quality Report.
        """
        logger.info("Exporting evaluation reports...")
        role_slug = _sanitize_slug(parsed_jd.role_name)
        if prefix:
            role_slug = f"{prefix}_{role_slug}"

        csv_path = REPORTS_DIR / f"{role_slug}_report.csv"
        json_path = REPORTS_DIR / f"{role_slug}_report.json"
        md_path = REPORTS_DIR / f"{role_slug}_report.md"
        quality_path = REPORTS_DIR / f"{role_slug}_parse_quality_report.csv"

        # 1. Export CSV Report
        try:
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Candidate Name", "Resume File", "Score", "Parse Quality", 
                    "Confidence", "CGPA", "Status", "Reasoning"
                ])
                for c in ranked_candidates:
                    status = "Shortlisted" if c.is_shortlisted else ("Reserve" if c.is_reserve else "Failed Parse")
                    reasoning_str = " | ".join(c.explanation)
                    writer.writerow([
                        c.candidate_name, c.resume_file, c.score, c.parse_quality,
                        c.confidence, f"{c.normalized_cgpa:.2f}", status, reasoning_str
                    ])
            logger.info(f"CSV Report exported to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")

        # 2. Export JSON Report
        try:
            candidates_dict = [c.model_dump() for c in ranked_candidates]
            with open(json_path, mode="w", encoding="utf-8") as f:
                json.dump({
                    "role_name": parsed_jd.role_name,
                    "slots": parsed_jd.slots,
                    "min_cgpa_required": parsed_jd.min_cgpa,
                    "candidates": candidates_dict
                }, f, indent=4)
            logger.info(f"JSON Report exported to {json_path}")
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")

        # 3. Export Markdown Report
        try:
            with open(md_path, mode="w", encoding="utf-8") as f:
                f.write(f"# Evaluation Report: {parsed_jd.role_name}\n\n")
                f.write(f"- **Available Slots**: {parsed_jd.slots}\n")
                f.write(f"- **Minimum CGPA Requirement**: {parsed_jd.min_cgpa:.2f}\n")
                f.write(f"- **Candidates Evaluated**: {len(ranked_candidates)}\n\n")
                
                f.write("## Shortlisted Candidates\n\n")
                shortlisted = [c for c in ranked_candidates if c.is_shortlisted]
                if shortlisted:
                    f.write("| Rank | Candidate Name | Score | CGPA | Parse Quality | Confidence |\n")
                    f.write("| --- | --- | --- | --- | --- | --- |\n")
                    for i, c in enumerate(shortlisted, 1):
                        f.write(f"| {i} | {c.candidate_name} | {c.score:.2f} | {c.normalized_cgpa:.2f} | {c.parse_quality} | {c.confidence} |\n")
                else:
                    f.write("*No candidates shortlisted.*\n")
                
                f.write("\n## Reserve List\n\n")
                reserve = [c for c in ranked_candidates if c.is_reserve]
                if reserve:
                    f.write("| Rank | Candidate Name | Score | CGPA | Parse Quality | Confidence |\n")
                    f.write("| --- | --- | --- | --- | --- | --- |\n")
                    for i, c in enumerate(reserve, 1):
                        f.write(f"| {i} | {c.candidate_name} | {c.score:.2f} | {c.normalized_cgpa:.2f} | {c.parse_quality} | {c.confidence} |\n")
                else:
                    f.write("*No candidates on the reserve list.*\n")

                f.write("\n## Failed Parse / Manual Review Needed\n\n")
                failed = [c for c in ranked_candidates if c.parse_quality == "Failed"]
                if failed:
                    f.write("| Candidate Name | Resume File | Error / Parse Quality | Reasoning |\n")
                    f.write("| --- | --- | --- | --- |\n")
                    for c in failed:
                        reasoning_str = " | ".join(c.explanation) if c.explanation else "Failed to parse text or compile credentials"
                        f.write(f"| {c.candidate_name} | {c.resume_file} | {c.parse_quality} | {reasoning_str} |\n")
                else:
                    f.write("*No failed parses.*\n")

                f.write("\n## Candidate Details & Explanations\n\n")
                for c in ranked_candidates:
                    status = "Shortlisted" if c.is_shortlisted else ("Reserve" if c.is_reserve else "Failed Parse")
                    f.write(f"### {c.candidate_name} ({status})\n")
                    f.write(f"- **Resume File**: {c.resume_file}\n")
                    f.write(f"- **Matching Score**: {c.score:.2f}/100\n")
                    f.write(f"- **Normalized CGPA**: {c.normalized_cgpa:.2f}\n")
                    f.write(f"- **Parse Quality**: {c.parse_quality} | **Confidence**: {c.confidence}\n")
                    f.write("- **Analysis Explanation**:\n")
                    for bullet in c.explanation:
                        f.write(f"  - {bullet}\n")
                    f.write("\n---\n\n")

            logger.info(f"Markdown Report exported to {md_path}")
        except Exception as e:
            logger.error(f"Failed to export Markdown: {e}")

        # 4. Export Parse Quality Report (Separate CSV containing parse tracking)
        try:
            with open(quality_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Resume File", "Candidate Name", "Parse Quality", "Confidence", "Reason"])
                for c in ranked_candidates:
                    # Simple heuristic for parse reason
                    reason = "Successfully extracted structured entities"
                    if c.parse_quality == "Failed":
                        reason = "Failed to extract text or missing vital contact credentials (name/email)"
                    elif c.confidence == "Low":
                        reason = "Extracted via OCR fallback or high number of missing fields"
                    writer.writerow([c.resume_file, c.candidate_name, c.parse_quality, c.confidence, reason])
            logger.info(f"Parse Quality Report exported to {quality_path}")
        except Exception as e:
            logger.error(f"Failed to export Parse Quality Report: {e}")

        return {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
            "quality_report": str(quality_path)
        }
