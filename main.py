import os
import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent))

from config import logger, REPORTS_DIR
from utils.llm import LLMClient
from agents.parser import PDFParserAgent
from agents.ocr import OCRFallbackAgent
from agents.extractor import ResumeExtractorAgent
from agents.grade_normalizer import GradeNormalizerAgent
from agents.skill_extractor import SkillExtractionAgent
from agents.jd_parser import JDParserAgent
from agents.matcher import MatchingAgent
from agents.scoring import ScoringAgent
from agents.confidence import ConfidenceAgent
from agents.explanation import ExplanationAgent
from agents.ranker import RankingAgent
from models.schemas import MatchResult, ParsedJD

# Default Job Descriptions for quick usage in demo
from jd_defaults import DEFAULT_JDS  # re-exported for backwards compatibility

def build_pipeline_agents() -> dict:
    """
    Constructs every pipeline agent once. Expensive to call (loads EasyOCR,
    SentenceTransformer, spaCy models) - callers that run this repeatedly
    (e.g. a Streamlit app across reruns) should cache the returned dict
    (e.g. via st.cache_resource) instead of calling this per-request.
    """
    logger.info("Initializing multi-agent pipeline...")
    llm_client = LLMClient()
    ocr_agent = OCRFallbackAgent()
    skill_extractor = SkillExtractionAgent()
    return {
        "llm_client": llm_client,
        "parser_agent": PDFParserAgent(ocr_agent),
        "extractor_agent": ResumeExtractorAgent(llm_client),
        "normalizer_agent": GradeNormalizerAgent(),
        "skill_extractor": skill_extractor,
        "matcher": MatchingAgent(skill_extractor),
        "scoring_agent": ScoringAgent(),
        "confidence_agent": ConfidenceAgent(),
        "explanation_agent": ExplanationAgent(llm_client),
        "ranker_agent": RankingAgent(),
    }


def process_resumes(resumes_dir: str, parsed_jd: ParsedJD, limit: int = -1, agents: dict = None, progress_cb=None, max_workers: int = None) -> tuple:
    """Orchestrates the multi-agent pipeline to parse, score, and analyze resumes.

    Args:
        agents: Optional pre-built agents dict from build_pipeline_agents(). If not
            provided, agents are built fresh (this is what the CLI path below does).
            Passing a cached dict avoids reloading EasyOCR/SentenceTransformer/spaCy
            on every call - important for the Streamlit app, where process_resumes
            is invoked on every button click.
    """
    # BUGFIX (perf): previously this function unconditionally rebuilt every agent -
    # including EasyOCR, SentenceTransformer('all-MiniLM-L6-v2'), and spaCy - on every
    # single call. In the CLI that happens once per run, which is fine. In the Streamlit
    # app it happened on every "Process & Rank" click, adding tens of seconds of pure
    # model-loading overhead each time. See app.py: agents are now built once via
    # st.cache_resource and passed in here.
    resumes_path = Path(resumes_dir)
    if not resumes_path.exists():
        raise FileNotFoundError(f"Resumes directory not found: {resumes_dir}")

    pdf_files = list(resumes_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {resumes_dir}")

    logger.info(f"Found {len(pdf_files)} PDF resumes to process.")
    
    if limit > 0:
        pdf_files = pdf_files[:limit]
        logger.info(f"Limiting execution to first {limit} files.")

    if agents is None:
        agents = build_pipeline_agents()

    llm_client = agents["llm_client"]
    parser_agent = agents["parser_agent"]
    extractor_agent = agents["extractor_agent"]
    normalizer_agent = agents["normalizer_agent"]
    skill_extractor = agents["skill_extractor"]
    matcher = agents["matcher"]
    scoring_agent = agents["scoring_agent"]
    confidence_agent = agents["confidence_agent"]
    explanation_agent = agents["explanation_agent"]
    ranker_agent = agents["ranker_agent"]

    results = []
    
    parse_lock = threading.Lock()  # OCR/torch models are not thread-safe; LLM + matching still run in parallel

    def _process_one(pdf_file):
        results = []
        t0 = time.perf_counter()
        logger.info(f"\nProcessing file: {pdf_file.name}")
        candidate_name = pdf_file.stem
        
        try:
            # Step 1: Parse PDF / OCR Fallback
            with parse_lock:
                raw_text, parse_status, parse_reason = parser_agent.parse_pdf(str(pdf_file))
            t_parse = time.perf_counter() - t0
            
            if parse_status == "Failed" or not raw_text.strip():
                # Parser and OCR failed completely
                logger.warning(f"Parse failed for {pdf_file.name}")
                results.append(MatchResult(
                    candidate_name=candidate_name,
                    resume_file=pdf_file.name,
                    role_name=parsed_jd.role_name,
                    score=0.0,
                    parse_quality="Failed",
                    confidence="Low",
                    normalized_cgpa=0.0,
                    explanation=["PDF text extraction failed completely.", "OCR failed to recover legible text.", "Manual CV review recommended."],
                    is_shortlisted=False,
                    is_reserve=False
                ))
                return results[-1]

            # Step 2: Information Extraction
            parsed_resume = extractor_agent.extract_resume_info(raw_text)
            
            # Update candidate name from extracted details if available
            cand_name = parsed_resume.name or candidate_name

            # Step 3: Grade Normalization
            normalized_cgpa, norm_reason = normalizer_agent.normalize(parsed_resume, raw_text)

            # Step 4: Parse Quality & Confidence Assessment
            quality, confidence = confidence_agent.evaluate_quality_and_confidence(parsed_resume, parse_status, parse_reason)

            # Safeguard: if the parse quality is deemed Failed (e.g. no name or email extracted), do not score
            if quality == "Failed":
                logger.warning(f"Extracted info failed validation for {pdf_file.name}")
                results.append(MatchResult(
                    candidate_name=cand_name,
                    resume_file=pdf_file.name,
                    role_name=parsed_jd.role_name,
                    score=0.0,
                    parse_quality="Failed",
                    confidence="Low",
                    normalized_cgpa=normalized_cgpa,
                    explanation=["Failed to extract crucial candidate credentials (Name/Email).", "Extracted content is insufficient for scoring.", "Manual CV review recommended."],
                    is_shortlisted=False,
                    is_reserve=False
                ))
                return results[-1]

            # Step 5: Skill Extraction & Synonym Mapping
            enriched_skills = skill_extractor.extract_and_enrich_skills(
                parsed_resume.skills, 
                raw_text, 
                parsed_resume.projects, 
                parsed_resume.experience or parsed_resume.internships
            )

            # Step 6: Semantic Skill Matching
            skill_matches = matcher.match_candidate_to_jd(enriched_skills, parsed_jd)

            # Step 7: Score Computation
            score, breakdown_str = scoring_agent.calculate_score(parsed_resume, normalized_cgpa, skill_matches, parsed_jd)

            # Step 8: Explanation Generation
            explanation_bullets = explanation_agent.generate_explanation(
                parsed_resume, parsed_jd, skill_matches, score, normalized_cgpa, confidence
            )

            # Construct MatchResult
            results.append(MatchResult(
                candidate_name=cand_name,
                resume_file=pdf_file.name,
                role_name=parsed_jd.role_name,
                score=score,
                parse_quality=quality,
                confidence=confidence,
                normalized_cgpa=normalized_cgpa,
                skills_extracted=[s["skill"] for s in enriched_skills],
                skills_matched=skill_matches,
                explanation=explanation_bullets,
                is_shortlisted=False, # Ranker will set this
                is_reserve=False     # Ranker will set this
            ))

        except Exception as e:
            logger.error(f"Pipeline crashed for {pdf_file.name}: {e}", exc_info=True)
            results.append(MatchResult(
                candidate_name=candidate_name,
                resume_file=pdf_file.name,
                role_name=parsed_jd.role_name,
                score=0.0,
                parse_quality="Failed",
                confidence="Low",
                normalized_cgpa=0.0,
                explanation=[f"Pipeline error: {str(e)}", "Please check system log files.", "Manual CV review recommended."],
                is_shortlisted=False,
                is_reserve=False
            ))
        logger.info(f"Finished {pdf_file.name} in {time.perf_counter() - t0:.1f}s")
        return results[-1]

    workers = max_workers or int(os.getenv("MAX_WORKERS", "3"))
    total = len(pdf_files)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_process_one, f): f for f in pdf_files}
        for done, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if progress_cb:
                progress_cb(done, total, futures[fut].name)  # called from the calling thread only

    # Step 9: Ranking & Slot Allocation
    ranked_candidates = ranker_agent.rank_and_allocate_slots(results, parsed_jd)
    
    # Step 10: Export Reports
    report_paths = ranker_agent.export_reports(ranked_candidates, parsed_jd)
    
    return ranked_candidates, report_paths

def main():
    parser = argparse.ArgumentParser(description="AI Resume Shortlisting Engine - Hackathon Edition")
    parser.add_argument("--resumes_dir", type=str, required=True, help="Folder containing raw PDF resumes")
    
    # JD Source Options
    jd_group = parser.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("--role_type", type=str, choices=list(DEFAULT_JDS.keys()), help="Select one of the 5 default hackathon JD roles")
    jd_group.add_argument("--jd_text_file", type=str, help="Path to plain text file containing Job Description")
    jd_group.add_argument("--jd_text", type=str, help="Raw Job Description text content")

    parser.add_argument("--limit", type=int, default=-1, help="Limit number of resumes processed (for rapid testing)")
    
    args = parser.parse_args()

    # Load environmental vars (API Keys)
    load_dotenv()

    # 1. Parse Job Description
    parsed_jd = None
    if args.role_type:
        parsed_jd = DEFAULT_JDS[args.role_type]
        logger.info(f"Using Default JD: {parsed_jd.role_name}")
    else:
        # Load raw text
        raw_jd = ""
        if args.jd_text_file:
            with open(args.jd_text_file, "r", encoding="utf-8") as f:
                raw_jd = f.read()
        else:
            raw_jd = args.jd_text

        # Call JD parser agent
        llm_client = LLMClient()
        jd_agent = JDParserAgent(llm_client)
        parsed_jd = jd_agent.parse_jd(raw_jd)

    logger.info(f"Job Description: Role={parsed_jd.role_name}, Slots={parsed_jd.slots}, Min CGPA={parsed_jd.min_cgpa}")

    # 2. Run Pipeline
    try:
        ranked_candidates, report_paths = process_resumes(args.resumes_dir, parsed_jd, args.limit)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    # 3. Print Leadboard Summary
    print("\n" + "="*80)
    print(f" LEADERBOARD: {parsed_jd.role_name} (Slots: {parsed_jd.slots}) ")
    print("="*80)
    print(f"{'Rank':<5} | {'Candidate Name':<25} | {'Score':<8} | {'CGPA':<6} | {'Status':<12} | {'Confidence':<10}")
    print("-"*80)
    
    shortlisted = [c for c in ranked_candidates if c.is_shortlisted]
    reserve = [c for c in ranked_candidates if c.is_reserve]
    failed = [c for c in ranked_candidates if c.parse_quality == "Failed"]

    rank = 1
    for c in shortlisted:
        print(f"{rank:<5} | {c.candidate_name[:25]:<25} | {c.score:<8.2f} | {c.normalized_cgpa:<6.2f} | {'SHORTLISTED':<12} | {c.confidence:<10}")
        rank += 1
        
    print("-"*80 + " (Reserve List)")
    for c in reserve:
        print(f"{rank:<5} | {c.candidate_name[:25]:<25} | {c.score:<8.2f} | {c.normalized_cgpa:<6.2f} | {'RESERVE':<12} | {c.confidence:<10}")
        rank += 1

    if failed:
        print("-"*80 + " (Parse Failures - Manual Review)")
        for c in failed:
            print(f"{'-':<5} | {c.candidate_name[:25]:<25} | {c.score:<8.2f} | {c.normalized_cgpa:<6.2f} | {'FAILED PARSE':<12} | {c.confidence:<10}")

    print("="*80)
    print(f"Reports exported successfully:")
    print(f"- CSV leaderboard: {report_paths['csv']}")
    print(f"- JSON data: {report_paths['json']}")
    print(f"- Markdown document: {report_paths['markdown']}")
    print(f"- Parse Quality Tracking: {report_paths['quality_report']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
