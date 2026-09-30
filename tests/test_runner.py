"""Lightweight tests. Run with:  python tests/test_runner.py   (or: pytest tests/)

Only needs pydantic, python-dotenv, rapidfuzz and pandas - no models, OCR or API keys.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SKILL_SYNONYMS
from models.schemas import ParsedResume, ParsedJD, SkillMatchDetail, MatchResult
from agents.grade_normalizer import GradeNormalizerAgent
from agents.scoring import ScoringAgent
from agents.ranker import RankingAgent

JD = ParsedJD(role_name="Backend Developer", required_skills=["Python", "Git"],
              preferred_skills=["Docker"], min_cgpa=6.5, slots=2)
MATCHES = [SkillMatchDetail(skill=s, match_type="exact", score=1.0, reason="t")
           for s in ("Python", "Git", "Docker")]


def _score(**resume_fields):
    return ScoringAgent().calculate_score(ParsedResume(**resume_fields), 8.0, MATCHES, JD)[0]


# ---- grade normalizer -------------------------------------------------------
def test_grade_direct_cgpa():
    assert GradeNormalizerAgent().normalize(ParsedResume(cgpa=8.4), "")[0] == 8.4

def test_grade_percentage_and_gpa():
    n = GradeNormalizerAgent()
    assert n.normalize(ParsedResume(percentage=85), "")[0] == round(85 / 9.5, 2)
    assert n.normalize(ParsedResume(gpa=3.5), "")[0] == 8.75

def test_missing_grade_is_unknown_not_passing():
    value, _ = GradeNormalizerAgent().normalize(ParsedResume(), "no grades here")
    assert value == 0.0 and value < JD.min_cgpa


# ---- scoring ----------------------------------------------------------------
def test_score_is_deterministic():
    kw = dict(branch="Computer Science", college="IIT Madras")
    assert len({_score(**kw) for _ in range(5)}) == 1

def test_substring_false_positives_rejected():
    base = _score(branch="Computer Science", college="Some College")
    assert _score(branch="Architecture", college="Institute of Commerce") < base
    assert _score(branch="Physics", college="Shapes Academy") < base

def test_real_cs_and_tier1_still_detected():
    cs = _score(branch="Information Technology", college="Some College")
    plain = _score(branch="Mechanical", college="Some College")
    assert cs > plain
    assert _score(branch="Mechanical", college="IIT Bombay") == cs


# ---- synonyms ---------------------------------------------------------------
def test_html_and_css_are_separate_skills():
    assert "css" in SKILL_SYNONYMS and "css" not in SKILL_SYNONYMS["html"]
    assert "s3" not in SKILL_SYNONYMS["aws"]


# ---- ranking ----------------------------------------------------------------
def _cand(name, score, cgpa, quality="Clean"):
    return MatchResult(candidate_name=name, resume_file=f"{name}.pdf", role_name="R",
                       score=score, normalized_cgpa=cgpa, parse_quality=quality)

def test_ranker_enforces_cgpa_floor_and_slots():
    out = RankingAgent().rank_and_allocate_slots(
        [_cand("a", 90, 5.0), _cand("b", 80, 8.0), _cand("c", 70, 8.0), _cand("d", 60, 8.0),
         _cand("x", 0, 0.0, "Failed")], JD)
    short = [c.candidate_name for c in out if c.is_shortlisted]
    assert short == ["b", "c"]                      # 'a' top score but below CGPA floor
    assert out[-1].candidate_name == "x" and not out[-1].is_shortlisted


# ---- pipeline errors (needs the heavy ML deps; skipped if not installed) -----
def test_process_resumes_raises_on_missing_dir():
    try:
        from main import process_resumes
    except ImportError:
        print("  SKIP (ML dependencies not installed)")
        return
    try:
        process_resumes("/definitely/not/a/dir", JD)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
