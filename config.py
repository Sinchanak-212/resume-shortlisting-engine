import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

# Create directories if they don't exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "app.log", encoding="utf-8")
    ]
)

logger = logging.getLogger("ResumeEngine")

# Skill Synonym Dictionary
SKILL_SYNONYMS = {
    "react": ["react.js", "reactjs", "react js"],
    "node": ["node.js", "nodejs", "node js"],
    "python": ["py", "python3", "python 3"],
    "javascript": ["js", "javascript", "ecmascript"],
    "typescript": ["ts", "typescript"],
    "mongodb": ["mongo", "mongodb"],
    "postgresql": ["postgres", "postgresql", "psql"],
    "mysql": ["mysql", "my sql"],
    "github": ["git", "github", "gitlab"],
    "redux": ["redux", "redux-toolkit", "redux toolkit"],
    "docker": ["docker", "dockerfile", "containerization"],
    "aws": ["aws", "amazon web services", "s3", "ec2"],
    "gcp": ["gcp", "google cloud platform", "google cloud"],
    "kubernetes": ["k8s", "kubernetes"],
    "html": ["html", "html5", "css", "css3"],
}

# Score Weights
SCORE_WEIGHTS = {
    "required_skills": 45,
    "preferred_skills": 15,
    "experience": 10,
    "projects": 10,
    "cgpa": 10,
    "certifications": 5,
    "education": 5
}
