# AI Resume Shortlisting Engine - Hackathon Winner

An industry-standard multi-agent AI resume shortlisting engine designed to parse, analyze, normalize, and rank candidates against job descriptions. It solves real-world layout challenges, manages scanned files via OCR, aligns grades to a 10-point scale, and performs semantic matching.

---

## Features
- **Layout-Aware PDF Parser**: Extracts multi-column templates by coordinate segmentation using PyMuPDF.
- **Automatic OCR Fallback**: Renders PDF pages to images and runs EasyOCR / Pytesseract when standard text yield is low (<50 words).
- **Hybrid Matching**: Computes Exact, Synonym, Partial, and Implicit (Semantic) similarities using local embeddings (`all-MiniLM-L6-v2`) and RapidFuzz.
- **10-Point Grade Normalizer**: Converts GPA, CGPA, and Percentages to a standardized 10-point Indian CGPA scale.
- **Deterministic Weighted Scoring**: Calculates stable candidate scores based on custom JD weights.
- **Explainable AI (XAI)**: Generates 3 clear bullet points explaining match results per candidate.
- **Streamlit Dashboard**: A high-fidelity UI to upload resumes, view Leaderboards, download reports, and inspect candidate drill-downs.

---

## 🛠️ Installation & Setup (Under 5 Minutes)

### 1. Clone & Set Active Workspace
Ensure your files are placed in:
`C:\Users\User\.gemini\antigravity\scratch\resume_shortlisting_engine`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory:
```env
# Provide at least one API Key:
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional Custom Pytesseract Location
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## 🚀 Running the Engine

### CLI Pipeline Run
To run the evaluation pipeline from the console:
```bash
# Run with one of the 5 predefined roles (frontend, backend, fullstack, database, api_integration)
python main.py --resumes_dir ./scratch/resumes --role_type backend

# Or run with a custom JD file
python main.py --resumes_dir ./scratch/resumes --jd_text_file ./scratch/jd.txt
```

### Running the Web Dashboard
```bash
streamlit run app.py
```

### Running Stability & Unit Tests
To run the score stability and grade normalizer tests:
```bash
python tests/test_runner.py
```

---

## 📋 Evaluation Reports Output
All evaluation outputs are saved in the `./reports/` directory:
1. `[role]_report.csv`: Table containing leaderboard scores and standings.
2. `[role]_report.json`: Detailed JSON representation of parsed resumes and score evaluations.
3. `[role]_report.md`: Markdown summary report with shortlisted/reserve statuses and analysis.
4. `[role]_parse_quality_report.csv`: Standalone parser quality audit log.
