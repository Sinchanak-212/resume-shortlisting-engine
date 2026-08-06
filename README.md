# AI Resume Shortlisting Engine - Enterprise ATS Edition

This project has been upgraded in place into an enterprise-grade ATS-style screening engine for modern hiring workflows. It preserves the existing architecture and folder structure while improving resume parsing, semantic skill matching, weighted scoring, ATS analysis, and recruiter-facing explanations.

---

## What is new
- Enterprise-grade resume parsing with structured extraction for contact, experience, education, skills, projects, certifications, achievements, leadership, links, and more.
- Semantic skill matching using normalization, aliases, hierarchies, and fuzzy/semantic similarity.
- Weighted scoring with explainable breakdowns for skills, experience, projects, education, certifications, achievements, leadership, and profile strength.
- ATS compliance analysis for tables, columns, missing sections, formatting issues, and readability.
- Duplicate detection based on email/phone/profile identifiers.
- Confidence metadata for extracted fields and recruiter-friendly summaries.
- Streamlit dashboard enhancements with filtering, ranking, and downloadable reports.

---

## Core workflow
1. Parse PDF resumes and OCR fallback when necessary.
2. Extract structured resume fields and confidence metadata.
3. Normalize and enrich skills with canonical mappings.
4. Match candidate skills semantically against the job description.
5. Score candidates with explainable breakdowns.
6. Generate ATS and recruiter-oriented insights.
7. Export reports and shortlist candidates.

---

## Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional environment variables:
```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## Usage

### CLI
```bash
python main.py --resumes_dir ./scratch/resumes --role_type backend
```

### Dashboard
```bash
streamlit run app.py
```

---

## Output reports
Reports are written into the reports directory:
1. [role]_report.csv
2. [role]_report.json
3. [role]_report.md
4. [role]_parse_quality_report.csv
