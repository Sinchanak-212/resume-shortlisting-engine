# AI Usage Log

We utilized Gemini/OpenAI API capabilities for unstructured entity extraction (Resume Information Extraction and Job Description Parsing) and explaining match results.

### Extensions Beyond AI Generations:
1. **Layout-Aware PDF Parser**: Built custom coordinate-aware geometric logic using PyMuPDF to split left/right column blocks on Canva templates, bypassing default left-to-right interleaving bugs.
2. **Hybrid Skill Matching**: Integrated exact lookup, synonym matching, RapidFuzz string similarity, and offline SentenceTransformers (`all-MiniLM-L6-v2`) to perform semantic matching.
3. **Deterministic Scoring**: Coded a fully deterministic scoring algorithm matching hackathon weights, ensuring score stability across runs.
