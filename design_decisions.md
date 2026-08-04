# Design Decisions: Resolving the Four Tricky Parts

This document outlines the engineering solutions implemented in the Resume Shortlisting Engine to handle real-world resume parsing and matching complexities.

## Tricky Part 1 — PDF Layout Chaos
To prevent text from different columns interleaving (e.g. merging a left-side Education section with a right-side Skills list), the parser performs a geometric bounding box analysis using PyMuPDF. The page width is bisected, and text blocks are classified into three bands: left-half blocks, right-half blocks, and full-width blocks (spanning across center). The parser reconstructs reading order by outputting header blocks, followed by the left column blocks (sorted top-to-bottom), then the right column blocks (sorted top-to-bottom), and finally the footer blocks. This layout-aware extraction successfully handles Canva templates and split-column tables.

## Tricky Part 2 — The Scanned Resume
Scanned resumes containing image layers only are handled via a local OCR fallback engine. The parser counts the extracted words; if the total is less than 50 words, the parser triggers the OCR fallback. Each page of the PDF is rendered to a 150 DPI image using PyMuPDF. EasyOCR is run as the primary OCR engine; if it yields zero words (e.g., due to engine initialization or installation gaps), the engine automatically falls back to Pytesseract. If text is recovered, it is marked as `Partial` parse quality with a flag indicating OCR was used.

## Tricky Part 3 — Skill Extraction from Unstructured Text
Instead of relying on a labeled "Skills" header, skills are extracted holistically. The engine first prompts the LLM to inspect the full resume text for skills. Then, the local `SkillExtractionAgent` performs regular expression searches across the titles and descriptions of projects, experience, and certifications using a pre-defined canonical synonym dictionary (mapping terms like `ReactJS` to `React`, `Py` to `Python`). Finally, a local spaCy named entity helper extracts unrecognized proper nouns in raw text, checking them against technology lists and assigning confidence levels based on the extraction source (e.g., 0.9 for direct listing, 0.7 for project descriptions).

## Tricky Part 4 — Parse Quality Affects Score Confidence
To prevent bad parses from producing high matching scores, the `ConfidenceAgent` implements a strict quality gate. Resumes where text is completely unreadable or where the name and email fields cannot be extracted are flagged as `Failed` quality, assigned a match score of 0.0, and pushed to the bottom of the list for manual review. For successful extractions, the parse quality is marked as `Partial` if OCR was triggered or if vital fields (like skills or experience lists) are empty. Any `Partial` parse is capped at a maximum confidence level of `Medium` or `Low` to prevent false positive shortlistings.
