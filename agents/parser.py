import fitz  # PyMuPDF
import pdfplumber
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from config import logger
from agents.ocr import OCRFallbackAgent

class PDFParserAgent:
    def __init__(self, ocr_agent: OCRFallbackAgent):
        self.ocr_agent = ocr_agent

    def parse_pdf(self, pdf_path: str) -> Tuple[str, str, str]:
        """
        Parses a PDF file.
        Returns:
            Tuple[str, str, str]: (extracted_text, parse_quality, parse_reason)
            parse_quality is one of: 'Clean', 'Partial', 'Failed'
        """
        logger.info(f"Parsing PDF: {pdf_path}")
        path = Path(pdf_path)
        if not path.exists():
            return "", "Failed", "File does not exist"

        text = ""
        is_scanned = False
        layout_type = "single-column"

        try:
            # 1. Attempt PyMuPDF text extraction with layout awareness
            doc = fitz.open(pdf_path)
            
            # Check if document is scanned (zero text or very few characters)
            raw_pymupdf_text = ""
            for page in doc:
                raw_pymupdf_text += page.get_text()

            word_count = len(raw_pymupdf_text.split())
            logger.info(f"Initial raw PyMuPDF word count for {path.name}: {word_count}")

            if word_count < 50:
                logger.info(f"Low-yield PDF detected ({word_count} words). Triggering OCR fallback...")
                is_scanned = True
                text, ocr_success = self.ocr_agent.run_ocr(pdf_path)
                if ocr_success:
                    return text, "Partial", "Scanned PDF parsed using OCR fallback"
                else:
                    return "", "Failed", "Scanned PDF and OCR fallback failed"

            # 2. Document has text, perform layout-aware extraction
            extracted_pages = []
            for i, page in enumerate(doc):
                page_width = page.rect.width
                mid_x = page_width / 2
                
                # Get text blocks: (x0, y0, x1, y1, "text", block_no, block_type)
                blocks = page.get_text("blocks")
                
                # Sort blocks top-to-bottom
                blocks.sort(key=lambda b: b[1])
                
                left_blocks = []
                right_blocks = []
                full_width_blocks = []
                
                for b in blocks:
                    x0, y0, x1, y1, content, block_no, block_type = b
                    content = content.strip()
                    if not content:
                        continue
                        
                    # Classify based on alignment
                    if x1 <= mid_x + 10:  # Margin of 10px
                        left_blocks.append(b)
                    elif x0 >= mid_x - 10:
                        right_blocks.append(b)
                    else:
                        # Spans across columns or is centered
                        # If block is wide but mostly on one side, we could split, but let's treat as full width
                        full_width_blocks.append(b)
                
                # Detect if multi-column
                # If we have both left and right blocks, it's highly likely multi-column
                if len(left_blocks) > 1 and len(right_blocks) > 1:
                    layout_type = "multi-column"
                    # Sort left and right blocks by vertical position (y0)
                    left_blocks.sort(key=lambda b: b[1])
                    right_blocks.sort(key=lambda b: b[1])
                    
                    # Reconstruction logic:
                    # We can iterate through page elements. If there are full-width headers, we keep them in order.
                    # Simple heuristic: Group blocks by bands of y, or just read left column first then right column.
                    # A robust approach is:
                    # - If full-width blocks exist, interlace them based on y-position.
                    # Let's sort all blocks by y coordinate but separate left/right blocks that overlap in y.
                    # Standard reading order: Left column from top to bottom, then right column from top to bottom.
                    # Extract left column text, then right column text.
                    # This is usually the best representation of two-column resumes (e.g. left sidebar: skills/contact, right: experience).
                    left_text = "\n".join([b[4] for b in left_blocks])
                    right_text = "\n".join([b[4] for b in right_blocks])
                    full_text_list = []
                    
                    # Add full width blocks in position
                    # Let's sort full width blocks and interlace them.
                    # For simplicity, if we have columns, let's output:
                    # Full width header blocks (y < minimum column y)
                    # Left column
                    # Right column
                    # Full width footer blocks (y > maximum column y)
                    min_col_y = min(min([b[1] for b in left_blocks], default=9999), min([b[1] for b in right_blocks], default=9999))
                    max_col_y = max(max([b[3] for b in left_blocks], default=0), max([b[3] for b in right_blocks], default=0))
                    
                    header_blocks = [b[4] for b in full_width_blocks if b[3] < min_col_y]
                    footer_blocks = [b[4] for b in full_width_blocks if b[1] > max_col_y]
                    middle_full = [b[4] for b in full_width_blocks if min_col_y <= b[1] <= max_col_y]
                    
                    if header_blocks:
                        full_text_list.append("\n".join(header_blocks))
                    if left_text:
                        full_text_list.append("--- COLUMN 1 --- \n" + left_text)
                    if right_text:
                        full_text_list.append("--- COLUMN 2 --- \n" + right_text)
                    if middle_full:
                        full_text_list.append("\n".join(middle_full))
                    if footer_blocks:
                        full_text_list.append("\n".join(footer_blocks))
                        
                    page_text_str = "\n\n".join(full_text_list)
                    extracted_pages.append(page_text_str)
                else:
                    # Single column text flow
                    page_text = "\n".join([b[4] for b in blocks])
                    extracted_pages.append(page_text)
            
            doc.close()
            text = "\n\n=== PAGE BREAK ===\n\n".join(extracted_pages)
            logger.info(f"Successfully parsed {path.name} as {layout_type} PDF.")
            return text, "Clean", f"Extracted successfully as {layout_type} layout"

        except Exception as e:
            logger.error(f"Error parsing PDF {path.name}: {e}")
            # Fallback to pdfplumber
            try:
                logger.info(f"Attempting pdfplumber fallback for {path.name}...")
                with pdfplumber.open(pdf_path) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            pages_text.append(extracted)
                    text = "\n\n".join(pages_text)
                if len(text.split()) > 50:
                    return text, "Partial", "Extracted using pdfplumber fallback (layout preservation might be degraded)"
                else:
                    # Try OCR fallback
                    text, ocr_success = self.ocr_agent.run_ocr(pdf_path)
                    if ocr_success:
                        return text, "Partial", "Parsed using OCR fallback after standard parser failed"
                    else:
                        return "", "Failed", f"Parser crashed: {str(e)} and OCR fallback failed"
            except Exception as ex:
                logger.error(f"pdfplumber fallback failed: {ex}")
                return "", "Failed", f"Parser crashed: {str(e)}: {str(ex)}"
