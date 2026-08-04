import os
import logging
from typing import Tuple
from config import logger
import fitz  # PyMuPDF to render pages
from PIL import Image
import io

class OCRFallbackAgent:
    def __init__(self):
        self.easyocr_reader = None
        self._init_easyocr()
        self._init_pytesseract()

    def _init_easyocr(self):
        try:
            import easyocr
            # Initialize reader; download model if not already cached
            logger.info("Initializing EasyOCR reader...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False) # Keep gpu=False for cpu compatibility in hackathons
            logger.info("EasyOCR initialized successfully.")
        except Exception as e:
            logger.warning(f"EasyOCR could not be initialized (PyTorch/EasyOCR might not be fully installed): {e}")

    def _init_pytesseract(self):
        try:
            import pytesseract
            # Check if custom path is provided in environment variables
            tess_cmd = os.getenv("TESSERACT_CMD")
            if tess_cmd:
                pytesseract.pytesseract.tesseract_cmd = tess_cmd
                logger.info(f"Pytesseract custom path set: {tess_cmd}")
        except ImportError:
            logger.warning("Pytesseract library not found in Python path.")

    def run_ocr(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Renders PDF pages as images and runs OCR.
        Returns:
            Tuple[str, bool]: (ocr_text, success_status)
        """
        logger.info(f"Starting OCR fallback for: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            ocr_text_pages = []
            
            for i, page in enumerate(doc):
                logger.info(f"Rendering page {i+1} for OCR...")
                # Render page to a high-resolution pixmap
                pix = page.get_pixmap(dpi=150)
                # Convert pixmap to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                page_text = ""
                
                # 1. Try EasyOCR first
                if self.easyocr_reader:
                    try:
                        logger.info("Running EasyOCR on rendered image...")
                        # easyocr expects numpy array or file path or bytes
                        # Let's pass the image bytes
                        results = self.easyocr_reader.readtext(img_data, detail=0)
                        page_text = " ".join(results)
                    except Exception as e:
                        logger.warning(f"EasyOCR failed on page {i+1}: {e}")
                
                # 2. Try Pytesseract fallback if EasyOCR failed or wasn't initialized
                if not page_text.strip():
                    try:
                        import pytesseract
                        logger.info("Running Pytesseract fallback on rendered image...")
                        page_text = pytesseract.image_to_string(img)
                    except Exception as e:
                        logger.error(f"Pytesseract OCR failed on page {i+1}: {e}")
                
                if page_text.strip():
                    ocr_text_pages.append(f"--- OCR PAGE {i+1} ---\n{page_text}")
                else:
                    logger.warning(f"No text recovered from page {i+1} during OCR.")

            doc.close()
            
            full_text = "\n\n".join(ocr_text_pages)
            if len(full_text.split()) > 10:
                logger.info("OCR successfully extracted text from PDF.")
                return full_text, True
            else:
                logger.warning("OCR extracted little to no text.")
                return "", False

        except Exception as e:
            logger.error(f"Error during OCR execution: {e}")
            return "", False
        
        return "", False
