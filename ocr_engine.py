import os
import sys
from ocr_pdf_pipeline import process_pdf

def extract_text(file_path: str) -> str:
    """
    Extracts text from a PDF or Image file using the existing OCR pipeline.
    
    Args:
        file_path (str): Path to the input file.
        
    Returns:
        str: The extracted text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if file_path.lower().endswith('.pdf'):
        # Use the existing process_pdf function
        # It returns a dictionary with pages and text
        try:
            result = process_pdf(file_path, use_ocr=True)
            full_text = ""
            for page in result.get('pages', []):
                # Check if 'text' key exists (PyPDF2 path) or 'detections' (OCR path)
                if 'text' in page:
                    full_text += page['text'] + "\n\n"
                elif 'detections' in page:
                    for detection in page['detections']:
                        full_text += detection['text'] + " "
                    full_text += "\n\n"
            return full_text.strip()
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return ""
    else:
        # Assume image
        # The existing pipeline is heavily PDF focused in ocr_pdf_pipeline.py
        # But ocr_order.py has CRAFTDetector and PaddleOCR logic we can reuse or just use PaddleOCR directly here for simplicity if needed.
        # However, ocr_pdf_pipeline.py's PDFTextExtractor uses CRAFTDetector internally.
        # Let's try to use the PDFTextExtractor's components for a single image if possible, 
        # or just use PaddleOCR directly for images to keep it simple as the user asked to "use this ocr model".
        # The "ocr model" implies the CRAFT+Paddle combo.
        
        # Let's import the detector and ocr from ocr_pdf_pipeline
        from ocr_pdf_pipeline import CRAFTDetector
        from paddleocr import PaddleOCR
        import cv2
        import numpy as np
        
        detector = CRAFTDetector()
        ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_batch_num=1)
        
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")
            
        # Detect text regions
        boxes = detector.detect_text_regions(image)
        # We should sort them, ocr_pdf_pipeline has a sort function
        from ocr_pdf_pipeline import sort_boxes_reading_order
        sorted_boxes = sort_boxes_reading_order(boxes)
        
        full_text = ""
        for idx, cy, cx, y_min, y_max, x_min, height, box in sorted_boxes:
             # Crop and OCR
             # Logic similar to ocr_pdf_pipeline.py lines 276-300
             try:
                arr = np.array(box).reshape(4, 2).astype(int)
                x1 = max(0, int(arr[:, 0].min()))
                y1 = max(0, int(arr[:, 1].min()))
                x2 = min(image.shape[1], int(arr[:, 0].max()))
                y2 = min(image.shape[0], int(arr[:, 1].max()))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                region = image[y1:y2, x1:x2]
                res = ocr.ocr(region, cls=True)
                if res and res[0]:
                    text = res[0][0][1][0] # Get text from first result
                    full_text += text + " "
             except Exception:
                 continue
                 
        return full_text.strip()
