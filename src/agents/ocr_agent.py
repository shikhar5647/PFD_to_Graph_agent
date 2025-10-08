import easyocr
import pytesseract
from PIL import Image
from typing import Dict, List
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI


class OCRAgent:
    """Agent for extracting text labels from PFD"""
    
    def __init__(self, llm_config: Dict):
        self.reader = easyocr.Reader(['en'])
        self.llm = ChatGoogleGenerativeAI(
            model=llm_config.get("model", "gemini-2.5-pro"),
            temperature=0,
            google_api_key=llm_config.get("api_key")
        )
    
    def extract(self, image: Image.Image, detected_symbols: List[Dict]) -> Dict:
        """Extract text labels from image"""
        img_array = np.array(image)
        
        # Run EasyOCR
        results_easyocr = self.reader.readtext(img_array)
        
        # Run Tesseract as backup
        results_tesseract = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        
        # Parse and associate labels with equipment
        labels = self._associate_labels_with_equipment(
            results_easyocr,
            results_tesseract,
            detected_symbols
        )
        
        return {
            "labels": labels,
            "regions": self._format_text_regions(results_easyocr)
        }
    
    def _associate_labels_with_equipment(
        self, 
        easyocr_results: List,
        tesseract_results: Dict,
        symbols: List[Dict]
    ) -> Dict[str, str]:
        """Match text labels to equipment symbols"""
        labels = {}
        
        for result in easyocr_results:
            bbox, text, confidence = result
            if confidence < 0.3:
                continue
            
            # Find nearest equipment symbol
            text_center = self._get_bbox_center(bbox)
            nearest_symbol = self._find_nearest_symbol(text_center, symbols)
            
            if nearest_symbol:
                symbol_id = nearest_symbol.get("id", f"eq_{len(labels)}")
                labels[symbol_id] = text.strip()
        
        return labels
    
    def _get_bbox_center(self, bbox: List) -> List[float]:
        """Calculate center of bounding box"""
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        return [sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords)]
    
    def _find_nearest_symbol(self, point: List[float], symbols: List[Dict]) -> Dict:
        """Find nearest symbol to a point"""
        min_dist = float('inf')
        nearest = None
        
        for symbol in symbols:
            if 'center' in symbol:
                dist = np.sqrt(
                    (point[0] - symbol['center'][0])**2 +
                    (point[1] - symbol['center'][1])**2
                )
                if dist < min_dist and dist < 100:  # Max distance threshold
                    min_dist = dist
                    nearest = symbol
        
        return nearest
    
    def _format_text_regions(self, results: List) -> List[Dict]:
        """Format OCR results"""
        regions = []
        for bbox, text, confidence in results:
            regions.append({
                "text": text,
                "bbox": bbox,
                "confidence": confidence
            })
        return regions