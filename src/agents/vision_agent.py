import cv2
import numpy as np
from PIL import Image
from typing import Dict, List
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
import base64
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


class VisionAgent:
    """Agent for detecting equipment symbols in PFD using vision models"""
    
    def __init__(self, llm_config: Dict):
        self.llm = ChatGoogleGenerativeAI(
            model=llm_config.get("model", "gemini-2.5-pro"),
            temperature=0,
            google_api_key=llm_config.get("api_key")
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in chemical engineering process flow diagrams (PFDs).
            Analyze the PFD image and identify all equipment symbols including:
            - Reactors (circles with special markings)
            - Heat exchangers (crossed symbols)
            - Distillation columns (tall cylindrical vessels)
            - Mixers (boxes with mixing symbols)
            - Splitters
            - Valves (V-shaped or throttle symbols)
            - Pumps (circular with arrows)
            - Tanks/vessels
            - Raw material inputs (arrows pointing in)
            - Product outputs (arrows pointing out)
            
            For each equipment, provide:
            1. Type of equipment
            2. Approximate bounding box coordinates (as percentage of image)
            3. Confidence level
            4. Any visible labels
            
            Return as structured JSON."""),
            ("user", "Analyze this PFD image and detect all equipment symbols: {image_description}")
        ])
    
    def analyze(self, image: Image.Image) -> Dict:
        """Analyze PFD image to detect equipment"""
        # Convert PIL Image to base64 and wrap as data URL for Gemini
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        
        # Send multimodal user message compatible with ChatGoogleGenerativeAI
        response = self.llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Identify all equipment symbols in this PFD"},
                {"type": "image_url", "image_url": data_url}
            ])
        ])
        
        # Also use traditional CV for backup
        cv_symbols = self._detect_with_cv(image)
        
        # Combine LLM and CV results
        symbols = self._merge_detections(response, cv_symbols)
        
        return {
            "symbols": symbols,
            "confidence": self._calculate_confidence(symbols)
        }
    
    def _detect_with_cv(self, image: Image.Image) -> List[Dict]:
        """Traditional computer vision detection as backup"""
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        symbols = []
        
        # Detect circles (reactors, pumps)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )
        
        if circles is not None:
            for circle in circles[0]:
                x, y, r = circle
                symbols.append({
                    "type": "circular_equipment",
                    "bbox": [int(x-r), int(y-r), int(2*r), int(2*r)],
                    "center": [int(x), int(y)],
                    "confidence": 0.7,
                    "detection_method": "cv"
                })
        
        # Detect rectangles (columns, tanks)
        contours, _ = cv2.findContours(
            cv2.Canny(gray, 50, 150),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if area > 500:  # Filter small noise
                symbols.append({
                    "type": "rectangular_equipment",
                    "bbox": [x, y, w, h],
                    "center": [x + w//2, y + h//2],
                    "confidence": 0.6,
                    "detection_method": "cv"
                })
        
        return symbols
    
    def _merge_detections(self, llm_response, cv_symbols: List[Dict]) -> List[Dict]:
        """Merge LLM and CV detections"""
        # Parse LLM response and combine with CV
        # For now, return CV symbols (in production, properly merge)
        return cv_symbols
    
    def _calculate_confidence(self, symbols: List[Dict]) -> float:
        """Calculate overall detection confidence"""
        if not symbols:
            return 0.0
        return sum(s.get("confidence", 0) for s in symbols) / len(symbols)