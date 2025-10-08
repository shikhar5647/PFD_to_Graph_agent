import cv2
import numpy as np
from typing import Dict, List
from langchain.chat_models import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from PIL import Image

class TopologyAgent:
    """Agent for detecting connections between equipment"""
    
    def __init__(self, llm_config: Dict):
        self.llm = ChatGoogleGenerativeAI(
            model=llm_config.get("model", "gemini-2.0-flash-exp"),
            temperature=0,
            google_api_key=llm_config.get("api_key")
        )
    
    def detect_connections(self, image: Image.Image, symbols: List[Dict]) -> Dict:
        """Detect stream connections between equipment"""
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect lines (streams)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 50,
            minLineLength=30, maxLineGap=10
        )
        
        connections = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Find source and target equipment
                source = self._find_equipment_at_point([x1, y1], symbols)
                target = self._find_equipment_at_point([x2, y2], symbols)
                
                if source and target and source != target:
                    connections.append({
                        "source": source.get("id"),
                        "target": target.get("id"),
                        "path": [[x1, y1], [x2, y2]],
                        "confidence": 0.8
                    })
        
        # Use LLM to verify and enhance connections
        enhanced_connections = self._enhance_with_llm(connections, symbols)
        
        return {
            "connections": enhanced_connections,
            "confidence": self._calculate_confidence(enhanced_connections)
        }
    
    def _find_equipment_at_point(self, point: List[int], symbols: List[Dict]) -> Dict:
        """Find equipment at or near a point"""
        for symbol in symbols:
            if self._point_in_bbox(point, symbol.get("bbox", [])):
                return symbol
        return None
    
    def _point_in_bbox(self, point: List[int], bbox: List[int]) -> bool:
        """Check if point is inside bounding box"""
        if len(bbox) != 4:
            return False
        x, y, w, h = bbox
        return x <= point[0] <= x+w and y <= point[1] <= y+h
    
    def _enhance_with_llm(self, connections: List[Dict], symbols: List[Dict]) -> List[Dict]:
        """Use LLM to verify and enhance detected connections"""
        # In production, send to LLM for verification
        return connections
    
    def _calculate_confidence(self, connections: List[Dict]) -> float:
        """Calculate connection detection confidence"""
        if not connections:
            return 0.0
        return sum(c.get("confidence", 0) for c in connections) / len(connections)