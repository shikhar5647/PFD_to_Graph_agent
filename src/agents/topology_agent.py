import cv2
import numpy as np
from typing import Dict, List
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
from io import BytesIO

class TopologyAgent:
    """Agent for detecting connections between equipment"""
    
    def __init__(self, llm_config: Dict):
        self.llm = ChatGoogleGenerativeAI(
            model=llm_config.get("model", "gemini-2.5-pro"),
            temperature=0,
            google_api_key=llm_config.get("api_key")
        )
    
    def detect_connections(self, image: Image.Image, symbols: List[Dict]) -> Dict:
        """Detect stream connections between equipment"""
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect lines (streams) using CV
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 50,
            minLineLength=30, maxLineGap=10
        )
        
        cv_connections = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Find source and target equipment
                source = self._find_equipment_at_point([x1, y1], symbols)
                target = self._find_equipment_at_point([x2, y2], symbols)
                
                if source and target and source != target:
                    cv_connections.append({
                        "source": source.get("id"),
                        "target": target.get("id"),
                        "path": [[x1, y1], [x2, y2]],
                        "confidence": 0.8
                    })
        
        # Use Gemini to verify and enhance connections
        gemini_connections = self._detect_with_gemini(image, symbols)
        
        # Merge connections
        enhanced_connections = self._merge_connections(cv_connections, gemini_connections)
        
        return {
            "connections": enhanced_connections,
            "confidence": self._calculate_confidence(enhanced_connections)
        }
    
    def _detect_with_gemini(self, image: Image.Image, symbols: List[Dict]) -> List[Dict]:
        """Use Gemini to detect connections"""
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Create symbol reference
            symbol_ref = "\n".join([
                f"- {s['id']}: {s.get('label', 'unlabeled')} at position {s.get('center')}"
                for s in symbols
            ])
            
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"""Analyze the stream connections (lines/arrows) in this PFD.

Equipment symbols identified:
{symbol_ref}

For each stream/connection you see:
1. Identify which equipment it connects (source to target)
2. Provide confidence (0-1)

Return as JSON array:
[
  {{"source": "eq_0", "target": "eq_1", "confidence": 0.9}}
]"""
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{img_str}"
                    }
                ]
            )
            
            response = self.llm.invoke([message])
            
            # Parse response
            import json
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content
            
            connections = json.loads(json_str)
            return connections
            
        except Exception as e:
            print(f"Gemini connection detection error: {e}")
            return []
    
    def _merge_connections(self, cv_connections: List[Dict], gemini_connections: List[Dict]) -> List[Dict]:
        """Merge CV and Gemini detected connections"""
        merged = {}
        
        # Add CV connections (validate first)
        for conn in cv_connections:
            source = conn.get('source')
            target = conn.get('target')
            
            # Skip invalid connections
            if not source or not target or source == target:
                continue
                
            key = (source, target)
            merged[key] = conn
        
        # Add or update with Gemini connections
        for conn in gemini_connections:
            source = conn.get('source')
            target = conn.get('target')
            
            # Skip invalid connections
            if not source or not target or source == target:
                continue
                
            key = (source, target)
            if key in merged:
                # Update confidence to average
                merged[key]['confidence'] = (merged[key]['confidence'] + conn.get('confidence', 0.8)) / 2
            else:
                merged[key] = conn
        
        # Return only valid connections
        return [conn for conn in merged.values() if conn.get('source') and conn.get('target')]
    
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
    
    def _calculate_confidence(self, connections: List[Dict]) -> float:
        """Calculate connection detection confidence"""
        if not connections:
            return 0.0
        return sum(c.get("confidence", 0) for c in connections) / len(connections)