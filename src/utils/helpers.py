import numpy as np
from typing import List, Tuple, Dict, Any
import cv2
from PIL import Image
import hashlib

class ImageHelpers:
    """Helper functions for image processing"""
    
    @staticmethod
    def resize_image(image: Image.Image, max_size: Tuple[int, int] = (1920, 1080)) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image
            max_size: Maximum dimensions (width, height)
        
        Returns:
            Resized image
        """
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    
    @staticmethod
    def preprocess_for_ocr(image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        
        Args:
            image: PIL Image
        
        Returns:
            Preprocessed image
        """
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        # Convert back to PIL
        return Image.fromarray(denoised)
    
    @staticmethod
    def enhance_contrast(image: Image.Image) -> Image.Image:
        """
        Enhance image contrast
        
        Args:
            image: PIL Image
        
        Returns:
            Enhanced image
        """
        img_array = np.array(image.convert('RGB'))
        
        # Apply CLAHE
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(enhanced)
    
    @staticmethod
    def get_image_hash(image: Image.Image) -> str:
        """
        Generate hash for image
        
        Args:
            image: PIL Image
        
        Returns:
            Hash string
        """
        img_bytes = image.tobytes()
        return hashlib.md5(img_bytes).hexdigest()

class GeometryHelpers:
    """Helper functions for geometry calculations"""
    
    @staticmethod
    def calculate_distance(point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    @staticmethod
    def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union for two bounding boxes
        
        Args:
            bbox1: [x, y, w, h]
            bbox2: [x, y, w, h]
        
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def point_in_bbox(point: List[float], bbox: List[int], margin: int = 0) -> bool:
        """
        Check if point is inside bounding box
        
        Args:
            point: [x, y]
            bbox: [x, y, w, h]
            margin: Extra margin around bbox
        
        Returns:
            True if point is inside bbox
        """
        x, y, w, h = bbox
        px, py = point
        
        return (x - margin <= px <= x + w + margin and
                y - margin <= py <= y + h + margin)
    
    @staticmethod
    def line_intersects_bbox(line: List[List[float]], bbox: List[int]) -> bool:
        """
        Check if line intersects with bounding box
        
        Args:
            line: [[x1, y1], [x2, y2]]
            bbox: [x, y, w, h]
        
        Returns:
            True if line intersects bbox
        """
        x, y, w, h = bbox
        (x1, y1), (x2, y2) = line
        
        # Check if either endpoint is inside bbox
        if (x <= x1 <= x + w and y <= y1 <= y + h) or \
           (x <= x2 <= x + w and y <= y2 <= y + h):
            return True
        
        # Check line-rectangle intersection (simplified)
        # This is a basic check, more sophisticated algorithms exist
        return False


class DataHelpers:
    """Helper functions for data processing"""
    
    @staticmethod
    def normalize_label(label: str) -> str:
        """
        Normalize equipment label
        
        Args:
            label: Raw label string
        
        Returns:
            Normalized label
        """
        # Remove extra whitespace
        label = ' '.join(label.split())
        
        # Convert to uppercase for consistency
        label = label.upper()
        
        # Remove special characters except hyphens and underscores
        label = ''.join(c for c in label if c.isalnum() or c in ['-', '_', ' '])
        
        return label.strip()
    
    @staticmethod
    def merge_overlapping_detections(
        detections: List[Dict],
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Merge overlapping detections using Non-Maximum Suppression
        
        Args:
            detections: List of detection dictionaries with 'bbox' and 'confidence'
            iou_threshold: IoU threshold for merging
        
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        kept = []
        geo_helpers = GeometryHelpers()
        
        for detection in detections:
            bbox = detection.get('bbox')
            if not bbox:
                continue
            
            # Check if it overlaps significantly with any kept detection
            should_keep = True
            for kept_detection in kept:
                kept_bbox = kept_detection.get('bbox')
                if geo_helpers.calculate_iou(bbox, kept_bbox) > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(detection)
        
        return kept
    
    @staticmethod
    def validate_graph_structure(G: Any) -> Dict[str, Any]:
        """
        Validate graph structure and return statistics
        
        Args:
            G: NetworkX graph
        
        Returns:
            Dictionary with validation results
        """
        import networkx as nx
        
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Basic statistics
            results['statistics']['num_nodes'] = G.number_of_nodes()
            results['statistics']['num_edges'] = G.number_of_edges()
            results['statistics']['is_connected'] = nx.is_weakly_connected(G)
            results['statistics']['num_components'] = nx.number_weakly_connected_components(G)
            
            # Check for issues
            isolated = list(nx.isolates(G))
            if isolated:
                results['warnings'].append(f"Found {len(isolated)} isolated nodes")
            
            if not nx.is_weakly_connected(G):
                results['warnings'].append("Graph is not fully connected")
            
            # Check degrees
            for node in G.nodes():
                if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                    results['errors'].append(f"Node {node} has no connections")
                    results['valid'] = False
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Error validating graph: {str(e)}")
        
        return results