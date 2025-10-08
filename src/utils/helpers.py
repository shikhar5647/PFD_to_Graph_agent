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
