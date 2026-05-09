"""
Image Processing Module - Document analysis and feature extraction
Handles document images (ID, statements, etc.) for banking application
Includes fallback if pytesseract unavailable
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Dict, Tuple, Optional

# Try to import pytesseract, but provide fallback
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("⚠️  Pytesseract not available - image OCR will be skipped")

class ImageProcessor:
    """Processes and extracts features from banking document images"""
    
    def __init__(self):
        """Initialize image processing components"""
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.pytesseract_available = PYTESSERACT_AVAILABLE
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path
        Returns: OpenCV image array or None if failed
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        return img
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OCR and feature extraction
        - Grayscale conversion
        - Contrast enhancement
        - Noise reduction
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        return denoised
    
    def extract_text_ocr(self, image: np.ndarray) -> Dict:
        """
        Extract text from document image using OCR
        Returns: {'text': str, 'confidence': float}
        Falls back to empty if pytesseract unavailable
        """
        if not self.pytesseract_available:
            return {'text': '', 'confidence': 0.0, 'word_count': 0}
        
        try:
            # Preprocess image
            processed = self.preprocess_image(image)
            
            # Apply OCR
            extracted_text = pytesseract.image_to_string(processed)
            
            # Get confidence score
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': extracted_text.strip(),
                'confidence': avg_confidence / 100.0,
                'word_count': len(extracted_text.split())
            }
        except Exception as e:
            print(f"OCR Error: {e}")
            return {'text': '', 'confidence': 0.0, 'word_count': 0}
    
    def detect_document_type(self, image: np.ndarray) -> Dict:
        """
        Detect document type based on image characteristics
        Returns: {'type': str, 'is_valid': bool, 'score': float}
        """
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Basic heuristics
        if 0.6 < aspect_ratio < 0.8:  # Portrait orientation
            doc_type = 'ID_CARD'
        elif 0.4 < aspect_ratio < 0.6:  # Very portrait
            doc_type = 'DOCUMENT'
        elif 1.2 < aspect_ratio < 1.5:  # Landscape
            doc_type = 'STATEMENT'
        else:
            doc_type = 'UNKNOWN'
        
        # Check if image is reasonable quality (not too small/large)
        is_valid = width > 200 and height > 200
        
        return {
            'type': doc_type,
            'is_valid': is_valid,
            'aspect_ratio': aspect_ratio
        }
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract numerical features from image
        Returns: [brightness, contrast, edges_density, document_validity]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Brightness (mean pixel value)
        brightness = np.mean(gray) / 255.0
        
        # Contrast (standard deviation)
        contrast = np.std(gray) / 255.0
        
        # Edge density (using Canny edge detection)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Document quality score
        doc_info = self.detect_document_type(image)
        validity_score = 1.0 if doc_info['is_valid'] else 0.0
        
        features = np.array([
            brightness,
            contrast,
            edge_density,
            validity_score
        ])
        
        return features
    
    def analyze_document(self, image_path: str) -> Dict:
        """
        Comprehensive document analysis
        Returns: OCR text, document type, and extracted features
        """
        image = self.load_image(image_path)
        if image is None:
            return {
                'success': False,
                'error': 'Failed to load image'
            }
        
        ocr_result = self.extract_text_ocr(image)
        doc_type = self.detect_document_type(image)
        features = self.extract_features(image)
        
        return {
            'success': True,
            'ocr': ocr_result,
            'document_type': doc_type,
            'image_features': features,
            'quality_score': (ocr_result['confidence'] + doc_type['aspect_ratio']/2) / 2
        }


# Example usage
if __name__ == "__main__":
    processor = ImageProcessor()
    print("Image Processor initialized and ready to use")
    print("Supported formats:", processor.supported_formats)
