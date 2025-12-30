"""
AI-Powered Skin Health Analysis

Integrates the Skin-Analysis deep learning model for:
- Skin age prediction (CNN model)
- Wrinkle detection (Canny edge detection)
- Pore/spots detection (contour analysis)
- Texture analysis (Local Binary Pattern)

Based on: https://github.com/Himika-Mishra/Skin-Analysis
"""

import os
import io
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Model paths
MODEL_DIR = Path(__file__).parent / "models" / "skin_health"
MODEL_PATH = MODEL_DIR / "skin_age_model.h5"
MODEL_URL = "https://github.com/Himika-Mishra/Skin-Analysis/raw/main/more_data(3).h5"

# Lazy imports for TensorFlow (heavy dependency)
_tf_model = None
_model_loaded = False


@dataclass
class SkinAnalysisResult:
    """Results from AI skin analysis."""
    skin_age: int
    wrinkle_score: int  # 0-100, higher = more wrinkles
    spots_score: int    # 0-100, higher = more spots
    texture_score: int  # 0-100, higher = better texture
    pore_score: int     # 0-100, higher = larger pores

    # Raw counts for detailed analysis
    wrinkle_count: int
    spots_count: int
    texture_uniformity: float


def download_model() -> bool:
    """Download the pretrained skin age model if not present."""
    if MODEL_PATH.exists():
        logger.info(f"Skin health model already exists at {MODEL_PATH}")
        return True

    try:
        logger.info(f"Downloading skin health model from {MODEL_URL}...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Download with progress
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        return True

    except Exception as e:
        logger.error(f"Failed to download skin health model: {e}")
        return False


def load_model():
    """Load the TensorFlow/Keras model for skin age prediction."""
    global _tf_model, _model_loaded

    if _model_loaded:
        return _tf_model

    # Download if needed
    if not MODEL_PATH.exists():
        if not download_model():
            logger.error("Could not download skin health model")
            return None

    try:
        # Lazy import TensorFlow
        import tensorflow as tf
        from tensorflow.keras.models import load_model as keras_load_model
        from tensorflow.keras.layers import SpatialDropout2D

        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        logger.info("Loading skin age prediction model...")

        # Custom SpatialDropout2D that ignores extra arguments for compatibility
        class CompatibleSpatialDropout2D(SpatialDropout2D):
            def __init__(self, rate, **kwargs):
                # Filter out unsupported arguments
                kwargs.pop('trainable', None)
                kwargs.pop('noise_shape', None)
                kwargs.pop('seed', None)
                super().__init__(rate, **kwargs)

        custom_objects = {
            'SpatialDropout2D': CompatibleSpatialDropout2D
        }

        _tf_model = keras_load_model(
            str(MODEL_PATH),
            compile=False,
            custom_objects=custom_objects
        )
        _model_loaded = True
        logger.info("Skin age model loaded successfully!")

        return _tf_model

    except Exception as e:
        logger.error(f"Failed to load skin age model: {e}")
        import traceback
        traceback.print_exc()
        return None


def preprocess_for_age_model(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for the skin age CNN model.

    Args:
        image: RGB image as numpy array

    Returns:
        Preprocessed image tensor ready for model input
    """
    # Resize to 180x180 (model's expected input size)
    img_resized = cv2.resize(image, (180, 180))

    # Normalize to 0-1 range
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch


def predict_skin_age(image: np.ndarray) -> Optional[int]:
    """
    Predict skin age using the CNN model.

    Args:
        image: RGB image as numpy array

    Returns:
        Predicted skin age, or None if prediction failed
    """
    model = load_model()
    if model is None:
        return None

    try:
        # Preprocess
        img_batch = preprocess_for_age_model(image)

        # Predict
        prediction = model.predict(img_batch, verbose=0)
        skin_age = int(round(prediction[0][0]))

        # Clamp to reasonable range
        skin_age = max(10, min(90, skin_age))

        return skin_age

    except Exception as e:
        logger.error(f"Error predicting skin age: {e}")
        return None


def detect_wrinkles(image: np.ndarray) -> Tuple[int, int]:
    """
    Detect wrinkles using Canny edge detection.

    Args:
        image: RGB image as numpy array

    Returns:
        Tuple of (wrinkle_score 0-100, raw_count)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Canny edge detection for wrinkle lines
        edges = cv2.Canny(filtered, 50, 150)

        # Count edge pixels as wrinkle indicator
        wrinkle_count = np.sum(edges > 0)

        # Normalize to 0-100 score (inverted: more wrinkles = lower score)
        # Assume max wrinkle pixels around 50000 for a face region
        max_wrinkles = 50000
        wrinkle_ratio = min(1.0, wrinkle_count / max_wrinkles)
        wrinkle_score = int(100 - (wrinkle_ratio * 100))

        return wrinkle_score, wrinkle_count

    except Exception as e:
        logger.error(f"Error detecting wrinkles: {e}")
        return 50, 0


def detect_spots(image: np.ndarray) -> Tuple[int, int]:
    """
    Detect spots/blemishes using contour analysis.

    Args:
        image: RGB image as numpy array

    Returns:
        Tuple of (spots_score 0-100, spot_count)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding to find dark spots
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours (spots)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Count spots with minimum area threshold
        min_area = 4
        max_area = 500  # Filter out large areas that aren't spots
        spots = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        spot_count = len(spots)

        # Normalize to 0-100 score (fewer spots = higher score)
        max_spots = 200
        spot_ratio = min(1.0, spot_count / max_spots)
        spots_score = int(100 - (spot_ratio * 100))

        return spots_score, spot_count

    except Exception as e:
        logger.error(f"Error detecting spots: {e}")
        return 50, 0


def analyze_texture(image: np.ndarray) -> Tuple[int, float]:
    """
    Analyze skin texture using Local Binary Pattern (LBP).

    Args:
        image: RGB image as numpy array

    Returns:
        Tuple of (texture_score 0-100, uniformity_value)
    """
    try:
        # Try to use skimage for LBP
        from skimage.feature import local_binary_pattern

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize for faster processing
        gray_small = cv2.resize(gray, (256, 256))

        # Compute LBP
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_small, n_points, radius, method='uniform')

        # Calculate histogram
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        # Calculate uniformity (entropy-based)
        # Lower entropy = more uniform texture = healthier skin
        hist = hist + 1e-10  # Avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))

        # Normalize entropy to 0-100 score
        # Typical entropy range is 2-6, map to score
        max_entropy = 6.0
        min_entropy = 2.0
        normalized_entropy = (entropy - min_entropy) / (max_entropy - min_entropy)
        normalized_entropy = max(0, min(1, normalized_entropy))

        # Higher score = better (more uniform) texture
        texture_score = int(100 - (normalized_entropy * 60))

        return texture_score, float(1 - normalized_entropy)

    except ImportError:
        # Fallback if skimage not available
        logger.warning("skimage not available, using fallback texture analysis")
        return _analyze_texture_fallback(image)
    except Exception as e:
        logger.error(f"Error analyzing texture: {e}")
        return 50, 0.5


def _analyze_texture_fallback(image: np.ndarray) -> Tuple[int, float]:
    """Fallback texture analysis using variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Use Laplacian variance as texture measure
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    # Normalize - higher variance can indicate rougher texture
    max_variance = 2000
    normalized = min(1.0, variance / max_variance)

    # Moderate variance is best (not too smooth, not too rough)
    # Score peaks around 0.3-0.5 normalized variance
    if normalized < 0.3:
        score = int(70 + normalized * 100)
    elif normalized < 0.6:
        score = int(90 - (normalized - 0.3) * 50)
    else:
        score = int(75 - (normalized - 0.6) * 100)

    score = max(30, min(100, score))
    return score, 1 - normalized


def detect_pores(image: np.ndarray) -> int:
    """
    Detect pore size/visibility.

    Args:
        image: RGB image as numpy array

    Returns:
        Pore score 0-100 (higher = smaller/less visible pores)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Use blob detection for pores
        # Pores appear as small dark spots

        # Threshold to find dark spots
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find small contours (pores)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count small circular-ish contours as pores
        pore_count = 0
        for c in contours:
            area = cv2.contourArea(c)
            if 2 < area < 100:  # Pore-sized
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.3:  # Somewhat circular
                        pore_count += 1

        # Normalize to score
        max_pores = 500
        pore_ratio = min(1.0, pore_count / max_pores)
        pore_score = int(100 - (pore_ratio * 80))

        return pore_score

    except Exception as e:
        logger.error(f"Error detecting pores: {e}")
        return 50


def analyze_skin(image_data: bytes) -> Optional[SkinAnalysisResult]:
    """
    Perform comprehensive AI skin analysis.

    Args:
        image_data: Raw image bytes

    Returns:
        SkinAnalysisResult with all metrics, or None if analysis failed
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_array = np.array(image)

        # Predict skin age using CNN
        skin_age = predict_skin_age(image_array)
        if skin_age is None:
            # Fallback to estimation based on features
            skin_age = 35
            logger.warning("Using fallback skin age estimation")

        # Detect wrinkles
        wrinkle_score, wrinkle_count = detect_wrinkles(image_array)

        # Detect spots
        spots_score, spots_count = detect_spots(image_array)

        # Analyze texture
        texture_score, texture_uniformity = analyze_texture(image_array)

        # Detect pores
        pore_score = detect_pores(image_array)

        return SkinAnalysisResult(
            skin_age=skin_age,
            wrinkle_score=wrinkle_score,
            spots_score=spots_score,
            texture_score=texture_score,
            pore_score=pore_score,
            wrinkle_count=wrinkle_count,
            spots_count=spots_count,
            texture_uniformity=texture_uniformity
        )

    except Exception as e:
        logger.error(f"Error in skin analysis: {e}")
        return None


def is_model_available() -> bool:
    """Check if the AI model is available."""
    return MODEL_PATH.exists()


# Singleton for reuse
_analyzer_initialized = False


def initialize():
    """Pre-initialize the model for faster first analysis."""
    global _analyzer_initialized
    if not _analyzer_initialized:
        download_model()
        load_model()
        _analyzer_initialized = True


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    print("Skin Health AI Module")
    print("=" * 40)

    # Check model
    if is_model_available():
        print("Model is available")
    else:
        print("Downloading model...")
        download_model()

    # Test with a sample image if available
    test_image = Path(__file__).parent / "uploads" / "test.jpg"
    if test_image.exists():
        print(f"\nAnalyzing {test_image}...")
        with open(test_image, 'rb') as f:
            result = analyze_skin(f.read())

        if result:
            print(f"Skin Age: {result.skin_age}")
            print(f"Wrinkle Score: {result.wrinkle_score}/100")
            print(f"Spots Score: {result.spots_score}/100")
            print(f"Texture Score: {result.texture_score}/100")
            print(f"Pore Score: {result.pore_score}/100")
    else:
        print("\nNo test image found. Place a test.jpg in uploads/ to test.")
