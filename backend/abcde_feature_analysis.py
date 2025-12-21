"""
ABCD(E) Feature Analysis Module

Provides quantitative analysis of the ABCDE criteria for melanoma detection:
- A: Asymmetry (shape asymmetry score)
- B: Border (irregularity, notching, blurring)
- C: Color (variance, number of colors, distribution)
- D: Diameter (estimated size in mm if calibration available)
- E: Evolution (requires longitudinal data - placeholder for comparison)

This module gives dermatologists the specific features driving AI decisions,
addressing the "explainability I trust" requirement.

All metrics are computed using standard image analysis techniques and
are designed to be clinically interpretable.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math


class RiskLevel(str, Enum):
    """Risk level for individual features"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AsymmetryAnalysis:
    """Asymmetry (A) analysis results"""
    overall_score: float  # 0-1, higher = more asymmetric
    horizontal_asymmetry: float  # Left-right asymmetry
    vertical_asymmetry: float  # Top-bottom asymmetry
    shape_asymmetry: float  # Overall shape irregularity
    color_asymmetry: float  # Color distribution asymmetry
    risk_level: RiskLevel
    description: str
    clinical_interpretation: str


@dataclass
class BorderAnalysis:
    """Border (B) analysis results"""
    overall_score: float  # 0-1, higher = more irregular
    irregularity_index: float  # Border irregularity measure
    notching_score: float  # Presence of notches/indentations
    blur_score: float  # Border definition/sharpness
    radial_variance: float  # Variance in border distance from center
    num_border_colors: int  # Color transitions at border
    risk_level: RiskLevel
    description: str
    clinical_interpretation: str


@dataclass
class ColorAnalysis:
    """Color (C) analysis results"""
    overall_score: float  # 0-1, higher = more concerning
    num_colors: int  # Number of distinct colors detected
    colors_detected: List[str]  # Named colors present
    color_variance: float  # Overall color heterogeneity
    has_blue_white_veil: bool  # Concerning dermoscopic feature
    has_regression: bool  # White/gray regression areas
    dominant_color: str  # Primary color
    color_distribution: Dict[str, float]  # Percentage of each color
    risk_level: RiskLevel
    description: str
    clinical_interpretation: str


@dataclass
class DiameterAnalysis:
    """Diameter (D) analysis results"""
    overall_score: float  # 0-1, higher = larger/more concerning
    estimated_diameter_mm: Optional[float]  # If calibration available
    pixel_diameter: int  # Diameter in pixels
    area_pixels: int  # Area in pixels
    is_above_6mm: Optional[bool]  # Clinical threshold
    calibration_available: bool
    risk_level: RiskLevel
    description: str
    clinical_interpretation: str


@dataclass
class EvolutionAnalysis:
    """Evolution (E) analysis results - requires longitudinal data"""
    has_comparison: bool
    change_detected: Optional[bool]
    change_description: Optional[str]
    risk_level: RiskLevel
    description: str
    clinical_interpretation: str


@dataclass
class ABCDEAnalysis:
    """Complete ABCDE analysis"""
    asymmetry: AsymmetryAnalysis
    border: BorderAnalysis
    color: ColorAnalysis
    diameter: DiameterAnalysis
    evolution: EvolutionAnalysis

    # Overall assessment
    total_score: float  # Weighted combination (0-10 scale, like TDS)
    risk_level: RiskLevel
    summary: str
    key_concerns: List[str]
    recommendation: str

    # For transparency
    methodology_notes: List[str]


def segment_lesion(image: Image.Image) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Segment the lesion from background using adaptive thresholding and morphology.
    Returns binary mask, contour, and metadata.
    """
    # Convert to numpy array
    img_array = np.array(image)

    # Convert to different color spaces
    if len(img_array.shape) == 2:
        gray = img_array
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's thresholding for automatic threshold selection
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: assume center region is lesion
        h, w = gray.shape
        center_mask = np.zeros_like(gray)
        cv2.ellipse(center_mask, (w//2, h//2), (w//4, h//4), 0, 0, 360, 255, -1)
        contours, _ = cv2.findContours(center_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        binary = center_mask

    # Get the largest contour (assumed to be the lesion)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create clean mask from largest contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    # Calculate metadata
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Centroid
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2

    metadata = {
        "area": area,
        "perimeter": perimeter,
        "bounding_box": (x, y, w, h),
        "centroid": (cx, cy),
        "aspect_ratio": w / h if h > 0 else 1.0,
    }

    return mask, largest_contour, metadata


def analyze_asymmetry(image: Image.Image, mask: np.ndarray, contour: np.ndarray) -> AsymmetryAnalysis:
    """
    Analyze asymmetry by comparing halves of the lesion.
    Uses both shape and color distribution.
    """
    img_array = np.array(image)
    h, w = mask.shape

    # Get centroid for splitting
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2

    # === Shape Asymmetry ===
    # Horizontal split (left vs right)
    left_half = mask[:, :cx]
    right_half = mask[:, cx:]
    right_flipped = cv2.flip(right_half, 1)

    # Resize to match if needed
    min_w = min(left_half.shape[1], right_flipped.shape[1])
    left_half = left_half[:, :min_w] if left_half.shape[1] > min_w else left_half
    right_flipped = right_flipped[:, :min_w] if right_flipped.shape[1] > min_w else right_flipped

    # Pad to match heights
    if left_half.shape[0] != right_flipped.shape[0]:
        max_h = max(left_half.shape[0], right_flipped.shape[0])
        left_half = np.pad(left_half, ((0, max_h - left_half.shape[0]), (0, 0)))
        right_flipped = np.pad(right_flipped, ((0, max_h - right_flipped.shape[0]), (0, 0)))

    # Calculate horizontal asymmetry
    diff_h = cv2.absdiff(left_half, right_flipped)
    horizontal_asymmetry = np.sum(diff_h > 0) / max(np.sum(mask > 0), 1)

    # Vertical split (top vs bottom)
    top_half = mask[:cy, :]
    bottom_half = mask[cy:, :]
    bottom_flipped = cv2.flip(bottom_half, 0)

    min_h = min(top_half.shape[0], bottom_flipped.shape[0])
    top_half = top_half[:min_h, :] if top_half.shape[0] > min_h else top_half
    bottom_flipped = bottom_flipped[:min_h, :] if bottom_flipped.shape[0] > min_h else bottom_flipped

    if top_half.shape[1] != bottom_flipped.shape[1]:
        max_w = max(top_half.shape[1], bottom_flipped.shape[1])
        top_half = np.pad(top_half, ((0, 0), (0, max_w - top_half.shape[1])))
        bottom_flipped = np.pad(bottom_flipped, ((0, 0), (0, max_w - bottom_flipped.shape[1])))

    diff_v = cv2.absdiff(top_half, bottom_flipped)
    vertical_asymmetry = np.sum(diff_v > 0) / max(np.sum(mask > 0), 1)

    # === Color Asymmetry ===
    color_asymmetry = 0.0
    if len(img_array.shape) == 3:
        # Compare color histograms of left vs right halves
        left_region = img_array[:, :cx][mask[:, :cx] > 0]
        right_region = img_array[:, cx:][mask[:, cx:] > 0]

        if len(left_region) > 0 and len(right_region) > 0:
            left_mean = np.mean(left_region, axis=0)
            right_mean = np.mean(right_region, axis=0)
            color_asymmetry = np.linalg.norm(left_mean - right_mean) / 255.0

    # === Overall Shape Asymmetry ===
    # Use circularity and convexity
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter > 0 else 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1

    shape_asymmetry = 1 - (circularity * 0.5 + solidity * 0.5)

    # === Overall Score ===
    overall_score = (
        horizontal_asymmetry * 0.3 +
        vertical_asymmetry * 0.3 +
        shape_asymmetry * 0.25 +
        color_asymmetry * 0.15
    )
    overall_score = min(max(overall_score, 0), 1)

    # Determine risk level
    if overall_score >= 0.6:
        risk_level = RiskLevel.HIGH
    elif overall_score >= 0.4:
        risk_level = RiskLevel.MODERATE
    elif overall_score >= 0.2:
        risk_level = RiskLevel.LOW
    else:
        risk_level = RiskLevel.LOW

    # Generate descriptions
    descriptions = []
    if horizontal_asymmetry > 0.3:
        descriptions.append("significant left-right asymmetry")
    if vertical_asymmetry > 0.3:
        descriptions.append("significant top-bottom asymmetry")
    if color_asymmetry > 0.2:
        descriptions.append("asymmetric color distribution")

    description = ", ".join(descriptions) if descriptions else "relatively symmetric"

    clinical_interpretation = {
        RiskLevel.HIGH: "Marked asymmetry in 2+ axes - concerning for dysplasia or malignancy",
        RiskLevel.MODERATE: "Moderate asymmetry present - warrants monitoring",
        RiskLevel.LOW: "Mild asymmetry - within normal variation for benign lesions",
    }.get(risk_level, "Unable to assess asymmetry")

    return AsymmetryAnalysis(
        overall_score=round(overall_score, 3),
        horizontal_asymmetry=round(horizontal_asymmetry, 3),
        vertical_asymmetry=round(vertical_asymmetry, 3),
        shape_asymmetry=round(shape_asymmetry, 3),
        color_asymmetry=round(color_asymmetry, 3),
        risk_level=risk_level,
        description=description.capitalize(),
        clinical_interpretation=clinical_interpretation
    )


def analyze_border(image: Image.Image, mask: np.ndarray, contour: np.ndarray) -> BorderAnalysis:
    """
    Analyze border characteristics including irregularity, notching, and blur.
    """
    img_array = np.array(image)

    # === Border Irregularity Index ===
    # Compare actual perimeter to convex hull perimeter
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)

    irregularity_index = (perimeter / hull_perimeter - 1) if hull_perimeter > 0 else 0
    irregularity_index = min(irregularity_index, 1)  # Cap at 1

    # === Notching Score ===
    # Detect concavities using convexity defects
    hull_indices = cv2.convexHull(contour, returnPoints=False)

    notching_score = 0.0
    if len(hull_indices) > 3 and len(contour) > 3:
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                significant_defects = 0
                for i in range(defects.shape[0]):
                    _, _, _, depth = defects[i, 0]
                    # Convert depth from fixed-point to float
                    depth_float = depth / 256.0
                    if depth_float > 5:  # Significant notch
                        significant_defects += 1
                notching_score = min(significant_defects / 10.0, 1.0)
        except:
            notching_score = 0.0

    # === Border Blur Score ===
    # Analyze gradient magnitude at border
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array

    # Dilate and erode mask to get border region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    eroded = cv2.erode(mask, kernel, iterations=2)
    border_region = dilated - eroded

    # Calculate gradient at border
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    border_gradient = gradient_magnitude[border_region > 0]
    if len(border_gradient) > 0:
        mean_gradient = np.mean(border_gradient)
        # Normalize: higher gradient = sharper border = lower blur
        blur_score = 1 - min(mean_gradient / 100, 1)
    else:
        blur_score = 0.5

    # === Radial Variance ===
    # Measure variance in distance from centroid to border points
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = img_array.shape[1] // 2, img_array.shape[0] // 2

    distances = []
    for point in contour:
        px, py = point[0]
        dist = math.sqrt((px - cx)**2 + (py - cy)**2)
        distances.append(dist)

    if distances:
        mean_dist = np.mean(distances)
        radial_variance = np.std(distances) / mean_dist if mean_dist > 0 else 0
    else:
        radial_variance = 0

    # === Border Color Transitions ===
    num_border_colors = 1
    if len(img_array.shape) == 3:
        border_pixels = img_array[border_region > 0]
        if len(border_pixels) > 10:
            # Simple color quantization
            border_pixels_float = border_pixels.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = min(5, len(border_pixels) // 10)
            if k >= 2:
                _, labels, _ = cv2.kmeans(border_pixels_float, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                num_border_colors = len(np.unique(labels))

    # === Overall Score ===
    overall_score = (
        irregularity_index * 0.35 +
        notching_score * 0.25 +
        blur_score * 0.15 +
        min(radial_variance, 1) * 0.25
    )
    overall_score = min(max(overall_score, 0), 1)

    # Determine risk level
    if overall_score >= 0.5:
        risk_level = RiskLevel.HIGH
    elif overall_score >= 0.3:
        risk_level = RiskLevel.MODERATE
    else:
        risk_level = RiskLevel.LOW

    # Generate descriptions
    descriptions = []
    if irregularity_index > 0.3:
        descriptions.append("irregular contour")
    if notching_score > 0.3:
        descriptions.append("notched/scalloped edges")
    if blur_score > 0.5:
        descriptions.append("poorly defined margins")
    if radial_variance > 0.4:
        descriptions.append("asymmetric border distances")

    description = ", ".join(descriptions) if descriptions else "well-defined, regular border"

    clinical_interpretation = {
        RiskLevel.HIGH: "Irregular, notched borders with poor definition - concerning feature",
        RiskLevel.MODERATE: "Some border irregularity present - monitor for changes",
        RiskLevel.LOW: "Regular, well-defined borders - reassuring feature",
    }.get(risk_level, "Unable to assess border")

    return BorderAnalysis(
        overall_score=round(overall_score, 3),
        irregularity_index=round(irregularity_index, 3),
        notching_score=round(notching_score, 3),
        blur_score=round(blur_score, 3),
        radial_variance=round(radial_variance, 3),
        num_border_colors=num_border_colors,
        risk_level=risk_level,
        description=description.capitalize(),
        clinical_interpretation=clinical_interpretation
    )


def analyze_color(image: Image.Image, mask: np.ndarray) -> ColorAnalysis:
    """
    Analyze color characteristics including number of colors, distribution, and concerning patterns.
    """
    img_array = np.array(image)

    if len(img_array.shape) != 3:
        # Grayscale image - limited color analysis
        return ColorAnalysis(
            overall_score=0.0,
            num_colors=1,
            colors_detected=["grayscale"],
            color_variance=0.0,
            has_blue_white_veil=False,
            has_regression=False,
            dominant_color="gray",
            color_distribution={"gray": 100.0},
            risk_level=RiskLevel.LOW,
            description="Grayscale image - color analysis limited",
            clinical_interpretation="Unable to assess color features in grayscale image"
        )

    # Get lesion pixels only
    lesion_pixels = img_array[mask > 0]

    if len(lesion_pixels) == 0:
        return ColorAnalysis(
            overall_score=0.0,
            num_colors=0,
            colors_detected=[],
            color_variance=0.0,
            has_blue_white_veil=False,
            has_regression=False,
            dominant_color="unknown",
            color_distribution={},
            risk_level=RiskLevel.LOW,
            description="No lesion pixels detected",
            clinical_interpretation="Unable to analyze color - no lesion segmented"
        )

    # Convert to different color spaces for analysis
    hsv_pixels = cv2.cvtColor(lesion_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # === Color Classification ===
    # Define color ranges in HSV
    color_ranges = {
        "tan/light_brown": {"h": (10, 30), "s": (30, 150), "v": (150, 255)},
        "dark_brown": {"h": (10, 30), "s": (50, 200), "v": (50, 150)},
        "black": {"h": (0, 180), "s": (0, 255), "v": (0, 50)},
        "white": {"h": (0, 180), "s": (0, 30), "v": (200, 255)},
        "red": {"h": (0, 10), "s": (100, 255), "v": (100, 255)},
        "red_2": {"h": (170, 180), "s": (100, 255), "v": (100, 255)},
        "blue_gray": {"h": (100, 130), "s": (20, 100), "v": (80, 180)},
        "pink": {"h": (0, 20), "s": (20, 100), "v": (180, 255)},
    }

    colors_detected = []
    color_distribution = {}
    total_pixels = len(hsv_pixels)

    for color_name, ranges in color_ranges.items():
        h_min, h_max = ranges["h"]
        s_min, s_max = ranges["s"]
        v_min, v_max = ranges["v"]

        in_range = (
            (hsv_pixels[:, 0] >= h_min) & (hsv_pixels[:, 0] <= h_max) &
            (hsv_pixels[:, 1] >= s_min) & (hsv_pixels[:, 1] <= s_max) &
            (hsv_pixels[:, 2] >= v_min) & (hsv_pixels[:, 2] <= v_max)
        )

        percentage = np.sum(in_range) / total_pixels * 100

        if percentage > 5:  # At least 5% of pixels
            # Merge red ranges
            if color_name == "red_2":
                if "red" in color_distribution:
                    color_distribution["red"] += percentage
                else:
                    color_distribution["red"] = percentage
                    colors_detected.append("red")
            else:
                color_distribution[color_name] = percentage
                colors_detected.append(color_name)

    num_colors = len(colors_detected)

    # === Color Variance ===
    color_variance = np.std(lesion_pixels.reshape(-1, 3), axis=0).mean() / 255.0

    # === Blue-White Veil Detection ===
    # Blue-gray pixels with low saturation
    blue_gray_mask = (
        (hsv_pixels[:, 0] >= 100) & (hsv_pixels[:, 0] <= 130) &
        (hsv_pixels[:, 1] >= 20) & (hsv_pixels[:, 1] <= 100) &
        (hsv_pixels[:, 2] >= 80) & (hsv_pixels[:, 2] <= 180)
    )
    has_blue_white_veil = np.sum(blue_gray_mask) / total_pixels > 0.1

    # === Regression Detection ===
    # White/gray areas within the lesion
    white_gray_mask = (
        (hsv_pixels[:, 1] <= 30) &
        (hsv_pixels[:, 2] >= 180)
    )
    has_regression = np.sum(white_gray_mask) / total_pixels > 0.05

    # === Dominant Color ===
    dominant_color = max(color_distribution, key=color_distribution.get) if color_distribution else "unknown"

    # === Overall Score ===
    # More colors and specific concerning features increase score
    color_count_score = min(num_colors / 5, 1)  # 5+ colors = max score
    concerning_features = (0.3 if has_blue_white_veil else 0) + (0.2 if has_regression else 0)

    overall_score = (
        color_count_score * 0.4 +
        color_variance * 0.3 +
        concerning_features +
        (0.1 if "black" in colors_detected else 0)
    )
    overall_score = min(max(overall_score, 0), 1)

    # Determine risk level
    if overall_score >= 0.6 or num_colors >= 4 or has_blue_white_veil:
        risk_level = RiskLevel.HIGH
    elif overall_score >= 0.4 or num_colors >= 3:
        risk_level = RiskLevel.MODERATE
    else:
        risk_level = RiskLevel.LOW

    # Generate descriptions
    descriptions = []
    descriptions.append(f"{num_colors} distinct color(s)")
    if has_blue_white_veil:
        descriptions.append("blue-white veil present")
    if has_regression:
        descriptions.append("possible regression areas")
    if "black" in colors_detected:
        descriptions.append("black pigmentation present")

    description = ", ".join(descriptions)

    clinical_interpretation = {
        RiskLevel.HIGH: f"Multiple colors ({num_colors}) with concerning features - high suspicion for melanoma",
        RiskLevel.MODERATE: f"Color heterogeneity present ({num_colors} colors) - warrants closer evaluation",
        RiskLevel.LOW: "Uniform coloration - reassuring feature for benign lesion",
    }.get(risk_level, "Unable to assess color")

    return ColorAnalysis(
        overall_score=round(overall_score, 3),
        num_colors=num_colors,
        colors_detected=colors_detected,
        color_variance=round(color_variance, 3),
        has_blue_white_veil=has_blue_white_veil,
        has_regression=has_regression,
        dominant_color=dominant_color,
        color_distribution={k: round(v, 1) for k, v in color_distribution.items()},
        risk_level=risk_level,
        description=description,
        clinical_interpretation=clinical_interpretation
    )


def analyze_diameter(
    image: Image.Image,
    mask: np.ndarray,
    contour: np.ndarray,
    pixels_per_mm: Optional[float] = None
) -> DiameterAnalysis:
    """
    Analyze diameter/size of the lesion.
    If calibration is available (pixels_per_mm), provides measurement in mm.
    """
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Use maximum dimension as diameter (more conservative)
    pixel_diameter = max(w, h)

    # Area in pixels
    area_pixels = cv2.contourArea(contour)

    # Calculate in mm if calibration available
    estimated_diameter_mm = None
    is_above_6mm = None
    calibration_available = pixels_per_mm is not None and pixels_per_mm > 0

    if calibration_available:
        estimated_diameter_mm = pixel_diameter / pixels_per_mm
        is_above_6mm = estimated_diameter_mm > 6.0

    # === Overall Score ===
    # Based on relative size in image (larger lesion = potentially more concerning)
    img_array = np.array(image)
    image_area = img_array.shape[0] * img_array.shape[1]
    relative_size = area_pixels / image_area

    if calibration_available and estimated_diameter_mm:
        # Use actual mm measurement
        if estimated_diameter_mm > 10:
            overall_score = 0.9
        elif estimated_diameter_mm > 6:
            overall_score = 0.6
        elif estimated_diameter_mm > 4:
            overall_score = 0.3
        else:
            overall_score = 0.1
    else:
        # Use relative size as proxy
        overall_score = min(relative_size * 10, 1)  # Rough scaling

    # Determine risk level
    if is_above_6mm or (calibration_available and estimated_diameter_mm and estimated_diameter_mm > 10):
        risk_level = RiskLevel.HIGH
    elif (calibration_available and estimated_diameter_mm and estimated_diameter_mm > 6) or relative_size > 0.15:
        risk_level = RiskLevel.MODERATE
    else:
        risk_level = RiskLevel.LOW

    # Generate description
    if calibration_available and estimated_diameter_mm:
        description = f"Estimated diameter: {estimated_diameter_mm:.1f}mm"
        if is_above_6mm:
            description += " (exceeds 6mm threshold)"
    else:
        description = f"Diameter: {pixel_diameter}px (calibration not available for mm measurement)"

    clinical_interpretation = {
        RiskLevel.HIGH: "Large lesion (>6mm) - size alone warrants evaluation",
        RiskLevel.MODERATE: "Moderate size - consider in context of other features",
        RiskLevel.LOW: "Small lesion - size not a concerning feature",
    }.get(risk_level, "Unable to assess diameter")

    return DiameterAnalysis(
        overall_score=round(overall_score, 3),
        estimated_diameter_mm=round(estimated_diameter_mm, 1) if estimated_diameter_mm else None,
        pixel_diameter=pixel_diameter,
        area_pixels=int(area_pixels),
        is_above_6mm=is_above_6mm,
        calibration_available=calibration_available,
        risk_level=risk_level,
        description=description,
        clinical_interpretation=clinical_interpretation
    )


def analyze_evolution(
    current_image: Image.Image,
    previous_image: Optional[Image.Image] = None,
    previous_analysis: Optional[Dict] = None
) -> EvolutionAnalysis:
    """
    Analyze evolution/change over time.
    Requires previous image or analysis data for comparison.
    """
    if previous_image is None and previous_analysis is None:
        return EvolutionAnalysis(
            has_comparison=False,
            change_detected=None,
            change_description=None,
            risk_level=RiskLevel.MODERATE,  # Unknown = moderate concern
            description="No prior images available for comparison",
            clinical_interpretation="Evolution cannot be assessed without prior images. Patient-reported history of change is important."
        )

    # If we have previous analysis data, compare metrics
    if previous_analysis:
        # This would compare ABCDE scores from previous analysis
        # For now, placeholder
        return EvolutionAnalysis(
            has_comparison=True,
            change_detected=None,
            change_description="Comparison with prior analysis available",
            risk_level=RiskLevel.MODERATE,
            description="Prior analysis data available for comparison",
            clinical_interpretation="Review prior analysis metrics for changes in asymmetry, border, color, or size"
        )

    # If we have previous image, perform comparison
    # This is a simplified comparison - a full implementation would align images and compare features
    return EvolutionAnalysis(
        has_comparison=True,
        change_detected=None,
        change_description="Image comparison available",
        risk_level=RiskLevel.MODERATE,
        description="Prior image available for comparison",
        clinical_interpretation="Compare current features with prior image for any changes"
    )


def calculate_total_dermoscopy_score(
    asymmetry: AsymmetryAnalysis,
    border: BorderAnalysis,
    color: ColorAnalysis,
    diameter: DiameterAnalysis
) -> Tuple[float, RiskLevel]:
    """
    Calculate a total dermoscopy score (TDS) similar to the ABCD rule scoring.

    Traditional TDS formula:
    TDS = (A × 1.3) + (B × 0.1) + (C × 0.5) + (D × 0.5)

    Where:
    - A: 0-2 (asymmetry in 0, 1, or 2 axes)
    - B: 0-8 (border score)
    - C: 1-6 (number of colors)
    - D: 1-5 (structural components) - we use diameter proxy

    TDS interpretation:
    - < 4.75: benign
    - 4.75-5.45: suspicious
    - > 5.45: high suspicion for melanoma

    We adapt this to our 0-1 scores.

    NOTE: This function intentionally does NOT factor in ML classification results.
    The ABCDE score should reflect purely image-based feature analysis for
    regulatory transparency. Combined risk assessment is handled separately.
    """
    # Convert our 0-1 scores to traditional scale
    a_score = asymmetry.overall_score * 2  # 0-2
    b_score = border.overall_score * 8  # 0-8
    c_score = min(color.num_colors, 6)  # 1-6
    d_score = diameter.overall_score * 5  # 0-5

    # Traditional TDS calculation
    tds = (a_score * 1.3) + (b_score * 0.1) + (c_score * 0.5) + (d_score * 0.5)

    # Normalize to 0-10 scale for display
    normalized_score = min(tds, 10)

    # Determine risk level based on TDS thresholds ONLY
    if tds > 5.45:
        risk_level = RiskLevel.VERY_HIGH
    elif tds > 4.75:
        risk_level = RiskLevel.HIGH
    elif tds > 3.5:
        risk_level = RiskLevel.MODERATE
    else:
        risk_level = RiskLevel.LOW

    return normalized_score, risk_level


def perform_abcde_analysis(
    image: Image.Image,
    pixels_per_mm: Optional[float] = None,
    previous_image: Optional[Image.Image] = None,
    previous_analysis: Optional[Dict] = None
) -> ABCDEAnalysis:
    """
    Perform complete ABCDE analysis on a lesion image.

    This function performs PURE image-based feature analysis without any
    influence from ML classification results. This separation is intentional
    for regulatory transparency - the ABCDE analysis should honestly reflect
    what can be determined from traditional image feature analysis alone.

    Args:
        image: PIL Image of the lesion
        pixels_per_mm: Calibration factor if available
        previous_image: Prior image for evolution comparison
        previous_analysis: Prior ABCDE analysis results for comparison

    Returns:
        Complete ABCDEAnalysis with all features and recommendations
    """
    # Segment the lesion
    mask, contour, metadata = segment_lesion(image)

    # Perform individual analyses
    asymmetry = analyze_asymmetry(image, mask, contour)
    border = analyze_border(image, mask, contour)
    color = analyze_color(image, mask)
    diameter = analyze_diameter(image, mask, contour, pixels_per_mm)
    evolution = analyze_evolution(image, previous_image, previous_analysis)

    # Calculate total score based purely on image features
    total_score, overall_risk = calculate_total_dermoscopy_score(
        asymmetry, border, color, diameter
    )

    # Compile key concerns
    key_concerns = []
    if asymmetry.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        key_concerns.append(f"Asymmetry: {asymmetry.description}")
    if border.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        key_concerns.append(f"Border: {border.description}")
    if color.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        key_concerns.append(f"Color: {color.description}")
    if diameter.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
        key_concerns.append(f"Diameter: {diameter.description}")
    if color.has_blue_white_veil:
        key_concerns.append("Blue-white veil detected - concerning dermoscopic feature")
    if color.has_regression:
        key_concerns.append("Regression areas detected")

    # Generate summary based ONLY on image feature analysis
    high_risk_count = sum(1 for a in [asymmetry, border, color, diameter]
                         if a.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH])

    if high_risk_count >= 3:
        summary = f"Multiple concerning ABCD features ({high_risk_count}/4) - high suspicion for dysplasia or melanoma"
    elif high_risk_count >= 2:
        summary = f"Some concerning features ({high_risk_count}/4) - warrants dermatology evaluation"
    elif high_risk_count == 1:
        summary = "One concerning feature identified - consider monitoring or evaluation based on clinical context"
    else:
        summary = "No highly concerning features identified based on traditional ABCD criteria"

    # Generate recommendation based ONLY on image features
    if overall_risk == RiskLevel.VERY_HIGH:
        recommendation = "Image features suggest high risk. Dermatology evaluation recommended."
    elif overall_risk == RiskLevel.HIGH:
        recommendation = "Some concerning image features. Consider dermatology evaluation."
    elif overall_risk == RiskLevel.MODERATE:
        recommendation = "Moderate features. Photo-document and reassess in 2-3 months."
    else:
        recommendation = "Image features consistent with benign lesion based on ABCD criteria."

    # Methodology notes - emphasize this is pure image analysis
    methodology_notes = [
        "Asymmetry: Calculated by comparing mirrored halves of segmented lesion (shape and color)",
        "Border: Analyzed using contour irregularity, convexity defects, and gradient sharpness",
        "Color: HSV color space analysis with predefined dermatological color categories",
        f"Diameter: {'Measured using provided calibration' if pixels_per_mm else 'Estimated from image (no calibration)'}",
        "Total Score: Adapted from traditional ABCD dermoscopy scoring system",
        "Note: This is pure image feature analysis independent of AI classification",
        "Important: This analysis supplements but does not replace clinical examination"
    ]

    return ABCDEAnalysis(
        asymmetry=asymmetry,
        border=border,
        color=color,
        diameter=diameter,
        evolution=evolution,
        total_score=round(total_score, 2),
        risk_level=overall_risk,
        summary=summary,
        key_concerns=key_concerns,
        recommendation=recommendation,
        methodology_notes=methodology_notes
    )


def format_abcde_for_response(analysis: ABCDEAnalysis) -> Dict[str, Any]:
    """Convert ABCDEAnalysis to dictionary for API response"""
    return {
        "asymmetry": {
            "overall_score": analysis.asymmetry.overall_score,
            "horizontal_asymmetry": analysis.asymmetry.horizontal_asymmetry,
            "vertical_asymmetry": analysis.asymmetry.vertical_asymmetry,
            "shape_asymmetry": analysis.asymmetry.shape_asymmetry,
            "color_asymmetry": analysis.asymmetry.color_asymmetry,
            "risk_level": analysis.asymmetry.risk_level.value,
            "description": analysis.asymmetry.description,
            "clinical_interpretation": analysis.asymmetry.clinical_interpretation,
        },
        "border": {
            "overall_score": analysis.border.overall_score,
            "irregularity_index": analysis.border.irregularity_index,
            "notching_score": analysis.border.notching_score,
            "blur_score": analysis.border.blur_score,
            "radial_variance": analysis.border.radial_variance,
            "num_border_colors": analysis.border.num_border_colors,
            "risk_level": analysis.border.risk_level.value,
            "description": analysis.border.description,
            "clinical_interpretation": analysis.border.clinical_interpretation,
        },
        "color": {
            "overall_score": analysis.color.overall_score,
            "num_colors": analysis.color.num_colors,
            "colors_detected": analysis.color.colors_detected,
            "color_variance": analysis.color.color_variance,
            "has_blue_white_veil": analysis.color.has_blue_white_veil,
            "has_regression": analysis.color.has_regression,
            "dominant_color": analysis.color.dominant_color,
            "color_distribution": analysis.color.color_distribution,
            "risk_level": analysis.color.risk_level.value,
            "description": analysis.color.description,
            "clinical_interpretation": analysis.color.clinical_interpretation,
        },
        "diameter": {
            "overall_score": analysis.diameter.overall_score,
            "estimated_diameter_mm": analysis.diameter.estimated_diameter_mm,
            "pixel_diameter": analysis.diameter.pixel_diameter,
            "area_pixels": analysis.diameter.area_pixels,
            "is_above_6mm": analysis.diameter.is_above_6mm,
            "calibration_available": analysis.diameter.calibration_available,
            "risk_level": analysis.diameter.risk_level.value,
            "description": analysis.diameter.description,
            "clinical_interpretation": analysis.diameter.clinical_interpretation,
        },
        "evolution": {
            "has_comparison": analysis.evolution.has_comparison,
            "change_detected": analysis.evolution.change_detected,
            "change_description": analysis.evolution.change_description,
            "risk_level": analysis.evolution.risk_level.value,
            "description": analysis.evolution.description,
            "clinical_interpretation": analysis.evolution.clinical_interpretation,
        },
        "total_score": analysis.total_score,
        "risk_level": analysis.risk_level.value,
        "summary": analysis.summary,
        "key_concerns": analysis.key_concerns,
        "recommendation": analysis.recommendation,
        "methodology_notes": analysis.methodology_notes,
    }


def build_combined_assessment(
    abcde_analysis: Dict[str, Any],
    ml_classification: Optional[str],
    ml_confidence: Optional[float]
) -> Dict[str, Any]:
    """
    Build a transparent combined assessment that shows both ML classification
    and image feature analysis results separately, with clear reasoning for
    the final recommendation.

    This function is designed for regulatory transparency - it does NOT hide
    discrepancies between ML and image analysis, but instead explains them.

    Args:
        abcde_analysis: The formatted ABCDE analysis dictionary
        ml_classification: The ML model's predicted class (e.g., "Melanoma")
        ml_confidence: The ML model's confidence score (0-1)

    Returns:
        Dictionary with separate sections for ML, image features, and combined assessment
    """
    # Determine if ML detected a malignant condition
    malignant_conditions = [
        "melanoma", "basal cell carcinoma", "squamous cell carcinoma",
        "malignant", "carcinoma", "bcc", "scc"
    ]

    ml_detected_malignancy = False
    ml_risk_level = "low"

    if ml_classification and ml_confidence:
        ml_detected_malignancy = any(
            condition in ml_classification.lower()
            for condition in malignant_conditions
        )

        if ml_detected_malignancy:
            if ml_confidence >= 0.7:
                ml_risk_level = "very_high"
            elif ml_confidence >= 0.5:
                ml_risk_level = "high"
            elif ml_confidence >= 0.3:
                ml_risk_level = "moderate"
            else:
                ml_risk_level = "low"
        else:
            # Non-malignant classification
            ml_risk_level = "low"

    # Get image feature risk level
    image_risk_level = abcde_analysis.get("risk_level", "low")

    # Determine if there's a discrepancy
    risk_levels_ordered = ["low", "moderate", "high", "very_high"]
    ml_risk_index = risk_levels_ordered.index(ml_risk_level) if ml_risk_level in risk_levels_ordered else 0
    image_risk_index = risk_levels_ordered.index(image_risk_level) if image_risk_level in risk_levels_ordered else 0

    has_discrepancy = abs(ml_risk_index - image_risk_index) >= 2

    # Build combined recommendation with transparent reasoning
    combined_risk_level = max(ml_risk_level, image_risk_level, key=lambda x: risk_levels_ordered.index(x))

    # Generate rationale explaining the assessment
    rationale_parts = []

    if ml_detected_malignancy and ml_confidence and ml_confidence >= 0.5:
        rationale_parts.append(
            f"AI pattern recognition detected {ml_classification} with {ml_confidence:.0%} confidence"
        )

    if image_risk_level in ["high", "very_high"]:
        rationale_parts.append(
            f"Traditional ABCD image features show {image_risk_level} risk (score: {abcde_analysis.get('total_score', 'N/A')})"
        )
    elif image_risk_level == "moderate":
        rationale_parts.append(
            f"Traditional ABCD image features show moderate risk (score: {abcde_analysis.get('total_score', 'N/A')})"
        )
    else:
        rationale_parts.append(
            f"Traditional ABCD image features appear low risk (score: {abcde_analysis.get('total_score', 'N/A')})"
        )

    if has_discrepancy:
        if ml_risk_index > image_risk_index:
            rationale_parts.append(
                "Note: AI classification indicates higher risk than traditional image features suggest. "
                "This may indicate subtle patterns not captured by ABCD criteria."
            )
        else:
            rationale_parts.append(
                "Note: Traditional image features indicate higher risk than AI classification. "
                "Clinical correlation recommended."
            )

    # Generate final recommendation
    if combined_risk_level == "very_high":
        recommendation = "Urgent dermatology referral recommended for evaluation and possible biopsy."
        urgency = "urgent"
    elif combined_risk_level == "high":
        recommendation = "Dermatology evaluation recommended within 2 weeks."
        urgency = "soon"
    elif combined_risk_level == "moderate":
        recommendation = "Consider dermatology evaluation. Photo-document and monitor for changes."
        urgency = "routine"
    else:
        recommendation = "Routine monitoring appropriate. Seek evaluation if changes occur."
        urgency = "routine"

    return {
        "ml_classification": {
            "result": ml_classification,
            "confidence": ml_confidence,
            "confidence_percent": f"{ml_confidence:.0%}" if ml_confidence else None,
            "risk_level": ml_risk_level,
            "is_malignant_type": ml_detected_malignancy
        },
        "image_feature_analysis": {
            "abcde_score": abcde_analysis.get("total_score"),
            "risk_level": image_risk_level,
            "summary": abcde_analysis.get("summary"),
            "key_concerns": abcde_analysis.get("key_concerns", [])
        },
        "combined_assessment": {
            "overall_risk_level": combined_risk_level,
            "recommendation": recommendation,
            "urgency": urgency,
            "rationale": rationale_parts,
            "has_discrepancy": has_discrepancy,
            "discrepancy_note": (
                "AI and image feature assessments differ significantly. "
                "Both analyses are provided for clinical correlation."
            ) if has_discrepancy else None
        },
        "regulatory_transparency": {
            "ml_and_image_analysis_independent": True,
            "abcde_not_modified_by_ml": True,
            "combined_risk_uses_higher_of_two": True,
            "disclaimer": (
                "This tool provides decision support only. It is not intended to diagnose "
                "or replace clinical judgment. All findings should be correlated with "
                "clinical examination by a qualified healthcare provider."
            )
        }
    }
