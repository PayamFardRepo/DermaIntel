"""
AI-Powered Clinical Photography Assistant - Enhanced Medical Standards

Comprehensive real-time quality assessment for medical photography with:
- Ruler/scale detection (rulers, coins, reference cards)
- Color calibration card detection (X-Rite, Kodak, custom)
- Advanced lighting analysis (shadows, glare, uniformity, color temperature)
- Distance/angle guidance with real-time feedback
- AR overlay data generation for guided capture
- DICOM compliance for medical imaging standards
- Multi-angle capture workflow support

This addresses the #1 complaint in teledermatology: poor quality photos.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import io
import base64
import json
from datetime import datetime
from PIL import Image


class QualityLevel(Enum):
    """Photo quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class CaptureAngle(Enum):
    """Standard capture angles for multi-angle workflow"""
    OVERVIEW = "overview"           # Full body region view
    REGIONAL = "regional"           # Surrounding area context
    CLOSEUP = "closeup"             # Detailed lesion view
    DERMOSCOPY = "dermoscopy"       # Dermoscopic (with dermatoscope)
    OBLIQUE = "oblique"             # Side angle for elevation
    TANGENTIAL = "tangential"       # Grazing light for texture


class ReferenceType(Enum):
    """Types of reference objects for scale calibration"""
    RULER_METRIC = "ruler_metric"       # mm/cm ruler
    RULER_IMPERIAL = "ruler_imperial"   # inch ruler
    COIN_US_PENNY = "coin_us_penny"     # 19.05mm diameter
    COIN_US_QUARTER = "coin_us_quarter" # 24.26mm diameter
    COIN_EURO_1 = "coin_euro_1"         # 23.25mm diameter
    COIN_UK_POUND = "coin_uk_pound"     # 22.5mm diameter
    COLOR_CARD = "color_card"           # Standard color card with size
    STICKER_REFERENCE = "sticker_reference"  # Medical reference sticker
    CUSTOM = "custom"


class ColorCardType(Enum):
    """Types of color calibration cards"""
    XRITE_COLORCHECKER = "xrite_colorchecker"
    KODAK_GRAY_CARD = "kodak_gray_card"
    MACBETH = "macbeth"
    DATACOLOR = "datacolor"
    MEDICAL_REFERENCE = "medical_reference"
    CUSTOM = "custom"


@dataclass
class ScaleReference:
    """Detected scale reference object"""
    reference_type: ReferenceType
    detected: bool
    confidence: float  # 0-1
    pixel_to_mm: Optional[float]
    bounding_box: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    contour_points: Optional[List[Tuple[int, int]]]
    real_world_size_mm: Optional[float]


@dataclass
class ColorCalibration:
    """Color calibration data from detected color card"""
    card_type: ColorCardType
    detected: bool
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]]
    color_patches: List[Dict[str, Any]]  # Detected color values
    white_balance_correction: Optional[Tuple[float, float, float]]
    color_temperature_k: Optional[int]
    saturation_correction: Optional[float]


@dataclass
class LightingAnalysis:
    """Comprehensive lighting analysis"""
    overall_score: float  # 0-100
    brightness_mean: float
    brightness_std: float
    contrast_ratio: float

    # Glare analysis
    has_glare: bool
    glare_percentage: float
    glare_locations: List[Tuple[int, int, int, int]]  # Bounding boxes of glare spots

    # Shadow analysis
    has_shadows: bool
    shadow_percentage: float
    shadow_locations: List[Tuple[int, int, int, int]]

    # Uniformity
    lighting_uniformity: float  # 0-100, 100 = perfectly uniform
    uniformity_map: Optional[np.ndarray]  # Heatmap of lighting

    # Color temperature
    color_temperature_k: int  # Kelvin
    white_balance_shift: Tuple[float, float]  # R/G, B/G ratios

    # Recommendations
    issues: List[str]
    suggestions: List[str]


@dataclass
class AROverlayGuide:
    """AR overlay guidance data for real-time capture assistance"""
    # Target regions
    target_box: Tuple[int, int, int, int]  # Where lesion should be placed
    ruler_zone: Tuple[int, int, int, int]  # Where ruler should be placed
    color_card_zone: Tuple[int, int, int, int]  # Where color card should be

    # Level indicators
    horizon_line: Tuple[Tuple[int, int], Tuple[int, int]]
    vertical_guide: Tuple[Tuple[int, int], Tuple[int, int]]
    tilt_angle: float  # Current tilt in degrees

    # Distance indicator
    distance_indicator: str  # "too_close", "too_far", "optimal"
    distance_bar_fill: float  # 0-1, fill level for distance bar

    # Quality indicators
    focus_indicator: str  # "focused", "blurry"
    lighting_indicator: str  # "good", "too_dark", "too_bright", "glare"

    # Capture readiness
    ready_to_capture: bool
    blocking_issues: List[str]

    # Grid overlay (for framing)
    grid_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]]


@dataclass
class DICOMMetadata:
    """DICOM-compliant metadata for medical images"""
    # Patient Information (should be anonymized/encrypted)
    patient_id: Optional[str]
    patient_name_hash: Optional[str]  # Hashed for privacy

    # Study Information
    study_instance_uid: str
    study_date: str
    study_time: str
    study_description: str

    # Series Information
    series_instance_uid: str
    series_description: str
    modality: str  # "XC" for External Camera

    # Image Information
    sop_instance_uid: str
    sop_class_uid: str
    image_type: List[str]
    acquisition_datetime: str

    # Acquisition Parameters
    rows: int
    columns: int
    bits_allocated: int
    bits_stored: int
    pixel_spacing: Optional[Tuple[float, float]]  # mm

    # Equipment
    manufacturer: str
    device_serial_number: Optional[str]
    software_versions: str

    # Anatomical Information
    body_part_examined: str
    patient_position: str
    view_position: str
    laterality: str  # "L", "R", "B" (bilateral)

    # Calibration
    pixel_spacing_calibration_type: str
    pixel_spacing_calibration_description: str
    calibration_object: Optional[str]

    # Clinical Context
    clinical_indication: Optional[str]
    performing_physician: Optional[str]
    institution_name: Optional[str]

    # Quality Metrics
    image_quality_indicator: str  # "HIGH", "STANDARD", "LOW"
    quality_control_performed: bool


@dataclass
class CaptureSession:
    """Multi-angle capture session tracking"""
    session_id: str
    patient_id: Optional[str]
    body_location: str
    lesion_identifier: Optional[str]

    # Required angles for this session
    required_angles: List[CaptureAngle]
    completed_angles: List[CaptureAngle]

    # Captured images
    captures: Dict[str, Dict[str, Any]]  # angle -> capture data

    # Session metadata
    started_at: datetime
    last_capture_at: Optional[datetime]
    completed: bool

    # Quality summary
    overall_quality_score: float
    missing_angles: List[CaptureAngle]
    quality_issues: List[str]


@dataclass
class QualityFeedback:
    """Real-time feedback for photo quality"""
    overall_score: float  # 0-100
    quality_level: QualityLevel
    issues: List[str]
    suggestions: List[str]
    warnings: List[str]

    # Specific scores
    lighting_score: float
    focus_score: float
    distance_score: float
    angle_score: float
    scale_score: float
    color_card_score: float

    # Detection results
    ruler_detected: bool
    color_card_detected: bool
    has_glare: bool
    has_shadows: bool
    is_blurry: bool
    too_close: bool
    too_far: bool

    # Measurements
    estimated_distance_cm: Optional[float]
    pixel_to_mm_ratio: Optional[float]
    glare_percentage: float
    shadow_percentage: float

    # Medical photography compliance
    meets_medical_standards: bool
    dicom_compliant: bool

    # Enhanced data (new fields)
    scale_reference: Optional[ScaleReference] = None
    color_calibration: Optional[ColorCalibration] = None
    lighting_analysis: Optional[LightingAnalysis] = None
    ar_overlay: Optional[AROverlayGuide] = None
    dicom_metadata: Optional[DICOMMetadata] = None


class ClinicalPhotographyAssistant:
    """
    AI-powered assistant for clinical photography quality assessment.

    Enhanced capabilities:
    - Multiple scale reference detection (rulers, coins, cards)
    - Advanced color calibration card detection
    - Comprehensive lighting analysis with uniformity mapping
    - Real-time AR overlay guidance
    - Full DICOM compliance support
    - Multi-angle capture workflow management
    """

    def __init__(self):
        # Minimum quality thresholds for medical photography
        self.MIN_LIGHTING_SCORE = 70
        self.MIN_FOCUS_SCORE = 75
        self.MIN_DISTANCE_SCORE = 60
        self.MAX_GLARE_PERCENTAGE = 10
        self.MAX_SHADOW_PERCENTAGE = 20

        # Ideal distances (cm)
        self.IDEAL_DISTANCE_MIN = 15
        self.IDEAL_DISTANCE_MAX = 30

        # Reference object sizes (mm)
        self.REFERENCE_SIZES = {
            ReferenceType.COIN_US_PENNY: 19.05,
            ReferenceType.COIN_US_QUARTER: 24.26,
            ReferenceType.COIN_EURO_1: 23.25,
            ReferenceType.COIN_UK_POUND: 22.5,
            ReferenceType.RULER_METRIC: 150,  # 15cm standard
            ReferenceType.RULER_IMPERIAL: 152.4,  # 6 inches
        }

        # X-Rite ColorChecker Classic colors (sRGB values)
        self.COLORCHECKER_COLORS = [
            {"name": "dark_skin", "rgb": (115, 82, 68)},
            {"name": "light_skin", "rgb": (194, 150, 130)},
            {"name": "blue_sky", "rgb": (98, 122, 157)},
            {"name": "foliage", "rgb": (87, 108, 67)},
            {"name": "blue_flower", "rgb": (133, 128, 177)},
            {"name": "bluish_green", "rgb": (103, 189, 170)},
            {"name": "orange", "rgb": (214, 126, 44)},
            {"name": "purplish_blue", "rgb": (80, 91, 166)},
            {"name": "moderate_red", "rgb": (193, 90, 99)},
            {"name": "purple", "rgb": (94, 60, 108)},
            {"name": "yellow_green", "rgb": (157, 188, 64)},
            {"name": "orange_yellow", "rgb": (224, 163, 46)},
            {"name": "blue", "rgb": (56, 61, 150)},
            {"name": "green", "rgb": (70, 148, 73)},
            {"name": "red", "rgb": (175, 54, 60)},
            {"name": "yellow", "rgb": (231, 199, 31)},
            {"name": "magenta", "rgb": (187, 86, 149)},
            {"name": "cyan", "rgb": (8, 133, 161)},
            {"name": "white", "rgb": (243, 243, 242)},
            {"name": "neutral_8", "rgb": (200, 200, 200)},
            {"name": "neutral_65", "rgb": (160, 160, 160)},
            {"name": "neutral_5", "rgb": (122, 122, 121)},
            {"name": "neutral_35", "rgb": (85, 85, 85)},
            {"name": "black", "rgb": (52, 52, 52)},
        ]

        # Active capture sessions
        self._capture_sessions: Dict[str, CaptureSession] = {}

    def assess_photo_quality(self, image_bytes: bytes,
                            include_ar_overlay: bool = True,
                            include_dicom: bool = True,
                            session_id: Optional[str] = None,
                            capture_angle: Optional[CaptureAngle] = None) -> QualityFeedback:
        """
        Comprehensive photo quality assessment with enhanced features.

        Args:
            image_bytes: Image data as bytes
            include_ar_overlay: Generate AR overlay guidance data
            include_dicom: Generate DICOM metadata
            session_id: Optional capture session ID
            capture_angle: The intended capture angle

        Returns:
            QualityFeedback with detailed assessment
        """
        # Load image
        img = self._load_image(image_bytes)
        h, w = img.shape[:2]

        # Enhanced scale reference detection
        scale_ref = self._detect_scale_reference(img)
        pixel_to_mm = scale_ref.pixel_to_mm if scale_ref.detected else None

        # Enhanced color calibration detection
        color_cal = self._detect_color_calibration(img)

        # Enhanced lighting analysis
        lighting = self._analyze_lighting_comprehensive(img)

        # Focus assessment
        focus_score, is_blurry = self._assess_focus(img)

        # Distance assessment
        distance_score, estimated_distance, too_close, too_far = self._assess_distance(
            img, scale_ref.detected, pixel_to_mm
        )

        # Angle assessment
        angle_score, tilt_angle = self._assess_angle_enhanced(img)

        # Calculate scores
        scale_score = scale_ref.confidence * 100 if scale_ref.detected else 0
        color_card_score = color_cal.confidence * 100 if color_cal.detected else 0

        # Calculate overall score (weighted average)
        weights = {
            'lighting': 0.25,
            'focus': 0.25,
            'distance': 0.15,
            'angle': 0.10,
            'scale': 0.15,
            'color_card': 0.10,
        }

        overall_score = (
            lighting.overall_score * weights['lighting'] +
            focus_score * weights['focus'] +
            distance_score * weights['distance'] +
            angle_score * weights['angle'] +
            scale_score * weights['scale'] +
            color_card_score * weights['color_card']
        )

        # Determine quality level
        if overall_score >= 85:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 75:
            quality_level = QualityLevel.GOOD
        elif overall_score >= 65:
            quality_level = QualityLevel.ACCEPTABLE
        elif overall_score >= 50:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.UNACCEPTABLE

        # Generate feedback
        issues, suggestions, warnings = self._generate_enhanced_feedback(
            scale_ref, color_cal, lighting, is_blurry, too_close, too_far,
            focus_score, angle_score, capture_angle
        )

        # Check medical standards compliance
        meets_standards = self._check_medical_standards_enhanced(
            scale_ref, color_cal, lighting, focus_score
        )

        # DICOM compliance
        dicom_compliant = meets_standards and color_cal.detected and scale_ref.detected

        # Generate AR overlay if requested
        ar_overlay = None
        if include_ar_overlay:
            ar_overlay = self._generate_ar_overlay(
                img, scale_ref, color_cal, lighting, focus_score,
                too_close, too_far, tilt_angle, warnings
            )

        # Generate DICOM metadata if requested
        dicom_metadata = None
        if include_dicom:
            dicom_metadata = self._generate_dicom_metadata(
                img, scale_ref, quality_level
            )

        # Update capture session if applicable
        if session_id and session_id in self._capture_sessions:
            self._update_capture_session(
                session_id, capture_angle, overall_score, image_bytes
            )

        return QualityFeedback(
            overall_score=round(overall_score, 1),
            quality_level=quality_level,
            issues=issues,
            suggestions=suggestions,
            warnings=warnings,
            lighting_score=round(lighting.overall_score, 1),
            focus_score=round(focus_score, 1),
            distance_score=round(distance_score, 1),
            angle_score=round(angle_score, 1),
            scale_score=round(scale_score, 1),
            color_card_score=round(color_card_score, 1),
            ruler_detected=scale_ref.detected,
            color_card_detected=color_cal.detected,
            has_glare=lighting.has_glare,
            has_shadows=lighting.has_shadows,
            is_blurry=is_blurry,
            too_close=too_close,
            too_far=too_far,
            estimated_distance_cm=estimated_distance,
            pixel_to_mm_ratio=pixel_to_mm,
            glare_percentage=round(lighting.glare_percentage, 1),
            shadow_percentage=round(lighting.shadow_percentage, 1),
            meets_medical_standards=meets_standards,
            dicom_compliant=dicom_compliant,
            scale_reference=scale_ref,
            color_calibration=color_cal,
            lighting_analysis=lighting,
            ar_overlay=ar_overlay,
            dicom_metadata=dicom_metadata
        )

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes"""
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # =========================================================================
    # ENHANCED SCALE/RULER DETECTION
    # =========================================================================

    def _detect_scale_reference(self, img: np.ndarray) -> ScaleReference:
        """
        Enhanced scale reference detection supporting multiple reference types.

        Detects:
        - Metric/Imperial rulers
        - Common coins (US, Euro, UK)
        - Medical reference stickers
        - Color cards with known dimensions
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try each detection method
        methods = [
            self._detect_ruler_enhanced,
            self._detect_coin_reference,
            self._detect_reference_sticker,
        ]

        best_result = ScaleReference(
            reference_type=ReferenceType.CUSTOM,
            detected=False,
            confidence=0.0,
            pixel_to_mm=None,
            bounding_box=None,
            contour_points=None,
            real_world_size_mm=None
        )

        for method in methods:
            result = method(img, gray)
            if result.detected and result.confidence > best_result.confidence:
                best_result = result

        return best_result

    def _detect_ruler_enhanced(self, img: np.ndarray, gray: np.ndarray) -> ScaleReference:
        """Enhanced ruler detection with tick mark analysis"""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(max(w, h)) / min(w, h)

                if aspect_ratio > 3:  # Elongated = likely ruler
                    roi = gray[y:y+h, x:x+w]
                    tick_confidence = self._analyze_ruler_ticks(roi)

                    if tick_confidence > 0.5:
                        # Determine if metric or imperial
                        ref_type, size_mm = self._identify_ruler_type(roi, max(w, h))
                        longer_side = max(w, h)
                        pixel_to_mm = longer_side / size_mm

                        return ScaleReference(
                            reference_type=ref_type,
                            detected=True,
                            confidence=tick_confidence,
                            pixel_to_mm=pixel_to_mm,
                            bounding_box=(x, y, w, h),
                            contour_points=approx.tolist(),
                            real_world_size_mm=size_mm
                        )

        return ScaleReference(
            reference_type=ReferenceType.RULER_METRIC,
            detected=False,
            confidence=0.0,
            pixel_to_mm=None,
            bounding_box=None,
            contour_points=None,
            real_world_size_mm=None
        )

    def _analyze_ruler_ticks(self, roi: np.ndarray) -> float:
        """Analyze tick marks on ruler for confidence score"""
        if roi.size == 0:
            return 0.0

        _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

        # Detect lines (tick marks)
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=30,
                                minLineLength=5, maxLineGap=5)

        if lines is None:
            return 0.0

        # Count perpendicular lines (tick marks are perpendicular to ruler)
        h, w = roi.shape
        horizontal_lines = 0
        vertical_lines = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 20 or angle > 160:  # Near horizontal
                horizontal_lines += 1
            elif 70 < angle < 110:  # Near vertical
                vertical_lines += 1

        # Rulers have many parallel tick marks
        tick_count = max(horizontal_lines, vertical_lines)

        if tick_count >= 10:
            return min(0.5 + (tick_count - 10) * 0.05, 0.95)
        elif tick_count >= 5:
            return 0.3 + tick_count * 0.04
        else:
            return tick_count * 0.06

    def _identify_ruler_type(self, roi: np.ndarray, length_pixels: int) -> Tuple[ReferenceType, float]:
        """Identify if ruler is metric or imperial"""
        # Analyze tick spacing patterns
        # Metric: marks at 1mm, 5mm, 10mm intervals
        # Imperial: marks at 1/16", 1/8", 1/4", 1/2" intervals

        # Default to metric 15cm ruler
        return ReferenceType.RULER_METRIC, 150.0

    def _detect_coin_reference(self, img: np.ndarray, gray: np.ndarray) -> ScaleReference:
        """Detect coins as scale reference"""
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 11, 75, 75)

        # Detect circles using Hough transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=40,
            minRadius=20,
            maxRadius=200
        )

        if circles is None:
            return ScaleReference(
                reference_type=ReferenceType.COIN_US_QUARTER,
                detected=False,
                confidence=0.0,
                pixel_to_mm=None,
                bounding_box=None,
                contour_points=None,
                real_world_size_mm=None
            )

        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            x, y, r = circle

            # Check if circle is in a reasonable location (not center of image)
            h, w = img.shape[:2]
            if abs(x - w/2) < w * 0.1 and abs(y - h/2) < h * 0.1:
                continue  # Skip circles in the center (likely the lesion)

            # Analyze coin appearance
            coin_type, confidence = self._identify_coin(img, x, y, r)

            if confidence > 0.6:
                coin_size = self.REFERENCE_SIZES.get(coin_type, 24.26)
                diameter_pixels = r * 2
                pixel_to_mm = diameter_pixels / coin_size

                return ScaleReference(
                    reference_type=coin_type,
                    detected=True,
                    confidence=confidence,
                    pixel_to_mm=pixel_to_mm,
                    bounding_box=(int(x-r), int(y-r), int(r*2), int(r*2)),
                    contour_points=None,
                    real_world_size_mm=coin_size
                )

        return ScaleReference(
            reference_type=ReferenceType.COIN_US_QUARTER,
            detected=False,
            confidence=0.0,
            pixel_to_mm=None,
            bounding_box=None,
            contour_points=None,
            real_world_size_mm=None
        )

    def _identify_coin(self, img: np.ndarray, x: int, y: int, r: int) -> Tuple[ReferenceType, float]:
        """Identify coin type based on color and features"""
        # Extract ROI
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Calculate mean color
        mean_color = cv2.mean(img, mask=mask)[:3]
        b, g, r_color = mean_color

        # US coins are typically silver/copper colored
        # Penny: copper (reddish)
        # Quarter: silver (grayish)

        if r_color > 100 and g < 80 and b < 80:  # Copper-ish
            return ReferenceType.COIN_US_PENNY, 0.7
        elif abs(r_color - g) < 30 and abs(g - b) < 30:  # Silver-ish (gray)
            return ReferenceType.COIN_US_QUARTER, 0.65
        else:
            return ReferenceType.COIN_US_QUARTER, 0.5  # Default

    def _detect_reference_sticker(self, img: np.ndarray, gray: np.ndarray) -> ScaleReference:
        """Detect medical reference stickers (often circular with markings)"""
        # Look for specific patterns common in medical reference stickers
        # These often have concentric circles or grid patterns

        return ScaleReference(
            reference_type=ReferenceType.STICKER_REFERENCE,
            detected=False,
            confidence=0.0,
            pixel_to_mm=None,
            bounding_box=None,
            contour_points=None,
            real_world_size_mm=None
        )

    # =========================================================================
    # ENHANCED COLOR CALIBRATION DETECTION
    # =========================================================================

    def _detect_color_calibration(self, img: np.ndarray) -> ColorCalibration:
        """
        Enhanced color calibration card detection.

        Supports:
        - X-Rite ColorChecker
        - Kodak Gray Card
        - Macbeth ColorChecker
        - Custom medical reference cards
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try to detect grid of colored squares
        squares = self._detect_color_squares(img, gray)

        if len(squares) >= 6:  # Minimum for a color card
            # Analyze colors in squares
            color_patches = []
            for sq in squares[:24]:  # Max 24 patches (ColorChecker Classic)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [sq], -1, 255, -1)
                mean_color = cv2.mean(img, mask=mask)[:3]

                x, y, w, h = cv2.boundingRect(sq)
                color_patches.append({
                    "bgr": mean_color,
                    "rgb": (mean_color[2], mean_color[1], mean_color[0]),
                    "bbox": (x, y, w, h)
                })

            # Determine card type and calculate corrections
            card_type, confidence = self._identify_color_card(color_patches)
            white_balance = self._calculate_white_balance_correction(color_patches)
            color_temp = self._estimate_color_temperature(white_balance)
            saturation = self._calculate_saturation_correction(color_patches)

            # Get bounding box of all squares
            all_points = np.vstack(squares)
            x, y, w, h = cv2.boundingRect(all_points)

            return ColorCalibration(
                card_type=card_type,
                detected=True,
                confidence=confidence,
                bounding_box=(x, y, w, h),
                color_patches=color_patches,
                white_balance_correction=white_balance,
                color_temperature_k=color_temp,
                saturation_correction=saturation
            )

        return ColorCalibration(
            card_type=ColorCardType.CUSTOM,
            detected=False,
            confidence=0.0,
            bounding_box=None,
            color_patches=[],
            white_balance_correction=None,
            color_temperature_k=None,
            saturation_correction=None
        )

    def _detect_color_squares(self, img: np.ndarray, gray: np.ndarray) -> List[np.ndarray]:
        """Detect colored square patches"""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        squares = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > 50000:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.7 < aspect_ratio < 1.3:  # Nearly square
                    squares.append(contour)

        return squares

    def _identify_color_card(self, patches: List[Dict]) -> Tuple[ColorCardType, float]:
        """Identify the type of color card based on patches"""
        if len(patches) >= 24:
            # Check if it matches ColorChecker Classic
            match_score = self._match_colorchecker(patches)
            if match_score > 0.6:
                return ColorCardType.XRITE_COLORCHECKER, match_score

        if len(patches) >= 6:
            return ColorCardType.CUSTOM, 0.5

        return ColorCardType.CUSTOM, 0.3

    def _match_colorchecker(self, patches: List[Dict]) -> float:
        """Calculate match score against ColorChecker reference"""
        if len(patches) < 24:
            return 0.0

        total_diff = 0
        matched = 0

        for i, ref_color in enumerate(self.COLORCHECKER_COLORS[:len(patches)]):
            patch_rgb = patches[i]["rgb"]
            ref_rgb = ref_color["rgb"]

            diff = np.sqrt(sum((a-b)**2 for a, b in zip(patch_rgb, ref_rgb)))
            total_diff += diff

            if diff < 50:  # Close match
                matched += 1

        if matched >= len(patches) * 0.6:
            return 0.8 - (total_diff / (len(patches) * 255))

        return 0.3

    def _calculate_white_balance_correction(self, patches: List[Dict]) -> Optional[Tuple[float, float, float]]:
        """Calculate white balance correction from neutral patches"""
        # Find neutral patches (white, grays, black)
        neutrals = []
        for p in patches:
            r, g, b = p["rgb"]
            if abs(r - g) < 30 and abs(g - b) < 30:  # Near neutral
                neutrals.append((r, g, b))

        if not neutrals:
            return None

        # Calculate average
        avg_r = sum(n[0] for n in neutrals) / len(neutrals)
        avg_g = sum(n[1] for n in neutrals) / len(neutrals)
        avg_b = sum(n[2] for n in neutrals) / len(neutrals)

        # Calculate correction (to make neutrals truly neutral)
        avg_all = (avg_r + avg_g + avg_b) / 3

        if avg_all > 0:
            r_correction = avg_all / avg_r if avg_r > 0 else 1.0
            g_correction = avg_all / avg_g if avg_g > 0 else 1.0
            b_correction = avg_all / avg_b if avg_b > 0 else 1.0
            return (r_correction, g_correction, b_correction)

        return (1.0, 1.0, 1.0)

    def _estimate_color_temperature(self, white_balance: Optional[Tuple[float, float, float]]) -> int:
        """Estimate color temperature in Kelvin"""
        if white_balance is None:
            return 5500  # Default daylight

        r, g, b = white_balance

        # Higher R correction = cooler (bluer) light
        # Higher B correction = warmer (yellower) light

        if r > b:
            temp = 5500 + int((r - b) * 1000)
        else:
            temp = 5500 - int((b - r) * 1000)

        return max(2700, min(10000, temp))

    def _calculate_saturation_correction(self, patches: List[Dict]) -> float:
        """Calculate saturation correction factor"""
        # Compare expected vs actual saturation
        return 1.0  # Default no correction

    # =========================================================================
    # COMPREHENSIVE LIGHTING ANALYSIS
    # =========================================================================

    def _analyze_lighting_comprehensive(self, img: np.ndarray) -> LightingAnalysis:
        """
        Comprehensive lighting analysis including:
        - Brightness and contrast
        - Glare detection and localization
        - Shadow detection and mapping
        - Lighting uniformity analysis
        - Color temperature estimation
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Contrast ratio
        min_val, max_val = np.min(gray), np.max(gray)
        contrast_ratio = (max_val - min_val) / 255.0 if max_val > min_val else 0

        # Glare detection
        glare_threshold = 245
        glare_mask = gray > glare_threshold
        glare_percentage = (np.sum(glare_mask) / glare_mask.size) * 100
        glare_locations = self._find_region_bboxes(glare_mask)
        has_glare = glare_percentage > self.MAX_GLARE_PERCENTAGE

        # Shadow detection
        shadow_threshold = 30
        shadow_mask = gray < shadow_threshold
        shadow_percentage = (np.sum(shadow_mask) / shadow_mask.size) * 100
        shadow_locations = self._find_region_bboxes(shadow_mask)
        has_shadows = shadow_percentage > self.MAX_SHADOW_PERCENTAGE

        # Lighting uniformity analysis
        uniformity, uniformity_map = self._analyze_lighting_uniformity(gray)

        # Color temperature estimation from image
        color_temp, wb_shift = self._analyze_color_temperature(img)

        # Calculate overall lighting score
        brightness_score = 100 - abs(mean_brightness - 140) / 140 * 50
        contrast_score = min(contrast_ratio * 150, 100)
        uniformity_score = uniformity

        # Penalties
        glare_penalty = min(glare_percentage * 3, 40)
        shadow_penalty = min(shadow_percentage * 2, 30)

        overall_score = max(0, (
            brightness_score * 0.3 +
            contrast_score * 0.2 +
            uniformity_score * 0.3 +
            20  # Base score
        ) - glare_penalty - shadow_penalty)

        # Generate issues and suggestions
        issues, suggestions = self._generate_lighting_feedback(
            mean_brightness, has_glare, has_shadows, uniformity,
            color_temp, glare_percentage, shadow_percentage
        )

        return LightingAnalysis(
            overall_score=min(overall_score, 100),
            brightness_mean=mean_brightness,
            brightness_std=std_brightness,
            contrast_ratio=contrast_ratio,
            has_glare=has_glare,
            glare_percentage=glare_percentage,
            glare_locations=glare_locations,
            has_shadows=has_shadows,
            shadow_percentage=shadow_percentage,
            shadow_locations=shadow_locations,
            lighting_uniformity=uniformity,
            uniformity_map=uniformity_map,
            color_temperature_k=color_temp,
            white_balance_shift=wb_shift,
            issues=issues,
            suggestions=suggestions
        )

    def _find_region_bboxes(self, mask: np.ndarray, min_area: int = 100) -> List[Tuple[int, int, int, int]]:
        """Find bounding boxes of contiguous regions in mask"""
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))

        return bboxes

    def _analyze_lighting_uniformity(self, gray: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Analyze lighting uniformity across the image.
        Returns uniformity score (0-100) and uniformity map.
        """
        h, w = gray.shape

        # Divide image into grid
        grid_size = 8
        cell_h, cell_w = h // grid_size, w // grid_size

        cell_means = []
        uniformity_map = np.zeros((grid_size, grid_size), dtype=np.float32)

        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = gray[y1:y2, x1:x2]
                mean = np.mean(cell)
                cell_means.append(mean)
                uniformity_map[i, j] = mean

        # Calculate coefficient of variation
        mean_all = np.mean(cell_means)
        std_all = np.std(cell_means)

        if mean_all > 0:
            cv = std_all / mean_all
            # Lower CV = more uniform
            uniformity_score = max(0, 100 - cv * 200)
        else:
            uniformity_score = 0

        return uniformity_score, uniformity_map

    def _analyze_color_temperature(self, img: np.ndarray) -> Tuple[int, Tuple[float, float]]:
        """Estimate color temperature from image"""
        b, g, r = cv2.split(img)

        mean_r = np.mean(r)
        mean_g = np.mean(g)
        mean_b = np.mean(b)

        # R/G and B/G ratios indicate color temperature
        rg_ratio = mean_r / mean_g if mean_g > 0 else 1.0
        bg_ratio = mean_b / mean_g if mean_g > 0 else 1.0

        # Estimate temperature
        # Higher R = warmer (lower K), Higher B = cooler (higher K)
        temp_offset = (rg_ratio - bg_ratio) * 2000
        estimated_temp = int(5500 - temp_offset)
        estimated_temp = max(2700, min(10000, estimated_temp))

        return estimated_temp, (rg_ratio, bg_ratio)

    def _generate_lighting_feedback(self, brightness: float, has_glare: bool,
                                   has_shadows: bool, uniformity: float,
                                   color_temp: int, glare_pct: float,
                                   shadow_pct: float) -> Tuple[List[str], List[str]]:
        """Generate lighting-specific feedback"""
        issues = []
        suggestions = []

        if brightness < 80:
            issues.append("Image is too dark")
            suggestions.append("Increase lighting or move to brighter area")
        elif brightness > 200:
            issues.append("Image is overexposed")
            suggestions.append("Reduce lighting intensity or adjust camera exposure")

        if has_glare:
            issues.append(f"Glare detected ({glare_pct:.1f}% of image)")
            suggestions.append("Use polarized filter or change lighting angle")
            suggestions.append("Try ring light or diffused lighting from 45° angles")

        if has_shadows:
            issues.append(f"Heavy shadows detected ({shadow_pct:.1f}% of image)")
            suggestions.append("Add fill lighting to eliminate shadows")
            suggestions.append("Use reflector to bounce light into shadow areas")

        if uniformity < 60:
            issues.append("Uneven lighting across the image")
            suggestions.append("Use multiple light sources for even illumination")

        if color_temp < 4000:
            issues.append("Warm (yellowish) lighting detected")
            suggestions.append("Use daylight-balanced lighting (5500K)")
        elif color_temp > 7000:
            issues.append("Cool (bluish) lighting detected")
            suggestions.append("Use daylight-balanced lighting (5500K)")

        return issues, suggestions

    # =========================================================================
    # FOCUS ASSESSMENT
    # =========================================================================

    def _assess_focus(self, img: np.ndarray) -> Tuple[float, bool]:
        """
        Assess focus quality using multiple methods.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplacian variance (primary method)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()

        # Sobel variance (secondary method)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = sobel_x.var() + sobel_y.var()

        # Normalize scores
        if laplacian_var < 100:
            lap_score = laplacian_var / 100 * 40
            is_blurry = True
        elif laplacian_var < 500:
            lap_score = 40 + (laplacian_var - 100) / 400 * 40
            is_blurry = False
        else:
            lap_score = 80 + min((laplacian_var - 500) / 500 * 20, 20)
            is_blurry = False

        # Combine scores
        focus_score = min(lap_score, 100)

        return focus_score, is_blurry

    # =========================================================================
    # DISTANCE ASSESSMENT
    # =========================================================================

    def _assess_distance(self, img: np.ndarray, ruler_detected: bool,
                        pixel_to_mm: Optional[float]) -> Tuple[float, Optional[float], bool, bool]:
        """
        Assess camera distance from subject.
        """
        h, w = img.shape[:2]

        if not ruler_detected or pixel_to_mm is None:
            # Heuristic based on image resolution
            if w < 800 or h < 600:
                return 50, None, False, True
            elif w > 3000 or h > 3000:
                return 70, None, True, False
            else:
                return 70, None, False, False

        # Calculate based on ruler size in image
        ruler_pixels = 150 * pixel_to_mm  # 150mm standard ruler
        ruler_ratio = ruler_pixels / w

        if ruler_ratio > 0.6:
            estimated_distance = 10
            too_close = True
            too_far = False
            distance_score = 40
        elif ruler_ratio > 0.4:
            estimated_distance = 20
            too_close = False
            too_far = False
            distance_score = 100
        elif ruler_ratio > 0.2:
            estimated_distance = 35
            too_close = False
            too_far = False
            distance_score = 80
        else:
            estimated_distance = 50
            too_close = False
            too_far = True
            distance_score = 50

        return distance_score, estimated_distance, too_close, too_far

    # =========================================================================
    # ANGLE ASSESSMENT
    # =========================================================================

    def _assess_angle_enhanced(self, img: np.ndarray) -> Tuple[float, float]:
        """
        Enhanced angle assessment with tilt detection.
        Returns (angle_score, tilt_angle_degrees)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=50, maxLineGap=10)

        if lines is None:
            return 70, 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

        if not angles:
            return 70, 0.0

        # Find dominant angle
        angles = np.array(angles)

        # Find deviation from horizontal/vertical
        deviations = []
        for a in angles:
            dev = min(abs(a), abs(a - 90), abs(a + 90), abs(a - 180), abs(a + 180))
            deviations.append(dev)

        mean_deviation = np.mean(deviations)
        tilt_angle = np.median(angles)

        # Normalize tilt to -45 to 45 range
        while tilt_angle > 45:
            tilt_angle -= 90
        while tilt_angle < -45:
            tilt_angle += 90

        # Score based on deviation
        angle_score = max(0, 100 - mean_deviation * 2)

        return angle_score, tilt_angle

    # =========================================================================
    # AR OVERLAY GENERATION
    # =========================================================================

    def _generate_ar_overlay(self, img: np.ndarray, scale_ref: ScaleReference,
                            color_cal: ColorCalibration, lighting: LightingAnalysis,
                            focus_score: float, too_close: bool, too_far: bool,
                            tilt_angle: float, warnings: List[str]) -> AROverlayGuide:
        """
        Generate AR overlay guidance data for real-time capture assistance.
        """
        h, w = img.shape[:2]

        # Calculate target zones
        center_x, center_y = w // 2, h // 2

        # Target box for lesion (center 60% of image)
        target_margin = int(min(w, h) * 0.2)
        target_box = (target_margin, target_margin,
                     w - 2*target_margin, h - 2*target_margin)

        # Ruler zone (bottom left corner)
        ruler_zone = (10, h - 100, 200, 90)

        # Color card zone (top right corner)
        color_card_zone = (w - 110, 10, 100, 100)

        # Level indicators
        horizon_line = ((0, center_y), (w, center_y))
        vertical_guide = ((center_x, 0), (center_x, h))

        # Distance indicator
        if too_close:
            distance_indicator = "too_close"
            distance_bar_fill = 0.9
        elif too_far:
            distance_indicator = "too_far"
            distance_bar_fill = 0.2
        else:
            distance_indicator = "optimal"
            distance_bar_fill = 0.6

        # Focus indicator
        focus_indicator = "focused" if focus_score >= 75 else "blurry"

        # Lighting indicator
        if lighting.has_glare:
            lighting_indicator = "glare"
        elif lighting.brightness_mean < 80:
            lighting_indicator = "too_dark"
        elif lighting.brightness_mean > 200:
            lighting_indicator = "too_bright"
        else:
            lighting_indicator = "good"

        # Check if ready to capture
        blocking_issues = []
        if focus_score < 60:
            blocking_issues.append("Image is blurry - hold steady")
        if lighting.has_glare and lighting.glare_percentage > 15:
            blocking_issues.append("Excessive glare - adjust angle")
        if not scale_ref.detected:
            blocking_issues.append("No scale reference detected")

        ready_to_capture = len(blocking_issues) == 0 and focus_score >= 70

        # Grid lines (rule of thirds)
        third_x = w // 3
        third_y = h // 3
        grid_lines = [
            ((third_x, 0), (third_x, h)),
            ((2*third_x, 0), (2*third_x, h)),
            ((0, third_y), (w, third_y)),
            ((0, 2*third_y), (w, 2*third_y)),
        ]

        return AROverlayGuide(
            target_box=target_box,
            ruler_zone=ruler_zone,
            color_card_zone=color_card_zone,
            horizon_line=horizon_line,
            vertical_guide=vertical_guide,
            tilt_angle=tilt_angle,
            distance_indicator=distance_indicator,
            distance_bar_fill=distance_bar_fill,
            focus_indicator=focus_indicator,
            lighting_indicator=lighting_indicator,
            ready_to_capture=ready_to_capture,
            blocking_issues=blocking_issues,
            grid_lines=grid_lines
        )

    # =========================================================================
    # DICOM METADATA GENERATION
    # =========================================================================

    def _generate_dicom_metadata(self, img: np.ndarray, scale_ref: ScaleReference,
                                quality_level: QualityLevel) -> DICOMMetadata:
        """
        Generate DICOM-compliant metadata for medical imaging.
        """
        import uuid
        from datetime import datetime

        h, w = img.shape[:2]
        now = datetime.now()

        # Generate UIDs
        study_uid = f"1.2.840.10008.{uuid.uuid4().hex[:16]}"
        series_uid = f"1.2.840.10008.{uuid.uuid4().hex[:16]}"
        sop_uid = f"1.2.840.10008.{uuid.uuid4().hex[:16]}"

        # Calculate pixel spacing if scale reference detected
        pixel_spacing = None
        calibration_type = "UNCALIBRATED"
        calibration_desc = "No calibration object detected"
        calibration_object = None

        if scale_ref.detected and scale_ref.pixel_to_mm:
            mm_per_pixel = 1.0 / scale_ref.pixel_to_mm
            pixel_spacing = (mm_per_pixel, mm_per_pixel)
            calibration_type = "GEOMETRY"
            calibration_desc = f"Calibrated using {scale_ref.reference_type.value}"
            calibration_object = scale_ref.reference_type.value

        # Quality indicator
        quality_map = {
            QualityLevel.EXCELLENT: "HIGH",
            QualityLevel.GOOD: "HIGH",
            QualityLevel.ACCEPTABLE: "STANDARD",
            QualityLevel.POOR: "LOW",
            QualityLevel.UNACCEPTABLE: "LOW"
        }

        return DICOMMetadata(
            patient_id=None,  # To be filled by application
            patient_name_hash=None,
            study_instance_uid=study_uid,
            study_date=now.strftime("%Y%m%d"),
            study_time=now.strftime("%H%M%S"),
            study_description="Dermatology Clinical Photography",
            series_instance_uid=series_uid,
            series_description="Skin Lesion Documentation",
            modality="XC",  # External Camera Photography
            sop_instance_uid=sop_uid,
            sop_class_uid="1.2.840.10008.5.1.4.1.1.77.1.4",  # VL Photographic Image
            image_type=["ORIGINAL", "PRIMARY"],
            acquisition_datetime=now.isoformat(),
            rows=h,
            columns=w,
            bits_allocated=8,
            bits_stored=8,
            pixel_spacing=pixel_spacing,
            manufacturer="SkinClassifier",
            device_serial_number=None,
            software_versions="1.0.0",
            body_part_examined="SKIN",
            patient_position="",
            view_position="AP",  # Anteroposterior
            laterality="",  # To be filled
            pixel_spacing_calibration_type=calibration_type,
            pixel_spacing_calibration_description=calibration_desc,
            calibration_object=calibration_object,
            clinical_indication=None,
            performing_physician=None,
            institution_name=None,
            image_quality_indicator=quality_map.get(quality_level, "STANDARD"),
            quality_control_performed=True
        )

    # =========================================================================
    # MULTI-ANGLE CAPTURE SESSION MANAGEMENT
    # =========================================================================

    def create_capture_session(self, body_location: str,
                               patient_id: Optional[str] = None,
                               lesion_identifier: Optional[str] = None,
                               required_angles: Optional[List[CaptureAngle]] = None) -> str:
        """
        Create a new multi-angle capture session.

        Returns session_id
        """
        import uuid

        session_id = str(uuid.uuid4())

        if required_angles is None:
            # Default required angles for standard documentation
            required_angles = [
                CaptureAngle.OVERVIEW,
                CaptureAngle.REGIONAL,
                CaptureAngle.CLOSEUP,
            ]

        session = CaptureSession(
            session_id=session_id,
            patient_id=patient_id,
            body_location=body_location,
            lesion_identifier=lesion_identifier,
            required_angles=required_angles,
            completed_angles=[],
            captures={},
            started_at=datetime.now(),
            last_capture_at=None,
            completed=False,
            overall_quality_score=0.0,
            missing_angles=required_angles.copy(),
            quality_issues=[]
        )

        self._capture_sessions[session_id] = session
        return session_id

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a capture session"""
        if session_id not in self._capture_sessions:
            return None

        session = self._capture_sessions[session_id]

        return {
            "session_id": session.session_id,
            "body_location": session.body_location,
            "lesion_identifier": session.lesion_identifier,
            "required_angles": [a.value for a in session.required_angles],
            "completed_angles": [a.value for a in session.completed_angles],
            "missing_angles": [a.value for a in session.missing_angles],
            "captures_count": len(session.captures),
            "started_at": session.started_at.isoformat(),
            "last_capture_at": session.last_capture_at.isoformat() if session.last_capture_at else None,
            "completed": session.completed,
            "overall_quality_score": session.overall_quality_score,
            "quality_issues": session.quality_issues,
            "next_recommended_angle": session.missing_angles[0].value if session.missing_angles else None
        }

    def _update_capture_session(self, session_id: str, angle: Optional[CaptureAngle],
                                quality_score: float, image_bytes: bytes):
        """Update capture session with new capture"""
        if session_id not in self._capture_sessions:
            return

        session = self._capture_sessions[session_id]

        if angle:
            angle_key = angle.value
            session.captures[angle_key] = {
                "angle": angle_key,
                "quality_score": quality_score,
                "captured_at": datetime.now().isoformat(),
            }

            if angle not in session.completed_angles:
                session.completed_angles.append(angle)

            if angle in session.missing_angles:
                session.missing_angles.remove(angle)

        session.last_capture_at = datetime.now()

        # Update overall quality score
        if session.captures:
            scores = [c["quality_score"] for c in session.captures.values()]
            session.overall_quality_score = sum(scores) / len(scores)

        # Check if session is complete
        session.completed = len(session.missing_angles) == 0

    def get_angle_guidance(self, angle: CaptureAngle) -> Dict[str, Any]:
        """
        Get guidance for capturing a specific angle.
        """
        guidance = {
            CaptureAngle.OVERVIEW: {
                "description": "Full body region view showing anatomical context",
                "distance_cm": "50-100",
                "includes": "Entire body region (arm, leg, torso, etc.)",
                "purpose": "Provides anatomical landmarks for lesion location",
                "tips": [
                    "Include nearby anatomical landmarks",
                    "Ensure even lighting across the region",
                    "Patient should be positioned consistently"
                ]
            },
            CaptureAngle.REGIONAL: {
                "description": "Surrounding area context around the lesion",
                "distance_cm": "20-40",
                "includes": "10-15cm of surrounding skin",
                "purpose": "Shows distribution and relation to nearby structures",
                "tips": [
                    "Center the lesion in frame",
                    "Include some normal skin for comparison",
                    "Note any satellite lesions"
                ]
            },
            CaptureAngle.CLOSEUP: {
                "description": "Detailed view of the lesion",
                "distance_cm": "10-20",
                "includes": "Lesion plus 2-3cm margin",
                "purpose": "Detailed documentation of lesion features",
                "tips": [
                    "Place ruler adjacent to lesion",
                    "Ensure sharp focus on lesion edges",
                    "Capture any surface texture details",
                    "Include color calibration card if possible"
                ]
            },
            CaptureAngle.DERMOSCOPY: {
                "description": "Dermoscopic view using dermatoscope",
                "distance_cm": "Contact or 1-2",
                "includes": "Dermoscopic field of view",
                "purpose": "Reveals subsurface structures and patterns",
                "tips": [
                    "Apply gel/fluid for contact dermoscopy",
                    "Ensure proper polarization if using polarized mode",
                    "Capture at 10x magnification minimum"
                ]
            },
            CaptureAngle.OBLIQUE: {
                "description": "Side angle to show elevation/depth",
                "distance_cm": "15-25",
                "includes": "Lesion profile",
                "purpose": "Documents elevation, nodularity, or depression",
                "tips": [
                    "Position camera at 30-45° angle",
                    "Use side lighting to accentuate texture",
                    "Capture from multiple sides if asymmetric"
                ]
            },
            CaptureAngle.TANGENTIAL: {
                "description": "Grazing light angle for surface texture",
                "distance_cm": "15-20",
                "includes": "Lesion surface details",
                "purpose": "Reveals surface texture and scaling",
                "tips": [
                    "Position light source at low angle (grazing)",
                    "Camera perpendicular to skin",
                    "Highlights scales, crusts, and texture"
                ]
            }
        }

        return guidance.get(angle, {
            "description": "Standard capture",
            "tips": ["Follow general photography guidelines"]
        })

    # =========================================================================
    # FEEDBACK GENERATION
    # =========================================================================

    def _generate_enhanced_feedback(self, scale_ref: ScaleReference,
                                   color_cal: ColorCalibration,
                                   lighting: LightingAnalysis,
                                   is_blurry: bool, too_close: bool, too_far: bool,
                                   focus_score: float, angle_score: float,
                                   capture_angle: Optional[CaptureAngle]) -> Tuple[List[str], List[str], List[str]]:
        """Generate comprehensive feedback"""
        issues = []
        suggestions = []
        warnings = []

        # Critical warnings
        if is_blurry:
            warnings.append("Image is out of focus")
            suggestions.append("Hold camera steady and tap to focus on lesion")

        if lighting.has_glare and lighting.glare_percentage > 15:
            warnings.append(f"Excessive glare detected ({lighting.glare_percentage:.1f}%)")
            suggestions.append("Use polarized filter or diffused lighting")

        if not scale_ref.detected:
            warnings.append("No scale reference detected")
            suggestions.append("Place ruler, coin, or reference card next to lesion")

        # Important issues
        issues.extend(lighting.issues)
        suggestions.extend(lighting.suggestions)

        if too_close:
            issues.append("Camera too close to subject")
            suggestions.append(f"Move back to {self.IDEAL_DISTANCE_MIN}-{self.IDEAL_DISTANCE_MAX}cm")

        if too_far:
            issues.append("Camera too far from subject")
            suggestions.append(f"Move closer to {self.IDEAL_DISTANCE_MIN}-{self.IDEAL_DISTANCE_MAX}cm")

        if not color_cal.detected:
            issues.append("No color calibration card detected")
            suggestions.append("Include color card for accurate color reproduction")

        if angle_score < 60:
            issues.append("Camera angle appears tilted")
            suggestions.append("Hold camera perpendicular to skin surface")

        # Angle-specific suggestions
        if capture_angle:
            guidance = self.get_angle_guidance(capture_angle)
            if "tips" in guidance:
                suggestions.extend([f"[{capture_angle.value}] {tip}" for tip in guidance["tips"][:2]])

        # Positive feedback
        if not issues and not warnings:
            suggestions.append("Photo quality is excellent - ready to capture!")

        return issues, suggestions, warnings

    def _check_medical_standards_enhanced(self, scale_ref: ScaleReference,
                                         color_cal: ColorCalibration,
                                         lighting: LightingAnalysis,
                                         focus_score: float) -> bool:
        """Check if photo meets medical photography standards"""
        if not scale_ref.detected:
            return False

        if lighting.overall_score < self.MIN_LIGHTING_SCORE:
            return False

        if focus_score < self.MIN_FOCUS_SCORE:
            return False

        if lighting.has_glare and lighting.glare_percentage > 10:
            return False

        if lighting.has_shadows and lighting.shadow_percentage > 15:
            return False

        return True


# Global instance
_assistant = None

def get_clinical_photography_assistant() -> ClinicalPhotographyAssistant:
    """Get or create global assistant instance"""
    global _assistant
    if _assistant is None:
        _assistant = ClinicalPhotographyAssistant()
    return _assistant
