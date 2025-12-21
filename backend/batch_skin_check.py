"""
Batch Processing & Full-Body Skin Check System

Features:
- Upload 20-30 photos of entire body surface
- AI automatically tags body location for each lesion
- Generate comprehensive full-body report
- Flag highest-risk lesions first
- Create "mole map" visualization
- Track total lesion count over time
"""

import json
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import re
import math
from collections import defaultdict


class BodyRegion(Enum):
    """Body regions for classification"""
    HEAD = "head"
    FACE = "face"
    NECK = "neck"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    BACK_UPPER = "back_upper"
    BACK_LOWER = "back_lower"
    SHOULDER_LEFT = "shoulder_left"
    SHOULDER_RIGHT = "shoulder_right"
    ARM_LEFT_UPPER = "arm_left_upper"
    ARM_RIGHT_UPPER = "arm_right_upper"
    ARM_LEFT_LOWER = "arm_left_lower"
    ARM_RIGHT_LOWER = "arm_right_lower"
    HAND_LEFT = "hand_left"
    HAND_RIGHT = "hand_right"
    HIP_LEFT = "hip_left"
    HIP_RIGHT = "hip_right"
    THIGH_LEFT = "thigh_left"
    THIGH_RIGHT = "thigh_right"
    KNEE_LEFT = "knee_left"
    KNEE_RIGHT = "knee_right"
    LEG_LEFT_LOWER = "leg_left_lower"
    LEG_RIGHT_LOWER = "leg_right_lower"
    FOOT_LEFT = "foot_left"
    FOOT_RIGHT = "foot_right"
    GENITAL = "genital"
    BUTTOCK = "buttock"


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LesionDetection:
    """Individual lesion detected in an image"""
    lesion_id: str
    image_index: int
    bounding_box: Dict[str, float]  # x, y, width, height (normalized 0-1)
    confidence: float
    predicted_class: str
    risk_level: RiskLevel
    risk_score: float  # 0-100
    body_location: Optional[BodyRegion] = None
    body_map_coordinates: Optional[Dict[str, float]] = None  # x, y on body map
    features: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ImageAnalysisResult:
    """Result of analyzing a single image"""
    image_id: str
    image_index: int
    filename: str
    body_region_detected: Optional[BodyRegion]
    body_region_confidence: float
    lesions_detected: List[LesionDetection]
    image_quality_score: float
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FullBodyCheckResult:
    """Complete full-body skin check result"""
    check_id: str
    user_id: int
    created_at: str
    total_images: int
    total_lesions: int
    images_analyzed: List[ImageAnalysisResult]
    risk_summary: Dict[str, int]  # Count by risk level
    body_coverage: Dict[str, bool]  # Which body regions were covered
    highest_risk_lesions: List[LesionDetection]
    mole_map: Dict[str, List[Dict]]  # Body region -> lesions for visualization
    lesion_count_by_region: Dict[str, int]
    recommendations: List[str]
    overall_risk_score: float
    comparison_with_previous: Optional[Dict[str, Any]] = None
    report_generated: bool = False


class BodyLocationClassifier:
    """
    AI-based body location classifier.
    Determines which body region an image shows based on visual features.
    """

    def __init__(self):
        # Keywords and patterns for body region detection from image metadata/filename
        self.region_keywords = {
            BodyRegion.HEAD: ["head", "scalp", "hair", "top"],
            BodyRegion.FACE: ["face", "forehead", "cheek", "nose", "chin", "eye", "lip", "temple"],
            BodyRegion.NECK: ["neck", "throat"],
            BodyRegion.CHEST: ["chest", "breast", "pectoral", "sternum"],
            BodyRegion.ABDOMEN: ["abdomen", "stomach", "belly", "navel", "umbilical", "tummy"],
            BodyRegion.BACK_UPPER: ["upper back", "shoulder blade", "scapula", "thoracic"],
            BodyRegion.BACK_LOWER: ["lower back", "lumbar", "sacral"],
            BodyRegion.SHOULDER_LEFT: ["left shoulder"],
            BodyRegion.SHOULDER_RIGHT: ["right shoulder"],
            BodyRegion.ARM_LEFT_UPPER: ["left upper arm", "left bicep", "left tricep"],
            BodyRegion.ARM_RIGHT_UPPER: ["right upper arm", "right bicep", "right tricep"],
            BodyRegion.ARM_LEFT_LOWER: ["left forearm", "left wrist"],
            BodyRegion.ARM_RIGHT_LOWER: ["right forearm", "right wrist"],
            BodyRegion.HAND_LEFT: ["left hand", "left palm", "left finger"],
            BodyRegion.HAND_RIGHT: ["right hand", "right palm", "right finger"],
            BodyRegion.HIP_LEFT: ["left hip"],
            BodyRegion.HIP_RIGHT: ["right hip"],
            BodyRegion.THIGH_LEFT: ["left thigh", "left quad"],
            BodyRegion.THIGH_RIGHT: ["right thigh", "right quad"],
            BodyRegion.KNEE_LEFT: ["left knee"],
            BodyRegion.KNEE_RIGHT: ["right knee"],
            BodyRegion.LEG_LEFT_LOWER: ["left calf", "left shin", "left lower leg"],
            BodyRegion.LEG_RIGHT_LOWER: ["right calf", "right shin", "right lower leg"],
            BodyRegion.FOOT_LEFT: ["left foot", "left ankle", "left toe", "left heel"],
            BodyRegion.FOOT_RIGHT: ["right foot", "right ankle", "right toe", "right heel"],
            BodyRegion.GENITAL: ["genital", "groin", "pubic"],
            BodyRegion.BUTTOCK: ["buttock", "gluteal", "bottom"],
        }

        # Body map coordinates for each region (percentage 0-100)
        self.region_coordinates = {
            BodyRegion.HEAD: {"x": 50, "y": 5},
            BodyRegion.FACE: {"x": 50, "y": 8},
            BodyRegion.NECK: {"x": 50, "y": 15},
            BodyRegion.CHEST: {"x": 50, "y": 28},
            BodyRegion.ABDOMEN: {"x": 50, "y": 40},
            BodyRegion.BACK_UPPER: {"x": 50, "y": 28},
            BodyRegion.BACK_LOWER: {"x": 50, "y": 42},
            BodyRegion.SHOULDER_LEFT: {"x": 25, "y": 22},
            BodyRegion.SHOULDER_RIGHT: {"x": 75, "y": 22},
            BodyRegion.ARM_LEFT_UPPER: {"x": 18, "y": 32},
            BodyRegion.ARM_RIGHT_UPPER: {"x": 82, "y": 32},
            BodyRegion.ARM_LEFT_LOWER: {"x": 14, "y": 44},
            BodyRegion.ARM_RIGHT_LOWER: {"x": 86, "y": 44},
            BodyRegion.HAND_LEFT: {"x": 10, "y": 52},
            BodyRegion.HAND_RIGHT: {"x": 90, "y": 52},
            BodyRegion.HIP_LEFT: {"x": 40, "y": 48},
            BodyRegion.HIP_RIGHT: {"x": 60, "y": 48},
            BodyRegion.THIGH_LEFT: {"x": 40, "y": 60},
            BodyRegion.THIGH_RIGHT: {"x": 60, "y": 60},
            BodyRegion.KNEE_LEFT: {"x": 40, "y": 72},
            BodyRegion.KNEE_RIGHT: {"x": 60, "y": 72},
            BodyRegion.LEG_LEFT_LOWER: {"x": 40, "y": 82},
            BodyRegion.LEG_RIGHT_LOWER: {"x": 60, "y": 82},
            BodyRegion.FOOT_LEFT: {"x": 40, "y": 94},
            BodyRegion.FOOT_RIGHT: {"x": 60, "y": 94},
            BodyRegion.GENITAL: {"x": 50, "y": 50},
            BodyRegion.BUTTOCK: {"x": 50, "y": 52},
        }

    def classify_from_filename(self, filename: str) -> Tuple[Optional[BodyRegion], float]:
        """Classify body region from filename"""
        filename_lower = filename.lower()

        for region, keywords in self.region_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return region, 0.9  # High confidence from filename

        return None, 0.0

    def classify_from_metadata(self, metadata: Dict[str, Any]) -> Tuple[Optional[BodyRegion], float]:
        """Classify body region from image metadata or user-provided data"""
        body_location = metadata.get("body_location", "").lower()
        body_sublocation = metadata.get("body_sublocation", "").lower()

        combined = f"{body_location} {body_sublocation}"

        for region, keywords in self.region_keywords.items():
            for keyword in keywords:
                if keyword in combined:
                    return region, 0.95  # Very high confidence from metadata

        return None, 0.0

    def classify_from_image_features(self, image_features: Dict[str, Any]) -> Tuple[Optional[BodyRegion], float]:
        """
        Classify body region from image visual features.
        In production, this would use a CNN trained on body region classification.
        For now, uses heuristics based on image characteristics.
        """
        # Placeholder - in production, integrate with body region CNN
        # Features could include:
        # - Skin texture patterns
        # - Hair presence/type
        # - Curvature/shape
        # - Relative position indicators

        # Simulated classification based on aspect ratio and other features
        aspect_ratio = image_features.get("aspect_ratio", 1.0)
        has_hair = image_features.get("has_hair", False)
        skin_texture = image_features.get("skin_texture", "smooth")

        # Basic heuristics (would be replaced by ML model)
        if has_hair and aspect_ratio > 1.2:
            return BodyRegion.HEAD, 0.6
        elif aspect_ratio < 0.7:
            return BodyRegion.ARM_LEFT_LOWER, 0.5  # Elongated images might be limbs

        return None, 0.0

    def get_body_map_coordinates(self, region: BodyRegion,
                                  offset_x: float = 0, offset_y: float = 0) -> Dict[str, float]:
        """Get body map coordinates for a region with optional offset"""
        base_coords = self.region_coordinates.get(region, {"x": 50, "y": 50})
        return {
            "x": base_coords["x"] + offset_x,
            "y": base_coords["y"] + offset_y
        }


class RiskScorer:
    """
    Calculate risk scores for lesions based on multiple factors.
    """

    # Risk weights for different conditions
    CONDITION_RISK_WEIGHTS = {
        "melanoma": 100,
        "basal cell carcinoma": 75,
        "squamous cell carcinoma": 80,
        "actinic keratosis": 50,
        "dysplastic nevus": 45,
        "atypical mole": 40,
        "seborrheic keratosis": 10,
        "dermatofibroma": 5,
        "benign nevus": 5,
        "cherry angioma": 2,
        "melanocytic nevi": 15,
        "vascular lesion": 10,
    }

    # ABCDE criteria weights
    ABCDE_WEIGHTS = {
        "asymmetry": 20,
        "border_irregularity": 20,
        "color_variation": 20,
        "diameter_large": 15,
        "evolution": 25,
    }

    def calculate_risk_score(self, predicted_class: str, confidence: float,
                             features: Dict[str, Any]) -> Tuple[float, RiskLevel]:
        """Calculate overall risk score (0-100) and risk level"""
        score = 0.0

        # Base score from predicted condition
        condition_lower = predicted_class.lower()
        for condition, weight in self.CONDITION_RISK_WEIGHTS.items():
            if condition in condition_lower:
                score += weight * confidence
                break
        else:
            score += 20 * confidence  # Default for unknown conditions

        # ABCDE criteria contribution
        abcde_score = 0
        for criterion, weight in self.ABCDE_WEIGHTS.items():
            if features.get(criterion, False):
                abcde_score += weight
        score += abcde_score * 0.3  # ABCDE contributes up to 30 points

        # Size contribution
        diameter_mm = features.get("diameter_mm", 0)
        if diameter_mm > 6:
            score += min(15, (diameter_mm - 6) * 2)

        # Growth rate contribution
        growth_rate = features.get("growth_rate_mm_per_month", 0)
        if growth_rate > 0:
            score += min(10, growth_rate * 5)

        # Cap at 100
        score = min(100, score)

        # Determine risk level
        if score >= 70:
            risk_level = RiskLevel.CRITICAL
        elif score >= 50:
            risk_level = RiskLevel.HIGH
        elif score >= 25:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return score, risk_level


class MoleMapGenerator:
    """
    Generate mole map visualization data.
    """

    def __init__(self):
        self.body_classifier = BodyLocationClassifier()

    def generate_mole_map(self, lesions: List[LesionDetection]) -> Dict[str, List[Dict]]:
        """Generate mole map data structure for visualization"""
        mole_map = defaultdict(list)

        for lesion in lesions:
            if lesion.body_location:
                region_key = lesion.body_location.value
                coords = lesion.body_map_coordinates or self.body_classifier.get_body_map_coordinates(
                    lesion.body_location
                )

                mole_map[region_key].append({
                    "lesion_id": lesion.lesion_id,
                    "x": coords.get("x", 50),
                    "y": coords.get("y", 50),
                    "risk_level": lesion.risk_level.value,
                    "risk_score": lesion.risk_score,
                    "predicted_class": lesion.predicted_class,
                    "confidence": lesion.confidence,
                })

        return dict(mole_map)

    def calculate_body_coverage(self, images: List[ImageAnalysisResult]) -> Dict[str, bool]:
        """Calculate which body regions have been covered"""
        covered = {region.value: False for region in BodyRegion}

        for image in images:
            if image.body_region_detected:
                covered[image.body_region_detected.value] = True

        return covered

    def get_missing_regions(self, coverage: Dict[str, bool]) -> List[str]:
        """Get list of body regions not yet photographed"""
        # Essential regions that should be checked
        essential_regions = [
            BodyRegion.FACE.value,
            BodyRegion.CHEST.value,
            BodyRegion.ABDOMEN.value,
            BodyRegion.BACK_UPPER.value,
            BodyRegion.BACK_LOWER.value,
            BodyRegion.ARM_LEFT_UPPER.value,
            BodyRegion.ARM_RIGHT_UPPER.value,
            BodyRegion.ARM_LEFT_LOWER.value,
            BodyRegion.ARM_RIGHT_LOWER.value,
            BodyRegion.THIGH_LEFT.value,
            BodyRegion.THIGH_RIGHT.value,
            BodyRegion.LEG_LEFT_LOWER.value,
            BodyRegion.LEG_RIGHT_LOWER.value,
        ]

        return [region for region in essential_regions if not coverage.get(region, False)]


class BatchSkinCheckProcessor:
    """
    Main processor for batch skin check operations.
    """

    def __init__(self):
        self.body_classifier = BodyLocationClassifier()
        self.risk_scorer = RiskScorer()
        self.mole_map_generator = MoleMapGenerator()

    async def process_batch(self, user_id: int, images: List[Dict[str, Any]],
                            classify_callback=None) -> FullBodyCheckResult:
        """
        Process a batch of images for full-body skin check.

        Args:
            user_id: User ID
            images: List of image data dicts with keys:
                    - image_data: base64 or file path
                    - filename: original filename
                    - metadata: optional metadata dict
            classify_callback: Async function to classify a single image
                             signature: async (image_data) -> classification_result

        Returns:
            FullBodyCheckResult with comprehensive analysis
        """
        check_id = str(uuid.uuid4())
        start_time = datetime.now()
        all_lesions: List[LesionDetection] = []
        image_results: List[ImageAnalysisResult] = []

        for idx, image_info in enumerate(images):
            result = await self._process_single_image(
                idx, image_info, classify_callback
            )
            image_results.append(result)
            all_lesions.extend(result.lesions_detected)

        # Generate mole map
        mole_map = self.mole_map_generator.generate_mole_map(all_lesions)

        # Calculate body coverage
        body_coverage = self.mole_map_generator.calculate_body_coverage(image_results)

        # Risk summary
        risk_summary = defaultdict(int)
        for lesion in all_lesions:
            risk_summary[lesion.risk_level.value] += 1

        # Lesion count by region
        lesion_count_by_region = defaultdict(int)
        for lesion in all_lesions:
            if lesion.body_location:
                lesion_count_by_region[lesion.body_location.value] += 1

        # Get highest risk lesions (sorted by risk score)
        highest_risk_lesions = sorted(
            all_lesions,
            key=lambda x: x.risk_score,
            reverse=True
        )[:10]  # Top 10

        # Calculate overall risk score
        if all_lesions:
            overall_risk = sum(l.risk_score for l in all_lesions) / len(all_lesions)
            # Boost if any high-risk lesions
            high_risk_count = risk_summary.get("high", 0) + risk_summary.get("critical", 0)
            if high_risk_count > 0:
                overall_risk = min(100, overall_risk + high_risk_count * 5)
        else:
            overall_risk = 0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            all_lesions, risk_summary, body_coverage
        )

        return FullBodyCheckResult(
            check_id=check_id,
            user_id=user_id,
            created_at=datetime.now().isoformat(),
            total_images=len(images),
            total_lesions=len(all_lesions),
            images_analyzed=image_results,
            risk_summary=dict(risk_summary),
            body_coverage=body_coverage,
            highest_risk_lesions=highest_risk_lesions,
            mole_map=mole_map,
            lesion_count_by_region=dict(lesion_count_by_region),
            recommendations=recommendations,
            overall_risk_score=overall_risk,
        )

    async def _process_single_image(self, idx: int, image_info: Dict[str, Any],
                                     classify_callback) -> ImageAnalysisResult:
        """Process a single image in the batch"""
        import time
        start = time.time()

        image_id = str(uuid.uuid4())
        filename = image_info.get("filename", f"image_{idx}.jpg")
        metadata = image_info.get("metadata", {})
        image_data = image_info.get("image_data")

        # Determine body region
        body_region, region_confidence = self._detect_body_region(filename, metadata)

        # Classify lesions in image
        lesions = []
        if classify_callback and image_data:
            try:
                classification = await classify_callback(image_data)
                lesions = self._extract_lesions_from_classification(
                    classification, idx, body_region
                )
            except Exception as e:
                print(f"Classification failed for image {idx}: {e}")

        # Calculate processing time
        processing_time = (time.time() - start) * 1000

        return ImageAnalysisResult(
            image_id=image_id,
            image_index=idx,
            filename=filename,
            body_region_detected=body_region,
            body_region_confidence=region_confidence,
            lesions_detected=lesions,
            image_quality_score=metadata.get("quality_score", 0.8),
            processing_time_ms=processing_time,
            metadata=metadata
        )

    def _detect_body_region(self, filename: str,
                            metadata: Dict[str, Any]) -> Tuple[Optional[BodyRegion], float]:
        """Detect body region from available information"""
        # Try metadata first (user-provided)
        region, confidence = self.body_classifier.classify_from_metadata(metadata)
        if region:
            return region, confidence

        # Try filename
        region, confidence = self.body_classifier.classify_from_filename(filename)
        if region:
            return region, confidence

        # Default to None if can't determine
        return None, 0.0

    def _extract_lesions_from_classification(self, classification: Dict[str, Any],
                                              image_index: int,
                                              body_region: Optional[BodyRegion]) -> List[LesionDetection]:
        """Extract lesion detections from classification result"""
        lesions = []

        # Handle single lesion classification (current system)
        if classification.get("is_lesion", False):
            predicted_class = classification.get("predicted_class", "unknown")
            confidence = classification.get("lesion_confidence", 0.5)
            features = classification.get("features", {})

            # Calculate risk
            risk_score, risk_level = self.risk_scorer.calculate_risk_score(
                predicted_class, confidence, features
            )

            # Get body map coordinates
            body_coords = None
            if body_region:
                body_coords = self.body_classifier.get_body_map_coordinates(body_region)

            lesion = LesionDetection(
                lesion_id=str(uuid.uuid4()),
                image_index=image_index,
                bounding_box={"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5},
                confidence=confidence,
                predicted_class=predicted_class,
                risk_level=risk_level,
                risk_score=risk_score,
                body_location=body_region,
                body_map_coordinates=body_coords,
                features=features,
                recommendations=self._get_lesion_recommendations(risk_level, predicted_class)
            )
            lesions.append(lesion)

        return lesions

    def _get_lesion_recommendations(self, risk_level: RiskLevel,
                                     predicted_class: str) -> List[str]:
        """Generate recommendations for a specific lesion"""
        recommendations = []

        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("URGENT: Schedule dermatologist appointment immediately")
            recommendations.append("Do not delay - biopsy may be needed")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Schedule dermatologist appointment within 2 weeks")
            recommendations.append("Monitor for any changes")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Schedule routine dermatology check within 1-2 months")
            recommendations.append("Take photos monthly to track changes")
        else:
            recommendations.append("Continue regular skin self-exams")
            recommendations.append("Recheck in 3-6 months")

        return recommendations

    def _generate_recommendations(self, lesions: List[LesionDetection],
                                   risk_summary: Dict[str, int],
                                   body_coverage: Dict[str, bool]) -> List[str]:
        """Generate overall recommendations for the full-body check"""
        recommendations = []

        # High-risk lesion recommendations
        critical_count = risk_summary.get("critical", 0)
        high_count = risk_summary.get("high", 0)

        if critical_count > 0:
            recommendations.append(
                f"URGENT: {critical_count} lesion(s) require immediate dermatologist evaluation"
            )
        if high_count > 0:
            recommendations.append(
                f"{high_count} lesion(s) should be evaluated by a dermatologist within 2 weeks"
            )

        # Coverage recommendations
        missing = self.mole_map_generator.get_missing_regions(body_coverage)
        if missing:
            missing_readable = [r.replace("_", " ").title() for r in missing[:5]]
            recommendations.append(
                f"Consider photographing: {', '.join(missing_readable)}"
            )

        # Total lesion count recommendations
        total = len(lesions)
        if total > 50:
            recommendations.append(
                "High lesion count detected. Regular dermatology follow-up recommended"
            )
        elif total > 20:
            recommendations.append(
                "Moderate lesion count. Annual full-body skin exam recommended"
            )

        # Sun protection
        recommendations.append("Use broad-spectrum SPF 30+ sunscreen daily")
        recommendations.append("Perform monthly skin self-exams")

        return recommendations


class FullBodyReportGenerator:
    """
    Generate comprehensive PDF/HTML reports for full-body skin checks.
    """

    def __init__(self):
        pass

    def generate_report_data(self, check_result: FullBodyCheckResult) -> Dict[str, Any]:
        """Generate report data structure for PDF/HTML generation"""
        return {
            "report_id": check_result.check_id,
            "generated_at": datetime.now().isoformat(),
            "patient_id": check_result.user_id,
            "check_date": check_result.created_at,

            "summary": {
                "total_images_analyzed": check_result.total_images,
                "total_lesions_detected": check_result.total_lesions,
                "overall_risk_score": round(check_result.overall_risk_score, 1),
                "risk_category": self._get_overall_risk_category(check_result.overall_risk_score),
            },

            "risk_breakdown": {
                "critical": check_result.risk_summary.get("critical", 0),
                "high": check_result.risk_summary.get("high", 0),
                "medium": check_result.risk_summary.get("medium", 0),
                "low": check_result.risk_summary.get("low", 0),
            },

            "body_coverage": {
                "regions_covered": sum(1 for v in check_result.body_coverage.values() if v),
                "total_regions": len(check_result.body_coverage),
                "coverage_percentage": round(
                    sum(1 for v in check_result.body_coverage.values() if v) /
                    len(check_result.body_coverage) * 100, 1
                ),
                "missing_regions": [
                    k for k, v in check_result.body_coverage.items() if not v
                ],
            },

            "lesion_distribution": check_result.lesion_count_by_region,

            "highest_priority_lesions": [
                {
                    "lesion_id": l.lesion_id,
                    "location": l.body_location.value if l.body_location else "Unknown",
                    "classification": l.predicted_class,
                    "risk_score": round(l.risk_score, 1),
                    "risk_level": l.risk_level.value,
                    "confidence": round(l.confidence * 100, 1),
                    "recommendations": l.recommendations,
                }
                for l in check_result.highest_risk_lesions[:5]
            ],

            "mole_map_data": check_result.mole_map,

            "recommendations": check_result.recommendations,

            "next_steps": self._get_next_steps(check_result),
        }

    def _get_overall_risk_category(self, score: float) -> str:
        """Get overall risk category label"""
        if score >= 70:
            return "High Risk - Immediate Action Required"
        elif score >= 50:
            return "Elevated Risk - Prompt Evaluation Recommended"
        elif score >= 25:
            return "Moderate Risk - Routine Follow-up Recommended"
        else:
            return "Low Risk - Continue Regular Monitoring"

    def _get_next_steps(self, result: FullBodyCheckResult) -> List[Dict[str, str]]:
        """Generate prioritized next steps"""
        steps = []

        if result.risk_summary.get("critical", 0) > 0:
            steps.append({
                "priority": "URGENT",
                "action": "Contact dermatologist immediately",
                "timeframe": "Within 24-48 hours",
                "reason": "Critical risk lesion(s) detected"
            })

        if result.risk_summary.get("high", 0) > 0:
            steps.append({
                "priority": "HIGH",
                "action": "Schedule dermatology appointment",
                "timeframe": "Within 2 weeks",
                "reason": "High risk lesion(s) require professional evaluation"
            })

        if result.risk_summary.get("medium", 0) > 0:
            steps.append({
                "priority": "MEDIUM",
                "action": "Schedule routine skin check",
                "timeframe": "Within 1-2 months",
                "reason": "Moderate risk lesions should be monitored"
            })

        steps.append({
            "priority": "ROUTINE",
            "action": "Schedule next full-body skin check",
            "timeframe": "In 3-6 months",
            "reason": "Regular monitoring recommended"
        })

        return steps


class LesionCountTracker:
    """
    Track lesion counts over time for trend analysis.
    """

    def __init__(self):
        pass

    def calculate_trends(self, current_check: FullBodyCheckResult,
                         previous_checks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate lesion count trends over time"""
        if not previous_checks:
            return {
                "has_history": False,
                "message": "First full-body skin check on record"
            }

        # Get counts from previous checks
        history = []
        for check in previous_checks:
            history.append({
                "date": check.get("created_at"),
                "total_lesions": check.get("total_lesions", 0),
                "high_risk_count": check.get("risk_summary", {}).get("high", 0) +
                                   check.get("risk_summary", {}).get("critical", 0)
            })

        # Sort by date
        history.sort(key=lambda x: x["date"])

        # Calculate trends
        current_total = current_check.total_lesions
        previous_total = history[-1]["total_lesions"] if history else 0

        change = current_total - previous_total
        change_percent = (change / previous_total * 100) if previous_total > 0 else 0

        return {
            "has_history": True,
            "total_previous_checks": len(history),
            "current_lesion_count": current_total,
            "previous_lesion_count": previous_total,
            "change": change,
            "change_percent": round(change_percent, 1),
            "trend": "increasing" if change > 0 else ("decreasing" if change < 0 else "stable"),
            "history": history,
            "recommendation": self._get_trend_recommendation(change, change_percent)
        }

    def _get_trend_recommendation(self, change: int, change_percent: float) -> str:
        """Get recommendation based on lesion count trend"""
        if change > 5 or change_percent > 20:
            return "Significant increase in lesion count. Consider more frequent monitoring."
        elif change > 0:
            return "Slight increase in lesion count. Continue regular monitoring."
        elif change < -5:
            return "Decrease in lesion count noted. Continue regular monitoring."
        else:
            return "Stable lesion count. Continue regular monitoring."


# Service instance
_batch_processor = None
_report_generator = None
_count_tracker = None


def get_batch_processor() -> BatchSkinCheckProcessor:
    """Get batch processor singleton"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchSkinCheckProcessor()
    return _batch_processor


def get_report_generator() -> FullBodyReportGenerator:
    """Get report generator singleton"""
    global _report_generator
    if _report_generator is None:
        _report_generator = FullBodyReportGenerator()
    return _report_generator


def get_count_tracker() -> LesionCountTracker:
    """Get count tracker singleton"""
    global _count_tracker
    if _count_tracker is None:
        _count_tracker = LesionCountTracker()
    return _count_tracker
