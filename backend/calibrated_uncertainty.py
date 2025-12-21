"""
Calibrated Uncertainty Module

This module converts raw model confidence scores into clinically meaningful
uncertainty categories. It addresses the problem that model confidence != diagnostic accuracy.

Key Concepts:
1. Model Confidence: What the neural network outputs (often overconfident)
2. Calibrated Probability: Adjusted probability that better reflects true accuracy
3. Clinical Uncertainty: Human-interpretable categories for decision-making

The goal is to NEVER provide false reassurance. When in doubt, recommend evaluation.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math


class ClinicalConcernLevel(str, Enum):
    """Clinical concern levels for skin lesion assessment"""
    HIGH_CONCERN = "high_concern"           # Urgent evaluation needed
    MODERATE_CONCERN = "moderate_concern"   # Schedule dermatology appointment
    LOW_CONCERN = "low_concern"             # Monitor for changes
    UNCERTAIN = "uncertain"                 # Clinical evaluation recommended
    INSUFFICIENT_DATA = "insufficient_data" # Cannot make assessment


@dataclass
class CalibratedResult:
    """Calibrated uncertainty result with clinical guidance"""
    concern_level: ClinicalConcernLevel
    concern_label: str
    concern_description: str
    action_recommendation: str

    # Uncertainty metrics
    model_confidence: float          # Raw model output
    calibrated_confidence: float     # After calibration adjustment
    uncertainty_score: float         # 0-1, higher = more uncertain

    # Clinical language (avoiding definitive statements)
    clinical_impression: str         # e.g., "Concerning features detected"
    clinical_caveats: List[str]      # Important limitations to communicate

    # For transparency
    calibration_applied: str         # Description of calibration method
    factors_considered: List[str]    # What influenced the assessment


# Calibration parameters based on typical neural network overconfidence
# These should ideally be learned from validation data
CALIBRATION_CONFIG = {
    # Temperature scaling factor (>1 reduces confidence, <1 increases)
    "temperature": 1.5,

    # Confidence thresholds for concern levels
    "high_concern_threshold": 0.70,      # Calibrated prob for high-risk class
    "moderate_concern_threshold": 0.40,
    "uncertainty_threshold": 0.25,       # Below this = uncertain

    # Entropy threshold for detecting uncertain predictions
    "high_entropy_threshold": 0.7,       # Normalized entropy

    # Minimum samples for reliable prediction
    "min_confidence_any_class": 0.20,    # If max prob < this, insufficient data
}


# High-risk conditions that should trigger elevated concern
HIGH_RISK_CONDITIONS = {
    "melanoma", "Melanoma",
    "squamous_cell_carcinoma", "Squamous Cell Carcinoma", "SCC",
    "basal_cell_carcinoma", "Basal Cell Carcinoma", "BCC",
    "merkel_cell_carcinoma", "Merkel Cell Carcinoma",
    "actinic_keratosis", "Actinic Keratosis",  # Pre-cancerous
}

# Conditions that require monitoring but lower immediate concern
MODERATE_RISK_CONDITIONS = {
    "dysplastic_nevus", "Dysplastic Nevus", "atypical_mole",
    "seborrheic_keratosis", "Seborrheic Keratosis",  # Benign but can mimic melanoma
    "dermatofibroma", "Dermatofibroma",
    "lentigo", "Lentigo",
}

# Conditions where we should never say "benign" definitively
NEVER_DISMISS = {
    "melanoma", "Melanoma",
    "dysplastic_nevus", "Dysplastic Nevus",
}


def calculate_entropy(probabilities: Dict[str, float]) -> float:
    """
    Calculate normalized entropy of probability distribution.
    Higher entropy = more uncertainty.
    Returns value between 0 and 1.
    """
    if not probabilities:
        return 1.0

    probs = [p for p in probabilities.values() if p > 0]
    if len(probs) <= 1:
        return 0.0

    # Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    # Normalize by max possible entropy (uniform distribution)
    max_entropy = math.log2(len(probs))
    if max_entropy == 0:
        return 0.0

    return min(entropy / max_entropy, 1.0)


def apply_temperature_scaling(
    probabilities: Dict[str, float],
    temperature: float = 1.5
) -> Dict[str, float]:
    """
    Apply temperature scaling to calibrate probabilities.
    Temperature > 1 reduces overconfidence (softens distribution)
    Temperature < 1 increases confidence (sharpens distribution)
    """
    if not probabilities or temperature <= 0:
        return probabilities

    # Convert to logits, apply temperature, convert back
    # For probabilities, we use: p_calibrated = p^(1/T) / sum(p^(1/T))
    scaled = {}
    for k, p in probabilities.items():
        if p > 0:
            scaled[k] = p ** (1.0 / temperature)
        else:
            scaled[k] = 0.0

    # Renormalize
    total = sum(scaled.values())
    if total > 0:
        scaled = {k: v / total for k, v in scaled.items()}

    return scaled


def get_uncertainty_score(
    probabilities: Dict[str, float],
    predicted_class: str,
    calibrated_confidence: float
) -> float:
    """
    Calculate overall uncertainty score (0-1, higher = more uncertain).
    Combines multiple signals:
    - Entropy of distribution
    - Gap between top predictions
    - Raw vs calibrated confidence difference
    """
    if not probabilities:
        return 1.0

    # Get sorted probabilities
    sorted_probs = sorted(probabilities.values(), reverse=True)

    # 1. Entropy component (0-1)
    entropy = calculate_entropy(probabilities)

    # 2. Confidence gap component
    # Small gap between top 2 predictions = more uncertain
    if len(sorted_probs) >= 2:
        confidence_gap = sorted_probs[0] - sorted_probs[1]
        gap_uncertainty = 1.0 - confidence_gap  # Invert: small gap = high uncertainty
    else:
        gap_uncertainty = 0.0

    # 3. Calibration adjustment component
    # Large difference between raw and calibrated = model was overconfident
    calibration_uncertainty = 1.0 - calibrated_confidence

    # Weighted combination
    uncertainty = (
        0.4 * entropy +
        0.3 * gap_uncertainty +
        0.3 * calibration_uncertainty
    )

    return min(max(uncertainty, 0.0), 1.0)


def determine_concern_level(
    predicted_class: str,
    calibrated_confidence: float,
    uncertainty_score: float,
    probabilities: Dict[str, float],
    clinical_context: Optional[Dict] = None
) -> ClinicalConcernLevel:
    """
    Determine clinical concern level based on prediction and uncertainty.
    Errs on the side of caution - when uncertain, recommends evaluation.
    """
    config = CALIBRATION_CONFIG

    # Check if we have enough signal
    max_prob = max(probabilities.values()) if probabilities else 0
    if max_prob < config["min_confidence_any_class"]:
        return ClinicalConcernLevel.INSUFFICIENT_DATA

    # High uncertainty = recommend clinical evaluation
    if uncertainty_score > 0.6:
        return ClinicalConcernLevel.UNCERTAIN

    # Check for high-risk conditions
    is_high_risk_predicted = predicted_class in HIGH_RISK_CONDITIONS

    # Check if any high-risk condition has significant probability
    high_risk_probability = sum(
        probabilities.get(cond, 0)
        for cond in HIGH_RISK_CONDITIONS
        if cond in probabilities
    )

    # HIGH CONCERN: High-risk condition with sufficient confidence
    # OR any meaningful probability of high-risk condition
    if is_high_risk_predicted and calibrated_confidence >= config["high_concern_threshold"]:
        return ClinicalConcernLevel.HIGH_CONCERN

    if high_risk_probability >= 0.30:  # >30% chance of any cancer = high concern
        return ClinicalConcernLevel.HIGH_CONCERN

    # Clinical context can elevate concern
    if clinical_context:
        risk_multiplier = clinical_context.get("calculated_risk_multiplier", 1.0)
        if risk_multiplier >= 2.0 and high_risk_probability >= 0.15:
            return ClinicalConcernLevel.HIGH_CONCERN

    # MODERATE CONCERN: Moderate confidence or moderate-risk condition
    if is_high_risk_predicted and calibrated_confidence >= config["moderate_concern_threshold"]:
        return ClinicalConcernLevel.MODERATE_CONCERN

    is_moderate_risk = predicted_class in MODERATE_RISK_CONDITIONS
    if is_moderate_risk and calibrated_confidence >= config["moderate_concern_threshold"]:
        return ClinicalConcernLevel.MODERATE_CONCERN

    if high_risk_probability >= 0.15:  # 15-30% chance = moderate concern
        return ClinicalConcernLevel.MODERATE_CONCERN

    # UNCERTAIN: Below thresholds but not clearly low risk
    if uncertainty_score > 0.4 or calibrated_confidence < config["uncertainty_threshold"]:
        return ClinicalConcernLevel.UNCERTAIN

    # LOW CONCERN: High confidence in low-risk condition
    # But never say "benign" - say "no obvious concerning features"
    return ClinicalConcernLevel.LOW_CONCERN


def get_clinical_impression(
    predicted_class: str,
    concern_level: ClinicalConcernLevel,
    calibrated_confidence: float
) -> str:
    """
    Generate clinical impression language that avoids definitive statements.
    Never says "benign" or "definitely not cancer".
    """
    impressions = {
        ClinicalConcernLevel.HIGH_CONCERN: (
            f"Features suggestive of {predicted_class.replace('_', ' ')} detected. "
            "Professional evaluation strongly recommended."
        ),
        ClinicalConcernLevel.MODERATE_CONCERN: (
            f"Some features consistent with {predicted_class.replace('_', ' ')} observed. "
            "Dermatology consultation advised."
        ),
        ClinicalConcernLevel.LOW_CONCERN: (
            "No obvious concerning features detected in this image. "
            "Continue monitoring for any changes."
        ),
        ClinicalConcernLevel.UNCERTAIN: (
            "The analysis is inconclusive for this image. "
            "Clinical evaluation is recommended for proper assessment."
        ),
        ClinicalConcernLevel.INSUFFICIENT_DATA: (
            "Unable to make a reliable assessment from this image. "
            "Please consult a healthcare provider."
        ),
    }
    return impressions.get(concern_level, impressions[ClinicalConcernLevel.UNCERTAIN])


def get_concern_details(concern_level: ClinicalConcernLevel) -> Tuple[str, str, str]:
    """
    Get label, description, and action for each concern level.
    Returns (label, description, action_recommendation)
    """
    details = {
        ClinicalConcernLevel.HIGH_CONCERN: (
            "High Concern - Urgent Evaluation Needed",
            "This lesion shows features that warrant prompt medical attention. "
            "This is NOT a diagnosis, but indicates features that a dermatologist should evaluate.",
            "Schedule an urgent appointment with a dermatologist within 1-2 weeks. "
            "If you notice rapid changes, bleeding, or pain, seek care sooner."
        ),
        ClinicalConcernLevel.MODERATE_CONCERN: (
            "Moderate Concern - Dermatology Appointment Recommended",
            "This lesion shows some features that should be evaluated by a professional. "
            "While not necessarily urgent, a dermatology consultation is advised.",
            "Schedule an appointment with a dermatologist within the next few weeks. "
            "Monitor for any changes in the meantime and note them for your appointment."
        ),
        ClinicalConcernLevel.LOW_CONCERN: (
            "Lower Concern - Monitor for Changes",
            "No obvious concerning features were detected in this image. However, this is "
            "NOT a guarantee that the lesion is benign. Any skin lesion can change over time.",
            "Continue self-monitoring using the ABCDE method (Asymmetry, Border, Color, "
            "Diameter, Evolution). If you notice any changes, seek professional evaluation. "
            "Consider annual skin checks with a dermatologist."
        ),
        ClinicalConcernLevel.UNCERTAIN: (
            "Uncertain - Clinical Evaluation Recommended",
            "The AI analysis could not make a confident assessment of this lesion. "
            "This may be due to image quality, unusual presentation, or overlapping features.",
            "We recommend clinical evaluation by a healthcare provider who can perform a "
            "hands-on examination. This uncertainty is not a diagnosis - it simply means "
            "the image requires expert review."
        ),
        ClinicalConcernLevel.INSUFFICIENT_DATA: (
            "Unable to Assess - Professional Consultation Needed",
            "There was insufficient information in this image to make any assessment. "
            "This could be due to image quality, lighting, or other technical factors.",
            "Please consult a healthcare provider for proper evaluation. Consider retaking "
            "the image with better lighting and focus if you wish to try again."
        ),
    }
    return details.get(concern_level, details[ClinicalConcernLevel.UNCERTAIN])


def get_clinical_caveats(
    predicted_class: str,
    concern_level: ClinicalConcernLevel,
    uncertainty_score: float
) -> List[str]:
    """
    Generate important caveats and limitations to communicate to users.
    These are critical for informed decision-making.
    """
    caveats = []

    # Universal caveats
    caveats.append(
        "This AI tool is for informational purposes only and is NOT a medical diagnosis."
    )
    caveats.append(
        "Only a qualified healthcare provider can diagnose skin conditions through "
        "clinical examination, and often biopsy is required for definitive diagnosis."
    )

    # Condition-specific caveats
    if predicted_class in NEVER_DISMISS:
        caveats.append(
            f"{predicted_class.replace('_', ' ')} can be difficult to distinguish from "
            "benign lesions. When in doubt, always seek professional evaluation."
        )

    # Uncertainty-based caveats
    if uncertainty_score > 0.5:
        caveats.append(
            "The AI showed significant uncertainty in this assessment. "
            "Professional evaluation is especially important in this case."
        )

    # Concern level caveats
    if concern_level == ClinicalConcernLevel.LOW_CONCERN:
        caveats.append(
            "'Low concern' does NOT mean 'definitely benign'. Skin cancer can "
            "sometimes have atypical presentations that fool both AI and clinicians."
        )
        caveats.append(
            "If you have any personal or family history of skin cancer, regular "
            "professional skin checks are recommended regardless of this result."
        )

    if concern_level == ClinicalConcernLevel.HIGH_CONCERN:
        caveats.append(
            "High concern is based on visual features only. Many benign conditions "
            "can mimic concerning lesions. Do not panic, but do seek prompt evaluation."
        )

    return caveats


def calibrate_prediction(
    predicted_class: str,
    raw_confidence: float,
    probabilities: Dict[str, float],
    clinical_context: Optional[Dict] = None,
    uncertainty_metrics: Optional[Dict] = None
) -> CalibratedResult:
    """
    Main function to convert raw model output into calibrated clinical assessment.

    Args:
        predicted_class: The model's top prediction
        raw_confidence: Raw model confidence (0-1)
        probabilities: Full probability distribution over classes
        clinical_context: Patient clinical context (from ClinicalContextForm)
        uncertainty_metrics: Monte Carlo dropout uncertainty if available

    Returns:
        CalibratedResult with clinical-grade uncertainty assessment
    """
    # Apply temperature scaling to calibrate probabilities
    calibrated_probs = apply_temperature_scaling(
        probabilities,
        CALIBRATION_CONFIG["temperature"]
    )
    calibrated_confidence = calibrated_probs.get(predicted_class, raw_confidence * 0.7)

    # Calculate uncertainty score
    uncertainty_score = get_uncertainty_score(
        calibrated_probs, predicted_class, calibrated_confidence
    )

    # Incorporate Monte Carlo uncertainty if available
    if uncertainty_metrics:
        mc_uncertainty = uncertainty_metrics.get("epistemic_uncertainty", 0)
        if mc_uncertainty > 0.3:  # High epistemic uncertainty
            uncertainty_score = max(uncertainty_score, 0.6)

    # Determine concern level
    concern_level = determine_concern_level(
        predicted_class,
        calibrated_confidence,
        uncertainty_score,
        calibrated_probs,
        clinical_context
    )

    # Get clinical language
    label, description, action = get_concern_details(concern_level)
    impression = get_clinical_impression(predicted_class, concern_level, calibrated_confidence)
    caveats = get_clinical_caveats(predicted_class, concern_level, uncertainty_score)

    # Track factors that influenced the assessment
    factors = []
    factors.append(f"Primary prediction: {predicted_class}")
    factors.append(f"Calibrated confidence: {calibrated_confidence:.1%}")
    factors.append(f"Uncertainty score: {uncertainty_score:.2f}")

    if clinical_context:
        risk_mult = clinical_context.get("calculated_risk_multiplier", 1.0)
        if risk_mult != 1.0:
            factors.append(f"Clinical risk adjustment: {risk_mult:.1f}x")

    if uncertainty_metrics:
        factors.append("Monte Carlo uncertainty analysis included")

    return CalibratedResult(
        concern_level=concern_level,
        concern_label=label,
        concern_description=description,
        action_recommendation=action,
        model_confidence=raw_confidence,
        calibrated_confidence=calibrated_confidence,
        uncertainty_score=uncertainty_score,
        clinical_impression=impression,
        clinical_caveats=caveats,
        calibration_applied="Temperature scaling (T=1.5) with entropy-based uncertainty",
        factors_considered=factors
    )


def format_calibrated_result_for_response(result: CalibratedResult) -> Dict:
    """Convert CalibratedResult to dictionary for API response"""
    return {
        "concern_level": result.concern_level.value,
        "concern_label": result.concern_label,
        "concern_description": result.concern_description,
        "action_recommendation": result.action_recommendation,
        "model_confidence": result.model_confidence,
        "calibrated_confidence": result.calibrated_confidence,
        "uncertainty_score": result.uncertainty_score,
        "clinical_impression": result.clinical_impression,
        "clinical_caveats": result.clinical_caveats,
        "calibration_applied": result.calibration_applied,
        "factors_considered": result.factors_considered,
        # Derived fields for easy frontend use
        "is_high_concern": result.concern_level == ClinicalConcernLevel.HIGH_CONCERN,
        "is_uncertain": result.concern_level == ClinicalConcernLevel.UNCERTAIN,
        "requires_evaluation": result.concern_level in [
            ClinicalConcernLevel.HIGH_CONCERN,
            ClinicalConcernLevel.MODERATE_CONCERN,
            ClinicalConcernLevel.UNCERTAIN,
            ClinicalConcernLevel.INSUFFICIENT_DATA
        ],
        "show_confidence_percentage": False,  # Signal to frontend to NOT show raw %
    }
