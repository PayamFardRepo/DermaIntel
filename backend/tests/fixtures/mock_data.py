"""
Mock data generators and factories for Skin Classifier tests.

This module provides factory classes and helper functions for generating
test data including users, analyses, and various domain-specific data.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


# =============================================================================
# CLASSIFICATION DATA
# =============================================================================

SKIN_CONDITIONS = {
    "lesion": [
        "melanocytic_nevus",
        "melanoma",
        "basal_cell_carcinoma",
        "actinic_keratosis",
        "benign_keratosis",
        "dermatofibroma",
        "vascular_lesion"
    ],
    "inflammatory": [
        "eczema",
        "psoriasis",
        "contact_dermatitis",
        "seborrheic_dermatitis",
        "rosacea",
        "acne",
        "urticaria"
    ],
    "infectious": [
        "ringworm",
        "impetigo",
        "cellulitis",
        "herpes_simplex",
        "molluscum_contagiosum",
        "scabies",
        "tinea_versicolor"
    ]
}

RISK_LEVELS = ["low", "medium", "high"]

BODY_LOCATIONS = [
    "face", "scalp", "neck", "chest", "back", "abdomen",
    "arm_left", "arm_right", "hand_left", "hand_right",
    "leg_left", "leg_right", "foot_left", "foot_right"
]

BODY_SUBLOCATIONS = {
    "face": ["forehead", "cheek", "nose", "chin", "ear"],
    "arm_left": ["upper_arm", "elbow", "forearm", "wrist"],
    "arm_right": ["upper_arm", "elbow", "forearm", "wrist"],
    "leg_left": ["thigh", "knee", "shin", "calf", "ankle"],
    "leg_right": ["thigh", "knee", "shin", "calf", "ankle"],
    "back": ["upper_back", "middle_back", "lower_back"],
    "chest": ["upper_chest", "lower_chest"]
}


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

def generate_classification_probabilities(
    predicted_class: str,
    confidence: float = 0.85,
    condition_type: str = "lesion"
) -> Dict[str, float]:
    """
    Generate realistic probability distributions for classification results.

    Args:
        predicted_class: The main predicted class
        confidence: Confidence level for the predicted class (0-1)
        condition_type: Type of condition ("lesion", "inflammatory", "infectious")

    Returns:
        Dictionary mapping class names to probabilities
    """
    conditions = SKIN_CONDITIONS.get(condition_type, SKIN_CONDITIONS["lesion"])
    remaining = 1.0 - confidence
    other_classes = [c for c in conditions if c != predicted_class]

    probabilities = {predicted_class: confidence}

    # Distribute remaining probability among other classes
    if other_classes:
        weights = [random.random() for _ in other_classes]
        total_weight = sum(weights)
        for cls, weight in zip(other_classes, weights):
            probabilities[cls] = round((weight / total_weight) * remaining, 4)

    return probabilities


def generate_mock_user_data(
    username: Optional[str] = None,
    email: Optional[str] = None,
    role: str = "patient"
) -> Dict[str, Any]:
    """
    Generate mock user registration data.

    Args:
        username: Optional specific username
        email: Optional specific email
        role: User role (patient, dermatologist, admin)

    Returns:
        Dictionary with user data
    """
    suffix = random.randint(1000, 9999)
    return {
        "username": username or f"user_{suffix}",
        "email": email or f"user_{suffix}@example.com",
        "password": f"SecurePass{suffix}!",
        "full_name": f"Test User {suffix}",
        "age": random.randint(18, 80),
        "gender": random.choice(["male", "female", "other"]),
        "role": role
    }


def generate_mock_analysis_data(
    user_id: int,
    analysis_type: str = "full",
    condition_type: str = "lesion",
    risk_level: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate mock analysis data.

    Args:
        user_id: ID of the user who owns the analysis
        analysis_type: Type of analysis ("binary", "full", "detailed")
        condition_type: Type of condition to generate
        risk_level: Optional specific risk level

    Returns:
        Dictionary with analysis data
    """
    conditions = SKIN_CONDITIONS.get(condition_type, SKIN_CONDITIONS["lesion"])
    predicted_class = random.choice(conditions)
    confidence = round(random.uniform(0.6, 0.95), 2)

    location = random.choice(BODY_LOCATIONS)
    sublocations = BODY_SUBLOCATIONS.get(location, [location])

    data = {
        "user_id": user_id,
        "image_filename": f"test_image_{random.randint(1000, 9999)}.jpg",
        "analysis_type": analysis_type,
        "is_lesion": True,
        "binary_confidence": round(random.uniform(0.85, 0.99), 2),
        "predicted_class": predicted_class,
        "lesion_confidence": confidence,
        "lesion_probabilities": generate_classification_probabilities(
            predicted_class, confidence, condition_type
        ),
        "risk_level": risk_level or random.choice(RISK_LEVELS),
        "image_quality_score": round(random.uniform(0.7, 0.95), 2),
        "image_quality_passed": True,
        "body_location": location,
        "body_sublocation": random.choice(sublocations),
        "body_side": random.choice(["left", "right", "center"])
    }

    return data


def generate_mock_burn_data(severity_level: int = 1) -> Dict[str, Any]:
    """
    Generate mock burn classification data.

    Args:
        severity_level: Burn severity (0=normal, 1=first, 2=second, 3=third)

    Returns:
        Dictionary with burn classification data
    """
    severity_names = ["normal", "first_degree", "second_degree", "third_degree"]
    urgency_messages = [
        "No burn detected",
        "Low urgency - home care appropriate",
        "Medium urgency - consider medical evaluation",
        "High urgency - seek immediate medical attention"
    ]
    treatment_advice = [
        "No treatment needed",
        "Cool water, aloe vera, over-the-counter pain relief",
        "Medical-grade burn cream, sterile dressings, pain management",
        "Emergency care required - do not treat at home"
    ]

    return {
        "burn_severity": severity_names[severity_level],
        "burn_confidence": round(random.uniform(0.75, 0.95), 2),
        "burn_severity_level": severity_level,
        "is_burn_detected": severity_level > 0,
        "burn_urgency": urgency_messages[severity_level],
        "burn_treatment_advice": treatment_advice[severity_level],
        "burn_medical_attention_required": severity_level >= 2
    }


def generate_mock_dermoscopy_data(
    malignant_features: bool = False
) -> Dict[str, Any]:
    """
    Generate mock dermoscopy analysis data.

    Args:
        malignant_features: Whether to include malignant-looking features

    Returns:
        Dictionary with dermoscopy analysis data
    """
    if malignant_features:
        seven_point = random.randint(4, 7)
        abcd_score = round(random.uniform(5.0, 8.0), 1)
        features = {
            "pigment_network": {"present": True, "atypical": True},
            "dots_globules": {"present": True, "irregular": True},
            "streaks": {"present": True},
            "blue_whitish_veil": {"present": True},
            "regression_structures": {"present": True}
        }
        recommendation = "Suspicious features detected. Recommend biopsy."
    else:
        seven_point = random.randint(0, 2)
        abcd_score = round(random.uniform(1.0, 4.0), 1)
        features = {
            "pigment_network": {"present": True, "atypical": False},
            "dots_globules": {"present": random.choice([True, False]), "irregular": False},
            "streaks": {"present": False},
            "blue_whitish_veil": {"present": False},
            "regression_structures": {"present": False}
        }
        recommendation = "Benign features. Routine monitoring recommended."

    return {
        "seven_point_score": seven_point,
        "abcd_score": abcd_score,
        "features": features,
        "recommendation": recommendation
    }


def generate_mock_quality_issues() -> List[str]:
    """Generate random image quality issues."""
    possible_issues = [
        "Image is too blurry",
        "Insufficient lighting",
        "Image is overexposed",
        "Image is too dark",
        "Motion blur detected",
        "Resolution too low",
        "Excessive shadows",
        "Poor focus on lesion",
        "Reflection or glare present"
    ]
    num_issues = random.randint(1, 3)
    return random.sample(possible_issues, num_issues)


# =============================================================================
# BATCH DATA GENERATORS
# =============================================================================

def generate_user_batch(count: int = 5, role: str = "patient") -> List[Dict[str, Any]]:
    """Generate a batch of mock user data."""
    return [generate_mock_user_data(role=role) for _ in range(count)]


def generate_analysis_batch(
    user_id: int,
    count: int = 10,
    condition_type: str = "lesion"
) -> List[Dict[str, Any]]:
    """Generate a batch of mock analysis data for a user."""
    return [
        generate_mock_analysis_data(user_id, condition_type=condition_type)
        for _ in range(count)
    ]


# =============================================================================
# DIFFERENTIAL DIAGNOSIS DATA
# =============================================================================

DIFFERENTIAL_DIAGNOSES = {
    "melanocytic_nevus": [
        {"condition": "melanocytic_nevus", "probability": 0.85, "description": "Common benign mole"},
        {"condition": "melanoma", "probability": 0.08, "description": "Malignant melanoma"},
        {"condition": "seborrheic_keratosis", "probability": 0.05, "description": "Benign skin growth"},
        {"condition": "dermatofibroma", "probability": 0.02, "description": "Benign fibrous nodule"}
    ],
    "melanoma": [
        {"condition": "melanoma", "probability": 0.72, "description": "Malignant melanoma"},
        {"condition": "melanocytic_nevus", "probability": 0.15, "description": "Common benign mole"},
        {"condition": "seborrheic_keratosis", "probability": 0.08, "description": "Benign skin growth"},
        {"condition": "basal_cell_carcinoma", "probability": 0.05, "description": "Common skin cancer"}
    ],
    "eczema": [
        {"condition": "eczema", "probability": 0.78, "description": "Atopic dermatitis"},
        {"condition": "psoriasis", "probability": 0.12, "description": "Autoimmune skin condition"},
        {"condition": "contact_dermatitis", "probability": 0.07, "description": "Allergic skin reaction"},
        {"condition": "seborrheic_dermatitis", "probability": 0.03, "description": "Sebaceous inflammation"}
    ]
}


def get_differential_diagnoses(condition: str) -> List[Dict[str, Any]]:
    """Get differential diagnoses for a condition."""
    return DIFFERENTIAL_DIAGNOSES.get(
        condition,
        [{"condition": condition, "probability": 0.85, "description": "Primary diagnosis"}]
    )
