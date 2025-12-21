"""
Lab-Skin Integration Module

Integrates lab results with skin analysis to provide more accurate diagnoses
by adjusting confidence scores based on systemic health indicators.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class LabInfluenceType(Enum):
    """Type of influence a lab value has on skin conditions."""
    INCREASES_LIKELIHOOD = "increases"
    DECREASES_LIKELIHOOD = "decreases"
    SUPPORTS_DIAGNOSIS = "supports"
    CONTRADICTS_DIAGNOSIS = "contradicts"


@dataclass
class LabSkinCorrelation:
    """Represents a correlation between a lab value and skin condition."""
    lab_name: str
    lab_value: float
    is_abnormal: bool
    abnormality_type: str  # "high" or "low"
    affected_conditions: List[str]
    influence_type: LabInfluenceType
    confidence_modifier: float  # -0.2 to +0.2 adjustment
    explanation: str


# Define lab-skin condition correlations
LAB_SKIN_CORRELATIONS = {
    # Vitamin D deficiency
    "vitamin_d": {
        "low_threshold": 30,
        "high_threshold": 100,
        "low_affects": {
            "psoriasis": {"modifier": 0.15, "explanation": "Low vitamin D is strongly associated with psoriasis flares and severity"},
            "eczema": {"modifier": 0.12, "explanation": "Vitamin D deficiency worsens atopic dermatitis symptoms"},
            "atopic_dermatitis": {"modifier": 0.12, "explanation": "Vitamin D deficiency worsens atopic dermatitis"},
            "seborrheic_dermatitis": {"modifier": 0.08, "explanation": "Low vitamin D may contribute to seborrheic dermatitis"},
            "acne": {"modifier": 0.05, "explanation": "Vitamin D deficiency may worsen inflammatory acne"},
            "wound": {"modifier": 0.10, "explanation": "Vitamin D is essential for wound healing"},
        },
    },

    # Elevated IgE (allergic conditions)
    "ige_total": {
        "low_threshold": 0,
        "high_threshold": 100,
        "high_affects": {
            "eczema": {"modifier": 0.20, "explanation": "Elevated IgE strongly suggests atopic/allergic skin condition"},
            "atopic_dermatitis": {"modifier": 0.20, "explanation": "High IgE is a hallmark of atopic dermatitis"},
            "urticaria": {"modifier": 0.18, "explanation": "Elevated IgE supports allergic urticaria diagnosis"},
            "contact_dermatitis": {"modifier": 0.15, "explanation": "High IgE suggests allergic contact dermatitis"},
            "allergic_reaction": {"modifier": 0.20, "explanation": "Elevated IgE confirms allergic component"},
        },
    },

    # Eosinophils (allergic/parasitic)
    "eosinophils": {
        "low_threshold": 0,
        "high_threshold": 4,
        "high_affects": {
            "eczema": {"modifier": 0.12, "explanation": "Elevated eosinophils suggest allergic skin condition"},
            "urticaria": {"modifier": 0.15, "explanation": "Eosinophilia supports allergic urticaria"},
            "parasitic_infection": {"modifier": 0.20, "explanation": "High eosinophils strongly suggest parasitic skin infection"},
            "drug_reaction": {"modifier": 0.12, "explanation": "Eosinophilia can indicate drug-induced skin reaction"},
            "scabies": {"modifier": 0.15, "explanation": "Eosinophilia common in scabies infestation"},
        },
    },

    # Thyroid - TSH
    "tsh": {
        "low_threshold": 0.4,
        "high_threshold": 4.5,
        "high_affects": {  # Hypothyroidism
            "dry_skin": {"modifier": 0.18, "explanation": "Hypothyroidism causes characteristic dry, coarse skin"},
            "xerosis": {"modifier": 0.18, "explanation": "Hypothyroidism is a common cause of xerosis"},
            "hair_loss": {"modifier": 0.20, "explanation": "Hypothyroidism causes diffuse hair thinning"},
            "alopecia": {"modifier": 0.15, "explanation": "Thyroid dysfunction associated with hair loss"},
            "brittle_nails": {"modifier": 0.15, "explanation": "Hypothyroidism affects nail health"},
            "myxedema": {"modifier": 0.25, "explanation": "High TSH strongly suggests myxedema"},
        },
        "low_affects": {  # Hyperthyroidism
            "hyperhidrosis": {"modifier": 0.18, "explanation": "Hyperthyroidism causes excessive sweating"},
            "warm_moist_skin": {"modifier": 0.15, "explanation": "Hyperthyroid patients have warm, moist skin"},
            "hair_loss": {"modifier": 0.12, "explanation": "Hyperthyroidism can cause hair thinning"},
            "pretibial_myxedema": {"modifier": 0.20, "explanation": "Associated with Graves' disease"},
        },
    },

    # Glucose/HbA1c (Diabetes)
    "hba1c": {
        "low_threshold": 0,
        "high_threshold": 5.7,
        "high_affects": {
            "diabetic_dermopathy": {"modifier": 0.25, "explanation": "Elevated HbA1c strongly suggests diabetic skin changes"},
            "acanthosis_nigricans": {"modifier": 0.20, "explanation": "High glucose associated with acanthosis nigricans"},
            "skin_infection": {"modifier": 0.15, "explanation": "Diabetes increases risk of skin infections"},
            "fungal_infection": {"modifier": 0.18, "explanation": "High blood sugar promotes fungal growth"},
            "candidiasis": {"modifier": 0.18, "explanation": "Diabetes is a major risk factor for candidiasis"},
            "bacterial_infection": {"modifier": 0.12, "explanation": "Diabetes impairs immune response to bacteria"},
            "slow_healing": {"modifier": 0.20, "explanation": "Elevated glucose impairs wound healing"},
            "necrobiosis_lipoidica": {"modifier": 0.22, "explanation": "Strongly associated with diabetes"},
        },
    },

    "glucose_fasting": {
        "low_threshold": 65,
        "high_threshold": 99,
        "high_affects": {
            "diabetic_dermopathy": {"modifier": 0.15, "explanation": "Elevated fasting glucose suggests diabetic skin changes"},
            "skin_infection": {"modifier": 0.10, "explanation": "High glucose increases infection risk"},
            "fungal_infection": {"modifier": 0.12, "explanation": "Elevated glucose promotes fungal overgrowth"},
        },
    },

    # Iron deficiency
    "ferritin": {
        "low_threshold": 12,
        "high_threshold": 300,
        "low_affects": {
            "hair_loss": {"modifier": 0.20, "explanation": "Iron deficiency is a major cause of hair loss"},
            "alopecia": {"modifier": 0.18, "explanation": "Low ferritin strongly associated with hair thinning"},
            "pale_skin": {"modifier": 0.15, "explanation": "Iron deficiency causes pallor"},
            "brittle_nails": {"modifier": 0.18, "explanation": "Iron deficiency causes koilonychia and brittle nails"},
            "angular_cheilitis": {"modifier": 0.15, "explanation": "Iron deficiency can cause angular cheilitis"},
            "glossitis": {"modifier": 0.12, "explanation": "Iron deficiency affects oral mucosa"},
        },
    },

    "iron": {
        "low_threshold": 60,
        "high_threshold": 170,
        "low_affects": {
            "hair_loss": {"modifier": 0.12, "explanation": "Low iron contributes to hair loss"},
            "pale_skin": {"modifier": 0.12, "explanation": "Iron deficiency causes skin pallor"},
            "brittle_nails": {"modifier": 0.10, "explanation": "Low iron affects nail integrity"},
        },
    },

    # Liver function
    "bilirubin_total": {
        "low_threshold": 0,
        "high_threshold": 1.2,
        "high_affects": {
            "jaundice": {"modifier": 0.30, "explanation": "Elevated bilirubin directly causes jaundice"},
            "pruritus": {"modifier": 0.20, "explanation": "Liver dysfunction causes generalized itching"},
            "spider_angioma": {"modifier": 0.18, "explanation": "Liver disease associated with spider angiomas"},
            "palmar_erythema": {"modifier": 0.15, "explanation": "Liver dysfunction causes palmar redness"},
        },
    },

    "alt": {
        "low_threshold": 0,
        "high_threshold": 46,
        "high_affects": {
            "pruritus": {"modifier": 0.12, "explanation": "Elevated liver enzymes can cause itching"},
            "jaundice": {"modifier": 0.10, "explanation": "Liver dysfunction may lead to jaundice"},
        },
    },

    # Inflammatory markers
    "crp": {
        "low_threshold": 0,
        "high_threshold": 3.0,
        "high_affects": {
            "psoriasis": {"modifier": 0.12, "explanation": "Elevated CRP indicates active inflammation in psoriasis"},
            "inflammatory_condition": {"modifier": 0.15, "explanation": "High CRP supports inflammatory skin diagnosis"},
            "cellulitis": {"modifier": 0.15, "explanation": "Elevated CRP suggests active infection/inflammation"},
            "vasculitis": {"modifier": 0.12, "explanation": "CRP elevation common in vasculitic conditions"},
            "lupus": {"modifier": 0.10, "explanation": "Inflammation marker elevated in lupus flares"},
        },
    },

    "esr": {
        "low_threshold": 0,
        "high_threshold": 20,
        "high_affects": {
            "inflammatory_condition": {"modifier": 0.12, "explanation": "Elevated ESR indicates systemic inflammation"},
            "vasculitis": {"modifier": 0.15, "explanation": "ESR elevation common in vasculitis"},
            "lupus": {"modifier": 0.12, "explanation": "ESR often elevated in lupus"},
            "dermatomyositis": {"modifier": 0.12, "explanation": "Elevated ESR supports inflammatory myopathy"},
        },
    },

    # Autoimmune markers
    "ana_positive": {
        "is_boolean": True,
        "positive_affects": {
            "lupus": {"modifier": 0.25, "explanation": "Positive ANA strongly supports lupus diagnosis"},
            "lupus_rash": {"modifier": 0.25, "explanation": "ANA positivity characteristic of lupus skin manifestations"},
            "malar_rash": {"modifier": 0.25, "explanation": "Butterfly rash with positive ANA highly suggestive of lupus"},
            "discoid_lupus": {"modifier": 0.20, "explanation": "ANA supports discoid lupus diagnosis"},
            "scleroderma": {"modifier": 0.18, "explanation": "Positive ANA common in scleroderma"},
            "dermatomyositis": {"modifier": 0.18, "explanation": "ANA positivity supports dermatomyositis"},
            "autoimmune_condition": {"modifier": 0.20, "explanation": "Positive ANA indicates autoimmune process"},
        },
    },

    # Kidney function
    "creatinine": {
        "low_threshold": 0.5,
        "high_threshold": 1.25,
        "high_affects": {
            "uremic_frost": {"modifier": 0.20, "explanation": "Elevated creatinine in severe cases causes uremic frost"},
            "pruritus": {"modifier": 0.15, "explanation": "Kidney dysfunction causes generalized itching"},
            "xerosis": {"modifier": 0.12, "explanation": "Uremia associated with dry skin"},
            "calciphylaxis": {"modifier": 0.18, "explanation": "Renal failure associated with calciphylaxis"},
        },
    },

    # B12 deficiency
    "vitamin_b12": {
        "low_threshold": 200,
        "high_threshold": 900,
        "low_affects": {
            "hyperpigmentation": {"modifier": 0.15, "explanation": "B12 deficiency can cause skin hyperpigmentation"},
            "glossitis": {"modifier": 0.18, "explanation": "B12 deficiency causes tongue inflammation"},
            "angular_cheilitis": {"modifier": 0.12, "explanation": "B12 deficiency affects oral corners"},
            "hair_loss": {"modifier": 0.10, "explanation": "B12 deficiency can contribute to hair loss"},
            "vitiligo": {"modifier": 0.08, "explanation": "B12 deficiency associated with vitiligo"},
        },
    },

    # Hemoglobin/anemia
    "hemoglobin": {
        "low_threshold": 12.0,
        "high_threshold": 17.5,
        "low_affects": {
            "pale_skin": {"modifier": 0.20, "explanation": "Low hemoglobin causes skin pallor"},
            "pallor": {"modifier": 0.20, "explanation": "Anemia is primary cause of pallor"},
            "koilonychia": {"modifier": 0.15, "explanation": "Anemia associated with spoon nails"},
            "brittle_nails": {"modifier": 0.12, "explanation": "Anemia affects nail health"},
        },
    },

    # Platelets
    "platelets": {
        "low_threshold": 140,
        "high_threshold": 400,
        "low_affects": {
            "petechiae": {"modifier": 0.25, "explanation": "Low platelets cause petechial hemorrhages"},
            "purpura": {"modifier": 0.25, "explanation": "Thrombocytopenia causes purpura"},
            "bruising": {"modifier": 0.20, "explanation": "Low platelets cause easy bruising"},
            "ecchymosis": {"modifier": 0.20, "explanation": "Thrombocytopenia leads to ecchymoses"},
        },
    },

    # Zinc (if added)
    "zinc": {
        "low_threshold": 60,
        "high_threshold": 130,
        "low_affects": {
            "acrodermatitis": {"modifier": 0.25, "explanation": "Zinc deficiency causes acrodermatitis enteropathica"},
            "hair_loss": {"modifier": 0.12, "explanation": "Zinc deficiency can cause hair loss"},
            "acne": {"modifier": 0.10, "explanation": "Low zinc may worsen acne"},
            "slow_healing": {"modifier": 0.15, "explanation": "Zinc essential for wound healing"},
            "eczema": {"modifier": 0.08, "explanation": "Zinc deficiency may worsen eczema"},
        },
    },
}


def get_user_lab_abnormalities(lab_data: Dict[str, Any]) -> List[LabSkinCorrelation]:
    """
    Analyze user's lab data and identify abnormalities relevant to skin conditions.

    Args:
        lab_data: Dictionary containing lab values from the database

    Returns:
        List of LabSkinCorrelation objects for abnormal values
    """
    correlations = []

    for lab_name, config in LAB_SKIN_CORRELATIONS.items():
        value = lab_data.get(lab_name)

        if value is None:
            continue

        # Handle boolean labs (like ANA)
        if config.get("is_boolean"):
            if value:  # Positive
                for condition, details in config.get("positive_affects", {}).items():
                    correlations.append(LabSkinCorrelation(
                        lab_name=lab_name.replace("_", " ").title(),
                        lab_value=1,
                        is_abnormal=True,
                        abnormality_type="positive",
                        affected_conditions=[condition],
                        influence_type=LabInfluenceType.INCREASES_LIKELIHOOD,
                        confidence_modifier=details["modifier"],
                        explanation=details["explanation"]
                    ))
            continue

        # Handle numeric labs
        low_threshold = config.get("low_threshold", 0)
        high_threshold = config.get("high_threshold", float('inf'))

        if value < low_threshold and "low_affects" in config:
            for condition, details in config["low_affects"].items():
                correlations.append(LabSkinCorrelation(
                    lab_name=lab_name.replace("_", " ").title(),
                    lab_value=value,
                    is_abnormal=True,
                    abnormality_type="low",
                    affected_conditions=[condition],
                    influence_type=LabInfluenceType.INCREASES_LIKELIHOOD,
                    confidence_modifier=details["modifier"],
                    explanation=details["explanation"]
                ))

        elif value > high_threshold and "high_affects" in config:
            for condition, details in config["high_affects"].items():
                correlations.append(LabSkinCorrelation(
                    lab_name=lab_name.replace("_", " ").title(),
                    lab_value=value,
                    is_abnormal=True,
                    abnormality_type="high",
                    affected_conditions=[condition],
                    influence_type=LabInfluenceType.INCREASES_LIKELIHOOD,
                    confidence_modifier=details["modifier"],
                    explanation=details["explanation"]
                ))

    return correlations


def adjust_skin_diagnosis_with_labs(
    predicted_class: str,
    class_probabilities: Dict[str, float],
    lab_correlations: List[LabSkinCorrelation],
    differential_diagnoses: Optional[List[Dict]] = None
) -> Tuple[Dict[str, float], List[Dict], List[Dict]]:
    """
    Adjust skin diagnosis probabilities based on lab correlations.

    Args:
        predicted_class: The AI's predicted skin condition
        class_probabilities: Dictionary of condition -> probability
        lab_correlations: List of relevant lab-skin correlations
        differential_diagnoses: Optional list of differential diagnoses

    Returns:
        Tuple of (adjusted_probabilities, lab_insights, adjusted_differentials)
    """
    adjusted_probs = class_probabilities.copy()
    lab_insights = []

    # Build a map of condition -> total modifier
    condition_modifiers = {}
    condition_explanations = {}

    for correlation in lab_correlations:
        for condition in correlation.affected_conditions:
            # Normalize condition name for matching
            condition_lower = condition.lower().replace("_", " ").replace("-", " ")

            if condition_lower not in condition_modifiers:
                condition_modifiers[condition_lower] = 0
                condition_explanations[condition_lower] = []

            condition_modifiers[condition_lower] += correlation.confidence_modifier
            condition_explanations[condition_lower].append({
                "lab": correlation.lab_name,
                "value": correlation.lab_value,
                "status": correlation.abnormality_type,
                "explanation": correlation.explanation,
                "modifier": correlation.confidence_modifier
            })

    # Apply modifiers to probabilities
    for prob_class in adjusted_probs:
        prob_class_lower = prob_class.lower().replace("_", " ").replace("-", " ")

        # Check for matching conditions
        for condition, modifier in condition_modifiers.items():
            # Check if condition matches or is contained in the class name
            if condition in prob_class_lower or prob_class_lower in condition:
                # Apply modifier (cap at reasonable bounds)
                original = adjusted_probs[prob_class]
                adjustment = min(modifier, 0.3)  # Cap adjustment at 30%
                adjusted_probs[prob_class] = min(0.99, max(0.01, original + adjustment))

                # Record insight
                lab_insights.append({
                    "condition": prob_class,
                    "original_confidence": round(original * 100, 1),
                    "adjusted_confidence": round(adjusted_probs[prob_class] * 100, 1),
                    "adjustment": round(adjustment * 100, 1),
                    "supporting_labs": condition_explanations[condition]
                })
                break

    # Normalize probabilities to sum to 1
    total = sum(adjusted_probs.values())
    if total > 0:
        adjusted_probs = {k: v/total for k, v in adjusted_probs.items()}

    # Adjust differential diagnoses if provided
    adjusted_differentials = []
    if differential_diagnoses:
        for diff in differential_diagnoses:
            diff_copy = diff.copy()
            diff_name = diff.get("condition", "").lower().replace("_", " ").replace("-", " ")

            for condition, modifier in condition_modifiers.items():
                if condition in diff_name or diff_name in condition:
                    original_prob = diff_copy.get("probability", 0)
                    adjustment = min(modifier, 0.3)
                    diff_copy["probability"] = min(0.99, max(0.01, original_prob + adjustment))
                    diff_copy["lab_supported"] = True
                    diff_copy["lab_notes"] = [exp["explanation"] for exp in condition_explanations[condition][:2]]
                    break

            adjusted_differentials.append(diff_copy)

        # Re-sort by probability
        adjusted_differentials.sort(key=lambda x: x.get("probability", 0), reverse=True)

    return adjusted_probs, lab_insights, adjusted_differentials


def generate_lab_context_summary(
    lab_correlations: List[LabSkinCorrelation],
    predicted_class: str
) -> Dict[str, Any]:
    """
    Generate a summary of how lab results relate to the skin analysis.

    Args:
        lab_correlations: List of relevant lab-skin correlations
        predicted_class: The predicted skin condition

    Returns:
        Dictionary containing lab context summary
    """
    if not lab_correlations:
        return {
            "has_lab_context": False,
            "message": "No recent lab results available. Add lab results for more personalized analysis.",
            "relevant_labs": [],
            "recommendations": []
        }

    # Group correlations by lab name to avoid duplicates
    # Each abnormal lab should only appear once, with all its affected conditions listed
    lab_groups: Dict[str, Dict[str, Any]] = {}

    predicted_lower = predicted_class.lower().replace("_", " ").replace("-", " ")

    for correlation in lab_correlations:
        lab_key = f"{correlation.lab_name}_{correlation.abnormality_type}"

        if lab_key not in lab_groups:
            lab_groups[lab_key] = {
                "name": correlation.lab_name,
                "value": correlation.lab_value,
                "status": correlation.abnormality_type,
                "explanation": correlation.explanation,
                "related_conditions": list(correlation.affected_conditions),
                "supports_diagnosis": False
            }
        else:
            # Add any new conditions to the existing entry
            for cond in correlation.affected_conditions:
                if cond not in lab_groups[lab_key]["related_conditions"]:
                    lab_groups[lab_key]["related_conditions"].append(cond)

        # Check if this lab supports the predicted diagnosis
        supports_prediction = any(
            cond.lower().replace("_", " ") in predicted_lower or
            predicted_lower in cond.lower().replace("_", " ")
            for cond in correlation.affected_conditions
        )

        if supports_prediction:
            lab_groups[lab_key]["supports_diagnosis"] = True
            # Use the explanation that matches the predicted condition
            lab_groups[lab_key]["explanation"] = correlation.explanation

    # Now separate into supporting vs other labs (no duplicates)
    supporting_labs = []
    other_relevant_labs = []

    for lab_info in lab_groups.values():
        lab_entry = {
            "name": lab_info["name"],
            "value": lab_info["value"],
            "status": lab_info["status"],
            "explanation": lab_info["explanation"],
            "related_conditions": lab_info["related_conditions"]
        }

        if lab_info["supports_diagnosis"]:
            supporting_labs.append(lab_entry)
        else:
            other_relevant_labs.append(lab_entry)

    # Generate recommendations based on abnormal labs
    recommendations = []

    # Check for specific patterns
    lab_names = [c.lab_name.lower() for c in lab_correlations]

    if any("vitamin d" in ln for ln in lab_names):
        recommendations.append("Consider vitamin D supplementation - deficiency worsens many skin conditions")

    if any("ige" in ln or "eosinophil" in ln for ln in lab_names):
        recommendations.append("Elevated allergy markers detected - consider allergy testing if not done")

    if any("hba1c" in ln or "glucose" in ln for ln in lab_names):
        recommendations.append("Blood sugar abnormality detected - good glycemic control important for skin health")

    if any("ferritin" in ln or "iron" in ln for ln in lab_names):
        recommendations.append("Iron status abnormal - may affect hair, nails, and skin healing")

    if any("tsh" in ln or "thyroid" in ln for ln in lab_names):
        recommendations.append("Thyroid abnormality detected - thyroid function significantly impacts skin health")

    # Count unique abnormal labs (not duplicate condition entries)
    unique_lab_count = len(lab_groups)

    return {
        "has_lab_context": True,
        "labs_analyzed": unique_lab_count,
        "supporting_labs": supporting_labs,
        "other_relevant_labs": other_relevant_labs[:5],  # Limit to top 5
        "recommendations": recommendations,
        "message": f"Analysis enhanced with {unique_lab_count} abnormal lab value(s)" +
                   (f", {len(supporting_labs)} directly support the diagnosis" if supporting_labs else "")
    }


def get_latest_user_labs(db_session, user_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent lab results for a user.

    Args:
        db_session: SQLAlchemy database session
        user_id: User ID to fetch labs for

    Returns:
        Dictionary of lab values or None if no labs found
    """
    from database import LabResults

    latest_lab = db_session.query(LabResults).filter(
        LabResults.user_id == user_id
    ).order_by(LabResults.test_date.desc()).first()

    if not latest_lab:
        return None

    # Convert to dictionary
    return {
        "id": latest_lab.id,
        "test_date": latest_lab.test_date.isoformat() if latest_lab.test_date else None,
        "lab_name": latest_lab.lab_name,

        # CBC
        "wbc": latest_lab.wbc,
        "rbc": latest_lab.rbc,
        "hemoglobin": latest_lab.hemoglobin,
        "hematocrit": latest_lab.hematocrit,
        "platelets": latest_lab.platelets,

        # WBC Differential
        "neutrophils": latest_lab.neutrophils,
        "lymphocytes": latest_lab.lymphocytes,
        "monocytes": latest_lab.monocytes,
        "eosinophils": latest_lab.eosinophils,
        "basophils": latest_lab.basophils,

        # Metabolic
        "glucose_fasting": latest_lab.glucose_fasting,
        "hba1c": latest_lab.hba1c,
        "bun": latest_lab.bun,
        "creatinine": latest_lab.creatinine,
        "egfr": latest_lab.egfr,
        "sodium": latest_lab.sodium,
        "potassium": latest_lab.potassium,
        "calcium": latest_lab.calcium,

        # Liver
        "alt": latest_lab.alt,
        "ast": latest_lab.ast,
        "bilirubin_total": latest_lab.bilirubin_total,
        "albumin": latest_lab.albumin,

        # Lipids (less relevant but included)
        "cholesterol_total": latest_lab.cholesterol_total,

        # Thyroid
        "tsh": latest_lab.tsh,
        "t3_free": latest_lab.t3_free,
        "t4_free": latest_lab.t4_free,

        # Iron
        "iron": latest_lab.iron,
        "ferritin": latest_lab.ferritin,
        "tibc": latest_lab.tibc,

        # Vitamins
        "vitamin_d": latest_lab.vitamin_d,
        "vitamin_b12": latest_lab.vitamin_b12,
        "folate": latest_lab.folate,

        # Inflammatory
        "crp": latest_lab.crp,
        "esr": latest_lab.esr,

        # Autoimmune
        "ana_positive": latest_lab.ana_positive,
        "rf": latest_lab.rf,

        # Allergy
        "ige_total": latest_lab.ige_total,
    }


def integrate_labs_with_skin_analysis(
    db_session,
    user_id: int,
    predicted_class: str,
    class_probabilities: Dict[str, float],
    differential_diagnoses: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Main function to integrate lab results with skin analysis.

    Args:
        db_session: SQLAlchemy database session
        user_id: User ID
        predicted_class: AI predicted skin condition
        class_probabilities: Dictionary of condition probabilities
        differential_diagnoses: Optional differential diagnoses list

    Returns:
        Dictionary containing integrated analysis results
    """
    # Fetch user's latest labs
    lab_data = get_latest_user_labs(db_session, user_id)

    if not lab_data:
        return {
            "lab_integrated": False,
            "lab_context": {
                "has_lab_context": False,
                "message": "No lab results available. Add lab results for enhanced, personalized skin analysis.",
            },
            "adjusted_probabilities": class_probabilities,
            "lab_insights": [],
            "adjusted_differentials": differential_diagnoses or []
        }

    # Get lab correlations
    correlations = get_user_lab_abnormalities(lab_data)

    if not correlations:
        return {
            "lab_integrated": True,
            "lab_data_date": lab_data.get("test_date"),
            "lab_context": {
                "has_lab_context": True,
                "message": "Lab results reviewed - all values within normal ranges. No significant impact on skin analysis.",
                "relevant_labs": [],
                "recommendations": []
            },
            "adjusted_probabilities": class_probabilities,
            "lab_insights": [],
            "adjusted_differentials": differential_diagnoses or []
        }

    # Adjust probabilities based on labs
    adjusted_probs, lab_insights, adjusted_diffs = adjust_skin_diagnosis_with_labs(
        predicted_class,
        class_probabilities,
        correlations,
        differential_diagnoses
    )

    # Generate context summary
    lab_context = generate_lab_context_summary(correlations, predicted_class)
    lab_context["lab_date"] = lab_data.get("test_date")
    lab_context["lab_source"] = lab_data.get("lab_name")

    return {
        "lab_integrated": True,
        "lab_data_date": lab_data.get("test_date"),
        "lab_context": lab_context,
        "adjusted_probabilities": adjusted_probs,
        "original_probabilities": class_probabilities,
        "lab_insights": lab_insights,
        "adjusted_differentials": adjusted_diffs,
        "abnormal_labs_count": len(correlations)
    }
