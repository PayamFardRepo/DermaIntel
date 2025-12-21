"""
Lab Results - Skin Condition Correlator

This module analyzes lab results (blood, urine, stool) and identifies
correlations with skin conditions to provide more comprehensive feedback.

The correlations are based on established medical literature connecting
systemic health markers with dermatological manifestations.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, date


class SeverityLevel(str, Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class RelevanceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class LabFinding:
    """A single lab finding with skin relevance"""
    lab_name: str
    value: float
    unit: str
    normal_range: str
    status: str  # "normal", "low", "high", "critical_low", "critical_high"
    skin_relevance: RelevanceLevel
    skin_implications: List[str]
    recommendations: List[str]


@dataclass
class SkinLabCorrelation:
    """Correlation between lab results and a skin condition"""
    condition: str
    correlation_strength: str  # "strong", "moderate", "weak"
    supporting_labs: List[str]
    explanation: str
    clinical_significance: str


@dataclass
class LabAnalysisResult:
    """Complete lab analysis result for skin relevance"""
    abnormal_findings: List[LabFinding]
    skin_correlations: List[SkinLabCorrelation]
    overall_skin_health_score: float  # 0-100
    key_concerns: List[str]
    recommendations: List[str]
    monitoring_suggestions: List[str]


# =============================================================================
# REFERENCE RANGES (gender-specific where applicable)
# =============================================================================

REFERENCE_RANGES = {
    # CBC
    "wbc": {"low": 4.5, "high": 11.0, "unit": "x10^9/L", "name": "White Blood Cells"},
    "rbc_male": {"low": 4.5, "high": 5.5, "unit": "x10^12/L", "name": "Red Blood Cells"},
    "rbc_female": {"low": 4.0, "high": 5.0, "unit": "x10^12/L", "name": "Red Blood Cells"},
    "hemoglobin_male": {"low": 14.0, "high": 18.0, "unit": "g/dL", "name": "Hemoglobin"},
    "hemoglobin_female": {"low": 12.0, "high": 16.0, "unit": "g/dL", "name": "Hemoglobin"},
    "platelets": {"low": 150, "high": 400, "unit": "x10^9/L", "name": "Platelets"},
    "eosinophils": {"low": 1.0, "high": 4.0, "unit": "%", "name": "Eosinophils"},

    # Metabolic
    "glucose_fasting": {"low": 70, "high": 100, "unit": "mg/dL", "name": "Fasting Glucose"},
    "hba1c": {"low": 4.0, "high": 5.7, "unit": "%", "name": "HbA1c"},
    "creatinine_male": {"low": 0.7, "high": 1.3, "unit": "mg/dL", "name": "Creatinine"},
    "creatinine_female": {"low": 0.6, "high": 1.1, "unit": "mg/dL", "name": "Creatinine"},
    "egfr": {"low": 90, "high": 999, "unit": "mL/min/1.73m²", "name": "eGFR"},

    # Liver
    "alt": {"low": 7, "high": 56, "unit": "U/L", "name": "ALT"},
    "ast": {"low": 10, "high": 40, "unit": "U/L", "name": "AST"},
    "bilirubin_total": {"low": 0.1, "high": 1.2, "unit": "mg/dL", "name": "Total Bilirubin"},

    # Thyroid
    "tsh": {"low": 0.4, "high": 4.0, "unit": "mIU/L", "name": "TSH"},
    "t4_free": {"low": 0.8, "high": 1.8, "unit": "ng/dL", "name": "Free T4"},

    # Iron
    "iron": {"low": 60, "high": 170, "unit": "µg/dL", "name": "Iron"},
    "ferritin_male": {"low": 12, "high": 300, "unit": "ng/mL", "name": "Ferritin"},
    "ferritin_female": {"low": 12, "high": 150, "unit": "ng/mL", "name": "Ferritin"},

    # Vitamins
    "vitamin_d": {"low": 30, "high": 100, "unit": "ng/mL", "name": "Vitamin D"},
    "vitamin_b12": {"low": 200, "high": 900, "unit": "pg/mL", "name": "Vitamin B12"},

    # Inflammatory
    "crp": {"low": 0, "high": 3.0, "unit": "mg/L", "name": "C-Reactive Protein"},
    "esr_male": {"low": 0, "high": 22, "unit": "mm/hr", "name": "ESR"},
    "esr_female": {"low": 0, "high": 29, "unit": "mm/hr", "name": "ESR"},

    # Allergy
    "ige_total": {"low": 0, "high": 100, "unit": "IU/mL", "name": "Total IgE"},
}


# =============================================================================
# SKIN CONDITION CORRELATIONS
# =============================================================================

LAB_SKIN_CORRELATIONS = {
    # Diabetes and Skin
    "high_glucose": {
        "conditions": [
            "Diabetic dermopathy (shin spots)",
            "Acanthosis nigricans (dark, velvety patches)",
            "Bacterial skin infections",
            "Fungal infections (candidiasis)",
            "Slow wound healing",
            "Diabetic foot ulcers",
            "Necrobiosis lipoidica",
            "Skin tags"
        ],
        "explanation": "Elevated blood glucose impairs immune function, promotes bacterial and fungal growth, and damages blood vessels affecting skin health.",
        "recommendations": [
            "Monitor blood glucose closely",
            "Keep skin clean and moisturized",
            "Inspect feet daily for wounds",
            "Treat cuts and scrapes promptly",
            "Consult endocrinologist for glucose management"
        ]
    },

    "high_hba1c": {
        "conditions": [
            "Diabetic skin complications",
            "Delayed wound healing",
            "Increased infection risk",
            "Neuropathic skin changes"
        ],
        "explanation": "HbA1c reflects long-term glucose control. Elevated levels indicate prolonged hyperglycemia affecting skin healing and immunity.",
        "recommendations": [
            "Work with healthcare provider to improve glucose control",
            "Regular skin examinations",
            "Prompt treatment of any skin infections"
        ]
    },

    # Thyroid and Skin
    "high_tsh": {
        "conditions": [
            "Dry, coarse skin (xerosis)",
            "Hair loss (telogen effluvium)",
            "Brittle nails",
            "Myxedema (skin thickening)",
            "Pale, cool skin",
            "Slow wound healing",
            "Eyebrow thinning (lateral third)"
        ],
        "explanation": "Hypothyroidism (high TSH) reduces metabolic rate, affecting skin cell turnover, oil production, and hair growth.",
        "recommendations": [
            "Consult endocrinologist for thyroid management",
            "Use rich moisturizers for dry skin",
            "Consider biotin supplements for hair/nails (after consulting doctor)"
        ]
    },

    "low_tsh": {
        "conditions": [
            "Warm, moist skin",
            "Excessive sweating (hyperhidrosis)",
            "Flushing",
            "Hair thinning",
            "Pretibial myxedema (Graves' disease)",
            "Onycholysis (nail separation)"
        ],
        "explanation": "Hyperthyroidism (low TSH) increases metabolism, causing excess sweating, heat intolerance, and skin changes.",
        "recommendations": [
            "Seek treatment for hyperthyroidism",
            "Use gentle, non-irritating skincare",
            "Stay hydrated"
        ]
    },

    # Iron Deficiency and Skin
    "low_iron": {
        "conditions": [
            "Pale skin",
            "Angular cheilitis (cracked mouth corners)",
            "Brittle, spoon-shaped nails (koilonychia)",
            "Hair loss",
            "Pruritus (itchy skin)",
            "Glossitis (smooth, sore tongue)"
        ],
        "explanation": "Iron is essential for oxygen delivery and cell function. Deficiency affects rapidly dividing cells including skin, hair, and nails.",
        "recommendations": [
            "Iron supplementation (under medical supervision)",
            "Vitamin C to enhance iron absorption",
            "Treat underlying cause of iron deficiency"
        ]
    },

    "low_ferritin": {
        "conditions": [
            "Hair loss (especially in women)",
            "Fatigue affecting skin appearance",
            "Pale skin",
            "Brittle nails"
        ],
        "explanation": "Low ferritin indicates depleted iron stores, often causing hair loss before anemia develops.",
        "recommendations": [
            "Check for iron deficiency causes",
            "Consider iron supplementation",
            "Follow up with ferritin monitoring"
        ]
    },

    # Vitamin D and Skin
    "low_vitamin_d": {
        "conditions": [
            "Psoriasis exacerbation",
            "Eczema flares",
            "Slow wound healing",
            "Increased skin infections",
            "Vitiligo progression",
            "Hair loss (alopecia areata)"
        ],
        "explanation": "Vitamin D is crucial for skin cell growth, immune function, and barrier integrity. Deficiency worsens inflammatory skin conditions.",
        "recommendations": [
            "Vitamin D supplementation (after blood test confirmation)",
            "Safe sun exposure (10-15 min)",
            "Vitamin D-rich foods (fatty fish, fortified dairy)",
            "Recheck levels after 3 months of supplementation"
        ]
    },

    # Vitamin B12 and Skin
    "low_b12": {
        "conditions": [
            "Hyperpigmentation",
            "Vitiligo",
            "Angular stomatitis",
            "Glossitis",
            "Hair changes",
            "Pallor"
        ],
        "explanation": "B12 is needed for DNA synthesis and red blood cell production. Deficiency causes skin pigment changes and mucosal symptoms.",
        "recommendations": [
            "B12 supplementation (oral or injection)",
            "Check for pernicious anemia",
            "Dietary sources (meat, fish, dairy)"
        ]
    },

    # Liver and Skin
    "high_bilirubin": {
        "conditions": [
            "Jaundice (yellow skin)",
            "Pruritus (severe itching)",
            "Spider angiomas",
            "Palmar erythema",
            "Xanthelasma"
        ],
        "explanation": "Elevated bilirubin causes yellow discoloration of skin and eyes. Liver dysfunction also causes itching and vascular skin changes.",
        "recommendations": [
            "Urgent evaluation for cause of jaundice",
            "Liver function assessment",
            "Avoid alcohol and hepatotoxic medications"
        ]
    },

    "high_liver_enzymes": {
        "conditions": [
            "Pruritus (itching)",
            "Easy bruising",
            "Spider angiomas",
            "Drug eruptions (if medication-related)"
        ],
        "explanation": "Elevated ALT/AST may indicate liver stress, affecting drug metabolism and causing skin manifestations.",
        "recommendations": [
            "Investigate cause of elevated enzymes",
            "Review medications",
            "Limit alcohol consumption"
        ]
    },

    # Kidney and Skin
    "low_egfr": {
        "conditions": [
            "Uremic frost (severe)",
            "Pruritus (uremic itching)",
            "Dry skin (xerosis)",
            "Pallor",
            "Half-and-half nails",
            "Calciphylaxis (severe cases)"
        ],
        "explanation": "Kidney dysfunction leads to toxin accumulation, causing itching and characteristic skin changes.",
        "recommendations": [
            "Nephrology consultation",
            "Careful medication dosing",
            "Moisturizers for dry, itchy skin",
            "Avoid nephrotoxic substances"
        ]
    },

    # Inflammatory Markers and Skin
    "high_crp": {
        "conditions": [
            "Psoriasis flares",
            "Hidradenitis suppurativa",
            "Pyoderma gangrenosum",
            "Systemic inflammatory skin conditions"
        ],
        "explanation": "Elevated CRP indicates systemic inflammation, which often manifests in or worsens inflammatory skin conditions.",
        "recommendations": [
            "Investigate source of inflammation",
            "Anti-inflammatory treatments as needed",
            "Monitor for cardiovascular risk"
        ]
    },

    "high_esr": {
        "conditions": [
            "Vasculitis",
            "Connective tissue diseases with skin manifestations",
            "Inflammatory skin conditions"
        ],
        "explanation": "Elevated ESR suggests chronic inflammation, potentially from autoimmune conditions affecting the skin.",
        "recommendations": [
            "Rheumatology evaluation if indicated",
            "Monitor for systemic symptoms"
        ]
    },

    # Allergy/Eosinophils and Skin
    "high_eosinophils": {
        "conditions": [
            "Atopic dermatitis (eczema)",
            "Urticaria (hives)",
            "Drug eruptions",
            "Parasitic skin infections",
            "Eosinophilic dermatoses"
        ],
        "explanation": "Elevated eosinophils indicate allergic or parasitic conditions, strongly associated with itchy skin conditions.",
        "recommendations": [
            "Allergy evaluation",
            "Check for parasitic infections",
            "Antihistamines for symptomatic relief"
        ]
    },

    "high_ige": {
        "conditions": [
            "Atopic dermatitis",
            "Allergic contact dermatitis",
            "Chronic urticaria",
            "Hyper-IgE syndrome (very high levels)"
        ],
        "explanation": "Elevated IgE indicates atopic tendency, predisposing to allergic skin conditions.",
        "recommendations": [
            "Allergy testing for specific triggers",
            "Allergen avoidance",
            "Consider immunotherapy if appropriate"
        ]
    },

    # Autoimmune Markers and Skin
    "positive_ana": {
        "conditions": [
            "Lupus rash (malar/discoid)",
            "Dermatomyositis (heliotrope rash, Gottron's papules)",
            "Scleroderma skin changes",
            "Photosensitivity",
            "Raynaud's phenomenon"
        ],
        "explanation": "Positive ANA suggests autoimmune disease, many of which have characteristic skin manifestations.",
        "recommendations": [
            "Rheumatology consultation",
            "Sun protection (autoimmune conditions often worsen with UV)",
            "Monitor for systemic symptoms"
        ]
    },

    # Platelets and Skin
    "low_platelets": {
        "conditions": [
            "Petechiae (tiny red spots)",
            "Purpura (larger bruises)",
            "Easy bruising",
            "Prolonged bleeding from cuts"
        ],
        "explanation": "Low platelets impair blood clotting, causing spontaneous bleeding into the skin.",
        "recommendations": [
            "Investigate cause of thrombocytopenia",
            "Avoid trauma and contact sports",
            "Report any unusual bleeding promptly"
        ]
    },

    # Stool Findings and Skin
    "stool_parasites": {
        "conditions": [
            "Urticaria (parasitic-induced)",
            "Pruritus ani",
            "Larva migrans (cutaneous)",
            "Eosinophilic skin reactions"
        ],
        "explanation": "Intestinal parasites can cause systemic allergic reactions manifesting as hives or itching.",
        "recommendations": [
            "Antiparasitic treatment",
            "Repeat stool tests after treatment",
            "Hygiene measures to prevent reinfection"
        ]
    },

    # Urinalysis and Skin
    "urine_protein": {
        "conditions": [
            "Edema affecting skin",
            "Nephrotic syndrome skin changes",
            "Poor wound healing"
        ],
        "explanation": "Proteinuria may indicate kidney problems that affect fluid balance and skin integrity.",
        "recommendations": [
            "Nephrology evaluation",
            "Blood pressure management",
            "Monitor for swelling"
        ]
    }
}


def analyze_lab_value(
    lab_name: str,
    value: float,
    gender: str = "unknown"
) -> Tuple[str, Optional[Dict]]:
    """
    Analyze a single lab value and return its status and skin implications.

    Returns:
        Tuple of (status, correlation_info or None)
    """
    # Get appropriate reference range
    range_key = lab_name
    if gender.lower() in ["m", "male"] and f"{lab_name}_male" in REFERENCE_RANGES:
        range_key = f"{lab_name}_male"
    elif gender.lower() in ["f", "female"] and f"{lab_name}_female" in REFERENCE_RANGES:
        range_key = f"{lab_name}_female"

    if range_key not in REFERENCE_RANGES and lab_name not in REFERENCE_RANGES:
        return ("unknown", None)

    ref = REFERENCE_RANGES.get(range_key, REFERENCE_RANGES.get(lab_name))
    low = ref["low"]
    high = ref["high"]

    # Determine status
    if value < low * 0.5:
        status = "critical_low"
    elif value < low:
        status = "low"
    elif value > high * 2:
        status = "critical_high"
    elif value > high:
        status = "high"
    else:
        status = "normal"

    # Get skin correlation
    correlation_key = None
    if status in ["low", "critical_low"]:
        correlation_key = f"low_{lab_name}"
    elif status in ["high", "critical_high"]:
        correlation_key = f"high_{lab_name}"

    correlation = LAB_SKIN_CORRELATIONS.get(correlation_key)

    return (status, correlation)


def analyze_lab_results(
    lab_data: Dict[str, Any],
    skin_condition: Optional[str] = None,
    gender: str = "unknown"
) -> LabAnalysisResult:
    """
    Analyze complete lab results for skin relevance.

    Args:
        lab_data: Dictionary of lab values
        skin_condition: Current skin diagnosis (if any)
        gender: Patient gender for gender-specific ranges

    Returns:
        LabAnalysisResult with findings and recommendations
    """
    abnormal_findings: List[LabFinding] = []
    skin_correlations: List[SkinLabCorrelation] = []
    key_concerns: List[str] = []
    all_recommendations: List[str] = []
    monitoring_suggestions: List[str] = []

    # Track which systems have issues
    affected_systems = set()

    # Analyze each lab value
    lab_mappings = {
        "wbc": "wbc",
        "hemoglobin": "hemoglobin",
        "platelets": "platelets",
        "eosinophils": "eosinophils",
        "glucose_fasting": "glucose",
        "hba1c": "hba1c",
        "creatinine": "creatinine",
        "egfr": "egfr",
        "alt": "liver_enzymes",
        "ast": "liver_enzymes",
        "bilirubin_total": "bilirubin",
        "tsh": "tsh",
        "iron": "iron",
        "ferritin": "ferritin",
        "vitamin_d": "vitamin_d",
        "vitamin_b12": "b12",
        "crp": "crp",
        "esr": "esr",
        "ige_total": "ige",
    }

    for lab_field, lab_category in lab_mappings.items():
        value = lab_data.get(lab_field)
        if value is None:
            continue

        status, correlation = analyze_lab_value(lab_field, value, gender)

        if status != "normal" and status != "unknown":
            # Get reference info
            range_key = lab_field
            if gender.lower() in ["m", "male"] and f"{lab_field}_male" in REFERENCE_RANGES:
                range_key = f"{lab_field}_male"
            elif gender.lower() in ["f", "female"] and f"{lab_field}_female" in REFERENCE_RANGES:
                range_key = f"{lab_field}_female"

            ref = REFERENCE_RANGES.get(range_key, REFERENCE_RANGES.get(lab_field, {}))

            skin_implications = []
            recommendations = []
            relevance = RelevanceLevel.LOW

            if correlation:
                skin_implications = correlation.get("conditions", [])[:5]  # Top 5
                recommendations = correlation.get("recommendations", [])
                relevance = RelevanceLevel.HIGH if len(skin_implications) > 3 else RelevanceLevel.MODERATE

                # Add to key concerns
                if relevance == RelevanceLevel.HIGH:
                    key_concerns.append(
                        f"{ref.get('name', lab_field)} is {status}: associated with skin conditions"
                    )

                # Track affected system
                if "glucose" in lab_field or "hba1c" in lab_field:
                    affected_systems.add("metabolic")
                elif "tsh" in lab_field or "t4" in lab_field:
                    affected_systems.add("thyroid")
                elif "iron" in lab_field or "ferritin" in lab_field:
                    affected_systems.add("iron")
                elif "vitamin" in lab_field:
                    affected_systems.add("nutritional")
                elif "crp" in lab_field or "esr" in lab_field:
                    affected_systems.add("inflammatory")

            finding = LabFinding(
                lab_name=ref.get("name", lab_field),
                value=value,
                unit=ref.get("unit", ""),
                normal_range=f"{ref.get('low', 'N/A')}-{ref.get('high', 'N/A')}",
                status=status,
                skin_relevance=relevance,
                skin_implications=skin_implications,
                recommendations=recommendations
            )
            abnormal_findings.append(finding)
            all_recommendations.extend(recommendations)

    # Check ANA separately (boolean)
    if lab_data.get("ana_positive"):
        correlation = LAB_SKIN_CORRELATIONS.get("positive_ana")
        if correlation:
            finding = LabFinding(
                lab_name="ANA (Antinuclear Antibody)",
                value=1,
                unit="positive",
                normal_range="negative",
                status="positive",
                skin_relevance=RelevanceLevel.HIGH,
                skin_implications=correlation["conditions"][:5],
                recommendations=correlation["recommendations"]
            )
            abnormal_findings.append(finding)
            key_concerns.append("Positive ANA: may indicate autoimmune condition with skin manifestations")
            all_recommendations.extend(correlation["recommendations"])
            affected_systems.add("autoimmune")

    # Check stool parasites
    if lab_data.get("stool_parasites") and lab_data["stool_parasites"].lower() not in ["none", "none detected", "negative"]:
        correlation = LAB_SKIN_CORRELATIONS.get("stool_parasites")
        if correlation:
            finding = LabFinding(
                lab_name="Stool Parasites",
                value=1,
                unit="detected",
                normal_range="none detected",
                status="positive",
                skin_relevance=RelevanceLevel.HIGH,
                skin_implications=correlation["conditions"],
                recommendations=correlation["recommendations"]
            )
            abnormal_findings.append(finding)
            key_concerns.append("Parasites detected: may cause skin reactions")
            all_recommendations.extend(correlation["recommendations"])

    # Build skin correlations based on current skin condition
    if skin_condition:
        skin_lower = skin_condition.lower()

        # Check if any lab findings correlate with current skin condition
        for finding in abnormal_findings:
            for implication in finding.skin_implications:
                if any(word in implication.lower() for word in skin_lower.split()):
                    correlation = SkinLabCorrelation(
                        condition=skin_condition,
                        correlation_strength="strong" if finding.skin_relevance == RelevanceLevel.HIGH else "moderate",
                        supporting_labs=[finding.lab_name],
                        explanation=f"Your {finding.lab_name} level ({finding.status}) may be contributing to or associated with your {skin_condition}.",
                        clinical_significance="This lab abnormality is known to be associated with your skin condition and addressing it may help improve your skin health."
                    )
                    skin_correlations.append(correlation)
                    break

    # Calculate overall skin health score
    # Start at 100, deduct for each abnormal finding
    score = 100.0
    for finding in abnormal_findings:
        if finding.skin_relevance == RelevanceLevel.HIGH:
            score -= 15
        elif finding.skin_relevance == RelevanceLevel.MODERATE:
            score -= 8
        else:
            score -= 3

    score = max(0, min(100, score))

    # Generate monitoring suggestions based on affected systems
    if "metabolic" in affected_systems:
        monitoring_suggestions.append("Recheck glucose/HbA1c in 3 months")
    if "thyroid" in affected_systems:
        monitoring_suggestions.append("Recheck thyroid panel in 6-8 weeks")
    if "iron" in affected_systems:
        monitoring_suggestions.append("Recheck iron studies in 3 months after supplementation")
    if "nutritional" in affected_systems:
        monitoring_suggestions.append("Recheck vitamin levels after 3 months of supplementation")
    if "inflammatory" in affected_systems:
        monitoring_suggestions.append("Monitor inflammatory markers if on treatment")
    if "autoimmune" in affected_systems:
        monitoring_suggestions.append("Follow up with rheumatology for autoimmune workup")

    # Deduplicate recommendations
    unique_recommendations = list(dict.fromkeys(all_recommendations))

    return LabAnalysisResult(
        abnormal_findings=abnormal_findings,
        skin_correlations=skin_correlations,
        overall_skin_health_score=round(score, 1),
        key_concerns=key_concerns,
        recommendations=unique_recommendations[:10],  # Top 10
        monitoring_suggestions=monitoring_suggestions
    )


def format_lab_analysis_for_response(analysis: LabAnalysisResult) -> Dict[str, Any]:
    """Format lab analysis result for API response."""
    return {
        "abnormal_findings": [
            {
                "lab_name": f.lab_name,
                "value": f.value,
                "unit": f.unit,
                "normal_range": f.normal_range,
                "status": f.status,
                "skin_relevance": f.skin_relevance.value,
                "skin_implications": f.skin_implications,
                "recommendations": f.recommendations
            }
            for f in analysis.abnormal_findings
        ],
        "skin_correlations": [
            {
                "condition": c.condition,
                "correlation_strength": c.correlation_strength,
                "supporting_labs": c.supporting_labs,
                "explanation": c.explanation,
                "clinical_significance": c.clinical_significance
            }
            for c in analysis.skin_correlations
        ],
        "overall_skin_health_score": analysis.overall_skin_health_score,
        "key_concerns": analysis.key_concerns,
        "recommendations": analysis.recommendations,
        "monitoring_suggestions": analysis.monitoring_suggestions,
        "disclaimer": (
            "This analysis is for educational purposes only and does not constitute medical advice. "
            "Lab result interpretation should be done by a qualified healthcare provider who can "
            "consider your complete medical history and clinical context."
        )
    }


def get_condition_relevant_labs(skin_condition: str) -> List[Dict[str, str]]:
    """
    Get list of lab tests that are relevant for a specific skin condition.
    Useful for suggesting which labs to check.
    """
    condition_lower = skin_condition.lower()

    relevant_labs = []

    # Eczema/Atopic Dermatitis
    if any(term in condition_lower for term in ["eczema", "atopic", "dermatitis"]):
        relevant_labs.extend([
            {"test": "Total IgE", "reason": "Elevated in atopic conditions"},
            {"test": "Eosinophils", "reason": "Often elevated in allergic conditions"},
            {"test": "Vitamin D", "reason": "Deficiency may worsen eczema"},
        ])

    # Psoriasis
    if "psoriasis" in condition_lower:
        relevant_labs.extend([
            {"test": "CRP/ESR", "reason": "Markers of systemic inflammation"},
            {"test": "Vitamin D", "reason": "Deficiency associated with psoriasis severity"},
            {"test": "Fasting Glucose/HbA1c", "reason": "Psoriasis associated with metabolic syndrome"},
            {"test": "Lipid Panel", "reason": "Cardiovascular risk monitoring"},
            {"test": "Liver Function", "reason": "Baseline before systemic therapy"},
        ])

    # Hair Loss
    if any(term in condition_lower for term in ["alopecia", "hair loss", "effluvium"]):
        relevant_labs.extend([
            {"test": "Thyroid Panel (TSH, T4)", "reason": "Thyroid disorders cause hair loss"},
            {"test": "Ferritin", "reason": "Low iron causes hair loss even without anemia"},
            {"test": "Vitamin D", "reason": "Deficiency associated with alopecia"},
            {"test": "Vitamin B12", "reason": "Deficiency can cause hair changes"},
            {"test": "ANA", "reason": "Screen for autoimmune causes"},
        ])

    # Acne
    if "acne" in condition_lower:
        relevant_labs.extend([
            {"test": "Hormones (if female)", "reason": "PCOS and hormonal causes"},
            {"test": "Fasting Glucose", "reason": "High glycemic load worsens acne"},
        ])

    # Wound healing issues
    if any(term in condition_lower for term in ["wound", "ulcer", "healing"]):
        relevant_labs.extend([
            {"test": "Fasting Glucose/HbA1c", "reason": "Diabetes impairs wound healing"},
            {"test": "CBC", "reason": "Anemia affects healing"},
            {"test": "Albumin", "reason": "Nutritional status for healing"},
            {"test": "Vitamin D", "reason": "Role in wound healing"},
        ])

    # Pigmentation changes
    if any(term in condition_lower for term in ["vitiligo", "pigment", "hyperpigmentation"]):
        relevant_labs.extend([
            {"test": "Thyroid Panel", "reason": "Associated with vitiligo"},
            {"test": "Vitamin B12", "reason": "Deficiency causes pigment changes"},
            {"test": "ANA", "reason": "Autoimmune screening"},
        ])

    # Remove duplicates
    seen = set()
    unique_labs = []
    for lab in relevant_labs:
        if lab["test"] not in seen:
            seen.add(lab["test"])
            unique_labs.append(lab)

    return unique_labs
