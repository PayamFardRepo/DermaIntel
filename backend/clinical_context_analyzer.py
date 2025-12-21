"""
Clinical Context Analyzer with Bayesian Prior Adjustment

This module analyzes clinical context information to calculate risk multipliers
that adjust the model's output probabilities using Bayesian reasoning.

The adjustments are based on established dermatological risk factors for
melanoma and other skin cancers from peer-reviewed literature.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class RiskFactor:
    """Represents a single risk factor with its relative risk"""
    name: str
    description: str
    relative_risk: float  # Multiplier for baseline risk (1.0 = no change)
    applies_to: List[str]  # Which conditions this affects


# ============================================================================
# RISK FACTOR DEFINITIONS
# Based on dermatological literature and clinical guidelines
# ============================================================================

# Age-based melanoma risk (baseline risk increases with age)
AGE_RISK_MULTIPLIERS = {
    (0, 20): 0.3,      # Very low risk in children/teens
    (20, 40): 0.8,     # Below average
    (40, 50): 1.0,     # Baseline
    (50, 60): 1.3,     # Elevated
    (60, 70): 1.6,     # High
    (70, 80): 1.8,     # Very high
    (80, 150): 2.0,    # Highest risk
}

# Fitzpatrick skin type risk multipliers for melanoma
FITZPATRICK_RISK = {
    "I": 2.0,    # Very fair - highest risk
    "II": 1.7,   # Fair - high risk
    "III": 1.3,  # Medium - moderate risk
    "IV": 0.8,   # Olive - lower risk
    "V": 0.5,    # Brown - low risk
    "VI": 0.3,   # Dark - lowest risk (but acral melanoma still possible)
}

# Lesion duration risk (new/changing lesions are more concerning)
DURATION_RISK = {
    "new": 1.5,        # New lesions need evaluation
    "recent": 1.3,     # Recent appearance
    "months": 1.1,     # Several months
    "one_year": 0.9,   # Stable for 1-2 years
    "years": 0.7,      # Stable for years
    "long_term": 0.5,  # Very stable, low concern
    "unknown": 1.0,    # No adjustment if unknown
}

# Body location risk multipliers
LOCATION_RISK = {
    # High-risk locations
    "back": 1.3,           # Common melanoma site
    "lower_leg": 1.2,      # Common in women
    "face": 1.2,           # Sun exposure
    "scalp": 1.3,          # Often missed
    "ear": 1.4,            # High UV exposure
    "neck": 1.2,           # Sun exposure

    # Acral locations (different melanoma subtype)
    "palm": 1.5,           # Acral melanoma risk
    "sole": 1.5,           # Acral melanoma risk
    "nail": 1.6,           # Subungual melanoma
    "mucosa": 1.7,         # Mucosal melanoma

    # Lower risk locations
    "trunk": 1.0,          # Baseline
    "arm": 0.9,
    "hand": 1.0,
    "foot": 1.1,
    "chest": 1.0,
    "abdomen": 0.9,
}


class ClinicalContextAnalyzer:
    """
    Analyzes clinical context to calculate Bayesian risk adjustments.

    Uses evidence-based risk factors to modify the pre-test probability
    (model output) to a post-test probability that accounts for clinical context.
    """

    def __init__(self):
        """Initialize the analyzer with risk factor definitions"""
        self.risk_factors_triggered: List[RiskFactor] = []

    def analyze(self, clinical_context: Dict) -> Dict:
        """
        Analyze clinical context and calculate risk adjustments.

        Args:
            clinical_context: Dictionary containing clinical information

        Returns:
            Dictionary with:
                - risk_multiplier: Overall risk multiplier
                - risk_factors: List of identified risk factors
                - risk_level: Categorized risk level
                - adjusted_priors: Suggested prior adjustments per condition
                - recommendations: Clinical recommendations
        """
        self.risk_factors_triggered = []

        if not clinical_context:
            return {
                "risk_multiplier": 1.0,
                "risk_factors": [],
                "risk_level": "unknown",
                "adjusted_priors": {},
                "recommendations": ["No clinical context provided. Consider completing the clinical questionnaire for more accurate risk assessment."]
            }

        # Calculate individual risk components
        age_risk = self._calculate_age_risk(clinical_context.get("patient_age"))
        skin_type_risk = self._calculate_skin_type_risk(clinical_context.get("fitzpatrick_skin_type"))
        duration_risk = self._calculate_duration_risk(clinical_context.get("lesion_duration"))
        location_risk = self._calculate_location_risk(clinical_context.get("body_location"))
        history_risk = self._calculate_history_risk(clinical_context)
        symptom_risk = self._calculate_symptom_risk(clinical_context.get("symptoms"))
        abcde_risk = self._calculate_abcde_risk(clinical_context.get("abcde_changes"))
        lifestyle_risk = self._calculate_lifestyle_risk(clinical_context)

        # Combine risk factors (multiplicative with dampening to avoid extreme values)
        raw_multiplier = (
            age_risk *
            skin_type_risk *
            duration_risk *
            location_risk *
            history_risk *
            symptom_risk *
            abcde_risk *
            lifestyle_risk
        )

        # Apply dampening to prevent extreme adjustments
        # Use logarithmic scaling to keep multiplier in reasonable range [0.2, 5.0]
        dampened_multiplier = self._dampen_multiplier(raw_multiplier)

        # Determine risk level
        risk_level = self._categorize_risk_level(dampened_multiplier)

        # Calculate condition-specific prior adjustments
        adjusted_priors = self._calculate_condition_priors(
            dampened_multiplier,
            clinical_context
        )

        # Generate clinical recommendations
        recommendations = self._generate_recommendations(
            dampened_multiplier,
            self.risk_factors_triggered,
            clinical_context
        )

        return {
            "risk_multiplier": round(dampened_multiplier, 3),
            "raw_multiplier": round(raw_multiplier, 3),
            "risk_factors": [
                {"name": rf.name, "description": rf.description, "relative_risk": rf.relative_risk}
                for rf in self.risk_factors_triggered
            ],
            "risk_level": risk_level,
            "adjusted_priors": adjusted_priors,
            "recommendations": recommendations,
            "component_risks": {
                "age": round(age_risk, 3),
                "skin_type": round(skin_type_risk, 3),
                "duration": round(duration_risk, 3),
                "location": round(location_risk, 3),
                "history": round(history_risk, 3),
                "symptoms": round(symptom_risk, 3),
                "abcde": round(abcde_risk, 3),
                "lifestyle": round(lifestyle_risk, 3),
            }
        }

    def _calculate_age_risk(self, age: Optional[int]) -> float:
        """Calculate risk multiplier based on patient age"""
        if age is None:
            return 1.0

        for (min_age, max_age), multiplier in AGE_RISK_MULTIPLIERS.items():
            if min_age <= age < max_age:
                if multiplier != 1.0:
                    self.risk_factors_triggered.append(RiskFactor(
                        name="Age",
                        description=f"Patient age {age} years",
                        relative_risk=multiplier,
                        applies_to=["Melanoma", "BCC", "SCC"]
                    ))
                return multiplier

        return 1.0

    def _calculate_skin_type_risk(self, skin_type: Optional[str]) -> float:
        """Calculate risk multiplier based on Fitzpatrick skin type"""
        if skin_type is None:
            return 1.0

        multiplier = FITZPATRICK_RISK.get(skin_type.upper(), 1.0)

        if multiplier != 1.0:
            risk_desc = "higher" if multiplier > 1.0 else "lower"
            self.risk_factors_triggered.append(RiskFactor(
                name="Skin Type",
                description=f"Fitzpatrick type {skin_type} ({risk_desc} melanoma risk)",
                relative_risk=multiplier,
                applies_to=["Melanoma"]
            ))

        return multiplier

    def _calculate_duration_risk(self, duration: Optional[str]) -> float:
        """Calculate risk multiplier based on lesion duration"""
        if duration is None:
            return 1.0

        multiplier = DURATION_RISK.get(duration.lower(), 1.0)

        if multiplier > 1.0:
            self.risk_factors_triggered.append(RiskFactor(
                name="New/Changing Lesion",
                description=f"Lesion duration: {duration}",
                relative_risk=multiplier,
                applies_to=["Melanoma", "BCC", "SCC"]
            ))

        return multiplier

    def _calculate_location_risk(self, location: Optional[str]) -> float:
        """Calculate risk multiplier based on body location"""
        if location is None:
            return 1.0

        # Normalize location string
        location_lower = location.lower().replace(" ", "_")

        multiplier = LOCATION_RISK.get(location_lower, 1.0)

        # Check for high-risk acral locations
        if location_lower in ["palm", "sole", "nail", "mucosa"]:
            self.risk_factors_triggered.append(RiskFactor(
                name="High-Risk Location",
                description=f"Acral/mucosal location: {location}",
                relative_risk=multiplier,
                applies_to=["Melanoma (Acral)"]
            ))
        elif multiplier > 1.1:
            self.risk_factors_triggered.append(RiskFactor(
                name="Location",
                description=f"Body location: {location}",
                relative_risk=multiplier,
                applies_to=["Melanoma", "BCC", "SCC"]
            ))

        return multiplier

    def _calculate_history_risk(self, context: Dict) -> float:
        """Calculate risk multiplier based on personal and family history"""
        multiplier = 1.0

        # Personal history of melanoma (MAJOR risk factor)
        if context.get("personal_history_melanoma"):
            multiplier *= 3.0
            self.risk_factors_triggered.append(RiskFactor(
                name="Personal Melanoma History",
                description="Previous melanoma diagnosis",
                relative_risk=3.0,
                applies_to=["Melanoma"]
            ))

        # Personal history of other skin cancers
        if context.get("personal_history_skin_cancer"):
            multiplier *= 1.5
            self.risk_factors_triggered.append(RiskFactor(
                name="Personal Skin Cancer History",
                description="Previous BCC/SCC diagnosis",
                relative_risk=1.5,
                applies_to=["BCC", "SCC", "Melanoma"]
            ))

        # Personal history of atypical moles
        if context.get("personal_history_atypical_moles"):
            multiplier *= 1.8
            self.risk_factors_triggered.append(RiskFactor(
                name="Atypical Mole Syndrome",
                description="History of dysplastic/atypical nevi",
                relative_risk=1.8,
                applies_to=["Melanoma"]
            ))

        # Family history of melanoma (first-degree relative)
        if context.get("family_history_melanoma"):
            multiplier *= 2.0
            self.risk_factors_triggered.append(RiskFactor(
                name="Family Melanoma History",
                description="First-degree relative with melanoma",
                relative_risk=2.0,
                applies_to=["Melanoma"]
            ))

        # Family history of other skin cancers
        if context.get("family_history_skin_cancer"):
            multiplier *= 1.3
            self.risk_factors_triggered.append(RiskFactor(
                name="Family Skin Cancer History",
                description="Family history of skin cancer",
                relative_risk=1.3,
                applies_to=["BCC", "SCC", "Melanoma"]
            ))

        return multiplier

    def _calculate_symptom_risk(self, symptoms: Optional[Dict]) -> float:
        """Calculate risk multiplier based on lesion symptoms"""
        if not symptoms:
            return 1.0

        multiplier = 1.0

        # Bleeding is a significant warning sign
        if symptoms.get("bleeding"):
            multiplier *= 1.8
            self.risk_factors_triggered.append(RiskFactor(
                name="Bleeding",
                description="Lesion is bleeding",
                relative_risk=1.8,
                applies_to=["Melanoma", "BCC", "SCC"]
            ))

        # Ulceration
        if symptoms.get("ulceration"):
            multiplier *= 1.7
            self.risk_factors_triggered.append(RiskFactor(
                name="Ulceration",
                description="Lesion is ulcerated",
                relative_risk=1.7,
                applies_to=["Melanoma", "BCC", "SCC"]
            ))

        # Itching can be a sign of melanoma
        if symptoms.get("itching"):
            multiplier *= 1.3
            self.risk_factors_triggered.append(RiskFactor(
                name="Itching",
                description="Lesion is itching",
                relative_risk=1.3,
                applies_to=["Melanoma"]
            ))

        # Crusting/oozing
        if symptoms.get("crusting") or symptoms.get("oozing"):
            multiplier *= 1.4
            self.risk_factors_triggered.append(RiskFactor(
                name="Crusting/Oozing",
                description="Lesion has crusting or oozing",
                relative_risk=1.4,
                applies_to=["BCC", "SCC"]
            ))

        return multiplier

    def _calculate_abcde_risk(self, abcde: Optional[Dict]) -> float:
        """Calculate risk multiplier based on ABCDE criteria changes"""
        if not abcde:
            return 1.0

        multiplier = 1.0
        changes_count = 0

        if abcde.get("asymmetry_changed"):
            multiplier *= 1.4
            changes_count += 1

        if abcde.get("border_changed"):
            multiplier *= 1.4
            changes_count += 1

        if abcde.get("color_changed"):
            multiplier *= 1.5
            changes_count += 1

        if abcde.get("diameter_changed"):
            multiplier *= 1.3
            changes_count += 1

        if abcde.get("evolving"):
            multiplier *= 1.6
            changes_count += 1

        if changes_count > 0:
            self.risk_factors_triggered.append(RiskFactor(
                name="ABCDE Changes",
                description=f"{changes_count} ABCDE criteria showing changes",
                relative_risk=multiplier,
                applies_to=["Melanoma"]
            ))

        return multiplier

    def _calculate_lifestyle_risk(self, context: Dict) -> float:
        """Calculate risk multiplier based on lifestyle factors"""
        multiplier = 1.0

        # History of severe sunburns
        if context.get("history_severe_sunburns"):
            multiplier *= 1.5
            self.risk_factors_triggered.append(RiskFactor(
                name="Severe Sunburn History",
                description="History of blistering sunburns",
                relative_risk=1.5,
                applies_to=["Melanoma"]
            ))

        # Tanning bed use
        if context.get("uses_tanning_beds"):
            multiplier *= 1.7
            self.risk_factors_triggered.append(RiskFactor(
                name="Tanning Bed Use",
                description="Uses indoor tanning beds",
                relative_risk=1.7,
                applies_to=["Melanoma", "BCC", "SCC"]
            ))

        # Immunosuppression
        if context.get("immunosuppressed"):
            multiplier *= 2.0
            self.risk_factors_triggered.append(RiskFactor(
                name="Immunosuppression",
                description="Patient is immunosuppressed",
                relative_risk=2.0,
                applies_to=["Melanoma", "BCC", "SCC"]
            ))

        # Many moles (>50)
        if context.get("many_moles"):
            multiplier *= 1.6
            self.risk_factors_triggered.append(RiskFactor(
                name="Multiple Moles",
                description="Patient has >50 moles",
                relative_risk=1.6,
                applies_to=["Melanoma"]
            ))

        return multiplier

    def _dampen_multiplier(self, raw_multiplier: float) -> float:
        """
        Apply dampening to prevent extreme risk adjustments.
        Uses logarithmic scaling to keep multiplier in [0.2, 5.0] range.
        """
        if raw_multiplier <= 0:
            return 0.2

        # Use log scaling for large multipliers
        if raw_multiplier > 5.0:
            dampened = 5.0 * (1 + math.log(raw_multiplier / 5.0))
            return min(dampened, 10.0)  # Cap at 10x
        elif raw_multiplier < 0.2:
            return 0.2

        return raw_multiplier

    def _categorize_risk_level(self, multiplier: float) -> str:
        """Categorize the overall risk level based on multiplier"""
        if multiplier >= 3.0:
            return "very_high"
        elif multiplier >= 2.0:
            return "high"
        elif multiplier >= 1.3:
            return "moderate"
        elif multiplier >= 0.8:
            return "average"
        else:
            return "low"

    def _calculate_condition_priors(self, multiplier: float, context: Dict) -> Dict:
        """
        Calculate adjusted prior probabilities for each condition.

        Returns suggested adjustments to apply to model outputs.
        """
        # Base melanoma probability adjustment (most affected by clinical context)
        melanoma_adjustment = multiplier

        # BCC/SCC adjustments (less affected by some factors)
        bcc_scc_adjustment = 1.0 + (multiplier - 1.0) * 0.6

        # Benign conditions get inverse adjustment
        benign_adjustment = 1.0 / (1.0 + (multiplier - 1.0) * 0.3)

        return {
            "Melanoma": {
                "prior_adjustment": round(melanoma_adjustment, 3),
                "interpretation": self._get_adjustment_interpretation(melanoma_adjustment)
            },
            "Basal Cell Carcinoma": {
                "prior_adjustment": round(bcc_scc_adjustment, 3),
                "interpretation": self._get_adjustment_interpretation(bcc_scc_adjustment)
            },
            "Actinic Keratoses": {
                "prior_adjustment": round(bcc_scc_adjustment, 3),
                "interpretation": self._get_adjustment_interpretation(bcc_scc_adjustment)
            },
            "Melanocytic Nevi": {
                "prior_adjustment": round(benign_adjustment, 3),
                "interpretation": self._get_adjustment_interpretation(benign_adjustment)
            },
            "Benign Keratoses-Like Lesions": {
                "prior_adjustment": round(benign_adjustment, 3),
                "interpretation": self._get_adjustment_interpretation(benign_adjustment)
            }
        }

    def _get_adjustment_interpretation(self, adjustment: float) -> str:
        """Get human-readable interpretation of adjustment"""
        if adjustment >= 2.0:
            return "Significantly increased risk based on clinical factors"
        elif adjustment >= 1.5:
            return "Moderately increased risk based on clinical factors"
        elif adjustment >= 1.2:
            return "Slightly increased risk based on clinical factors"
        elif adjustment <= 0.7:
            return "Decreased risk based on clinical factors"
        elif adjustment <= 0.85:
            return "Slightly decreased risk based on clinical factors"
        else:
            return "No significant adjustment based on clinical factors"

    def _generate_recommendations(
        self,
        multiplier: float,
        risk_factors: List[RiskFactor],
        context: Dict
    ) -> List[str]:
        """Generate clinical recommendations based on risk assessment"""
        recommendations = []

        # High risk recommendations
        if multiplier >= 2.0:
            recommendations.append(
                "⚠️ ELEVATED RISK: Clinical factors suggest increased risk for skin cancer. "
                "Professional dermatological evaluation is strongly recommended."
            )

        # Specific recommendations based on risk factors
        if any(rf.name == "Personal Melanoma History" for rf in risk_factors):
            recommendations.append(
                "Due to personal melanoma history, regular full-body skin examinations "
                "every 3-6 months are recommended."
            )

        if any(rf.name == "Family Melanoma History" for rf in risk_factors):
            recommendations.append(
                "Family history of melanoma warrants enhanced surveillance. "
                "Consider genetic counseling if multiple family members affected."
            )

        if any(rf.name == "ABCDE Changes" for rf in risk_factors):
            recommendations.append(
                "Changes in ABCDE criteria are concerning. This lesion should be "
                "evaluated by a dermatologist promptly."
            )

        if any(rf.name in ["Bleeding", "Ulceration"] for rf in risk_factors):
            recommendations.append(
                "Bleeding or ulceration in a skin lesion requires prompt evaluation. "
                "Please see a dermatologist as soon as possible."
            )

        if any(rf.name == "High-Risk Location" for rf in risk_factors):
            recommendations.append(
                "Lesions on palms, soles, nails, or mucosal surfaces may represent "
                "acral melanoma. These locations require specialized evaluation."
            )

        if any(rf.name == "Tanning Bed Use" for rf in risk_factors):
            recommendations.append(
                "Tanning bed use significantly increases skin cancer risk. "
                "Consider discontinuing use and implement sun protection measures."
            )

        if any(rf.name == "Immunosuppression" for rf in risk_factors):
            recommendations.append(
                "Immunosuppressed patients have increased skin cancer risk. "
                "Regular dermatological screening is essential."
            )

        # Default recommendation if no specific ones
        if not recommendations:
            if multiplier >= 1.3:
                recommendations.append(
                    "Some risk factors identified. Consider professional evaluation "
                    "for any changing or concerning lesions."
                )
            else:
                recommendations.append(
                    "No major risk factors identified. Continue regular self-examination "
                    "and sun protection practices."
                )

        return recommendations


def apply_bayesian_adjustment(
    model_probabilities: Dict[str, float],
    clinical_context: Dict
) -> Tuple[Dict[str, float], Dict]:
    """
    Apply Bayesian adjustment to model probabilities based on clinical context.

    This is the main function to call from the classification endpoint.

    Args:
        model_probabilities: Raw probabilities from the model
        clinical_context: Clinical context dictionary

    Returns:
        Tuple of (adjusted_probabilities, analysis_report)
    """
    analyzer = ClinicalContextAnalyzer()
    analysis = analyzer.analyze(clinical_context)

    if not model_probabilities:
        return {}, analysis

    adjusted_probs = {}
    priors = analysis.get("adjusted_priors", {})

    # Apply adjustments to each condition
    total = 0.0
    for condition, prob in model_probabilities.items():
        adjustment = 1.0

        # Find matching prior adjustment
        for prior_condition, prior_data in priors.items():
            if prior_condition.lower() in condition.lower() or condition.lower() in prior_condition.lower():
                adjustment = prior_data.get("prior_adjustment", 1.0)
                break

        # Apply adjustment
        adjusted_prob = prob * adjustment
        adjusted_probs[condition] = adjusted_prob
        total += adjusted_prob

    # Renormalize to ensure probabilities sum to 1
    if total > 0:
        adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}

    # Round for display
    adjusted_probs = {k: round(v, 4) for k, v in adjusted_probs.items()}

    return adjusted_probs, analysis


# Singleton instance
_analyzer_instance = None


def get_clinical_context_analyzer() -> ClinicalContextAnalyzer:
    """Get or create the clinical context analyzer singleton"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ClinicalContextAnalyzer()
    return _analyzer_instance
