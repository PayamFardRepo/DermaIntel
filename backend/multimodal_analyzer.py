"""
Multimodal Analysis Service

Combines image analysis with patient history, lab results, and lesion tracking
to provide comprehensive, context-aware skin analysis.

This service orchestrates:
1. Image-based classification (existing models)
2. Clinical context analysis (patient history, demographics)
3. Lab result integration (systemic health markers)
4. Historical lesion comparison (change detection)
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# Import existing analyzers
from clinical_context_analyzer import (
    ClinicalContextAnalyzer,
    apply_bayesian_adjustment
)
from lab_skin_integration import (
    integrate_labs_with_skin_analysis,
    adjust_skin_diagnosis_with_labs,
    get_user_lab_abnormalities,
    LabSkinCorrelation
)

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources for multimodal analysis."""
    IMAGE = "image"
    CLINICAL_HISTORY = "clinical_history"
    LAB_RESULTS = "lab_results"
    LESION_TRACKING = "lesion_tracking"


@dataclass
class ConfidenceAdjustment:
    """Tracks how confidence was adjusted by each data source."""
    source: str
    original_confidence: float
    adjusted_confidence: float
    delta: float
    factors: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""


@dataclass
class MultimodalResult:
    """Result of multimodal analysis combining all data sources."""
    # Core prediction
    predicted_class: str
    final_confidence: float

    # Raw image analysis
    image_predicted_class: str
    image_confidence: float
    image_probabilities: Dict[str, float]

    # Data sources used
    data_sources_used: List[str]
    multimodal_enabled: bool = True

    # Clinical context adjustments
    clinical_adjustments_applied: bool = False
    clinical_factors: List[Dict[str, Any]] = field(default_factory=list)
    clinical_confidence_delta: float = 0.0

    # Lab adjustments
    labs_integrated: bool = False
    lab_date: Optional[str] = None
    lab_correlations: List[Dict[str, Any]] = field(default_factory=list)
    lab_confidence_delta: float = 0.0
    lab_recommendations: List[str] = field(default_factory=list)

    # Historical comparison
    history_compared: bool = False
    previous_analyses_count: int = 0
    change_detected: bool = False
    growth_rate: Optional[str] = None
    trend: Optional[str] = None

    # Confidence breakdown
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)

    # Final adjusted probabilities
    adjusted_probabilities: Dict[str, float] = field(default_factory=dict)

    # Risk assessment
    risk_level: str = "unknown"
    risk_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Genetic risk data
    genetic_risk_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.final_confidence,

            "multimodal_analysis": {
                "enabled": self.multimodal_enabled,
                "data_sources": self.data_sources_used,

                "image_analysis": {
                    "predicted_class": self.image_predicted_class,
                    "raw_confidence": self.image_confidence,
                    "probabilities": self.image_probabilities
                },

                "clinical_adjustments": {
                    "applied": self.clinical_adjustments_applied,
                    "factors": [
                        {
                            "factor": f.get("name", f.get("factor", "Unknown")),
                            "multiplier": f.get("relative_risk", f.get("multiplier", 1.0))
                        }
                        for f in self.clinical_factors
                    ],
                    "confidence_delta": self.clinical_confidence_delta
                },

                "lab_adjustments": {
                    "applied": self.labs_integrated,
                    "lab_date": self.lab_date,
                    "correlations": self.lab_correlations,
                    "confidence_delta": self.lab_confidence_delta,
                    "recommendations": self.lab_recommendations
                },

                "historical_comparison": {
                    "compared": self.history_compared,
                    "previous_analyses": self.previous_analyses_count,
                    "change_detected": self.change_detected,
                    "growth_rate": self.growth_rate,
                    "trend": self.trend
                },

                "confidence_breakdown": self.confidence_breakdown,
                "adjusted_probabilities": self.adjusted_probabilities,

                "genetic_risk": self.genetic_risk_data
            },

            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations,
            "genetic_risk_alert": self.genetic_risk_data
        }


class MultimodalAnalyzer:
    """
    Orchestrates multimodal skin analysis by combining:
    - Image classification results
    - Patient clinical history
    - Lab results
    - Historical lesion data
    """

    def __init__(self):
        """Initialize the multimodal analyzer."""
        self.clinical_analyzer = ClinicalContextAnalyzer()
        self.lab_lookback_days = 90  # Consider labs from last 90 days

    def analyze(
        self,
        db_session,
        user_id: int,
        image_results: Dict[str, Any],
        clinical_context: Optional[Dict[str, Any]] = None,
        body_location: Optional[str] = None,
        lesion_group_id: Optional[int] = None,
        include_labs: bool = True,
        include_history: bool = True,
        include_lesion_tracking: bool = True
    ) -> MultimodalResult:
        """
        Perform multimodal analysis combining all available data sources.

        Args:
            db_session: Database session
            user_id: User ID for fetching history and labs
            image_results: Results from image classification containing:
                - predicted_class: str
                - confidence: float
                - probabilities: Dict[str, float]
            clinical_context: Optional clinical context from form/stored profile
            body_location: Body location of the lesion
            lesion_group_id: Optional lesion group for tracking
            include_labs: Whether to integrate lab results
            include_history: Whether to use clinical history
            include_lesion_tracking: Whether to compare with previous analyses

        Returns:
            MultimodalResult with comprehensive analysis
        """
        # Import here to avoid circular imports
        from database import UserProfile, LabResults, LesionGroup, AnalysisHistory

        # Initialize result with image analysis
        data_sources = [DataSource.IMAGE.value]

        predicted_class = image_results.get("predicted_class", "unknown")
        image_confidence = image_results.get("confidence", 0.0)
        image_probabilities = image_results.get("probabilities", {})

        # Track running probability adjustments
        current_probabilities = image_probabilities.copy()
        confidence_breakdown = {"image_model": image_confidence}

        # Initialize result
        result = MultimodalResult(
            predicted_class=predicted_class,
            final_confidence=image_confidence,
            image_predicted_class=predicted_class,
            image_confidence=image_confidence,
            image_probabilities=image_probabilities,
            data_sources_used=data_sources
        )

        # ================================================================
        # 1. FETCH AND INTEGRATE PATIENT HISTORY
        # ================================================================
        if include_history:
            try:
                history_context = self._fetch_patient_history(
                    db_session, user_id, clinical_context
                )

                if history_context:
                    data_sources.append(DataSource.CLINICAL_HISTORY.value)

                    # Apply Bayesian adjustment
                    adjusted_probs, clinical_analysis = apply_bayesian_adjustment(
                        model_probabilities=current_probabilities,
                        clinical_context=history_context
                    )

                    # Calculate confidence delta
                    new_confidence = adjusted_probs.get(predicted_class, image_confidence)
                    clinical_delta = new_confidence - image_confidence

                    # Update result
                    result.clinical_adjustments_applied = True
                    result.clinical_factors = clinical_analysis.get("risk_factors", [])
                    result.clinical_confidence_delta = round(clinical_delta, 4)

                    # Update running state
                    current_probabilities = adjusted_probs
                    confidence_breakdown["clinical_adjustment"] = round(clinical_delta, 4)

                    # Add risk factors to result
                    result.risk_factors.extend([
                        f.get("name", f.get("description", "Unknown factor"))
                        for f in result.clinical_factors[:5]  # Top 5
                    ])

                    logger.info(f"Clinical context applied: {clinical_delta:+.2%} adjustment")

            except Exception as e:
                logger.warning(f"Failed to integrate patient history: {e}")

        # ================================================================
        # 2. INTEGRATE GENETIC RISK FACTORS
        # ================================================================
        try:
            genetic_result = self._integrate_genetic_risk(
                db_session, user_id, current_probabilities
            )

            if genetic_result and genetic_result.get("has_genetic_data"):
                data_sources.append("genetic_testing")

                # Apply genetic risk adjustments
                genetic_adjusted_probs = genetic_result.get("adjusted_probabilities", {})
                if genetic_adjusted_probs:
                    genetic_delta = genetic_adjusted_probs.get("Melanoma", 0) - current_probabilities.get("Melanoma", 0)
                    current_probabilities = genetic_adjusted_probs
                    confidence_breakdown["genetic_adjustment"] = round(genetic_delta, 4)

                # Add genetic risk factors to clinical_factors (for UI display)
                if genetic_result.get("melanoma_multiplier", 1.0) > 1.0:
                    result.clinical_adjustments_applied = True
                    multiplier = genetic_result['melanoma_multiplier']

                    # Add each risk gene as a clinical factor
                    for gene in genetic_result.get("risk_genes", ["MC1R"]):
                        result.clinical_factors.append({
                            "name": f"Genetic Risk ({gene})",
                            "relative_risk": multiplier
                        })

                    # Add prominent risk factor warning
                    result.risk_factors.append(
                        f"⚠️ ELEVATED GENETIC RISK: {multiplier:.1f}x increased melanoma susceptibility"
                    )

                    # Add genetic-specific recommendations
                    result.recommendations.insert(0,
                        f"GENETIC ALERT: MC1R variants detected - {multiplier:.1f}x melanoma risk. "
                        "Enhanced surveillance recommended regardless of current lesion classification."
                    )

                    if multiplier >= 3.0:
                        result.recommendations.insert(1,
                            "HIGH GENETIC RISK: Consider dermatologist referral for full-body skin exam "
                            "and establish baseline mole mapping due to significantly elevated melanoma risk."
                        )

                    # Store genetic data for frontend display
                    result.genetic_risk_data = {
                        "has_genetic_risk": True,
                        "melanoma_multiplier": multiplier,
                        "risk_genes": genetic_result.get("risk_genes", ["MC1R"]),
                        "risk_level": "high" if multiplier >= 3.0 else "moderate" if multiplier >= 2.0 else "elevated"
                    }

                logger.info(f"Genetic data integrated: {genetic_result.get('melanoma_multiplier', 1.0):.1f}x melanoma risk")

        except Exception as e:
            logger.warning(f"Failed to integrate genetic data: {e}")

        # ================================================================
        # 4. FETCH AND INTEGRATE LAB RESULTS
        # ================================================================
        if include_labs:
            try:
                lab_result = self._integrate_lab_results(
                    db_session, user_id, predicted_class,
                    current_probabilities, image_results.get("differential_diagnoses", [])
                )

                if lab_result and lab_result.get("labs_found"):
                    data_sources.append(DataSource.LAB_RESULTS.value)

                    # Update result
                    result.labs_integrated = True
                    result.lab_date = lab_result.get("lab_date")
                    result.lab_correlations = lab_result.get("correlations", [])
                    result.lab_recommendations = lab_result.get("recommendations", [])

                    # Get adjusted probabilities from lab integration
                    lab_adjusted_probs = lab_result.get("adjusted_probabilities", {})
                    if lab_adjusted_probs:
                        new_confidence = lab_adjusted_probs.get(predicted_class, 0)
                        old_confidence = current_probabilities.get(predicted_class, 0)
                        lab_delta = new_confidence - old_confidence

                        result.lab_confidence_delta = round(lab_delta, 4)
                        current_probabilities = lab_adjusted_probs
                        confidence_breakdown["lab_adjustment"] = round(lab_delta, 4)

                    logger.info(f"Lab results integrated from {result.lab_date}")

            except Exception as e:
                logger.warning(f"Failed to integrate lab results: {e}")

        # ================================================================
        # 5. COMPARE WITH HISTORICAL LESION DATA
        # ================================================================
        if include_lesion_tracking and (lesion_group_id or body_location):
            try:
                history_comparison = self._compare_with_history(
                    db_session, user_id, lesion_group_id, body_location
                )

                if history_comparison and history_comparison.get("found"):
                    data_sources.append(DataSource.LESION_TRACKING.value)

                    result.history_compared = True
                    result.previous_analyses_count = history_comparison.get("count", 0)
                    result.change_detected = history_comparison.get("change_detected", False)
                    result.growth_rate = history_comparison.get("growth_rate")
                    result.trend = history_comparison.get("trend")

                    # If significant change detected, adjust risk
                    if result.change_detected:
                        result.risk_factors.append("Recent change detected in lesion")

                    logger.info(f"Historical comparison: {result.previous_analyses_count} previous analyses")

            except Exception as e:
                logger.warning(f"Failed to compare with history: {e}")

        # ================================================================
        # 6. CALCULATE FINAL RESULTS
        # ================================================================

        # Update final probabilities and confidence
        result.adjusted_probabilities = current_probabilities
        result.final_confidence = current_probabilities.get(predicted_class, image_confidence)
        result.confidence_breakdown = confidence_breakdown
        result.data_sources_used = data_sources

        # Recalculate predicted class if probabilities changed significantly
        if current_probabilities:
            max_class = max(current_probabilities, key=current_probabilities.get)
            if max_class != predicted_class and current_probabilities[max_class] > result.final_confidence + 0.1:
                result.predicted_class = max_class
                result.final_confidence = current_probabilities[max_class]

        # Determine risk level
        result.risk_level = self._calculate_risk_level(
            result.final_confidence,
            result.predicted_class,
            result.risk_factors,
            result.change_detected,
            result.genetic_risk_data
        )

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def _fetch_patient_history(
        self,
        db_session,
        user_id: int,
        form_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Fetch patient history from database and merge with form-provided context.

        Returns:
            Combined clinical context dictionary
        """
        from database import UserProfile, User, FamilyMember

        context = form_context.copy() if form_context else {}

        # Fetch user profile
        profile = db_session.query(UserProfile).filter_by(user_id=user_id).first()
        user = db_session.query(User).filter_by(id=user_id).first()

        if profile:
            # Add profile data if not already in form context
            if not context.get("fitzpatrick_skin_type") and profile.skin_type:
                context["fitzpatrick_skin_type"] = profile.skin_type

            # Parse medical history if available
            if profile.medical_history:
                context["medical_history_text"] = profile.medical_history

                # Try to extract structured info from text
                history_lower = profile.medical_history.lower()
                if "melanoma" in history_lower:
                    context.setdefault("personal_history_melanoma", True)
                if "skin cancer" in history_lower:
                    context.setdefault("personal_history_skin_cancer", True)
                if "immunosuppres" in history_lower:
                    context.setdefault("immunosuppressed", True)

            # Parse family history text if available
            if profile.family_history:
                context["family_history_text"] = profile.family_history

                family_lower = profile.family_history.lower()
                if "melanoma" in family_lower:
                    context.setdefault("family_history_melanoma", True)
                if "skin cancer" in family_lower:
                    context.setdefault("family_history_skin_cancer", True)

        # Check FamilyMember table for structured family history
        family_members = db_session.query(FamilyMember).filter_by(user_id=user_id).all()
        if family_members:
            for member in family_members:
                if member.has_melanoma:
                    context["family_history_melanoma"] = True
                    # First-degree relatives have higher impact
                    if member.relationship_type in ["parent", "sibling", "child"]:
                        context["first_degree_melanoma"] = True
                if member.has_skin_cancer:
                    context["family_history_skin_cancer"] = True

        if user:
            # Calculate age from date of birth or use stored age
            if user.age:
                context.setdefault("patient_age", user.age)
            if user.gender:
                context["gender"] = user.gender

        return context if context else None

    def _integrate_lab_results(
        self,
        db_session,
        user_id: int,
        predicted_class: str,
        probabilities: Dict[str, float],
        differential_diagnoses: List[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch and integrate recent lab results.

        Returns:
            Dictionary with lab integration results or None
        """
        from database import LabResults

        # Find recent lab results
        cutoff_date = datetime.now() - timedelta(days=self.lab_lookback_days)

        recent_labs = db_session.query(LabResults).filter(
            LabResults.user_id == user_id,
            LabResults.test_date >= cutoff_date
        ).order_by(LabResults.test_date.desc()).first()

        if not recent_labs:
            return {"labs_found": False}

        # Convert lab record to dictionary
        lab_data = self._lab_record_to_dict(recent_labs)

        # Get lab abnormalities
        lab_correlations = get_user_lab_abnormalities(lab_data)

        if not lab_correlations:
            return {
                "labs_found": True,
                "lab_date": recent_labs.test_date.isoformat() if recent_labs.test_date else None,
                "correlations": [],
                "adjusted_probabilities": probabilities,
                "recommendations": []
            }

        # Adjust diagnosis with labs
        adjusted_probs, lab_insights, adjusted_differentials = adjust_skin_diagnosis_with_labs(
            predicted_class=predicted_class,
            class_probabilities=probabilities,
            lab_correlations=lab_correlations,
            differential_diagnoses=differential_diagnoses
        )

        # Extract recommendations from correlations
        recommendations = []
        for corr in lab_correlations[:5]:
            if hasattr(corr, 'explanation') and corr.explanation:
                recommendations.append(corr.explanation)

        # Format correlations for response
        formatted_correlations = []
        for corr in lab_correlations[:10]:
            formatted_correlations.append({
                "lab": corr.lab_name if hasattr(corr, 'lab_name') else str(corr),
                "value": corr.lab_value if hasattr(corr, 'lab_value') else None,
                "status": corr.abnormality_type if hasattr(corr, 'abnormality_type') else "abnormal",
                "modifier": corr.confidence_modifier if hasattr(corr, 'confidence_modifier') else 0,
                "explanation": corr.explanation if hasattr(corr, 'explanation') else ""
            })

        return {
            "labs_found": True,
            "lab_date": recent_labs.test_date.isoformat() if recent_labs.test_date else None,
            "correlations": formatted_correlations,
            "adjusted_probabilities": adjusted_probs,
            "insights": lab_insights,
            "recommendations": recommendations
        }

    def _lab_record_to_dict(self, lab_record) -> Dict[str, Any]:
        """Convert SQLAlchemy lab record to dictionary."""
        lab_fields = [
            # CBC
            'wbc', 'rbc', 'hemoglobin', 'hematocrit', 'platelets',
            'neutrophils', 'lymphocytes', 'monocytes', 'eosinophils', 'basophils',
            # Metabolic
            'glucose_fasting', 'glucose_random', 'hba1c', 'bun', 'creatinine', 'egfr',
            'sodium', 'potassium', 'chloride', 'co2', 'calcium', 'magnesium', 'phosphorus',
            # Liver
            'alt', 'ast', 'alp', 'bilirubin_total', 'bilirubin_direct',
            'albumin', 'total_protein', 'globulin',
            # Lipid
            'cholesterol_total', 'ldl', 'hdl', 'triglycerides',
            # Thyroid
            'tsh', 't4_total', 't4_free', 't3_total', 't3_free',
            # Iron
            'iron', 'ferritin', 'tibc', 'transferrin_saturation',
            # Vitamins
            'vitamin_d', 'vitamin_b12', 'folate',
            # Inflammatory
            'crp', 'esr', 'homocysteine',
            # Autoimmune
            'ana_positive', 'ana_titer', 'rheumatoid_factor',
            # Allergy
            'ige_total'
        ]

        result = {}
        for field in lab_fields:
            value = getattr(lab_record, field, None)
            if value is not None:
                result[field] = value

        return result

    def _integrate_genetic_risk(
        self,
        db_session,
        user_id: int,
        probabilities: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Integrate genetic risk factors from genetic testing results.

        Returns:
            Dictionary with genetic risk adjustments
        """
        from database import GeneticVariant

        # Find genetic variants with melanoma risk modifiers
        variants = db_session.query(GeneticVariant).filter(
            GeneticVariant.user_id == user_id,
            GeneticVariant.melanoma_risk_modifier != None,
            GeneticVariant.melanoma_risk_modifier > 1.0
        ).all()

        if not variants:
            return {"has_genetic_data": False}

        # Calculate combined melanoma risk multiplier
        max_multiplier = 1.0
        risk_genes = []

        for variant in variants:
            if variant.melanoma_risk_modifier and variant.melanoma_risk_modifier > max_multiplier:
                max_multiplier = variant.melanoma_risk_modifier
                risk_genes.append(variant.gene_symbol)

        if max_multiplier <= 1.0:
            return {"has_genetic_data": False}

        # Adjust probabilities - boost melanoma probability
        adjusted_probs = probabilities.copy()

        # Find melanoma key (might be "Melanoma", "mel", or "MEL")
        melanoma_key = None
        for key in adjusted_probs.keys():
            if "melanoma" in key.lower() or key.lower() == "mel":
                melanoma_key = key
                break

        if melanoma_key:
            # Apply multiplier to melanoma probability
            original_melanoma = adjusted_probs.get(melanoma_key, 0)
            adjusted_melanoma = min(original_melanoma * max_multiplier, 0.95)  # Cap at 95%
            adjusted_probs[melanoma_key] = adjusted_melanoma

            # Renormalize
            total = sum(adjusted_probs.values())
            if total > 0:
                adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}

        return {
            "has_genetic_data": True,
            "melanoma_multiplier": max_multiplier,
            "risk_genes": list(set(risk_genes)),
            "adjusted_probabilities": adjusted_probs
        }

    def _compare_with_history(
        self,
        db_session,
        user_id: int,
        lesion_group_id: Optional[int],
        body_location: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Compare current analysis with historical data for the same lesion.

        Returns:
            Dictionary with comparison results or None
        """
        from database import LesionGroup, AnalysisHistory

        # Try to find existing lesion group
        if lesion_group_id:
            lesion_group = db_session.query(LesionGroup).filter_by(
                id=lesion_group_id,
                user_id=user_id
            ).first()
        elif body_location:
            # Try to find by location
            lesion_group = db_session.query(LesionGroup).filter_by(
                user_id=user_id,
                body_location=body_location
            ).order_by(LesionGroup.created_at.desc()).first()
        else:
            return None

        if not lesion_group:
            return {"found": False}

        # Get previous analyses in this group
        previous_analyses = db_session.query(AnalysisHistory).filter_by(
            lesion_group_id=lesion_group.id
        ).order_by(AnalysisHistory.created_at.desc()).limit(10).all()

        if not previous_analyses:
            return {"found": False}

        return {
            "found": True,
            "count": len(previous_analyses),
            "change_detected": lesion_group.change_detected if hasattr(lesion_group, 'change_detected') else False,
            "growth_rate": f"{lesion_group.growth_rate:.1f}mm/month" if hasattr(lesion_group, 'growth_rate') and lesion_group.growth_rate else None,
            "trend": self._determine_trend(previous_analyses),
            "lesion_group_id": lesion_group.id
        }

    def _determine_trend(self, analyses: List) -> str:
        """Determine trend from historical analyses."""
        if len(analyses) < 2:
            return "insufficient_data"

        # Compare confidence levels over time
        confidences = [
            a.lesion_confidence or a.binary_confidence or 0
            for a in analyses
        ]

        if len(confidences) >= 2:
            recent_avg = sum(confidences[:3]) / min(3, len(confidences))
            older_avg = sum(confidences[3:6]) / max(1, min(3, len(confidences) - 3)) if len(confidences) > 3 else recent_avg

            if recent_avg > older_avg + 0.1:
                return "increasing"
            elif recent_avg < older_avg - 0.1:
                return "decreasing"

        return "stable"

    def _calculate_risk_level(
        self,
        confidence: float,
        predicted_class: str,
        risk_factors: List[str],
        change_detected: bool,
        genetic_risk_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Calculate overall risk level based on all factors including genetic risk."""
        # High-risk conditions
        high_risk_conditions = [
            "melanoma", "squamous_cell_carcinoma", "basal_cell_carcinoma",
            "merkel_cell_carcinoma", "malignant"
        ]

        # Start with base risk from prediction
        risk_score = 0.0

        # Add risk from predicted class
        for condition in high_risk_conditions:
            if condition.lower() in predicted_class.lower():
                risk_score += confidence * 50
                break

        # Add risk from confidence level
        if confidence > 0.8:
            risk_score += 20
        elif confidence > 0.6:
            risk_score += 10

        # Add risk from factors
        risk_score += len(risk_factors) * 5

        # Add risk from change detection
        if change_detected:
            risk_score += 15

        # IMPORTANT: Add significant risk from genetic factors
        if genetic_risk_data and genetic_risk_data.get("has_genetic_risk"):
            multiplier = genetic_risk_data.get("melanoma_multiplier", 1.0)
            if multiplier >= 3.0:
                risk_score += 30  # High genetic risk adds significant score
            elif multiplier >= 2.0:
                risk_score += 20  # Moderate genetic risk
            elif multiplier > 1.0:
                risk_score += 10  # Elevated genetic risk

        # Categorize
        if risk_score >= 60:
            return "high"
        elif risk_score >= 35:
            return "moderate"
        elif risk_score >= 15:
            return "low"
        else:
            return "minimal"

    def _generate_recommendations(self, result: MultimodalResult) -> List[str]:
        """Generate clinical recommendations based on analysis results."""
        recommendations = []

        # Risk-based recommendations
        if result.risk_level == "high":
            recommendations.append("Urgent dermatologist consultation recommended")
            recommendations.append("Consider biopsy for definitive diagnosis")
        elif result.risk_level == "moderate":
            recommendations.append("Schedule dermatologist appointment within 2-4 weeks")
            recommendations.append("Monitor for any changes in size, color, or shape")
        elif result.risk_level == "low":
            recommendations.append("Continue regular skin self-examinations")
            recommendations.append("Follow up if any changes occur")

        # Change-based recommendations
        if result.change_detected:
            recommendations.append("Lesion has shown changes - closer monitoring advised")

        # Lab-based recommendations
        if result.lab_recommendations:
            recommendations.extend(result.lab_recommendations[:3])

        # Confidence-based recommendations
        if result.final_confidence < 0.6:
            recommendations.append("Low confidence - consider professional evaluation for accurate diagnosis")

        return recommendations[:6]  # Limit to top 6 recommendations


# Convenience function for use in endpoints
def perform_multimodal_analysis(
    db_session,
    user_id: int,
    image_results: Dict[str, Any],
    clinical_context: Optional[Dict] = None,
    body_location: Optional[str] = None,
    lesion_group_id: Optional[int] = None,
    include_labs: bool = True,
    include_history: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to perform multimodal analysis and return dict result.

    Use this in endpoints for easy integration.
    """
    analyzer = MultimodalAnalyzer()
    result = analyzer.analyze(
        db_session=db_session,
        user_id=user_id,
        image_results=image_results,
        clinical_context=clinical_context,
        body_location=body_location,
        lesion_group_id=lesion_group_id,
        include_labs=include_labs,
        include_history=include_history
    )
    return result.to_dict()
