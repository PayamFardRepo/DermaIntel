"""
Skin Cancer Risk Calculator Module

Combines multiple validated risk factors to calculate personalized skin cancer risk:
- Fitzpatrick skin type
- Sun exposure history
- Family history of skin cancer
- Personal history of skin cancer
- Number of moles/nevi
- History of sunburns
- Tanning bed use
- Immunosuppression status
- Age and gender
- Geographic location (UV index)
- Occupational sun exposure
- AI analysis findings

Based on validated risk models:
- Fears & Saraiya (2011) - CDC Melanoma Risk Assessment
- Usher-Smith et al. (2014) - Melanoma Risk Prediction Models Systematic Review
- Olsen et al. (2018) - Skin Cancer Risk Factors
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date
import math


class FitzpatrickType(Enum):
    """Fitzpatrick Skin Phototypes"""
    TYPE_I = 1    # Very fair, always burns, never tans
    TYPE_II = 2   # Fair, usually burns, tans minimally
    TYPE_III = 3  # Medium, sometimes burns, tans uniformly
    TYPE_IV = 4   # Olive, rarely burns, tans easily
    TYPE_V = 5    # Brown, very rarely burns, tans very easily
    TYPE_VI = 6   # Dark brown/black, never burns


class SunExposureLevel(Enum):
    """Lifetime sun exposure levels"""
    MINIMAL = 1      # Indoor lifestyle, minimal outdoor time
    LOW = 2          # Occasional outdoor activities
    MODERATE = 3     # Regular outdoor activities, some sun protection
    HIGH = 4         # Frequent outdoor activities, inconsistent protection
    VERY_HIGH = 5    # Outdoor occupation or lifestyle, minimal protection


class CancerType(Enum):
    """Types of skin cancer"""
    MELANOMA = "melanoma"
    BASAL_CELL = "basal_cell_carcinoma"
    SQUAMOUS_CELL = "squamous_cell_carcinoma"
    MERKEL_CELL = "merkel_cell_carcinoma"
    ANY = "any"


@dataclass
class FamilyHistoryEntry:
    """Family history of skin cancer"""
    relation: str  # first_degree, second_degree
    cancer_type: CancerType
    age_at_diagnosis: Optional[int] = None
    multiple_primaries: bool = False


@dataclass
class PersonalHistoryEntry:
    """Personal history of skin cancer or precancers"""
    condition: str  # melanoma, bcc, scc, actinic_keratosis, dysplastic_nevus
    year_diagnosed: Optional[int] = None
    location: Optional[str] = None
    recurrence: bool = False


@dataclass
class SunburnHistory:
    """History of sunburns"""
    childhood_severe_burns: int = 0  # Before age 18, blistering burns
    childhood_moderate_burns: int = 0  # Before age 18, peeling burns
    adult_severe_burns: int = 0  # After 18, blistering burns
    adult_moderate_burns: int = 0  # After 18, peeling burns


@dataclass
class AIAnalysisFindings:
    """Findings from AI skin analysis"""
    total_lesions_analyzed: int = 0
    high_risk_lesions: int = 0
    medium_risk_lesions: int = 0
    atypical_moles: int = 0
    suspicious_features: List[str] = field(default_factory=list)
    abcde_flags: Dict[str, int] = field(default_factory=dict)
    highest_malignancy_probability: float = 0.0
    conditions_detected: List[str] = field(default_factory=list)


@dataclass
class GeneticTestFindings:
    """Findings from genetic/NGS testing"""
    has_genetic_data: bool = False
    test_date: Optional[date] = None
    lab_name: Optional[str] = None

    # Detected variants by gene
    variants: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Risk multipliers derived from variants
    melanoma_genetic_risk_multiplier: float = 1.0
    nmsc_genetic_risk_multiplier: float = 1.0

    # High-impact genes detected
    high_risk_genes: List[str] = field(default_factory=list)  # e.g., CDKN2A, BAP1
    moderate_risk_genes: List[str] = field(default_factory=list)  # e.g., MC1R variants
    pharmacogenomic_flags: List[str] = field(default_factory=list)  # e.g., TPMT, DPYD

    # Specific conditions
    familial_melanoma_syndrome: bool = False
    gorlin_syndrome: bool = False
    xeroderma_pigmentosum: bool = False

    # Summary
    pathogenic_variant_count: int = 0
    likely_pathogenic_count: int = 0
    vus_count: int = 0


@dataclass
class LabResultFindings:
    """Findings from lab results relevant to skin cancer risk"""
    has_lab_data: bool = False
    test_date: Optional[date] = None
    lab_name: Optional[str] = None

    # Vitamin D status (low levels associated with increased cancer risk)
    vitamin_d_level: Optional[float] = None  # ng/mL
    vitamin_d_status: str = "unknown"  # deficient (<20), insufficient (20-29), sufficient (30+)

    # Immunosuppression markers
    wbc_count: Optional[float] = None
    lymphocyte_count: Optional[float] = None
    immunosuppressed_by_labs: bool = False

    # Inflammatory markers (chronic inflammation may affect risk)
    crp_level: Optional[float] = None  # mg/L
    esr_level: Optional[float] = None  # mm/hr
    elevated_inflammation: bool = False

    # Autoimmune markers (some autoimmune conditions increase skin cancer risk)
    ana_positive: bool = False

    # Liver function (affects drug metabolism for treatment)
    liver_function_normal: bool = True

    # Risk multipliers derived from lab values
    lab_risk_multiplier: float = 1.0
    risk_factors_from_labs: List[str] = field(default_factory=list)


@dataclass
class RiskFactorInput:
    """Complete input for risk calculation"""
    # Demographics
    age: int
    gender: str  # male, female, other

    # Skin characteristics
    fitzpatrick_type: FitzpatrickType
    natural_hair_color: str  # red, blonde, light_brown, dark_brown, black
    natural_eye_color: str  # blue, green, hazel, brown, black
    freckles: str  # none, few, moderate, many
    total_mole_count: str  # none, few_1_10, some_11_25, moderate_26_50, many_51_100, very_many_100plus

    # Sun exposure
    sun_exposure_level: SunExposureLevel
    outdoor_occupation: bool = False
    geographic_latitude: Optional[float] = None  # For UV index estimation
    sunscreen_use: str = "sometimes"  # never, rarely, sometimes, usually, always
    protective_clothing: str = "sometimes"  # never, rarely, sometimes, usually, always
    peak_sun_avoidance: bool = False  # Avoids 10am-4pm sun

    # Sunburn history
    sunburn_history: SunburnHistory = field(default_factory=SunburnHistory)

    # Tanning
    tanning_bed_use: str = "never"  # never, past, occasional, regular
    tanning_bed_years: int = 0

    # Medical history
    immunosuppressed: bool = False
    immunosuppression_reason: Optional[str] = None  # transplant, hiv, medication, cancer_treatment
    radiation_therapy_history: bool = False
    chronic_skin_conditions: List[str] = field(default_factory=list)

    # Family history
    family_history: List[FamilyHistoryEntry] = field(default_factory=list)

    # Personal history
    personal_history: List[PersonalHistoryEntry] = field(default_factory=list)

    # AI findings
    ai_findings: Optional[AIAnalysisFindings] = None

    # Genetic test findings (from NGS/VCF)
    genetic_findings: Optional[GeneticTestFindings] = None

    # Lab result findings
    lab_findings: Optional[LabResultFindings] = None


@dataclass
class RiskScore:
    """Calculated risk score with breakdown"""
    overall_risk_score: float  # 0-100
    risk_category: str  # very_low, low, moderate, high, very_high

    # Individual component scores (0-100)
    genetic_score: float
    phenotype_score: float
    sun_exposure_score: float
    behavioral_score: float
    medical_history_score: float
    ai_findings_score: float

    # Relative risks
    melanoma_relative_risk: float
    bcc_relative_risk: float
    scc_relative_risk: float

    # Lifetime risk estimates
    melanoma_lifetime_risk_percent: float
    nmsc_lifetime_risk_percent: float

    # Percentile (compared to general population)
    population_percentile: int

    # Risk factors identified
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    protective_factors: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    screening_frequency: str = ""
    recommendations: List[str] = field(default_factory=list)
    urgent_concerns: List[str] = field(default_factory=list)

    # Confidence
    confidence_level: str = "moderate"  # low, moderate, high
    data_completeness: float = 0.0  # Percentage of fields provided


class SkinCancerRiskCalculator:
    """
    Comprehensive skin cancer risk calculator using validated risk models.
    """

    # Baseline lifetime risks (US population)
    BASELINE_MELANOMA_RISK = 2.6  # ~1 in 38 for men, 1 in 60 for women
    BASELINE_NMSC_RISK = 20.0  # ~1 in 5 Americans

    # Relative risk multipliers based on research
    FITZPATRICK_RR = {
        FitzpatrickType.TYPE_I: 10.0,
        FitzpatrickType.TYPE_II: 5.0,
        FitzpatrickType.TYPE_III: 2.5,
        FitzpatrickType.TYPE_IV: 1.0,
        FitzpatrickType.TYPE_V: 0.5,
        FitzpatrickType.TYPE_VI: 0.3,
    }

    HAIR_COLOR_RR = {
        "red": 3.64,
        "blonde": 1.96,
        "light_brown": 1.62,
        "dark_brown": 1.0,
        "black": 0.8,
    }

    EYE_COLOR_RR = {
        "blue": 1.47,
        "green": 1.61,
        "hazel": 1.52,
        "brown": 1.0,
        "black": 0.9,
    }

    MOLE_COUNT_RR = {
        "none": 0.5,
        "few_1_10": 1.0,
        "some_11_25": 1.5,
        "moderate_26_50": 3.0,
        "many_51_100": 6.0,
        "very_many_100plus": 10.0,
    }

    FRECKLES_RR = {
        "none": 1.0,
        "few": 1.3,
        "moderate": 1.8,
        "many": 2.5,
    }

    def __init__(self):
        self.calculation_date = datetime.now()

    def calculate_risk(self, input_data: RiskFactorInput) -> RiskScore:
        """Calculate comprehensive skin cancer risk score."""

        # Calculate individual component scores
        genetic_score = self._calculate_genetic_score(input_data)
        phenotype_score = self._calculate_phenotype_score(input_data)
        sun_exposure_score = self._calculate_sun_exposure_score(input_data)
        behavioral_score = self._calculate_behavioral_score(input_data)
        medical_history_score = self._calculate_medical_history_score(input_data)
        ai_findings_score = self._calculate_ai_findings_score(input_data)

        # Calculate relative risks first (needed for combination bonus)
        melanoma_rr = self._calculate_melanoma_relative_risk(input_data)
        bcc_rr = self._calculate_bcc_relative_risk(input_data)
        scc_rr = self._calculate_scc_relative_risk(input_data)

        # Weight the components
        weights = {
            'genetic': 0.20,
            'phenotype': 0.20,
            'sun_exposure': 0.20,
            'behavioral': 0.15,
            'medical_history': 0.15,
            'ai_findings': 0.10,
        }

        base_score = (
            genetic_score * weights['genetic'] +
            phenotype_score * weights['phenotype'] +
            sun_exposure_score * weights['sun_exposure'] +
            behavioral_score * weights['behavioral'] +
            medical_history_score * weights['medical_history'] +
            ai_findings_score * weights['ai_findings']
        )

        # =================================================================
        # HIGH-RISK COMBINATION BONUS
        # When multiple high-risk factors combine, the overall risk is greater
        # than the sum of individual components. Add bonus based on relative risk.
        # =================================================================
        combination_bonus = 0
        max_rr = max(melanoma_rr, bcc_rr, scc_rr)

        # Count high-risk factors
        high_risk_count = self._count_high_risk_factors(input_data)

        # Bonus based on combined relative risk (logarithmic scaling)
        if max_rr >= 100:
            combination_bonus += 30  # Extreme risk (e.g., XP genes)
        elif max_rr >= 20:
            combination_bonus += 20  # Very high risk
        elif max_rr >= 10:
            combination_bonus += 15  # High risk
        elif max_rr >= 5:
            combination_bonus += 10  # Elevated risk
        elif max_rr >= 2:
            combination_bonus += 5   # Moderate risk

        # Additional bonus for multiple high-risk factors
        if high_risk_count >= 5:
            combination_bonus += 15
        elif high_risk_count >= 3:
            combination_bonus += 10
        elif high_risk_count >= 2:
            combination_bonus += 5

        overall_score = min(base_score + combination_bonus, 100)

        # Calculate lifetime risks
        melanoma_lifetime = self._calculate_lifetime_melanoma_risk(input_data, melanoma_rr)
        nmsc_lifetime = self._calculate_lifetime_nmsc_risk(input_data, bcc_rr, scc_rr)

        # Determine risk category
        risk_category = self._determine_risk_category(overall_score)

        # Calculate population percentile
        percentile = self._calculate_percentile(overall_score)

        # Identify risk and protective factors
        risk_factors = self._identify_risk_factors(input_data)
        protective_factors = self._identify_protective_factors(input_data)

        # Generate recommendations
        recommendations = self._generate_recommendations(input_data, risk_category, risk_factors)
        screening_frequency = self._determine_screening_frequency(risk_category, input_data)
        urgent_concerns = self._identify_urgent_concerns(input_data)

        # Calculate data completeness
        completeness = self._calculate_data_completeness(input_data)

        return RiskScore(
            overall_risk_score=round(overall_score, 1),
            risk_category=risk_category,
            genetic_score=round(genetic_score, 1),
            phenotype_score=round(phenotype_score, 1),
            sun_exposure_score=round(sun_exposure_score, 1),
            behavioral_score=round(behavioral_score, 1),
            medical_history_score=round(medical_history_score, 1),
            ai_findings_score=round(ai_findings_score, 1),
            melanoma_relative_risk=round(melanoma_rr, 2),
            bcc_relative_risk=round(bcc_rr, 2),
            scc_relative_risk=round(scc_rr, 2),
            melanoma_lifetime_risk_percent=round(melanoma_lifetime, 2),
            nmsc_lifetime_risk_percent=round(nmsc_lifetime, 2),
            population_percentile=percentile,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            screening_frequency=screening_frequency,
            recommendations=recommendations,
            urgent_concerns=urgent_concerns,
            confidence_level=self._determine_confidence_level(completeness),
            data_completeness=round(completeness, 1),
        )

    def _calculate_genetic_score(self, input_data: RiskFactorInput) -> float:
        """Calculate genetic/family history risk score (0-100)."""
        score = 20  # Baseline

        # Family history
        first_degree_melanoma = 0
        second_degree_melanoma = 0
        first_degree_other = 0

        for fh in input_data.family_history:
            if fh.relation == "first_degree":
                if fh.cancer_type == CancerType.MELANOMA:
                    first_degree_melanoma += 1
                    if fh.multiple_primaries:
                        first_degree_melanoma += 0.5
                else:
                    first_degree_other += 1
            elif fh.relation == "second_degree":
                if fh.cancer_type == CancerType.MELANOMA:
                    second_degree_melanoma += 1

        # First-degree relative with melanoma: +25 points each (max 2)
        score += min(first_degree_melanoma, 2) * 25

        # Second-degree relative with melanoma: +10 points each (max 3)
        score += min(second_degree_melanoma, 3) * 10

        # First-degree with other skin cancer: +5 points each (max 3)
        score += min(first_degree_other, 3) * 5

        # =================================================================
        # GENETIC TEST FINDINGS (NGS/VCF data)
        # =================================================================
        if input_data.genetic_findings and input_data.genetic_findings.has_genetic_data:
            gf = input_data.genetic_findings

            # High-risk genes (CDKN2A, BAP1, CDK4, PTCH1, XPA)
            # These are very high-impact variants
            for gene in gf.high_risk_genes:
                if gene in ["CDKN2A", "CDK4"]:
                    score += 30  # Familial melanoma genes
                elif gene == "BAP1":
                    score += 25  # BAP1 tumor predisposition
                elif gene == "PTCH1":
                    score += 25  # Gorlin syndrome
                elif gene in ["XPA", "XPB", "XPC", "XPD", "XPE", "XPF", "XPG"]:
                    score += 35  # Xeroderma pigmentosum

            # Moderate-risk genes (MC1R, MITF)
            for gene in gf.moderate_risk_genes:
                if gene == "MC1R":
                    # MC1R variants are cumulative
                    mc1r_variants = len(gf.variants.get("MC1R", []))
                    score += min(mc1r_variants * 8, 20)  # Up to 20 points
                elif gene == "MITF":
                    score += 12

            # Pathogenic variant count
            score += min(gf.pathogenic_variant_count * 10, 25)
            score += min(gf.likely_pathogenic_count * 5, 15)

            # Syndrome flags
            if gf.familial_melanoma_syndrome:
                score += 15  # Additional points for confirmed syndrome
            if gf.gorlin_syndrome:
                score += 20
            if gf.xeroderma_pigmentosum:
                score += 30

        return min(score, 100)

    def _calculate_phenotype_score(self, input_data: RiskFactorInput) -> float:
        """Calculate phenotype risk score (0-100)."""
        score = 0

        # Fitzpatrick type
        fitzpatrick_scores = {
            FitzpatrickType.TYPE_I: 40,
            FitzpatrickType.TYPE_II: 30,
            FitzpatrickType.TYPE_III: 15,
            FitzpatrickType.TYPE_IV: 5,
            FitzpatrickType.TYPE_V: 2,
            FitzpatrickType.TYPE_VI: 0,
        }
        score += fitzpatrick_scores.get(input_data.fitzpatrick_type, 15)

        # Hair color
        hair_scores = {
            "red": 25,
            "blonde": 18,
            "light_brown": 10,
            "dark_brown": 3,
            "black": 0,
        }
        score += hair_scores.get(input_data.natural_hair_color, 5)

        # Eye color
        eye_scores = {
            "blue": 15,
            "green": 12,
            "hazel": 10,
            "brown": 3,
            "black": 0,
        }
        score += eye_scores.get(input_data.natural_eye_color, 5)

        # Freckles
        freckle_scores = {
            "none": 0,
            "few": 5,
            "moderate": 10,
            "many": 15,
        }
        score += freckle_scores.get(input_data.freckles, 0)

        # Mole count (major risk factor)
        mole_scores = {
            "none": 0,
            "few_1_10": 2,
            "some_11_25": 5,
            "moderate_26_50": 12,
            "many_51_100": 20,
            "very_many_100plus": 30,
        }
        score += mole_scores.get(input_data.total_mole_count, 5)

        return min(score, 100)

    def _calculate_sun_exposure_score(self, input_data: RiskFactorInput) -> float:
        """Calculate sun exposure risk score (0-100)."""
        score = 0

        # Lifetime sun exposure
        exposure_scores = {
            SunExposureLevel.MINIMAL: 5,
            SunExposureLevel.LOW: 15,
            SunExposureLevel.MODERATE: 30,
            SunExposureLevel.HIGH: 50,
            SunExposureLevel.VERY_HIGH: 70,
        }
        score += exposure_scores.get(input_data.sun_exposure_level, 25)

        # Outdoor occupation
        if input_data.outdoor_occupation:
            score += 15

        # Geographic location (latitude as proxy for UV index)
        if input_data.geographic_latitude is not None:
            # Closer to equator = higher risk
            lat = abs(input_data.geographic_latitude)
            if lat < 25:  # Tropical
                score += 15
            elif lat < 35:  # Subtropical
                score += 10
            elif lat < 45:  # Temperate
                score += 5

        return min(score, 100)

    def _calculate_behavioral_score(self, input_data: RiskFactorInput) -> float:
        """Calculate behavioral risk score (0-100)."""
        score = 50  # Start at midpoint

        # Sunscreen use (reduces score)
        sunscreen_adjustment = {
            "always": -25,
            "usually": -15,
            "sometimes": 0,
            "rarely": 15,
            "never": 25,
        }
        score += sunscreen_adjustment.get(input_data.sunscreen_use, 0)

        # Protective clothing (reduces score)
        clothing_adjustment = {
            "always": -15,
            "usually": -10,
            "sometimes": 0,
            "rarely": 10,
            "never": 15,
        }
        score += clothing_adjustment.get(input_data.protective_clothing, 0)

        # Peak sun avoidance
        if input_data.peak_sun_avoidance:
            score -= 10

        # Sunburn history (major risk factor)
        burns = input_data.sunburn_history
        # Childhood blistering sunburns are especially dangerous
        score += min(burns.childhood_severe_burns, 5) * 8
        score += min(burns.childhood_moderate_burns, 10) * 3
        score += min(burns.adult_severe_burns, 5) * 4
        score += min(burns.adult_moderate_burns, 10) * 1.5

        # Tanning bed use
        tanning_scores = {
            "never": 0,
            "past": 10,
            "occasional": 20,
            "regular": 35,
        }
        score += tanning_scores.get(input_data.tanning_bed_use, 0)

        # Additional risk for prolonged tanning bed use
        if input_data.tanning_bed_years > 0:
            score += min(input_data.tanning_bed_years, 10) * 2

        return max(0, min(score, 100))

    def _calculate_medical_history_score(self, input_data: RiskFactorInput) -> float:
        """Calculate medical history risk score (0-100)."""
        score = 10  # Baseline

        # Personal history of skin cancer
        melanoma_history = False
        nmsc_history = False
        precancer_history = False

        for ph in input_data.personal_history:
            if ph.condition == "melanoma":
                melanoma_history = True
                score += 40
                if ph.recurrence:
                    score += 20
            elif ph.condition in ["bcc", "scc", "basal_cell_carcinoma", "squamous_cell_carcinoma"]:
                nmsc_history = True
                score += 20
            elif ph.condition in ["actinic_keratosis", "dysplastic_nevus"]:
                precancer_history = True
                score += 10

        # Immunosuppression
        if input_data.immunosuppressed:
            score += 25
            if input_data.immunosuppression_reason == "transplant":
                score += 15  # Organ transplant recipients have very high risk
            elif input_data.immunosuppression_reason == "hiv":
                score += 10

        # Radiation therapy
        if input_data.radiation_therapy_history:
            score += 15

        # Age factor (risk increases with age)
        if input_data.age >= 65:
            score += 10
        elif input_data.age >= 50:
            score += 5

        # Gender (men have slightly higher risk)
        if input_data.gender == "male":
            score += 5

        # =================================================================
        # LAB RESULT FINDINGS
        # =================================================================
        if input_data.lab_findings and input_data.lab_findings.has_lab_data:
            lf = input_data.lab_findings

            # Vitamin D deficiency (associated with increased cancer risk)
            if lf.vitamin_d_status == "deficient":
                score += 10
            elif lf.vitamin_d_status == "insufficient":
                score += 5

            # Immunosuppression from lab values
            if lf.immunosuppressed_by_labs:
                score += 15  # Already immunosuppressed adds to existing check

            # Chronic inflammation markers
            if lf.elevated_inflammation:
                score += 8

            # Autoimmune markers (some autoimmune conditions increase skin cancer risk)
            if lf.ana_positive:
                score += 5

        return min(score, 100)

    def _calculate_ai_findings_score(self, input_data: RiskFactorInput) -> float:
        """Calculate risk score based on AI analysis findings (0-100)."""
        if input_data.ai_findings is None:
            return 25  # Neutral score when no AI data available

        ai = input_data.ai_findings
        score = 10  # Baseline

        # High-risk lesions
        score += min(ai.high_risk_lesions, 5) * 15

        # Medium-risk lesions
        score += min(ai.medium_risk_lesions, 10) * 5

        # Atypical moles
        score += min(ai.atypical_moles, 10) * 3

        # ABCDE flags
        abcde_flags_total = sum(ai.abcde_flags.values())
        score += min(abcde_flags_total, 10) * 3

        # Highest malignancy probability
        if ai.highest_malignancy_probability >= 0.8:
            score += 30
        elif ai.highest_malignancy_probability >= 0.6:
            score += 20
        elif ai.highest_malignancy_probability >= 0.4:
            score += 10
        elif ai.highest_malignancy_probability >= 0.2:
            score += 5

        # Suspicious conditions detected
        high_risk_conditions = [
            "melanoma", "basal cell carcinoma", "squamous cell carcinoma",
            "actinic keratosis", "dysplastic nevus"
        ]
        for condition in ai.conditions_detected:
            if any(hrc in condition.lower() for hrc in high_risk_conditions):
                score += 10

        return min(score, 100)

    def _count_high_risk_factors(self, input_data: RiskFactorInput) -> int:
        """Count the number of significant high-risk factors present."""
        count = 0

        # Phenotype factors
        if input_data.fitzpatrick_type in [FitzpatrickType.TYPE_I, FitzpatrickType.TYPE_II]:
            count += 1
        if input_data.natural_hair_color in ["red", "blonde"]:
            count += 1
        if input_data.natural_eye_color in ["blue", "green"]:
            count += 1
        if input_data.freckles in ["moderate", "many"]:
            count += 1
        if input_data.total_mole_count in ["many_51_100", "very_many_100plus"]:
            count += 1

        # Family history
        if any(fh.cancer_type == CancerType.MELANOMA for fh in input_data.family_history):
            count += 1

        # Personal history
        if any(ph.condition == "melanoma" for ph in input_data.personal_history):
            count += 1
        if any(ph.condition in ["bcc", "scc"] for ph in input_data.personal_history):
            count += 1

        # Behavioral factors
        if input_data.sunburn_history.childhood_severe_burns >= 1:
            count += 1
        if input_data.tanning_bed_use in ["regular", "occasional"]:
            count += 1
        if input_data.immunosuppressed:
            count += 1

        # Genetic findings
        if input_data.genetic_findings and input_data.genetic_findings.has_genetic_data:
            if input_data.genetic_findings.high_risk_genes:
                count += 2  # High-risk genes count double
            if input_data.genetic_findings.moderate_risk_genes:
                count += 1

        # Lab findings
        if input_data.lab_findings and input_data.lab_findings.has_lab_data:
            if input_data.lab_findings.vitamin_d_status == "deficient":
                count += 1
            if input_data.lab_findings.immunosuppressed_by_labs:
                count += 1

        return count

    def _calculate_melanoma_relative_risk(self, input_data: RiskFactorInput) -> float:
        """Calculate relative risk for melanoma compared to baseline population."""
        rr = 1.0

        # Fitzpatrick type
        rr *= self.FITZPATRICK_RR.get(input_data.fitzpatrick_type, 1.0)

        # Hair color
        rr *= self.HAIR_COLOR_RR.get(input_data.natural_hair_color, 1.0)

        # Mole count
        rr *= self.MOLE_COUNT_RR.get(input_data.total_mole_count, 1.0)

        # Family history
        first_degree_melanoma = sum(
            1 for fh in input_data.family_history
            if fh.relation == "first_degree" and fh.cancer_type == CancerType.MELANOMA
        )
        if first_degree_melanoma >= 2:
            rr *= 5.0
        elif first_degree_melanoma == 1:
            rr *= 2.24

        # Personal history
        if any(ph.condition == "melanoma" for ph in input_data.personal_history):
            rr *= 9.0  # Prior melanoma dramatically increases risk

        # Sunburn history
        burns = input_data.sunburn_history
        if burns.childhood_severe_burns >= 5:
            rr *= 2.0
        elif burns.childhood_severe_burns >= 1:
            rr *= 1.5

        # Tanning bed use
        if input_data.tanning_bed_use == "regular":
            rr *= 1.75
        elif input_data.tanning_bed_use in ["occasional", "past"]:
            rr *= 1.2

        # Immunosuppression
        if input_data.immunosuppressed:
            rr *= 2.5

        # =================================================================
        # GENETIC TEST FINDINGS - Apply genetic risk multipliers
        # =================================================================
        if input_data.genetic_findings and input_data.genetic_findings.has_genetic_data:
            gf = input_data.genetic_findings

            # Apply the calculated melanoma genetic risk multiplier
            if gf.melanoma_genetic_risk_multiplier > 1.0:
                rr *= gf.melanoma_genetic_risk_multiplier

            # Specific high-penetrance genes
            if "CDKN2A" in gf.high_risk_genes:
                rr *= 10.0  # 60-90% lifetime risk
            elif "CDK4" in gf.high_risk_genes:
                rr *= 8.0
            elif "BAP1" in gf.high_risk_genes:
                rr *= 5.0

            # XP genes cause extreme risk
            xp_genes = ["XPA", "XPB", "XPC", "XPD", "XPE", "XPF", "XPG"]
            if any(g in gf.high_risk_genes for g in xp_genes):
                rr *= 100.0  # 1000x increased risk

            # MC1R variants (moderate risk, cumulative)
            if "MC1R" in gf.moderate_risk_genes:
                mc1r_count = len(gf.variants.get("MC1R", []))
                rr *= (1.0 + (mc1r_count * 0.5))  # ~1.5-2.5x per variant

        # =================================================================
        # LAB RESULT FINDINGS - Apply lab risk multipliers
        # =================================================================
        if input_data.lab_findings and input_data.lab_findings.has_lab_data:
            lf = input_data.lab_findings
            rr *= lf.lab_risk_multiplier  # Pre-calculated from vitamin D, immunosuppression, etc.

        return rr

    def _calculate_bcc_relative_risk(self, input_data: RiskFactorInput) -> float:
        """Calculate relative risk for basal cell carcinoma."""
        rr = 1.0

        rr *= self.FITZPATRICK_RR.get(input_data.fitzpatrick_type, 1.0) * 0.8

        # Sun exposure is major factor for BCC
        exposure_rr = {
            SunExposureLevel.MINIMAL: 0.5,
            SunExposureLevel.LOW: 0.8,
            SunExposureLevel.MODERATE: 1.0,
            SunExposureLevel.HIGH: 2.0,
            SunExposureLevel.VERY_HIGH: 3.5,
        }
        rr *= exposure_rr.get(input_data.sun_exposure_level, 1.0)

        if input_data.immunosuppressed:
            rr *= 10.0  # Very high BCC risk with immunosuppression

        # Age
        if input_data.age >= 70:
            rr *= 2.0
        elif input_data.age >= 50:
            rr *= 1.5

        # =================================================================
        # GENETIC TEST FINDINGS - BCC-specific genes
        # =================================================================
        if input_data.genetic_findings and input_data.genetic_findings.has_genetic_data:
            gf = input_data.genetic_findings

            # PTCH1 mutations cause Gorlin syndrome (multiple BCCs)
            if "PTCH1" in gf.high_risk_genes or gf.gorlin_syndrome:
                rr *= 20.0  # Very high BCC risk

            # XP genes dramatically increase NMSC risk
            xp_genes = ["XPA", "XPB", "XPC", "XPD", "XPE", "XPF", "XPG"]
            if any(g in gf.high_risk_genes for g in xp_genes):
                rr *= 100.0

            # Apply NMSC genetic risk multiplier
            if gf.nmsc_genetic_risk_multiplier > 1.0:
                rr *= gf.nmsc_genetic_risk_multiplier

        # LAB FINDINGS
        if input_data.lab_findings and input_data.lab_findings.has_lab_data:
            lf = input_data.lab_findings
            # Immunosuppression from labs is especially important for BCC
            if lf.immunosuppressed_by_labs:
                rr *= 2.0
            rr *= lf.lab_risk_multiplier

        return rr

    def _calculate_scc_relative_risk(self, input_data: RiskFactorInput) -> float:
        """Calculate relative risk for squamous cell carcinoma."""
        rr = 1.0

        rr *= self.FITZPATRICK_RR.get(input_data.fitzpatrick_type, 1.0) * 0.7

        # Cumulative sun exposure is the major factor for SCC
        exposure_rr = {
            SunExposureLevel.MINIMAL: 0.3,
            SunExposureLevel.LOW: 0.6,
            SunExposureLevel.MODERATE: 1.0,
            SunExposureLevel.HIGH: 2.5,
            SunExposureLevel.VERY_HIGH: 5.0,
        }
        rr *= exposure_rr.get(input_data.sun_exposure_level, 1.0)

        if input_data.immunosuppressed:
            rr *= 65.0  # Extremely high SCC risk with immunosuppression

        if input_data.outdoor_occupation:
            rr *= 2.0

        # Age (SCC increases dramatically with age)
        if input_data.age >= 70:
            rr *= 3.0
        elif input_data.age >= 60:
            rr *= 2.0
        elif input_data.age >= 50:
            rr *= 1.5

        # =================================================================
        # GENETIC TEST FINDINGS - SCC-specific genes
        # =================================================================
        if input_data.genetic_findings and input_data.genetic_findings.has_genetic_data:
            gf = input_data.genetic_findings

            # XP genes dramatically increase SCC risk
            xp_genes = ["XPA", "XPB", "XPC", "XPD", "XPE", "XPF", "XPG"]
            if any(g in gf.high_risk_genes for g in xp_genes):
                rr *= 100.0

            # Apply NMSC genetic risk multiplier
            if gf.nmsc_genetic_risk_multiplier > 1.0:
                rr *= gf.nmsc_genetic_risk_multiplier

        # LAB FINDINGS - especially important for SCC
        if input_data.lab_findings and input_data.lab_findings.has_lab_data:
            lf = input_data.lab_findings
            # Immunosuppression dramatically increases SCC risk
            if lf.immunosuppressed_by_labs:
                rr *= 3.0  # Higher multiplier for SCC
            rr *= lf.lab_risk_multiplier

        return rr

    def _calculate_lifetime_melanoma_risk(self, input_data: RiskFactorInput, rr: float) -> float:
        """Calculate lifetime melanoma risk percentage."""
        base_risk = self.BASELINE_MELANOMA_RISK

        # Adjust for gender
        if input_data.gender == "male":
            base_risk = 2.8  # 1 in 36
        elif input_data.gender == "female":
            base_risk = 1.9  # 1 in 53

        # Adjust for age (remaining lifetime risk)
        age_factor = max(0.2, (85 - input_data.age) / 85)

        lifetime_risk = base_risk * rr * age_factor

        return min(lifetime_risk, 50)  # Cap at 50%

    def _calculate_lifetime_nmsc_risk(self, input_data: RiskFactorInput,
                                       bcc_rr: float, scc_rr: float) -> float:
        """Calculate lifetime non-melanoma skin cancer risk percentage."""
        base_risk = self.BASELINE_NMSC_RISK

        # Combined RR (weighted average - BCC is more common)
        combined_rr = (bcc_rr * 0.8 + scc_rr * 0.2)

        # Age adjustment
        age_factor = max(0.2, (85 - input_data.age) / 85)

        lifetime_risk = base_risk * combined_rr * age_factor

        return min(lifetime_risk, 80)  # Cap at 80%

    def _determine_risk_category(self, score: float) -> str:
        """Determine risk category based on overall score."""
        if score >= 75:
            return "very_high"
        elif score >= 55:
            return "high"
        elif score >= 35:
            return "moderate"
        elif score >= 20:
            return "low"
        else:
            return "very_low"

    def _calculate_percentile(self, score: float) -> int:
        """Calculate population percentile for risk score."""
        # Simplified percentile calculation
        # In reality, this would use population distribution data
        if score >= 80:
            return 99
        elif score >= 70:
            return 95
        elif score >= 60:
            return 90
        elif score >= 50:
            return 80
        elif score >= 40:
            return 65
        elif score >= 30:
            return 50
        elif score >= 20:
            return 35
        elif score >= 10:
            return 20
        else:
            return 10

    def _identify_risk_factors(self, input_data: RiskFactorInput) -> List[Dict[str, Any]]:
        """Identify individual risk factors."""
        risk_factors = []

        # Fitzpatrick type
        if input_data.fitzpatrick_type in [FitzpatrickType.TYPE_I, FitzpatrickType.TYPE_II]:
            risk_factors.append({
                "factor": "Fair skin",
                "category": "phenotype",
                "impact": "high",
                "description": f"Fitzpatrick Type {input_data.fitzpatrick_type.value} skin burns easily",
                "relative_risk": self.FITZPATRICK_RR.get(input_data.fitzpatrick_type, 1.0)
            })

        # Hair color
        if input_data.natural_hair_color in ["red", "blonde"]:
            risk_factors.append({
                "factor": f"{input_data.natural_hair_color.capitalize()} hair",
                "category": "phenotype",
                "impact": "high" if input_data.natural_hair_color == "red" else "moderate",
                "description": "Natural hair color associated with increased melanoma risk",
                "relative_risk": self.HAIR_COLOR_RR.get(input_data.natural_hair_color, 1.0)
            })

        # Mole count
        if input_data.total_mole_count in ["many_51_100", "very_many_100plus"]:
            risk_factors.append({
                "factor": "High mole count",
                "category": "phenotype",
                "impact": "very_high",
                "description": "Having many moles significantly increases melanoma risk",
                "relative_risk": self.MOLE_COUNT_RR.get(input_data.total_mole_count, 1.0)
            })
        elif input_data.total_mole_count == "moderate_26_50":
            risk_factors.append({
                "factor": "Moderate mole count",
                "category": "phenotype",
                "impact": "moderate",
                "description": "Multiple moles warrant regular monitoring",
                "relative_risk": self.MOLE_COUNT_RR.get(input_data.total_mole_count, 1.0)
            })

        # Family history
        melanoma_family = [fh for fh in input_data.family_history
                          if fh.cancer_type == CancerType.MELANOMA]
        if melanoma_family:
            first_degree = [fh for fh in melanoma_family if fh.relation == "first_degree"]
            risk_factors.append({
                "factor": "Family history of melanoma",
                "category": "genetic",
                "impact": "very_high" if first_degree else "high",
                "description": f"{len(melanoma_family)} family member(s) with melanoma",
                "relative_risk": 5.0 if len(first_degree) >= 2 else (2.24 if first_degree else 1.5)
            })

        # Personal history
        if any(ph.condition == "melanoma" for ph in input_data.personal_history):
            risk_factors.append({
                "factor": "Prior melanoma",
                "category": "medical_history",
                "impact": "very_high",
                "description": "Personal history of melanoma greatly increases risk of recurrence",
                "relative_risk": 9.0
            })

        # Sunburn history
        burns = input_data.sunburn_history
        if burns.childhood_severe_burns >= 3:
            risk_factors.append({
                "factor": "Childhood blistering sunburns",
                "category": "behavioral",
                "impact": "very_high",
                "description": f"{burns.childhood_severe_burns} severe sunburns before age 18",
                "relative_risk": 2.0
            })
        elif burns.childhood_severe_burns >= 1:
            risk_factors.append({
                "factor": "Childhood sunburns",
                "category": "behavioral",
                "impact": "high",
                "description": "Blistering sunburns in childhood increase melanoma risk",
                "relative_risk": 1.5
            })

        # Tanning bed use
        if input_data.tanning_bed_use in ["regular", "occasional"]:
            risk_factors.append({
                "factor": "Tanning bed use",
                "category": "behavioral",
                "impact": "high",
                "description": "Indoor tanning increases melanoma risk by 59% (WHO carcinogen)",
                "relative_risk": 1.59
            })

        # Immunosuppression
        if input_data.immunosuppressed:
            risk_factors.append({
                "factor": "Immunosuppression",
                "category": "medical_history",
                "impact": "very_high",
                "description": f"Immunosuppressed state ({input_data.immunosuppression_reason or 'unspecified'})",
                "relative_risk": 2.5
            })

        # Sun exposure
        if input_data.sun_exposure_level in [SunExposureLevel.HIGH, SunExposureLevel.VERY_HIGH]:
            risk_factors.append({
                "factor": "High sun exposure",
                "category": "behavioral",
                "impact": "high",
                "description": "Lifetime high UV exposure increases skin cancer risk",
                "relative_risk": 2.0 if input_data.sun_exposure_level == SunExposureLevel.HIGH else 3.0
            })

        # Outdoor occupation
        if input_data.outdoor_occupation:
            risk_factors.append({
                "factor": "Outdoor occupation",
                "category": "behavioral",
                "impact": "moderate",
                "description": "Occupational sun exposure increases cumulative UV dose",
                "relative_risk": 1.5
            })

        # AI findings
        if input_data.ai_findings and input_data.ai_findings.high_risk_lesions > 0:
            risk_factors.append({
                "factor": "AI-detected high-risk lesions",
                "category": "ai_findings",
                "impact": "high",
                "description": f"{input_data.ai_findings.high_risk_lesions} lesion(s) flagged as high risk",
                "relative_risk": 2.0
            })

        # =================================================================
        # GENETIC TEST FINDINGS
        # =================================================================
        if input_data.genetic_findings and input_data.genetic_findings.has_genetic_data:
            gf = input_data.genetic_findings

            # High-risk genes
            for gene in gf.high_risk_genes:
                gene_info = self._get_gene_risk_info(gene)
                risk_factors.append({
                    "factor": f"{gene} pathogenic variant",
                    "category": "genetic_testing",
                    "impact": "very_high",
                    "description": gene_info["description"],
                    "relative_risk": gene_info["relative_risk"],
                    "gene": gene,
                    "source": "genetic_test"
                })

            # Moderate-risk genes
            for gene in gf.moderate_risk_genes:
                gene_info = self._get_gene_risk_info(gene)
                variant_count = len(gf.variants.get(gene, []))
                risk_factors.append({
                    "factor": f"{gene} variant(s)",
                    "category": "genetic_testing",
                    "impact": "moderate",
                    "description": f"{variant_count} {gene} variant(s) detected - {gene_info['description']}",
                    "relative_risk": gene_info["relative_risk"],
                    "gene": gene,
                    "source": "genetic_test"
                })

            # Pharmacogenomic flags (important for treatment planning)
            for gene in gf.pharmacogenomic_flags:
                risk_factors.append({
                    "factor": f"{gene} pharmacogenomic variant",
                    "category": "pharmacogenomics",
                    "impact": "moderate",
                    "description": f"{gene} variant affects drug metabolism - consult before treatment",
                    "gene": gene,
                    "source": "genetic_test"
                })

            # Syndrome flags
            if gf.familial_melanoma_syndrome:
                risk_factors.append({
                    "factor": "Familial melanoma syndrome",
                    "category": "genetic_testing",
                    "impact": "very_high",
                    "description": "Genetic testing confirms familial melanoma predisposition",
                    "relative_risk": 10.0,
                    "source": "genetic_test"
                })

            if gf.gorlin_syndrome:
                risk_factors.append({
                    "factor": "Gorlin syndrome (NBCCS)",
                    "category": "genetic_testing",
                    "impact": "very_high",
                    "description": "PTCH1 mutation causes nevoid basal cell carcinoma syndrome",
                    "relative_risk": 20.0,
                    "source": "genetic_test"
                })

            if gf.xeroderma_pigmentosum:
                risk_factors.append({
                    "factor": "Xeroderma pigmentosum",
                    "category": "genetic_testing",
                    "impact": "very_high",
                    "description": "DNA repair deficiency causes extreme UV sensitivity and skin cancer risk",
                    "relative_risk": 100.0,
                    "source": "genetic_test"
                })

        # =================================================================
        # LAB RESULT FINDINGS
        # =================================================================
        if input_data.lab_findings and input_data.lab_findings.has_lab_data:
            lf = input_data.lab_findings

            # Vitamin D status
            if lf.vitamin_d_status == "deficient":
                risk_factors.append({
                    "factor": "Vitamin D deficiency",
                    "category": "lab_results",
                    "impact": "moderate",
                    "description": f"Vitamin D level {lf.vitamin_d_level} ng/mL (deficient <20) - associated with increased cancer risk",
                    "relative_risk": 1.3,
                    "source": "lab_test"
                })
            elif lf.vitamin_d_status == "insufficient":
                risk_factors.append({
                    "factor": "Vitamin D insufficiency",
                    "category": "lab_results",
                    "impact": "low",
                    "description": f"Vitamin D level {lf.vitamin_d_level} ng/mL (insufficient 20-29)",
                    "relative_risk": 1.1,
                    "source": "lab_test"
                })

            # Immunosuppression from labs
            if lf.immunosuppressed_by_labs:
                risk_factors.append({
                    "factor": "Lab-indicated immunosuppression",
                    "category": "lab_results",
                    "impact": "high",
                    "description": "Low WBC or lymphocyte count indicates possible immunosuppression",
                    "relative_risk": 2.0,
                    "source": "lab_test"
                })

            # Inflammation markers
            if lf.elevated_inflammation:
                risk_factors.append({
                    "factor": "Elevated inflammatory markers",
                    "category": "lab_results",
                    "impact": "low",
                    "description": "Elevated CRP/ESR may indicate chronic inflammation",
                    "relative_risk": 1.15,
                    "source": "lab_test"
                })

            # ANA positive
            if lf.ana_positive:
                risk_factors.append({
                    "factor": "Positive ANA",
                    "category": "lab_results",
                    "impact": "low",
                    "description": "Autoimmune markers may affect skin cancer risk and treatment options",
                    "relative_risk": 1.2,
                    "source": "lab_test"
                })

        return risk_factors

    def _get_gene_risk_info(self, gene: str) -> Dict[str, Any]:
        """Get risk information for a specific gene."""
        gene_info = {
            "CDKN2A": {
                "description": "Familial melanoma gene - 60-90% lifetime melanoma risk",
                "relative_risk": 10.0
            },
            "CDK4": {
                "description": "Familial melanoma gene - high lifetime melanoma risk",
                "relative_risk": 8.0
            },
            "BAP1": {
                "description": "BAP1 tumor predisposition - uveal and cutaneous melanoma risk",
                "relative_risk": 5.0
            },
            "MC1R": {
                "description": "Red hair/fair skin gene - 2-4x melanoma risk per variant",
                "relative_risk": 2.5
            },
            "MITF": {
                "description": "Melanocyte transcription factor - melanoma susceptibility",
                "relative_risk": 3.0
            },
            "PTCH1": {
                "description": "Gorlin syndrome gene - multiple basal cell carcinomas",
                "relative_risk": 20.0
            },
            "XPA": {
                "description": "Xeroderma pigmentosum gene - extreme skin cancer risk",
                "relative_risk": 100.0
            },
            "TPMT": {
                "description": "Thiopurine metabolism - affects azathioprine dosing",
                "relative_risk": 1.0
            },
            "DPYD": {
                "description": "5-FU metabolism - affects topical chemotherapy",
                "relative_risk": 1.0
            }
        }
        return gene_info.get(gene, {
            "description": f"{gene} variant detected",
            "relative_risk": 1.5
        })

    def _identify_protective_factors(self, input_data: RiskFactorInput) -> List[Dict[str, Any]]:
        """Identify protective factors."""
        protective = []

        # Dark skin
        if input_data.fitzpatrick_type in [FitzpatrickType.TYPE_V, FitzpatrickType.TYPE_VI]:
            protective.append({
                "factor": "Darker skin type",
                "category": "phenotype",
                "impact": "high",
                "description": "Melanin provides natural UV protection"
            })

        # Regular sunscreen use
        if input_data.sunscreen_use in ["always", "usually"]:
            protective.append({
                "factor": "Regular sunscreen use",
                "category": "behavioral",
                "impact": "moderate",
                "description": "Consistent sun protection reduces UV damage"
            })

        # Protective clothing
        if input_data.protective_clothing in ["always", "usually"]:
            protective.append({
                "factor": "Protective clothing",
                "category": "behavioral",
                "impact": "moderate",
                "description": "Physical sun protection is highly effective"
            })

        # Peak sun avoidance
        if input_data.peak_sun_avoidance:
            protective.append({
                "factor": "Avoids peak sun hours",
                "category": "behavioral",
                "impact": "moderate",
                "description": "Avoiding 10am-4pm sun reduces UV exposure by ~60%"
            })

        # Low mole count
        if input_data.total_mole_count in ["none", "few_1_10"]:
            protective.append({
                "factor": "Low mole count",
                "category": "phenotype",
                "impact": "moderate",
                "description": "Fewer moles correlates with lower melanoma risk"
            })

        # No family history
        if not input_data.family_history:
            protective.append({
                "factor": "No family history of skin cancer",
                "category": "genetic",
                "impact": "moderate",
                "description": "Absence of family history reduces baseline risk"
            })

        return protective

    def _generate_recommendations(self, input_data: RiskFactorInput,
                                  risk_category: str,
                                  risk_factors: List[Dict]) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []

        # Universal recommendations
        recommendations.append(
            "Perform monthly skin self-examinations, checking for new or changing moles"
        )

        # Risk-level specific
        if risk_category in ["very_high", "high"]:
            recommendations.append(
                "Schedule a full-body skin examination with a dermatologist within 1-2 months"
            )
            recommendations.append(
                "Consider dermoscopy and/or mole mapping for baseline documentation"
            )
            recommendations.append(
                "Discuss genetic counseling if you have family history of melanoma"
            )
        elif risk_category == "moderate":
            recommendations.append(
                "Schedule an annual full-body skin examination with a dermatologist"
            )
        else:
            recommendations.append(
                "Consider a baseline skin examination with a dermatologist"
            )

        # Behavioral recommendations
        if input_data.sunscreen_use in ["never", "rarely", "sometimes"]:
            recommendations.append(
                "Use broad-spectrum SPF 30+ sunscreen daily, reapplying every 2 hours outdoors"
            )

        if input_data.tanning_bed_use in ["occasional", "regular"]:
            recommendations.append(
                "Stop using tanning beds immediately - they are classified as carcinogenic"
            )

        if not input_data.peak_sun_avoidance:
            recommendations.append(
                "Avoid direct sun exposure during peak hours (10am-4pm) when possible"
            )

        if input_data.protective_clothing in ["never", "rarely"]:
            recommendations.append(
                "Wear protective clothing: wide-brimmed hats, long sleeves, UV-blocking sunglasses"
            )

        # Mole-specific
        if input_data.total_mole_count in ["many_51_100", "very_many_100plus"]:
            recommendations.append(
                "Document your moles with photos to track changes over time"
            )
            recommendations.append(
                "Learn the ABCDE criteria for identifying suspicious moles"
            )

        # AI findings
        if input_data.ai_findings and input_data.ai_findings.high_risk_lesions > 0:
            recommendations.append(
                f"Have the {input_data.ai_findings.high_risk_lesions} AI-flagged lesion(s) evaluated by a dermatologist promptly"
            )

        # Immunosuppression
        if input_data.immunosuppressed:
            recommendations.append(
                "Due to immunosuppression, skin checks should be more frequent (every 3-6 months)"
            )

        # =================================================================
        # GENETIC TEST FINDINGS - Specific recommendations
        # =================================================================
        if input_data.genetic_findings and input_data.genetic_findings.has_genetic_data:
            gf = input_data.genetic_findings

            # High-risk gene carriers need intensive surveillance
            if gf.high_risk_genes:
                recommendations.append(
                    "Based on your genetic test results, you should have full-body skin exams every 3-6 months"
                )

            # CDKN2A/CDK4 carriers
            if "CDKN2A" in gf.high_risk_genes or "CDK4" in gf.high_risk_genes:
                recommendations.append(
                    "Consider annual pancreatic cancer screening (CDKN2A carriers have increased pancreatic cancer risk)"
                )
                recommendations.append(
                    "Family members should be offered genetic testing for familial melanoma"
                )

            # BAP1 carriers need eye exams
            if "BAP1" in gf.high_risk_genes:
                recommendations.append(
                    "Annual dilated eye examination to screen for uveal melanoma"
                )
                recommendations.append(
                    "Consider abdominal imaging for renal cell carcinoma screening"
                )

            # Gorlin syndrome / PTCH1
            if "PTCH1" in gf.high_risk_genes or gf.gorlin_syndrome:
                recommendations.append(
                    "Strict sun avoidance - BCCs can appear at young age in Gorlin syndrome"
                )
                recommendations.append(
                    "Avoid radiation therapy if possible - increased sensitivity"
                )
                recommendations.append(
                    "Consider baseline brain MRI (medulloblastoma screening)"
                )

            # XP genes
            xp_genes = ["XPA", "XPB", "XPC", "XPD", "XPE", "XPF", "XPG"]
            if any(g in gf.high_risk_genes for g in xp_genes) or gf.xeroderma_pigmentosum:
                recommendations.append(
                    "CRITICAL: Complete UV avoidance required - use UV-protective window film, protective clothing"
                )
                recommendations.append(
                    "Skin exams every 1-3 months by a dermatologist experienced with XP"
                )
                recommendations.append(
                    "Neurological monitoring may be needed depending on XP complementation group"
                )

            # MC1R variants - moderate risk
            if "MC1R" in gf.moderate_risk_genes:
                recommendations.append(
                    "MC1R variants increase UV sensitivity - be extra vigilant with sun protection"
                )

            # Pharmacogenomic variants
            if "TPMT" in gf.pharmacogenomic_flags:
                recommendations.append(
                    "TPMT variant detected - inform your doctor before taking azathioprine or 6-mercaptopurine"
                )
            if "DPYD" in gf.pharmacogenomic_flags:
                recommendations.append(
                    "DPYD variant detected - inform your doctor before using 5-fluorouracil (topical or systemic)"
                )

        return recommendations

    def _determine_screening_frequency(self, risk_category: str,
                                       input_data: RiskFactorInput) -> str:
        """Determine recommended screening frequency."""
        if input_data.immunosuppressed:
            return "Every 3-6 months"

        if any(ph.condition == "melanoma" for ph in input_data.personal_history):
            return "Every 3-6 months for 5 years, then annually"

        frequencies = {
            "very_high": "Every 3-6 months",
            "high": "Every 6 months",
            "moderate": "Annually",
            "low": "Every 1-2 years",
            "very_low": "Every 2-3 years or as needed"
        }

        return frequencies.get(risk_category, "Annually")

    def _identify_urgent_concerns(self, input_data: RiskFactorInput) -> List[str]:
        """Identify any urgent concerns requiring immediate attention."""
        urgent = []

        # AI findings with high malignancy probability
        if input_data.ai_findings:
            if input_data.ai_findings.highest_malignancy_probability >= 0.7:
                urgent.append(
                    f"AI analysis detected lesion(s) with {input_data.ai_findings.highest_malignancy_probability*100:.0f}% "
                    "malignancy probability - seek dermatologist evaluation within 1-2 weeks"
                )

            if input_data.ai_findings.high_risk_lesions >= 3:
                urgent.append(
                    f"Multiple high-risk lesions detected ({input_data.ai_findings.high_risk_lesions}) - "
                    "schedule comprehensive skin examination promptly"
                )

        # Suspicious symptoms in conditions
        if input_data.ai_findings:
            suspicious_conditions = ["melanoma", "squamous cell", "basal cell"]
            for condition in input_data.ai_findings.conditions_detected:
                if any(sc in condition.lower() for sc in suspicious_conditions):
                    urgent.append(
                        f"'{condition}' detected by AI - requires professional evaluation"
                    )
                    break

        return urgent

    def _calculate_data_completeness(self, input_data: RiskFactorInput) -> float:
        """Calculate how complete the input data is."""
        total_fields = 15
        completed = 0

        if input_data.fitzpatrick_type:
            completed += 1
        if input_data.natural_hair_color:
            completed += 1
        if input_data.natural_eye_color:
            completed += 1
        if input_data.total_mole_count:
            completed += 1
        if input_data.sun_exposure_level:
            completed += 1
        if input_data.sunburn_history:
            completed += 1
        if input_data.family_history is not None:  # Even empty list counts
            completed += 1
        if input_data.personal_history is not None:
            completed += 1
        if input_data.sunscreen_use:
            completed += 1
        if input_data.protective_clothing:
            completed += 1
        if input_data.tanning_bed_use:
            completed += 1
        if input_data.age:
            completed += 1
        if input_data.gender:
            completed += 1
        if input_data.ai_findings:
            completed += 2  # Worth 2 points

        return (completed / total_fields) * 100

    def _determine_confidence_level(self, completeness: float) -> str:
        """Determine confidence level based on data completeness."""
        if completeness >= 80:
            return "high"
        elif completeness >= 50:
            return "moderate"
        else:
            return "low"


# Convenience function for API use
def calculate_skin_cancer_risk(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate skin cancer risk from dictionary input.
    Convenience function for API endpoints.
    """
    calculator = SkinCancerRiskCalculator()

    # Parse fitzpatrick type
    fitzpatrick_map = {
        1: FitzpatrickType.TYPE_I,
        2: FitzpatrickType.TYPE_II,
        3: FitzpatrickType.TYPE_III,
        4: FitzpatrickType.TYPE_IV,
        5: FitzpatrickType.TYPE_V,
        6: FitzpatrickType.TYPE_VI,
    }

    # Parse sun exposure level
    exposure_map = {
        "minimal": SunExposureLevel.MINIMAL,
        "low": SunExposureLevel.LOW,
        "moderate": SunExposureLevel.MODERATE,
        "high": SunExposureLevel.HIGH,
        "very_high": SunExposureLevel.VERY_HIGH,
    }

    # Parse sunburn history
    sunburn_data = data.get("sunburn_history", {})
    sunburn_history = SunburnHistory(
        childhood_severe_burns=sunburn_data.get("childhood_severe_burns", 0),
        childhood_moderate_burns=sunburn_data.get("childhood_moderate_burns", 0),
        adult_severe_burns=sunburn_data.get("adult_severe_burns", 0),
        adult_moderate_burns=sunburn_data.get("adult_moderate_burns", 0),
    )

    # Parse family history
    family_history = []
    for fh in data.get("family_history", []):
        cancer_type_map = {
            "melanoma": CancerType.MELANOMA,
            "basal_cell": CancerType.BASAL_CELL,
            "squamous_cell": CancerType.SQUAMOUS_CELL,
            "any": CancerType.ANY,
        }
        family_history.append(FamilyHistoryEntry(
            relation=fh.get("relation", "second_degree"),
            cancer_type=cancer_type_map.get(fh.get("cancer_type", "any"), CancerType.ANY),
            age_at_diagnosis=fh.get("age_at_diagnosis"),
            multiple_primaries=fh.get("multiple_primaries", False),
        ))

    # Parse personal history
    personal_history = []
    for ph in data.get("personal_history", []):
        personal_history.append(PersonalHistoryEntry(
            condition=ph.get("condition", ""),
            year_diagnosed=ph.get("year_diagnosed"),
            location=ph.get("location"),
            recurrence=ph.get("recurrence", False),
        ))

    # Parse AI findings
    ai_data = data.get("ai_findings")
    ai_findings = None
    if ai_data:
        ai_findings = AIAnalysisFindings(
            total_lesions_analyzed=ai_data.get("total_lesions_analyzed", 0),
            high_risk_lesions=ai_data.get("high_risk_lesions", 0),
            medium_risk_lesions=ai_data.get("medium_risk_lesions", 0),
            atypical_moles=ai_data.get("atypical_moles", 0),
            suspicious_features=ai_data.get("suspicious_features", []),
            abcde_flags=ai_data.get("abcde_flags", {}),
            highest_malignancy_probability=ai_data.get("highest_malignancy_probability", 0.0),
            conditions_detected=ai_data.get("conditions_detected", []),
        )

    # Parse genetic test findings (NGS/VCF data)
    genetic_data = data.get("genetic_findings")
    genetic_findings = None
    if genetic_data:
        genetic_findings = GeneticTestFindings(
            has_genetic_data=genetic_data.get("has_genetic_data", True),
            test_date=genetic_data.get("test_date"),
            lab_name=genetic_data.get("lab_name"),
            variants=genetic_data.get("variants", {}),
            melanoma_genetic_risk_multiplier=genetic_data.get("melanoma_genetic_risk_multiplier", 1.0),
            nmsc_genetic_risk_multiplier=genetic_data.get("nmsc_genetic_risk_multiplier", 1.0),
            high_risk_genes=genetic_data.get("high_risk_genes", []),
            moderate_risk_genes=genetic_data.get("moderate_risk_genes", []),
            pharmacogenomic_flags=genetic_data.get("pharmacogenomic_flags", []),
            familial_melanoma_syndrome=genetic_data.get("familial_melanoma_syndrome", False),
            gorlin_syndrome=genetic_data.get("gorlin_syndrome", False),
            xeroderma_pigmentosum=genetic_data.get("xeroderma_pigmentosum", False),
            pathogenic_variant_count=genetic_data.get("pathogenic_variant_count", 0),
            likely_pathogenic_count=genetic_data.get("likely_pathogenic_count", 0),
            vus_count=genetic_data.get("vus_count", 0),
        )

    # Parse lab result findings
    lab_data = data.get("lab_findings")
    lab_findings = None
    if lab_data:
        lab_findings = LabResultFindings(
            has_lab_data=lab_data.get("has_lab_data", True),
            test_date=lab_data.get("test_date"),
            lab_name=lab_data.get("lab_name"),
            vitamin_d_level=lab_data.get("vitamin_d_level"),
            vitamin_d_status=lab_data.get("vitamin_d_status", "unknown"),
            wbc_count=lab_data.get("wbc_count"),
            lymphocyte_count=lab_data.get("lymphocyte_count"),
            immunosuppressed_by_labs=lab_data.get("immunosuppressed_by_labs", False),
            crp_level=lab_data.get("crp_level"),
            esr_level=lab_data.get("esr_level"),
            elevated_inflammation=lab_data.get("elevated_inflammation", False),
            ana_positive=lab_data.get("ana_positive", False),
            liver_function_normal=lab_data.get("liver_function_normal", True),
            lab_risk_multiplier=lab_data.get("lab_risk_multiplier", 1.0),
            risk_factors_from_labs=lab_data.get("risk_factors_from_labs", []),
        )

    # Create input object
    input_data = RiskFactorInput(
        age=data.get("age", 40),
        gender=data.get("gender", "other"),
        fitzpatrick_type=fitzpatrick_map.get(data.get("fitzpatrick_type", 3), FitzpatrickType.TYPE_III),
        natural_hair_color=data.get("natural_hair_color", "dark_brown"),
        natural_eye_color=data.get("natural_eye_color", "brown"),
        freckles=data.get("freckles", "none"),
        total_mole_count=data.get("total_mole_count", "few_1_10"),
        sun_exposure_level=exposure_map.get(data.get("sun_exposure_level", "moderate"), SunExposureLevel.MODERATE),
        outdoor_occupation=data.get("outdoor_occupation", False),
        geographic_latitude=data.get("geographic_latitude"),
        sunscreen_use=data.get("sunscreen_use", "sometimes"),
        protective_clothing=data.get("protective_clothing", "sometimes"),
        peak_sun_avoidance=data.get("peak_sun_avoidance", False),
        sunburn_history=sunburn_history,
        tanning_bed_use=data.get("tanning_bed_use", "never"),
        tanning_bed_years=data.get("tanning_bed_years", 0),
        immunosuppressed=data.get("immunosuppressed", False),
        immunosuppression_reason=data.get("immunosuppression_reason"),
        radiation_therapy_history=data.get("radiation_therapy_history", False),
        chronic_skin_conditions=data.get("chronic_skin_conditions", []),
        family_history=family_history,
        personal_history=personal_history,
        ai_findings=ai_findings,
        genetic_findings=genetic_findings,
        lab_findings=lab_findings,
    )

    # Calculate risk
    result = calculator.calculate_risk(input_data)

    # Convert to dictionary for JSON response
    response = {
        "overall_risk_score": result.overall_risk_score,
        "risk_category": result.risk_category,
        "component_scores": {
            "genetic": result.genetic_score,
            "phenotype": result.phenotype_score,
            "sun_exposure": result.sun_exposure_score,
            "behavioral": result.behavioral_score,
            "medical_history": result.medical_history_score,
            "ai_findings": result.ai_findings_score,
        },
        "relative_risks": {
            "melanoma": result.melanoma_relative_risk,
            "basal_cell_carcinoma": result.bcc_relative_risk,
            "squamous_cell_carcinoma": result.scc_relative_risk,
        },
        "lifetime_risk_percent": {
            "melanoma": result.melanoma_lifetime_risk_percent,
            "non_melanoma_skin_cancer": result.nmsc_lifetime_risk_percent,
        },
        "population_percentile": result.population_percentile,
        "risk_factors": result.risk_factors,
        "protective_factors": result.protective_factors,
        "screening_frequency": result.screening_frequency,
        "recommendations": result.recommendations,
        "urgent_concerns": result.urgent_concerns,
        "confidence_level": result.confidence_level,
        "data_completeness": result.data_completeness,
    }

    # Add genetic testing summary if available
    if genetic_findings and genetic_findings.has_genetic_data:
        response["genetic_testing_summary"] = {
            "has_genetic_data": True,
            "test_date": str(genetic_findings.test_date) if genetic_findings.test_date else None,
            "lab_name": genetic_findings.lab_name,
            "high_risk_genes": genetic_findings.high_risk_genes,
            "moderate_risk_genes": genetic_findings.moderate_risk_genes,
            "pharmacogenomic_flags": genetic_findings.pharmacogenomic_flags,
            "pathogenic_variants": genetic_findings.pathogenic_variant_count,
            "likely_pathogenic_variants": genetic_findings.likely_pathogenic_count,
            "variants_of_uncertain_significance": genetic_findings.vus_count,
            "syndromes": {
                "familial_melanoma": genetic_findings.familial_melanoma_syndrome,
                "gorlin_syndrome": genetic_findings.gorlin_syndrome,
                "xeroderma_pigmentosum": genetic_findings.xeroderma_pigmentosum,
            },
            "genetic_risk_multipliers": {
                "melanoma": genetic_findings.melanoma_genetic_risk_multiplier,
                "nmsc": genetic_findings.nmsc_genetic_risk_multiplier,
            }
        }
    else:
        response["genetic_testing_summary"] = {
            "has_genetic_data": False,
            "message": "No genetic test data available. Consider genetic testing if you have family history of melanoma."
        }

    return response
