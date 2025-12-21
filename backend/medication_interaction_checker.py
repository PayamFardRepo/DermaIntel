"""
Medication Interaction Checker

Comprehensive drug safety checking for dermatological treatments:
- Drug-drug interactions
- Contraindications with patient history
- Photosensitivity warnings for sun exposure
- Age/pregnancy warnings
- Dosage verification
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, date
import re


class SeverityLevel(Enum):
    """Severity levels for interactions and warnings"""
    CONTRAINDICATED = "contraindicated"  # Should not be used together
    SEVERE = "severe"  # Serious interaction, avoid if possible
    MODERATE = "moderate"  # Monitor closely, may need dose adjustment
    MILD = "mild"  # Minor interaction, generally safe
    INFO = "info"  # Informational only


class InteractionType(Enum):
    """Types of drug interactions"""
    PHARMACOKINETIC = "pharmacokinetic"  # Affects absorption/metabolism
    PHARMACODYNAMIC = "pharmacodynamic"  # Affects drug action
    SYNERGISTIC = "synergistic"  # Increased effect
    ANTAGONISTIC = "antagonistic"  # Decreased effect
    ADDITIVE_TOXICITY = "additive_toxicity"  # Combined side effects


@dataclass
class DrugInteraction:
    """Represents an interaction between two drugs"""
    drug1: str
    drug2: str
    severity: SeverityLevel
    interaction_type: InteractionType
    description: str
    mechanism: str
    clinical_effects: List[str]
    management: str
    references: List[str] = field(default_factory=list)


@dataclass
class Contraindication:
    """Represents a contraindication for a drug"""
    drug: str
    condition: str
    severity: SeverityLevel
    description: str
    reason: str
    alternatives: List[str] = field(default_factory=list)


@dataclass
class PhotosensitivityWarning:
    """Warning for drug-induced photosensitivity"""
    drug: str
    photosensitivity_type: str  # "phototoxic" or "photoallergic"
    severity: SeverityLevel
    onset_timeframe: str
    duration_after_stopping: str
    uva_uvb_sensitivity: str
    clinical_presentation: str
    precautions: List[str]
    spf_recommendation: int


@dataclass
class AgeWarning:
    """Age-related warnings for medications"""
    drug: str
    min_age: Optional[int]
    max_age: Optional[int]
    severity: SeverityLevel
    pediatric_concerns: List[str]
    geriatric_concerns: List[str]
    dose_adjustment: str


@dataclass
class PregnancyWarning:
    """Pregnancy and lactation warnings"""
    drug: str
    pregnancy_category: str  # A, B, C, D, X or new PLLR format
    severity: SeverityLevel
    first_trimester_risk: str
    second_trimester_risk: str
    third_trimester_risk: str
    lactation_safe: bool
    lactation_notes: str
    alternatives_during_pregnancy: List[str]
    alternatives_during_lactation: List[str]


@dataclass
class DosageInfo:
    """Standard dosage information for verification"""
    drug: str
    formulation: str  # cream, ointment, tablet, etc.
    strength: str
    adult_dose: str
    adult_max_daily: str
    pediatric_dose: str
    pediatric_max_daily: str
    renal_adjustment: str
    hepatic_adjustment: str
    elderly_adjustment: str
    application_frequency: str
    duration_typical: str
    duration_max: str


@dataclass
class InteractionCheckResult:
    """Complete result of medication interaction check"""
    medication: str
    is_safe: bool
    overall_risk_level: SeverityLevel
    drug_interactions: List[DrugInteraction]
    contraindications: List[Contraindication]
    photosensitivity_warnings: List[PhotosensitivityWarning]
    age_warnings: List[AgeWarning]
    pregnancy_warnings: List[PregnancyWarning]
    dosage_issues: List[Dict[str, Any]]
    recommendations: List[str]
    requires_provider_review: bool
    checked_at: datetime = field(default_factory=datetime.now)


class DrugInteractionDatabase:
    """
    Comprehensive database of dermatological drug interactions
    """

    def __init__(self):
        self._load_interaction_data()
        self._load_contraindication_data()
        self._load_photosensitivity_data()
        self._load_age_pregnancy_data()
        self._load_dosage_data()

    def _load_interaction_data(self):
        """Load drug-drug interaction data"""
        self.interactions: Dict[Tuple[str, str], DrugInteraction] = {}

        # Common dermatological drug interactions
        interactions_list = [
            # Retinoids interactions
            DrugInteraction(
                drug1="isotretinoin",
                drug2="tetracycline",
                severity=SeverityLevel.CONTRAINDICATED,
                interaction_type=InteractionType.ADDITIVE_TOXICITY,
                description="Risk of pseudotumor cerebri (increased intracranial pressure)",
                mechanism="Both drugs can increase intracranial pressure independently",
                clinical_effects=["Severe headache", "Visual disturbances", "Papilledema", "Nausea/vomiting"],
                management="Do not use together. Choose alternative antibiotic.",
                references=["FDA prescribing information", "AAD Guidelines"]
            ),
            DrugInteraction(
                drug1="isotretinoin",
                drug2="doxycycline",
                severity=SeverityLevel.CONTRAINDICATED,
                interaction_type=InteractionType.ADDITIVE_TOXICITY,
                description="Risk of pseudotumor cerebri",
                mechanism="Both drugs can increase intracranial pressure independently",
                clinical_effects=["Severe headache", "Visual disturbances", "Papilledema"],
                management="Do not use together. Choose alternative antibiotic if needed.",
                references=["FDA prescribing information"]
            ),
            DrugInteraction(
                drug1="isotretinoin",
                drug2="minocycline",
                severity=SeverityLevel.CONTRAINDICATED,
                interaction_type=InteractionType.ADDITIVE_TOXICITY,
                description="Risk of pseudotumor cerebri",
                mechanism="Both drugs can increase intracranial pressure",
                clinical_effects=["Severe headache", "Visual changes", "Increased intracranial pressure"],
                management="Contraindicated combination. Wait 30 days after stopping tetracycline.",
                references=["Accutane PI"]
            ),
            DrugInteraction(
                drug1="isotretinoin",
                drug2="vitamin_a",
                severity=SeverityLevel.SEVERE,
                interaction_type=InteractionType.ADDITIVE_TOXICITY,
                description="Risk of vitamin A toxicity (hypervitaminosis A)",
                mechanism="Isotretinoin is a vitamin A derivative; additional vitamin A increases toxicity",
                clinical_effects=["Headache", "Liver damage", "Skin peeling", "Joint pain", "Bone changes"],
                management="Avoid vitamin A supplements during isotretinoin therapy",
                references=["FDA prescribing information"]
            ),
            DrugInteraction(
                drug1="isotretinoin",
                drug2="phenytoin",
                severity=SeverityLevel.MODERATE,
                interaction_type=InteractionType.PHARMACOKINETIC,
                description="Phenytoin may decrease isotretinoin levels",
                mechanism="Phenytoin induces hepatic metabolism",
                clinical_effects=["Reduced isotretinoin efficacy"],
                management="Monitor response; may need dose adjustment",
                references=["Drug interaction databases"]
            ),

            # Methotrexate interactions (for psoriasis)
            DrugInteraction(
                drug1="methotrexate",
                drug2="nsaids",
                severity=SeverityLevel.SEVERE,
                interaction_type=InteractionType.PHARMACOKINETIC,
                description="NSAIDs can increase methotrexate toxicity",
                mechanism="NSAIDs reduce renal clearance of methotrexate",
                clinical_effects=["Bone marrow suppression", "Mucositis", "Hepatotoxicity", "Nephrotoxicity"],
                management="Avoid high-dose NSAIDs. Low-dose aspirin may be acceptable with monitoring.",
                references=["ACR Guidelines"]
            ),
            DrugInteraction(
                drug1="methotrexate",
                drug2="trimethoprim",
                severity=SeverityLevel.CONTRAINDICATED,
                interaction_type=InteractionType.ADDITIVE_TOXICITY,
                description="Additive antifolate toxicity",
                mechanism="Both drugs inhibit folate metabolism",
                clinical_effects=["Severe pancytopenia", "Megaloblastic anemia", "Mucositis"],
                management="Use alternative antibiotic. If unavoidable, increase leucovorin rescue.",
                references=["FDA warnings"]
            ),
            DrugInteraction(
                drug1="methotrexate",
                drug2="alcohol",
                severity=SeverityLevel.SEVERE,
                interaction_type=InteractionType.ADDITIVE_TOXICITY,
                description="Increased risk of hepatotoxicity",
                mechanism="Both are hepatotoxic; combined effect on liver",
                clinical_effects=["Liver fibrosis", "Cirrhosis", "Elevated liver enzymes"],
                management="Avoid or strictly limit alcohol during methotrexate therapy",
                references=["AAD Guidelines"]
            ),

            # Corticosteroid interactions
            DrugInteraction(
                drug1="prednisone",
                drug2="nsaids",
                severity=SeverityLevel.MODERATE,
                interaction_type=InteractionType.ADDITIVE_TOXICITY,
                description="Increased risk of GI bleeding and ulceration",
                mechanism="Both drugs impair gastric mucosal protection",
                clinical_effects=["GI bleeding", "Peptic ulcers", "GI perforation"],
                management="Use gastroprotection (PPI) if combination necessary",
                references=["Clinical guidelines"]
            ),
            DrugInteraction(
                drug1="prednisone",
                drug2="warfarin",
                severity=SeverityLevel.MODERATE,
                interaction_type=InteractionType.PHARMACODYNAMIC,
                description="Variable effect on anticoagulation",
                mechanism="Steroids may affect vitamin K-dependent clotting factors",
                clinical_effects=["Increased or decreased INR"],
                management="Monitor INR closely; adjust warfarin as needed",
                references=["Drug interaction databases"]
            ),
            DrugInteraction(
                drug1="prednisone",
                drug2="diabetes_medications",
                severity=SeverityLevel.MODERATE,
                interaction_type=InteractionType.PHARMACODYNAMIC,
                description="Steroids increase blood glucose",
                mechanism="Glucocorticoids cause insulin resistance and gluconeogenesis",
                clinical_effects=["Hyperglycemia", "Loss of glycemic control"],
                management="Monitor blood glucose; may need to increase diabetes medication doses",
                references=["Endocrine guidelines"]
            ),

            # Antifungal interactions (common in dermatology)
            DrugInteraction(
                drug1="ketoconazole",
                drug2="simvastatin",
                severity=SeverityLevel.CONTRAINDICATED,
                interaction_type=InteractionType.PHARMACOKINETIC,
                description="Risk of rhabdomyolysis",
                mechanism="Ketoconazole inhibits CYP3A4, dramatically increasing statin levels",
                clinical_effects=["Rhabdomyolysis", "Myopathy", "Renal failure"],
                management="Contraindicated. Hold statin or use alternative antifungal.",
                references=["FDA Black Box Warning"]
            ),
            DrugInteraction(
                drug1="itraconazole",
                drug2="calcium_channel_blockers",
                severity=SeverityLevel.SEVERE,
                interaction_type=InteractionType.PHARMACOKINETIC,
                description="Increased CCB levels causing hypotension",
                mechanism="Itraconazole inhibits CYP3A4 metabolism of CCBs",
                clinical_effects=["Severe hypotension", "Bradycardia", "Peripheral edema"],
                management="Monitor BP closely; consider dose reduction of CCB",
                references=["Drug interaction databases"]
            ),
            DrugInteraction(
                drug1="fluconazole",
                drug2="warfarin",
                severity=SeverityLevel.SEVERE,
                interaction_type=InteractionType.PHARMACOKINETIC,
                description="Significantly increased warfarin effect",
                mechanism="Fluconazole inhibits CYP2C9 metabolism of S-warfarin",
                clinical_effects=["Elevated INR", "Bleeding risk"],
                management="Reduce warfarin dose by 25-50%; monitor INR closely",
                references=["FDA prescribing information"]
            ),
            DrugInteraction(
                drug1="terbinafine",
                drug2="caffeine",
                severity=SeverityLevel.MILD,
                interaction_type=InteractionType.PHARMACOKINETIC,
                description="Mild increase in caffeine levels",
                mechanism="Terbinafine inhibits CYP1A2",
                clinical_effects=["Jitteriness", "Insomnia", "Palpitations"],
                management="May need to reduce caffeine intake if symptomatic",
                references=["Drug interaction databases"]
            ),

            # Immunosuppressant interactions
            DrugInteraction(
                drug1="cyclosporine",
                drug2="grapefruit",
                severity=SeverityLevel.MODERATE,
                interaction_type=InteractionType.PHARMACOKINETIC,
                description="Increased cyclosporine levels",
                mechanism="Grapefruit inhibits intestinal CYP3A4",
                clinical_effects=["Nephrotoxicity", "Hypertension", "Neurotoxicity"],
                management="Avoid grapefruit and grapefruit juice consistently",
                references=["FDA prescribing information"]
            ),
            DrugInteraction(
                drug1="azathioprine",
                drug2="allopurinol",
                severity=SeverityLevel.CONTRAINDICATED,
                interaction_type=InteractionType.PHARMACOKINETIC,
                description="Severe bone marrow suppression",
                mechanism="Allopurinol inhibits xanthine oxidase, preventing azathioprine breakdown",
                clinical_effects=["Pancytopenia", "Severe infections", "Death"],
                management="Reduce azathioprine dose by 75% or use alternative",
                references=["FDA Black Box Warning"]
            ),

            # Topical medication interactions
            DrugInteraction(
                drug1="topical_retinoids",
                drug2="benzoyl_peroxide",
                severity=SeverityLevel.MILD,
                interaction_type=InteractionType.PHARMACOKINETIC,
                description="Benzoyl peroxide may degrade tretinoin",
                mechanism="Oxidation of tretinoin by benzoyl peroxide",
                clinical_effects=["Reduced retinoid efficacy"],
                management="Apply at different times (retinoid PM, BP AM) or use stable retinoids",
                references=["Dermatology guidelines"]
            ),
            DrugInteraction(
                drug1="topical_corticosteroids",
                drug2="topical_retinoids",
                severity=SeverityLevel.MILD,
                interaction_type=InteractionType.SYNERGISTIC,
                description="May reduce irritation but also efficacy",
                mechanism="Corticosteroids may decrease retinoid-induced cell turnover",
                clinical_effects=["Reduced irritation", "Potentially reduced efficacy"],
                management="Use short-term if needed for irritation management",
                references=["AAD Guidelines"]
            ),

            # Biologics interactions
            DrugInteraction(
                drug1="adalimumab",
                drug2="live_vaccines",
                severity=SeverityLevel.CONTRAINDICATED,
                interaction_type=InteractionType.PHARMACODYNAMIC,
                description="Risk of infection from live vaccines",
                mechanism="TNF inhibitors impair immune response to live pathogens",
                clinical_effects=["Disseminated infection from vaccine strain"],
                management="Complete live vaccines before starting; use inactivated vaccines",
                references=["FDA prescribing information", "CDC guidelines"]
            ),
            DrugInteraction(
                drug1="secukinumab",
                drug2="live_vaccines",
                severity=SeverityLevel.CONTRAINDICATED,
                interaction_type=InteractionType.PHARMACODYNAMIC,
                description="Risk of infection from live vaccines",
                mechanism="IL-17 inhibition impairs immune response",
                clinical_effects=["Potential infection from live vaccine strains"],
                management="Avoid live vaccines; inactivated vaccines acceptable",
                references=["FDA prescribing information"]
            ),
        ]

        # Build lookup dictionary (both directions)
        for interaction in interactions_list:
            key1 = (interaction.drug1.lower(), interaction.drug2.lower())
            key2 = (interaction.drug2.lower(), interaction.drug1.lower())
            self.interactions[key1] = interaction
            self.interactions[key2] = interaction

    def _load_contraindication_data(self):
        """Load contraindication data"""
        self.contraindications: Dict[str, List[Contraindication]] = {}

        contraindications_list = [
            # Isotretinoin contraindications
            Contraindication(
                drug="isotretinoin",
                condition="pregnancy",
                severity=SeverityLevel.CONTRAINDICATED,
                description="Severe teratogenic effects - Category X",
                reason="Causes severe birth defects including CNS, cardiac, and craniofacial malformations",
                alternatives=["Topical retinoids (with caution)", "Azelaic acid", "Benzoyl peroxide"]
            ),
            Contraindication(
                drug="isotretinoin",
                condition="breastfeeding",
                severity=SeverityLevel.CONTRAINDICATED,
                description="Excreted in breast milk",
                reason="Potential harm to nursing infant",
                alternatives=["Topical acne treatments"]
            ),
            Contraindication(
                drug="isotretinoin",
                condition="liver_disease",
                severity=SeverityLevel.SEVERE,
                description="Risk of hepatotoxicity",
                reason="Isotretinoin is metabolized by the liver and can elevate liver enzymes",
                alternatives=["Non-systemic acne treatments"]
            ),
            Contraindication(
                drug="isotretinoin",
                condition="hyperlipidemia",
                severity=SeverityLevel.MODERATE,
                description="Can worsen lipid profile",
                reason="Isotretinoin commonly elevates triglycerides and cholesterol",
                alternatives=["Topical retinoids", "Other systemic options"]
            ),
            Contraindication(
                drug="isotretinoin",
                condition="depression",
                severity=SeverityLevel.MODERATE,
                description="Possible association with depression and suicidal ideation",
                reason="Reports of psychiatric adverse events; causality debated",
                alternatives=["Topical treatments", "Other systemic options with monitoring"]
            ),

            # Methotrexate contraindications
            Contraindication(
                drug="methotrexate",
                condition="pregnancy",
                severity=SeverityLevel.CONTRAINDICATED,
                description="Teratogenic and abortifacient",
                reason="Causes neural tube defects, craniofacial abnormalities, limb defects",
                alternatives=["Cyclosporine", "Phototherapy", "Biologics (some)"]
            ),
            Contraindication(
                drug="methotrexate",
                condition="liver_disease",
                severity=SeverityLevel.CONTRAINDICATED,
                description="Hepatotoxic drug in hepatic impairment",
                reason="Risk of severe liver damage including fibrosis and cirrhosis",
                alternatives=["Cyclosporine", "Biologics", "Phototherapy"]
            ),
            Contraindication(
                drug="methotrexate",
                condition="kidney_disease",
                severity=SeverityLevel.SEVERE,
                description="Renally cleared - risk of accumulation",
                reason="Reduced clearance leads to toxicity",
                alternatives=["Cyclosporine (with monitoring)", "Biologics"]
            ),
            Contraindication(
                drug="methotrexate",
                condition="immunodeficiency",
                severity=SeverityLevel.SEVERE,
                description="Further immunosuppression dangerous",
                reason="Risk of severe infections",
                alternatives=["Phototherapy", "Topical treatments"]
            ),
            Contraindication(
                drug="methotrexate",
                condition="alcohol_use_disorder",
                severity=SeverityLevel.SEVERE,
                description="Synergistic hepatotoxicity",
                reason="Both damage liver; combination greatly increases fibrosis risk",
                alternatives=["Biologics", "Phototherapy"]
            ),

            # Systemic corticosteroid contraindications
            Contraindication(
                drug="prednisone",
                condition="active_infection",
                severity=SeverityLevel.SEVERE,
                description="Can worsen infections",
                reason="Immunosuppression allows infection spread",
                alternatives=["Treat infection first", "Use lowest effective dose"]
            ),
            Contraindication(
                drug="prednisone",
                condition="uncontrolled_diabetes",
                severity=SeverityLevel.MODERATE,
                description="Worsens glycemic control",
                reason="Increases insulin resistance and gluconeogenesis",
                alternatives=["Steroid-sparing agents", "Topical steroids"]
            ),
            Contraindication(
                drug="prednisone",
                condition="osteoporosis",
                severity=SeverityLevel.MODERATE,
                description="Accelerates bone loss",
                reason="Steroids inhibit bone formation and calcium absorption",
                alternatives=["Steroid-sparing agents", "Add bone protection if necessary"]
            ),
            Contraindication(
                drug="prednisone",
                condition="peptic_ulcer",
                severity=SeverityLevel.MODERATE,
                description="Can worsen or reactivate ulcers",
                reason="Impairs gastric mucosal defense",
                alternatives=["Use with PPI protection", "Steroid-sparing agents"]
            ),

            # Antifungal contraindications
            Contraindication(
                drug="ketoconazole",
                condition="liver_disease",
                severity=SeverityLevel.CONTRAINDICATED,
                description="Severe hepatotoxicity risk",
                reason="FDA black box warning for liver failure",
                alternatives=["Terbinafine", "Itraconazole", "Fluconazole"]
            ),
            Contraindication(
                drug="terbinafine",
                condition="liver_disease",
                severity=SeverityLevel.SEVERE,
                description="Risk of hepatotoxicity",
                reason="Can cause liver failure; monitoring required",
                alternatives=["Topical antifungals", "Itraconazole (with caution)"]
            ),

            # Biologic contraindications
            Contraindication(
                drug="adalimumab",
                condition="active_tuberculosis",
                severity=SeverityLevel.CONTRAINDICATED,
                description="Risk of TB reactivation and dissemination",
                reason="TNF is critical for TB granuloma maintenance",
                alternatives=["Non-biologic systemic agents", "Complete TB treatment first"]
            ),
            Contraindication(
                drug="adalimumab",
                condition="heart_failure",
                severity=SeverityLevel.SEVERE,
                description="May worsen heart failure",
                reason="TNF inhibitors associated with CHF exacerbation (NYHA III-IV)",
                alternatives=["IL-17 or IL-23 inhibitors", "Non-biologic agents"]
            ),
            Contraindication(
                drug="adalimumab",
                condition="demyelinating_disease",
                severity=SeverityLevel.CONTRAINDICATED,
                description="May cause or worsen MS or similar conditions",
                reason="Reports of new-onset or worsening demyelinating disease",
                alternatives=["IL-17 inhibitors", "Non-biologic agents"]
            ),

            # Dapsone contraindications
            Contraindication(
                drug="dapsone",
                condition="g6pd_deficiency",
                severity=SeverityLevel.CONTRAINDICATED,
                description="Risk of severe hemolytic anemia",
                reason="Dapsone causes oxidative stress to RBCs",
                alternatives=["Sulfasalazine", "Other sulfonamide alternatives"]
            ),
            Contraindication(
                drug="dapsone",
                condition="sulfonamide_allergy",
                severity=SeverityLevel.MODERATE,
                description="Possible cross-reactivity",
                reason="Dapsone is a sulfone; some cross-sensitivity exists",
                alternatives=["Non-sulfonamide alternatives"]
            ),
        ]

        for contra in contraindications_list:
            drug_key = contra.drug.lower()
            if drug_key not in self.contraindications:
                self.contraindications[drug_key] = []
            self.contraindications[drug_key].append(contra)

    def _load_photosensitivity_data(self):
        """Load photosensitivity warning data"""
        self.photosensitivity_drugs: Dict[str, PhotosensitivityWarning] = {}

        photosensitivity_list = [
            # Tetracyclines
            PhotosensitivityWarning(
                drug="doxycycline",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MODERATE,
                onset_timeframe="Within hours of sun exposure",
                duration_after_stopping="48-72 hours after last dose",
                uva_uvb_sensitivity="Both UVA and UVB",
                clinical_presentation="Exaggerated sunburn, blistering, erythema on sun-exposed areas",
                precautions=[
                    "Use broad-spectrum SPF 30+ sunscreen",
                    "Wear protective clothing and hat",
                    "Avoid peak sun hours (10am-4pm)",
                    "Avoid tanning beds",
                    "Window glass does not block UVA"
                ],
                spf_recommendation=50
            ),
            PhotosensitivityWarning(
                drug="minocycline",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MILD,
                onset_timeframe="Variable",
                duration_after_stopping="Days after stopping",
                uva_uvb_sensitivity="Primarily UVA",
                clinical_presentation="Blue-gray pigmentation in sun-exposed areas (prolonged use)",
                precautions=[
                    "Less photosensitizing than doxycycline",
                    "Still use sun protection",
                    "Monitor for hyperpigmentation"
                ],
                spf_recommendation=30
            ),
            PhotosensitivityWarning(
                drug="tetracycline",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MODERATE,
                onset_timeframe="Hours after exposure",
                duration_after_stopping="48-72 hours",
                uva_uvb_sensitivity="Both UVA and UVB",
                clinical_presentation="Severe sunburn reaction",
                precautions=[
                    "Avoid direct sunlight",
                    "Use high SPF sunscreen",
                    "Wear sun-protective clothing"
                ],
                spf_recommendation=50
            ),

            # Retinoids
            PhotosensitivityWarning(
                drug="isotretinoin",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MODERATE,
                onset_timeframe="Increased sensitivity throughout treatment",
                duration_after_stopping="Several weeks after stopping",
                uva_uvb_sensitivity="Both UVA and UVB",
                clinical_presentation="Enhanced sunburn, skin peeling, increased skin fragility",
                precautions=[
                    "Avoid prolonged sun exposure",
                    "Use SPF 30+ broad-spectrum sunscreen daily",
                    "Avoid waxing (skin fragility)",
                    "Avoid laser treatments during and after therapy"
                ],
                spf_recommendation=50
            ),
            PhotosensitivityWarning(
                drug="tretinoin",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MODERATE,
                onset_timeframe="During active use",
                duration_after_stopping="Days after stopping",
                uva_uvb_sensitivity="Both UVA and UVB",
                clinical_presentation="Enhanced sunburn, irritation, hyperpigmentation risk",
                precautions=[
                    "Apply at night only",
                    "Use daily sunscreen",
                    "Start slowly to build tolerance"
                ],
                spf_recommendation=30
            ),
            PhotosensitivityWarning(
                drug="adapalene",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MILD,
                onset_timeframe="During active use",
                duration_after_stopping="Days after stopping",
                uva_uvb_sensitivity="Both",
                clinical_presentation="Mild increase in sun sensitivity",
                precautions=[
                    "Less photosensitizing than tretinoin",
                    "Still use daily sunscreen",
                    "Apply at night"
                ],
                spf_recommendation=30
            ),

            # NSAIDs (topical)
            PhotosensitivityWarning(
                drug="ketoprofen",
                photosensitivity_type="photoallergic",
                severity=SeverityLevel.SEVERE,
                onset_timeframe="24-48 hours after exposure",
                duration_after_stopping="Weeks to months (persistent)",
                uva_uvb_sensitivity="Primarily UVA",
                clinical_presentation="Eczematous reaction, may spread beyond application site",
                precautions=[
                    "HIGH RISK - avoid sun completely on treated areas",
                    "Cover treated area from sun",
                    "Reaction can persist long after stopping"
                ],
                spf_recommendation=50
            ),
            PhotosensitivityWarning(
                drug="piroxicam",
                photosensitivity_type="photoallergic",
                severity=SeverityLevel.MODERATE,
                onset_timeframe="24-72 hours",
                duration_after_stopping="Variable",
                uva_uvb_sensitivity="UVA",
                clinical_presentation="Photoallergic dermatitis",
                precautions=[
                    "Avoid sun on application area",
                    "Cross-reactivity with other NSAIDs possible"
                ],
                spf_recommendation=50
            ),

            # Antifungals
            PhotosensitivityWarning(
                drug="voriconazole",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.SEVERE,
                onset_timeframe="Weeks to months of use",
                duration_after_stopping="May persist",
                uva_uvb_sensitivity="Primarily UVA",
                clinical_presentation="Severe photosensitivity, risk of skin cancer with prolonged use",
                precautions=[
                    "STRICT sun avoidance required",
                    "Regular dermatology surveillance for skin cancer",
                    "Use SPF 50+ and protective clothing"
                ],
                spf_recommendation=50
            ),
            PhotosensitivityWarning(
                drug="griseofulvin",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MODERATE,
                onset_timeframe="Variable",
                duration_after_stopping="48-72 hours",
                uva_uvb_sensitivity="Both UVA and UVB",
                clinical_presentation="Enhanced sunburn reaction",
                precautions=[
                    "Use sun protection",
                    "Limit sun exposure during treatment"
                ],
                spf_recommendation=30
            ),

            # Fluoroquinolones
            PhotosensitivityWarning(
                drug="ciprofloxacin",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MODERATE,
                onset_timeframe="Hours after sun exposure",
                duration_after_stopping="48-72 hours",
                uva_uvb_sensitivity="Primarily UVA",
                clinical_presentation="Exaggerated sunburn, blisters in severe cases",
                precautions=[
                    "Avoid prolonged sun exposure",
                    "Use broad-spectrum sunscreen"
                ],
                spf_recommendation=30
            ),
            PhotosensitivityWarning(
                drug="levofloxacin",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MILD,
                onset_timeframe="Hours after exposure",
                duration_after_stopping="Days",
                uva_uvb_sensitivity="UVA",
                clinical_presentation="Mild photosensitivity",
                precautions=["Use sunscreen as precaution"],
                spf_recommendation=30
            ),

            # Thiazide diuretics
            PhotosensitivityWarning(
                drug="hydrochlorothiazide",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MODERATE,
                onset_timeframe="Variable",
                duration_after_stopping="Days",
                uva_uvb_sensitivity="UVA",
                clinical_presentation="Photosensitivity reaction, increased skin cancer risk with prolonged use",
                precautions=[
                    "Use daily sun protection",
                    "Monitor for skin changes",
                    "Associated with increased SCC and melanoma risk"
                ],
                spf_recommendation=30
            ),

            # Sulfonamides
            PhotosensitivityWarning(
                drug="sulfamethoxazole",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.MODERATE,
                onset_timeframe="Hours to days",
                duration_after_stopping="48-72 hours",
                uva_uvb_sensitivity="Both",
                clinical_presentation="Sunburn-like reaction",
                precautions=["Avoid excessive sun exposure", "Use sunscreen"],
                spf_recommendation=30
            ),

            # Psoralens (used in phototherapy)
            PhotosensitivityWarning(
                drug="methoxsalen",
                photosensitivity_type="phototoxic",
                severity=SeverityLevel.SEVERE,
                onset_timeframe="2 hours after ingestion (intended effect)",
                duration_after_stopping="8-24 hours (eyes protected for 24h)",
                uva_uvb_sensitivity="UVA (therapeutic use)",
                clinical_presentation="Severe burns if uncontrolled exposure - THERAPEUTIC PHOTOSENSITIZER",
                precautions=[
                    "Used for PUVA therapy under medical supervision",
                    "Wear UVA-blocking eyewear for 24 hours",
                    "Avoid sun for 8 hours minimum after dosing"
                ],
                spf_recommendation=50
            ),

            # Antipsychotics (sometimes used off-label)
            PhotosensitivityWarning(
                drug="chlorpromazine",
                photosensitivity_type="both",
                severity=SeverityLevel.SEVERE,
                onset_timeframe="Variable",
                duration_after_stopping="Variable",
                uva_uvb_sensitivity="Both",
                clinical_presentation="Severe burns, blue-gray discoloration with prolonged use",
                precautions=["Strict sun avoidance", "Protective clothing required"],
                spf_recommendation=50
            ),
        ]

        for warning in photosensitivity_list:
            self.photosensitivity_drugs[warning.drug.lower()] = warning

    def _load_age_pregnancy_data(self):
        """Load age and pregnancy warning data"""
        self.age_warnings: Dict[str, AgeWarning] = {}
        self.pregnancy_warnings: Dict[str, PregnancyWarning] = {}

        # Age warnings
        age_warnings_list = [
            AgeWarning(
                drug="isotretinoin",
                min_age=12,
                max_age=None,
                severity=SeverityLevel.MODERATE,
                pediatric_concerns=["Premature epiphyseal closure", "Bone changes", "Psychiatric effects"],
                geriatric_concerns=["May not be necessary for acne in elderly"],
                dose_adjustment="Standard dosing applies; monitor growth in adolescents"
            ),
            AgeWarning(
                drug="tetracycline",
                min_age=8,
                max_age=None,
                severity=SeverityLevel.SEVERE,
                pediatric_concerns=["Permanent tooth discoloration", "Enamel hypoplasia", "Bone growth effects"],
                geriatric_concerns=["Increased photosensitivity"],
                dose_adjustment="Contraindicated under age 8"
            ),
            AgeWarning(
                drug="doxycycline",
                min_age=8,
                max_age=None,
                severity=SeverityLevel.SEVERE,
                pediatric_concerns=["Tooth staining if used before age 8", "Some flexibility for short courses"],
                geriatric_concerns=["Esophageal ulceration risk - take with water, stay upright"],
                dose_adjustment="Contraindicated under 8 for prolonged use; short courses may be acceptable"
            ),
            AgeWarning(
                drug="fluoroquinolones",
                min_age=18,
                max_age=None,
                severity=SeverityLevel.SEVERE,
                pediatric_concerns=["Tendon damage", "Cartilage toxicity in growing joints"],
                geriatric_concerns=["Increased tendon rupture risk", "QT prolongation", "CNS effects"],
                dose_adjustment="Generally avoid in children; reduce dose in elderly"
            ),
            AgeWarning(
                drug="topical_corticosteroids",
                min_age=None,
                max_age=None,
                severity=SeverityLevel.MODERATE,
                pediatric_concerns=["Higher systemic absorption", "HPA axis suppression", "Growth suppression"],
                geriatric_concerns=["Skin atrophy more likely", "Use lower potency"],
                dose_adjustment="Use lowest effective potency; limit duration in children"
            ),
            AgeWarning(
                drug="methotrexate",
                min_age=3,
                max_age=None,
                severity=SeverityLevel.MODERATE,
                pediatric_concerns=["Growth effects", "Hepatotoxicity monitoring"],
                geriatric_concerns=["Increased toxicity due to decreased renal function"],
                dose_adjustment="Adjust for renal function in elderly; weight-based dosing in children"
            ),
        ]

        for warning in age_warnings_list:
            self.age_warnings[warning.drug.lower()] = warning

        # Pregnancy warnings
        pregnancy_warnings_list = [
            PregnancyWarning(
                drug="isotretinoin",
                pregnancy_category="X",
                severity=SeverityLevel.CONTRAINDICATED,
                first_trimester_risk="Extremely high - severe teratogenicity",
                second_trimester_risk="Extremely high",
                third_trimester_risk="High",
                lactation_safe=False,
                lactation_notes="Contraindicated during breastfeeding",
                alternatives_during_pregnancy=["Azelaic acid", "Topical erythromycin", "Benzoyl peroxide"],
                alternatives_during_lactation=["Azelaic acid", "Topical erythromycin", "Benzoyl peroxide"]
            ),
            PregnancyWarning(
                drug="tretinoin",
                pregnancy_category="C (topical) / X (oral)",
                severity=SeverityLevel.SEVERE,
                first_trimester_risk="Avoid - theoretical teratogenic risk",
                second_trimester_risk="Avoid",
                third_trimester_risk="Avoid",
                lactation_safe=False,
                lactation_notes="Unknown excretion; avoid application to breast area",
                alternatives_during_pregnancy=["Azelaic acid", "Glycolic acid"],
                alternatives_during_lactation=["Azelaic acid", "Glycolic acid"]
            ),
            PregnancyWarning(
                drug="methotrexate",
                pregnancy_category="X",
                severity=SeverityLevel.CONTRAINDICATED,
                first_trimester_risk="Extremely high - teratogenic and abortifacient",
                second_trimester_risk="High risk of fetal abnormalities",
                third_trimester_risk="High risk",
                lactation_safe=False,
                lactation_notes="Contraindicated - excreted in breast milk",
                alternatives_during_pregnancy=["Cyclosporine", "Phototherapy", "Topical treatments"],
                alternatives_during_lactation=["Phototherapy", "Topical treatments"]
            ),
            PregnancyWarning(
                drug="doxycycline",
                pregnancy_category="D",
                severity=SeverityLevel.SEVERE,
                first_trimester_risk="May affect bone development",
                second_trimester_risk="Tooth discoloration in fetus",
                third_trimester_risk="Tooth and bone effects",
                lactation_safe=False,
                lactation_notes="Excreted in milk; may affect infant teeth/bones",
                alternatives_during_pregnancy=["Azithromycin", "Erythromycin", "Amoxicillin"],
                alternatives_during_lactation=["Azithromycin", "Erythromycin"]
            ),
            PregnancyWarning(
                drug="fluconazole",
                pregnancy_category="D (high dose) / C (single dose)",
                severity=SeverityLevel.MODERATE,
                first_trimester_risk="High-dose associated with birth defects; single low-dose likely safe",
                second_trimester_risk="Moderate risk with prolonged use",
                third_trimester_risk="Use with caution",
                lactation_safe=True,
                lactation_notes="Compatible with breastfeeding in usual doses",
                alternatives_during_pregnancy=["Topical antifungals", "Clotrimazole"],
                alternatives_during_lactation=["Can use fluconazole", "Topical preferred"]
            ),
            PregnancyWarning(
                drug="prednisone",
                pregnancy_category="C",
                severity=SeverityLevel.MODERATE,
                first_trimester_risk="Small increased risk of cleft palate",
                second_trimester_risk="Generally acceptable for serious conditions",
                third_trimester_risk="Risk of adrenal suppression in newborn",
                lactation_safe=True,
                lactation_notes="Low doses compatible; wait 4 hours after high dose before nursing",
                alternatives_during_pregnancy=["Use if benefits outweigh risks"],
                alternatives_during_lactation=["Can use with timing precautions"]
            ),
            PregnancyWarning(
                drug="hydrocortisone_topical",
                pregnancy_category="C",
                severity=SeverityLevel.MILD,
                first_trimester_risk="Limited systemic absorption; likely safe",
                second_trimester_risk="Safe for short-term use",
                third_trimester_risk="Safe for short-term use",
                lactation_safe=True,
                lactation_notes="Safe; avoid application to nipple area",
                alternatives_during_pregnancy=["Can use low-potency topical steroids"],
                alternatives_during_lactation=["Can use; avoid nipple area"]
            ),
            PregnancyWarning(
                drug="cyclosporine",
                pregnancy_category="C",
                severity=SeverityLevel.MODERATE,
                first_trimester_risk="Limited data; use if clearly needed",
                second_trimester_risk="Monitor for preterm birth and low birth weight",
                third_trimester_risk="Monitor closely",
                lactation_safe=False,
                lactation_notes="Excreted in breast milk; avoid breastfeeding",
                alternatives_during_pregnancy=["Phototherapy if possible"],
                alternatives_during_lactation=["Phototherapy", "Topical treatments"]
            ),
            PregnancyWarning(
                drug="adalimumab",
                pregnancy_category="B",
                severity=SeverityLevel.MODERATE,
                first_trimester_risk="Limited data; appears relatively safe",
                second_trimester_risk="Use if needed; monitor",
                third_trimester_risk="Avoid in third trimester - crosses placenta",
                lactation_safe=True,
                lactation_notes="Minimal transfer to breast milk; generally compatible",
                alternatives_during_pregnancy=["Consider holding in third trimester", "Certolizumab (less placental transfer)"],
                alternatives_during_lactation=["Can continue if needed"]
            ),
        ]

        for warning in pregnancy_warnings_list:
            self.pregnancy_warnings[warning.drug.lower()] = warning

    def _load_dosage_data(self):
        """Load standard dosage information"""
        self.dosage_info: Dict[str, List[DosageInfo]] = {}

        dosage_list = [
            # Isotretinoin
            DosageInfo(
                drug="isotretinoin",
                formulation="oral capsule",
                strength="10mg, 20mg, 30mg, 40mg",
                adult_dose="0.5-1 mg/kg/day divided twice daily",
                adult_max_daily="2 mg/kg/day (rarely used)",
                pediatric_dose="Same as adult (weight-based)",
                pediatric_max_daily="1 mg/kg/day typically",
                renal_adjustment="Use with caution; reduce dose if CrCl <30",
                hepatic_adjustment="Contraindicated in severe hepatic disease",
                elderly_adjustment="No specific adjustment; monitor closely",
                application_frequency="Twice daily with fatty meal",
                duration_typical="15-20 weeks (cumulative dose 120-150 mg/kg)",
                duration_max="Continue until cumulative dose achieved or 8 months"
            ),

            # Doxycycline
            DosageInfo(
                drug="doxycycline",
                formulation="oral tablet/capsule",
                strength="50mg, 100mg",
                adult_dose="100mg twice daily or 200mg once daily",
                adult_max_daily="200mg (300mg rarely for severe infections)",
                pediatric_dose="2.2 mg/kg/dose twice daily (>8 years)",
                pediatric_max_daily="200mg",
                renal_adjustment="No adjustment needed",
                hepatic_adjustment="Use with caution in severe liver disease",
                elderly_adjustment="No specific adjustment",
                application_frequency="Once or twice daily",
                duration_typical="6-12 weeks for acne; 7-14 days for infections",
                duration_max="Generally safe for long-term use with monitoring"
            ),

            # Tretinoin topical
            DosageInfo(
                drug="tretinoin",
                formulation="topical cream/gel",
                strength="0.025%, 0.05%, 0.1%",
                adult_dose="Pea-sized amount to affected area once daily",
                adult_max_daily="Once daily application",
                pediatric_dose="Same as adult (≥12 years)",
                pediatric_max_daily="Once daily",
                renal_adjustment="Not applicable (topical)",
                hepatic_adjustment="Not applicable (topical)",
                elderly_adjustment="Start with lower concentration",
                application_frequency="Once daily at bedtime",
                duration_typical="8-12 weeks for initial improvement",
                duration_max="Can be used long-term for maintenance"
            ),

            # Hydrocortisone topical
            DosageInfo(
                drug="hydrocortisone",
                formulation="topical cream/ointment",
                strength="0.5%, 1%, 2.5%",
                adult_dose="Apply thin layer 1-4 times daily",
                adult_max_daily="4 times daily; limit to 45g/week",
                pediatric_dose="Apply thin layer 1-2 times daily",
                pediatric_max_daily="Limit duration and area; avoid face",
                renal_adjustment="Not applicable (topical)",
                hepatic_adjustment="Not applicable (topical)",
                elderly_adjustment="Use lower potency; skin atrophy risk",
                application_frequency="2-4 times daily as needed",
                duration_typical="1-2 weeks for acute flares",
                duration_max="2 weeks continuous; then taper or intermittent"
            ),

            # Methotrexate
            DosageInfo(
                drug="methotrexate",
                formulation="oral tablet/injection",
                strength="2.5mg, 5mg, 7.5mg, 10mg, 15mg tablets",
                adult_dose="7.5-25mg once weekly",
                adult_max_daily="25-30mg weekly (not daily!)",
                pediatric_dose="10-15 mg/m² once weekly",
                pediatric_max_daily="25mg weekly",
                renal_adjustment="CrCl 30-50: reduce 50%; CrCl <30: avoid",
                hepatic_adjustment="Avoid in significant liver disease",
                elderly_adjustment="Start low, increase slowly; monitor renal function",
                application_frequency="ONCE WEEKLY (NOT daily - common error!)",
                duration_typical="Months to years for chronic conditions",
                duration_max="Long-term use with monitoring"
            ),

            # Fluconazole
            DosageInfo(
                drug="fluconazole",
                formulation="oral tablet/capsule",
                strength="50mg, 100mg, 150mg, 200mg",
                adult_dose="150mg single dose (vaginal); 100-400mg daily (other)",
                adult_max_daily="400mg (800mg in severe infections)",
                pediatric_dose="3-12 mg/kg/day",
                pediatric_max_daily="400mg",
                renal_adjustment="CrCl <50: reduce dose by 50%",
                hepatic_adjustment="Use with caution; monitor LFTs",
                elderly_adjustment="Adjust for renal function",
                application_frequency="Once daily (single dose for some indications)",
                duration_typical="1-2 weeks for skin/nail infections",
                duration_max="Varies by indication"
            ),

            # Terbinafine
            DosageInfo(
                drug="terbinafine",
                formulation="oral tablet",
                strength="250mg",
                adult_dose="250mg once daily",
                adult_max_daily="250mg",
                pediatric_dose="<20kg: 62.5mg; 20-40kg: 125mg; >40kg: 250mg daily",
                pediatric_max_daily="250mg",
                renal_adjustment="CrCl <50: reduce by 50%",
                hepatic_adjustment="Not recommended in liver disease",
                elderly_adjustment="Adjust for renal function",
                application_frequency="Once daily",
                duration_typical="Fingernail: 6 weeks; Toenail: 12 weeks",
                duration_max="16 weeks for resistant cases"
            ),

            # Prednisone
            DosageInfo(
                drug="prednisone",
                formulation="oral tablet",
                strength="1mg, 2.5mg, 5mg, 10mg, 20mg, 50mg",
                adult_dose="5-60mg daily depending on condition",
                adult_max_daily="Varies; 60mg common starting dose for severe disease",
                pediatric_dose="0.5-2 mg/kg/day",
                pediatric_max_daily="60mg or 2mg/kg (whichever less)",
                renal_adjustment="No specific adjustment",
                hepatic_adjustment="May be preferred over prednisolone in liver disease",
                elderly_adjustment="Use lowest effective dose; monitor for side effects",
                application_frequency="Once daily (morning) or divided doses",
                duration_typical="Days to weeks for acute conditions",
                duration_max="Taper to lowest effective dose; chronic use needs monitoring"
            ),
        ]

        for dosage in dosage_list:
            drug_key = dosage.drug.lower()
            if drug_key not in self.dosage_info:
                self.dosage_info[drug_key] = []
            self.dosage_info[drug_key].append(dosage)

    def find_interactions(self, drug: str, other_medications: List[str]) -> List[DrugInteraction]:
        """Find drug-drug interactions"""
        interactions = []
        drug_lower = drug.lower()

        for other_med in other_medications:
            other_lower = other_med.lower()
            key = (drug_lower, other_lower)

            if key in self.interactions:
                interactions.append(self.interactions[key])

            # Also check drug class interactions
            interactions.extend(self._check_class_interactions(drug_lower, other_lower))

        return interactions

    def _check_class_interactions(self, drug1: str, drug2: str) -> List[DrugInteraction]:
        """Check for drug class-based interactions"""
        interactions = []

        # NSAID class
        nsaids = ['ibuprofen', 'naproxen', 'aspirin', 'celecoxib', 'meloxicam', 'diclofenac', 'indomethacin', 'ketoprofen']
        # Statin class
        statins = ['simvastatin', 'atorvastatin', 'lovastatin', 'rosuvastatin', 'pravastatin']
        # ACE inhibitors
        ace_inhibitors = ['lisinopril', 'enalapril', 'ramipril', 'captopril', 'benazepril']
        # Diabetes medications
        diabetes_meds = ['metformin', 'glipizide', 'glyburide', 'insulin', 'sitagliptin', 'empagliflozin']

        # Check methotrexate + NSAIDs
        if drug1 == 'methotrexate' and drug2 in nsaids:
            key = ('methotrexate', 'nsaids')
            if key in self.interactions:
                interactions.append(self.interactions[key])

        # Check azole antifungals + statins
        azoles = ['ketoconazole', 'itraconazole', 'fluconazole', 'voriconazole', 'posaconazole']
        if drug1 in azoles and drug2 in statins:
            key = ('ketoconazole', 'simvastatin')  # Use as template
            if key in self.interactions:
                interaction = self.interactions[key]
                # Create specific interaction for this pair
                specific = DrugInteraction(
                    drug1=drug1,
                    drug2=drug2,
                    severity=interaction.severity,
                    interaction_type=interaction.interaction_type,
                    description=f"CYP3A4 inhibition increases {drug2} levels - rhabdomyolysis risk",
                    mechanism=interaction.mechanism,
                    clinical_effects=interaction.clinical_effects,
                    management=f"Avoid combination or hold {drug2} during {drug1} therapy",
                    references=interaction.references
                )
                interactions.append(specific)

        # Check prednisone + diabetes meds
        if drug1 in ['prednisone', 'prednisolone', 'methylprednisolone', 'dexamethasone'] and drug2 in diabetes_meds:
            key = ('prednisone', 'diabetes_medications')
            if key in self.interactions:
                interactions.append(self.interactions[key])

        return interactions

    def get_contraindications(self, drug: str, patient_conditions: List[str]) -> List[Contraindication]:
        """Find contraindications based on patient conditions"""
        result = []
        drug_lower = drug.lower()

        if drug_lower in self.contraindications:
            for contra in self.contraindications[drug_lower]:
                for condition in patient_conditions:
                    condition_lower = condition.lower()
                    if self._condition_matches(contra.condition, condition_lower):
                        result.append(contra)

        return result

    def _condition_matches(self, contra_condition: str, patient_condition: str) -> bool:
        """Check if a patient condition matches a contraindication"""
        contra_lower = contra_condition.lower()
        patient_lower = patient_condition.lower()

        # Direct match
        if contra_lower in patient_lower or patient_lower in contra_lower:
            return True

        # Synonym matching
        condition_synonyms = {
            'liver_disease': ['hepatitis', 'cirrhosis', 'liver failure', 'hepatic', 'fatty liver', 'elevated liver enzymes', 'alt', 'ast'],
            'kidney_disease': ['renal', 'ckd', 'kidney failure', 'dialysis', 'nephropathy', 'creatinine', 'gfr'],
            'pregnancy': ['pregnant', 'gestation', 'expecting', 'trimester'],
            'breastfeeding': ['lactation', 'nursing', 'breast milk', 'lactating'],
            'depression': ['depressed', 'suicidal', 'mental health', 'psychiatric', 'mood disorder'],
            'heart_failure': ['chf', 'cardiac failure', 'cardiomyopathy', 'heart disease', 'reduced ejection fraction'],
            'immunodeficiency': ['hiv', 'aids', 'immunocompromised', 'transplant', 'immunosuppressed'],
            'diabetes': ['diabetic', 'dm', 'type 2', 'type 1', 'hyperglycemia', 'a1c'],
        }

        for key, synonyms in condition_synonyms.items():
            if key in contra_lower:
                if any(syn in patient_lower for syn in synonyms):
                    return True

        return False

    def get_photosensitivity_warning(self, drug: str) -> Optional[PhotosensitivityWarning]:
        """Get photosensitivity warning for a drug"""
        return self.photosensitivity_drugs.get(drug.lower())

    def get_age_warning(self, drug: str, patient_age: int) -> Optional[AgeWarning]:
        """Get age-specific warnings"""
        warning = self.age_warnings.get(drug.lower())
        if not warning:
            return None

        # Check if patient age triggers warning
        if warning.min_age and patient_age < warning.min_age:
            return warning
        if warning.max_age and patient_age > warning.max_age:
            return warning

        # Also return for pediatric (<18) or geriatric (>65) general concerns
        if patient_age < 18 and warning.pediatric_concerns:
            return warning
        if patient_age > 65 and warning.geriatric_concerns:
            return warning

        return None

    def get_pregnancy_warning(self, drug: str) -> Optional[PregnancyWarning]:
        """Get pregnancy/lactation warning"""
        return self.pregnancy_warnings.get(drug.lower())

    def get_dosage_info(self, drug: str) -> List[DosageInfo]:
        """Get dosage information"""
        return self.dosage_info.get(drug.lower(), [])

    def verify_dosage(self, drug: str, dose: str, frequency: str, patient_age: int,
                      renal_function: Optional[str] = None,
                      hepatic_function: Optional[str] = None) -> List[Dict[str, Any]]:
        """Verify if dosage is within recommended range"""
        issues = []
        dosage_infos = self.get_dosage_info(drug)

        if not dosage_infos:
            issues.append({
                'type': 'info',
                'message': f'No standard dosage information available for {drug}',
                'severity': 'info'
            })
            return issues

        for dosage_info in dosage_infos:
            # Check special warnings
            if drug.lower() == 'methotrexate' and 'daily' in frequency.lower():
                issues.append({
                    'type': 'critical_error',
                    'message': 'CRITICAL: Methotrexate should be dosed WEEKLY, not daily! Daily dosing can be fatal.',
                    'severity': 'contraindicated',
                    'correct_frequency': dosage_info.application_frequency
                })

            # Check pediatric dosing
            if patient_age < 18:
                if dosage_info.pediatric_dose:
                    issues.append({
                        'type': 'pediatric_note',
                        'message': f'Pediatric dosing applies: {dosage_info.pediatric_dose}',
                        'severity': 'info',
                        'max_daily': dosage_info.pediatric_max_daily
                    })

            # Check geriatric concerns
            if patient_age > 65:
                if dosage_info.elderly_adjustment != "No specific adjustment":
                    issues.append({
                        'type': 'geriatric_adjustment',
                        'message': f'Elderly adjustment recommended: {dosage_info.elderly_adjustment}',
                        'severity': 'moderate'
                    })

            # Check renal adjustment
            if renal_function and renal_function.lower() != 'normal':
                if 'no adjustment' not in dosage_info.renal_adjustment.lower():
                    issues.append({
                        'type': 'renal_adjustment',
                        'message': f'Renal adjustment needed: {dosage_info.renal_adjustment}',
                        'severity': 'moderate'
                    })

            # Check hepatic adjustment
            if hepatic_function and hepatic_function.lower() != 'normal':
                if 'no adjustment' not in dosage_info.hepatic_adjustment.lower() and 'not applicable' not in dosage_info.hepatic_adjustment.lower():
                    issues.append({
                        'type': 'hepatic_adjustment',
                        'message': f'Hepatic adjustment needed: {dosage_info.hepatic_adjustment}',
                        'severity': 'moderate' if 'caution' in dosage_info.hepatic_adjustment.lower() else 'severe'
                    })

        return issues


class MedicationInteractionChecker:
    """
    Main service for checking medication interactions and safety
    """

    def __init__(self):
        self.database = DrugInteractionDatabase()

    def check_medication(
        self,
        medication: str,
        current_medications: List[str] = None,
        patient_conditions: List[str] = None,
        patient_age: int = None,
        is_pregnant: bool = False,
        is_breastfeeding: bool = False,
        dose: str = None,
        frequency: str = None,
        renal_function: str = None,
        hepatic_function: str = None,
        sun_exposure_level: str = None
    ) -> InteractionCheckResult:
        """
        Comprehensive medication safety check

        Args:
            medication: Name of medication to check
            current_medications: List of patient's current medications
            patient_conditions: List of patient's medical conditions
            patient_age: Patient's age in years
            is_pregnant: Whether patient is pregnant
            is_breastfeeding: Whether patient is breastfeeding
            dose: Prescribed dose
            frequency: Dosing frequency
            renal_function: "normal", "mild", "moderate", "severe"
            hepatic_function: "normal", "mild", "moderate", "severe"
            sun_exposure_level: "minimal", "moderate", "high", "very_high"

        Returns:
            InteractionCheckResult with all findings
        """
        current_medications = current_medications or []
        patient_conditions = patient_conditions or []

        # Collect all findings
        drug_interactions = []
        contraindications = []
        photosensitivity_warnings = []
        age_warnings = []
        pregnancy_warnings = []
        dosage_issues = []
        recommendations = []

        # 1. Check drug-drug interactions
        drug_interactions = self.database.find_interactions(medication, current_medications)

        # 2. Check contraindications
        # Add pregnancy/breastfeeding to conditions if applicable
        conditions_to_check = patient_conditions.copy()
        if is_pregnant:
            conditions_to_check.append('pregnancy')
        if is_breastfeeding:
            conditions_to_check.append('breastfeeding')

        contraindications = self.database.get_contraindications(medication, conditions_to_check)

        # 3. Check photosensitivity
        photo_warning = self.database.get_photosensitivity_warning(medication)
        if photo_warning:
            photosensitivity_warnings.append(photo_warning)

            # Add recommendations based on sun exposure level
            if sun_exposure_level in ['high', 'very_high']:
                recommendations.append(
                    f"IMPORTANT: {medication} causes photosensitivity. With your high sun exposure level, "
                    f"strict sun protection is essential. Use SPF {photo_warning.spf_recommendation}+ "
                    "and protective clothing."
                )

        # 4. Check age warnings
        if patient_age is not None:
            age_warning = self.database.get_age_warning(medication, patient_age)
            if age_warning:
                age_warnings.append(age_warning)

        # 5. Check pregnancy/lactation warnings
        if is_pregnant or is_breastfeeding:
            preg_warning = self.database.get_pregnancy_warning(medication)
            if preg_warning:
                pregnancy_warnings.append(preg_warning)

        # 6. Verify dosage
        if dose or frequency:
            dosage_issues = self.database.verify_dosage(
                medication,
                dose or "unknown",
                frequency or "unknown",
                patient_age or 40,  # Default age for dosing
                renal_function,
                hepatic_function
            )

        # Determine overall safety and risk level
        is_safe, risk_level, requires_review = self._assess_overall_safety(
            drug_interactions,
            contraindications,
            photosensitivity_warnings,
            age_warnings,
            pregnancy_warnings,
            dosage_issues,
            is_pregnant,
            is_breastfeeding
        )

        # Generate recommendations
        recommendations.extend(self._generate_recommendations(
            medication,
            drug_interactions,
            contraindications,
            photosensitivity_warnings,
            age_warnings,
            pregnancy_warnings,
            dosage_issues
        ))

        return InteractionCheckResult(
            medication=medication,
            is_safe=is_safe,
            overall_risk_level=risk_level,
            drug_interactions=drug_interactions,
            contraindications=contraindications,
            photosensitivity_warnings=photosensitivity_warnings,
            age_warnings=age_warnings,
            pregnancy_warnings=pregnancy_warnings,
            dosage_issues=dosage_issues,
            recommendations=recommendations,
            requires_provider_review=requires_review
        )

    def _assess_overall_safety(
        self,
        drug_interactions: List[DrugInteraction],
        contraindications: List[Contraindication],
        photosensitivity_warnings: List[PhotosensitivityWarning],
        age_warnings: List[AgeWarning],
        pregnancy_warnings: List[PregnancyWarning],
        dosage_issues: List[Dict],
        is_pregnant: bool,
        is_breastfeeding: bool
    ) -> Tuple[bool, SeverityLevel, bool]:
        """Assess overall safety based on all findings"""

        # Check for contraindicated findings
        for interaction in drug_interactions:
            if interaction.severity == SeverityLevel.CONTRAINDICATED:
                return False, SeverityLevel.CONTRAINDICATED, True

        for contra in contraindications:
            if contra.severity == SeverityLevel.CONTRAINDICATED:
                return False, SeverityLevel.CONTRAINDICATED, True

        for preg in pregnancy_warnings:
            if preg.severity == SeverityLevel.CONTRAINDICATED:
                if is_pregnant or is_breastfeeding:
                    return False, SeverityLevel.CONTRAINDICATED, True

        # Check for severe findings
        has_severe = any(
            interaction.severity == SeverityLevel.SEVERE
            for interaction in drug_interactions
        ) or any(
            contra.severity == SeverityLevel.SEVERE
            for contra in contraindications
        ) or any(
            warning.severity == SeverityLevel.SEVERE
            for warning in photosensitivity_warnings
        ) or any(
            issue.get('severity') == 'critical_error'
            for issue in dosage_issues
        )

        if has_severe:
            return False, SeverityLevel.SEVERE, True

        # Check for moderate findings
        has_moderate = any(
            interaction.severity == SeverityLevel.MODERATE
            for interaction in drug_interactions
        ) or any(
            contra.severity == SeverityLevel.MODERATE
            for contra in contraindications
        ) or any(
            warning.severity == SeverityLevel.MODERATE
            for warning in photosensitivity_warnings
        )

        if has_moderate:
            return True, SeverityLevel.MODERATE, True

        # Check for mild findings
        has_mild = any(
            interaction.severity == SeverityLevel.MILD
            for interaction in drug_interactions
        ) or len(age_warnings) > 0 or len(photosensitivity_warnings) > 0

        if has_mild:
            return True, SeverityLevel.MILD, False

        # No significant findings
        return True, SeverityLevel.INFO, False

    def _generate_recommendations(
        self,
        medication: str,
        drug_interactions: List[DrugInteraction],
        contraindications: List[Contraindication],
        photosensitivity_warnings: List[PhotosensitivityWarning],
        age_warnings: List[AgeWarning],
        pregnancy_warnings: List[PregnancyWarning],
        dosage_issues: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Drug interaction recommendations
        for interaction in drug_interactions:
            if interaction.severity in [SeverityLevel.CONTRAINDICATED, SeverityLevel.SEVERE]:
                recommendations.append(
                    f"AVOID: {interaction.drug1} + {interaction.drug2}: {interaction.management}"
                )
            elif interaction.severity == SeverityLevel.MODERATE:
                recommendations.append(
                    f"MONITOR: {interaction.drug1} + {interaction.drug2}: {interaction.management}"
                )

        # Contraindication recommendations
        for contra in contraindications:
            recommendations.append(
                f"{contra.severity.value.upper()}: {medication} and {contra.condition} - "
                f"{contra.description}. "
                + (f"Consider alternatives: {', '.join(contra.alternatives)}" if contra.alternatives else "")
            )

        # Photosensitivity recommendations
        for photo in photosensitivity_warnings:
            recommendations.append(
                f"SUN PROTECTION REQUIRED: {medication} causes {photo.photosensitivity_type} reaction. "
                f"Use SPF {photo.spf_recommendation}+ sunscreen and avoid peak sun hours."
            )

        # Age-related recommendations
        for age_warn in age_warnings:
            if age_warn.pediatric_concerns:
                recommendations.append(
                    f"PEDIATRIC ALERT: {medication} - Concerns: {', '.join(age_warn.pediatric_concerns)}"
                )
            if age_warn.geriatric_concerns:
                recommendations.append(
                    f"GERIATRIC ALERT: {medication} - Concerns: {', '.join(age_warn.geriatric_concerns)}"
                )

        # Pregnancy/lactation recommendations
        for preg in pregnancy_warnings:
            if preg.alternatives_during_pregnancy:
                recommendations.append(
                    f"PREGNANCY: Consider alternatives - {', '.join(preg.alternatives_during_pregnancy)}"
                )
            if preg.alternatives_during_lactation:
                recommendations.append(
                    f"LACTATION: Consider alternatives - {', '.join(preg.alternatives_during_lactation)}"
                )

        # Dosage issue recommendations
        for issue in dosage_issues:
            if issue['type'] == 'critical_error':
                recommendations.append(f"CRITICAL DOSING ERROR: {issue['message']}")
            elif issue['type'] in ['renal_adjustment', 'hepatic_adjustment']:
                recommendations.append(f"DOSE ADJUSTMENT NEEDED: {issue['message']}")

        return recommendations

    def check_treatment_safety(
        self,
        treatment_plan: Dict[str, Any],
        patient_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check safety of a complete treatment plan

        Args:
            treatment_plan: Dict with medications, doses, frequencies
            patient_profile: Dict with age, conditions, current_medications, pregnancy status, etc.

        Returns:
            Comprehensive safety report
        """
        all_results = []
        overall_safe = True
        highest_severity = SeverityLevel.INFO

        medications = treatment_plan.get('medications', [])

        for med in medications:
            result = self.check_medication(
                medication=med.get('name'),
                current_medications=patient_profile.get('current_medications', []),
                patient_conditions=patient_profile.get('conditions', []),
                patient_age=patient_profile.get('age'),
                is_pregnant=patient_profile.get('is_pregnant', False),
                is_breastfeeding=patient_profile.get('is_breastfeeding', False),
                dose=med.get('dose'),
                frequency=med.get('frequency'),
                renal_function=patient_profile.get('renal_function'),
                hepatic_function=patient_profile.get('hepatic_function'),
                sun_exposure_level=patient_profile.get('sun_exposure_level')
            )

            all_results.append(result)

            if not result.is_safe:
                overall_safe = False

            # Track highest severity
            severity_order = [
                SeverityLevel.INFO,
                SeverityLevel.MILD,
                SeverityLevel.MODERATE,
                SeverityLevel.SEVERE,
                SeverityLevel.CONTRAINDICATED
            ]
            if severity_order.index(result.overall_risk_level) > severity_order.index(highest_severity):
                highest_severity = result.overall_risk_level

        # Check medication-medication interactions within the treatment plan
        plan_meds = [med.get('name') for med in medications]
        internal_interactions = []
        for i, med1 in enumerate(plan_meds):
            for med2 in plan_meds[i+1:]:
                interactions = self.database.find_interactions(med1, [med2])
                internal_interactions.extend(interactions)

        return {
            'overall_safe': overall_safe,
            'highest_severity': highest_severity.value,
            'medication_results': [self._result_to_dict(r) for r in all_results],
            'internal_interactions': [self._interaction_to_dict(i) for i in internal_interactions],
            'requires_provider_review': any(r.requires_provider_review for r in all_results),
            'total_warnings': sum(
                len(r.drug_interactions) + len(r.contraindications) +
                len(r.photosensitivity_warnings) + len(r.age_warnings) +
                len(r.pregnancy_warnings) + len(r.dosage_issues)
                for r in all_results
            ),
            'recommendations': [rec for r in all_results for rec in r.recommendations]
        }

    def _result_to_dict(self, result: InteractionCheckResult) -> Dict[str, Any]:
        """Convert InteractionCheckResult to dictionary"""
        return {
            'medication': result.medication,
            'is_safe': result.is_safe,
            'overall_risk_level': result.overall_risk_level.value,
            'drug_interactions': [self._interaction_to_dict(i) for i in result.drug_interactions],
            'contraindications': [self._contraindication_to_dict(c) for c in result.contraindications],
            'photosensitivity_warnings': [self._photo_warning_to_dict(p) for p in result.photosensitivity_warnings],
            'age_warnings': [self._age_warning_to_dict(a) for a in result.age_warnings],
            'pregnancy_warnings': [self._pregnancy_warning_to_dict(p) for p in result.pregnancy_warnings],
            'dosage_issues': result.dosage_issues,
            'recommendations': result.recommendations,
            'requires_provider_review': result.requires_provider_review,
            'checked_at': result.checked_at.isoformat()
        }

    def _interaction_to_dict(self, interaction: DrugInteraction) -> Dict[str, Any]:
        return {
            'drug1': interaction.drug1,
            'drug2': interaction.drug2,
            'severity': interaction.severity.value,
            'interaction_type': interaction.interaction_type.value,
            'description': interaction.description,
            'mechanism': interaction.mechanism,
            'clinical_effects': interaction.clinical_effects,
            'management': interaction.management
        }

    def _contraindication_to_dict(self, contra: Contraindication) -> Dict[str, Any]:
        return {
            'drug': contra.drug,
            'condition': contra.condition,
            'severity': contra.severity.value,
            'description': contra.description,
            'reason': contra.reason,
            'alternatives': contra.alternatives
        }

    def _photo_warning_to_dict(self, warning: PhotosensitivityWarning) -> Dict[str, Any]:
        return {
            'drug': warning.drug,
            'type': warning.photosensitivity_type,
            'severity': warning.severity.value,
            'onset_timeframe': warning.onset_timeframe,
            'duration_after_stopping': warning.duration_after_stopping,
            'uva_uvb_sensitivity': warning.uva_uvb_sensitivity,
            'clinical_presentation': warning.clinical_presentation,
            'precautions': warning.precautions,
            'spf_recommendation': warning.spf_recommendation
        }

    def _age_warning_to_dict(self, warning: AgeWarning) -> Dict[str, Any]:
        return {
            'drug': warning.drug,
            'min_age': warning.min_age,
            'max_age': warning.max_age,
            'severity': warning.severity.value,
            'pediatric_concerns': warning.pediatric_concerns,
            'geriatric_concerns': warning.geriatric_concerns,
            'dose_adjustment': warning.dose_adjustment
        }

    def _pregnancy_warning_to_dict(self, warning: PregnancyWarning) -> Dict[str, Any]:
        return {
            'drug': warning.drug,
            'pregnancy_category': warning.pregnancy_category,
            'severity': warning.severity.value,
            'first_trimester_risk': warning.first_trimester_risk,
            'second_trimester_risk': warning.second_trimester_risk,
            'third_trimester_risk': warning.third_trimester_risk,
            'lactation_safe': warning.lactation_safe,
            'lactation_notes': warning.lactation_notes,
            'alternatives_during_pregnancy': warning.alternatives_during_pregnancy,
            'alternatives_during_lactation': warning.alternatives_during_lactation
        }


# Singleton instance
medication_checker = MedicationInteractionChecker()
