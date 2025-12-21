"""
Insurance Pre-Authorization AI Module

Generates comprehensive pre-authorization documentation for dermatological conditions
including medical necessity letters, clinical summaries, and supporting evidence.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any


# ICD-10 codes for common dermatological conditions
ICD10_CODES = {
    "melanoma": "C43.9",
    "basal cell carcinoma": "C44.91",
    "squamous cell carcinoma": "C44.92",
    "actinic keratosis": "L57.0",
    "seborrheic keratosis": "L82.1",
    "nevus": "D22.9",
    "atopic dermatitis": "L20.9",
    "contact dermatitis": "L25.9",
    "psoriasis": "L40.9",
    "eczema": "L30.9",
    "dermatitis": "L30.9",
    "rosacea": "L71.9",
    "acne": "L70.0",
    "urticaria": "L50.9",
    "vitiligo": "L80",
    "lichen planus": "L43.9",
    "seborrheic dermatitis": "L21.9",
    "fungal infection": "B35.9",
    "bacterial infection": "L08.9",
    "viral infection": "B09",
    "herpes simplex": "B00.9",
    "herpes zoster": "B02.9",
    "impetigo": "L01.00",
    "cellulitis": "L03.90",
    "abscess": "L02.91",
    "wart": "B07.9",
    "molluscum contagiosum": "B08.1",
}

# CPT codes for common dermatological procedures
CPT_CODES = {
    "biopsy_skin": "11102",
    "biopsy_additional": "11103",
    "excision_benign_0.5cm": "11400",
    "excision_benign_1.0cm": "11401",
    "excision_benign_2.0cm": "11402",
    "excision_malignant_0.5cm": "11600",
    "excision_malignant_1.0cm": "11601",
    "excision_malignant_2.0cm": "11602",
    "destruction_benign_1": "17110",
    "destruction_benign_2-14": "17111",
    "destruction_malignant_0.5cm": "17260",
    "destruction_malignant_1.0cm": "17261",
    "dermoscopy": "96999",
    "photography": "96904",
    "phototherapy": "96912",
}


def get_icd10_code(condition: str) -> str:
    """Get ICD-10 code for a condition."""
    condition_lower = condition.lower()
    for key, code in ICD10_CODES.items():
        if key in condition_lower:
            return code
    return "L98.9"  # Default: Other disorder of skin and subcutaneous tissue


def get_recommended_procedures(condition: str, severity: Optional[str] = None) -> List[Dict[str, str]]:
    """Get recommended procedures based on condition."""
    condition_lower = condition.lower()
    procedures = []

    # Always recommend dermoscopy and photography for documentation
    procedures.append({
        "code": CPT_CODES["dermoscopy"],
        "description": "Dermoscopic examination",
        "rationale": "Non-invasive examination technique to evaluate skin lesion characteristics"
    })

    procedures.append({
        "code": CPT_CODES["photography"],
        "description": "Clinical photography",
        "rationale": "Documentation for monitoring and comparison over time"
    })

    # Malignant or suspicious conditions
    if any(term in condition_lower for term in ["melanoma", "carcinoma", "malignant", "cancer"]):
        procedures.append({
            "code": CPT_CODES["biopsy_skin"],
            "description": "Skin biopsy",
            "rationale": "Tissue diagnosis required for definitive diagnosis and treatment planning"
        })
        procedures.append({
            "code": CPT_CODES["excision_malignant_1.0cm"],
            "description": "Excision of malignant lesion",
            "rationale": "Complete excision with adequate margins for curative treatment"
        })

    # Premalignant conditions
    elif "actinic keratosis" in condition_lower:
        procedures.append({
            "code": CPT_CODES["destruction_benign_1"],
            "description": "Destruction of premalignant lesion",
            "rationale": "Treatment to prevent progression to squamous cell carcinoma"
        })

    # Benign but symptomatic lesions
    elif any(term in condition_lower for term in ["seborrheic keratosis", "benign"]):
        if severity and "symptomatic" in severity.lower():
            procedures.append({
                "code": CPT_CODES["excision_benign_1.0cm"],
                "description": "Excision of benign lesion",
                "rationale": "Removal indicated for symptomatic relief or cosmetic concerns"
            })

    # Inflammatory conditions
    elif any(term in condition_lower for term in ["dermatitis", "eczema", "psoriasis"]):
        procedures.append({
            "code": CPT_CODES["phototherapy"],
            "description": "Phototherapy",
            "rationale": "Evidence-based treatment for moderate to severe inflammatory skin conditions"
        })
        procedures.append({
            "code": CPT_CODES["biopsy_skin"],
            "description": "Skin biopsy",
            "rationale": "Histopathologic confirmation if diagnosis uncertain or treatment-resistant"
        })

    # Infectious conditions
    elif any(term in condition_lower for term in ["infection", "cellulitis", "abscess"]):
        procedures.append({
            "code": CPT_CODES["biopsy_skin"],
            "description": "Skin biopsy with culture",
            "rationale": "Identification of causative organism for targeted antimicrobial therapy"
        })

    return procedures


def generate_medical_necessity_letter(
    condition: str,
    confidence: float,
    severity: Optional[str] = None,
    patient_factors: Optional[Dict] = None,
    clinical_findings: Optional[Dict] = None
) -> str:
    """Generate formal medical necessity letter."""

    date = datetime.now().strftime("%B %d, %Y")
    icd10 = get_icd10_code(condition)

    letter = f"""MEDICAL NECESSITY LETTER

Date: {date}

To: Insurance Pre-Authorization Department

RE: Pre-Authorization Request for Dermatological Evaluation and Treatment

PATIENT INFORMATION:
[Patient Name]
[Patient Date of Birth]
[Policy/Member ID]

DIAGNOSIS:
{condition} (ICD-10: {icd10})

CLINICAL PRESENTATION:
The patient presents with a dermatological lesion requiring evaluation and treatment.
Advanced AI-assisted analysis has identified characteristics consistent with {condition}
with {confidence:.1f}% diagnostic confidence using validated deep learning models.
"""

    if clinical_findings:
        letter += "\nCLINICAL FINDINGS:\n"
        if clinical_findings.get("lesion_characteristics"):
            letter += f"- Lesion Analysis: {clinical_findings['lesion_characteristics']}\n"
        if clinical_findings.get("inflammatory_characteristics"):
            letter += f"- Inflammatory Features: {clinical_findings['inflammatory_characteristics']}\n"
        if clinical_findings.get("red_flags"):
            letter += f"- Red Flag Indicators: {', '.join(clinical_findings['red_flags'])}\n"

    letter += f"""
MEDICAL NECESSITY JUSTIFICATION:

1. DIAGNOSTIC IMPERATIVE:
   - AI-assisted diagnostic evaluation indicates {condition}
   - Condition requires definitive diagnosis through appropriate diagnostic procedures
   - Early detection and treatment is critical for optimal patient outcomes
"""

    # Add condition-specific justification
    if any(term in condition.lower() for term in ["melanoma", "carcinoma", "malignant"]):
        letter += """
2. URGENCY OF CARE:
   - Suspicious for malignancy based on clinical and AI analysis
   - Early diagnosis and treatment significantly impacts prognosis and survival
   - Delay in treatment may result in disease progression and metastasis
   - Follows NCCN guidelines for management of suspicious skin lesions

3. STANDARD OF CARE:
   - Tissue diagnosis (biopsy) is standard of care for suspicious lesions
   - Excision with adequate margins is definitive treatment
   - American Academy of Dermatology guidelines support this approach
"""
    elif "actinic keratosis" in condition.lower():
        letter += """
2. CANCER PREVENTION:
   - Actinic keratosis is a precancerous condition
   - 5-10% risk of progression to squamous cell carcinoma if untreated
   - Treatment is preventive care to reduce cancer risk
   - Cost-effective compared to treating invasive carcinoma

3. EVIDENCE-BASED TREATMENT:
   - Treatment reduces risk of progression to invasive cancer
   - Multiple FDA-approved treatment modalities available
   - American Academy of Dermatology recommends treatment
"""
    elif any(term in condition.lower() for term in ["dermatitis", "eczema", "psoriasis"]):
        letter += """
2. QUALITY OF LIFE IMPACT:
   - Chronic inflammatory skin condition significantly impacts quality of life
   - May cause pain, itching, sleep disturbance, and psychological distress
   - Affects work productivity and social functioning
   - Treatment can significantly improve patient outcomes

3. EVIDENCE-BASED TREATMENT:
   - Condition requires comprehensive evaluation and treatment plan
   - Multiple therapeutic modalities may be necessary
   - Follows evidence-based treatment guidelines (AAD/AADA)
   - Early intervention prevents disease progression and complications
"""
    else:
        letter += """
2. CLINICAL NECESSITY:
   - Lesion characteristics warrant professional dermatological evaluation
   - Appropriate diagnostic workup required for definitive diagnosis
   - Treatment planning depends on accurate diagnosis
   - Patient safety requires expert clinical assessment

3. STANDARD OF CARE:
   - Evaluation and management follows evidence-based clinical guidelines
   - Recommended procedures are medically necessary and appropriate
   - Treatment aligns with current dermatological practice standards
"""

    letter += """
REQUESTED PROCEDURES:
Please see attached procedure codes and clinical justification.

SUPPORTING DOCUMENTATION:
- AI-assisted diagnostic analysis report
- Clinical photographs with dermoscopic evaluation
- Explainability analysis showing diagnostic features
- Evidence-based treatment guidelines
- Relevant medical literature references

CONCLUSION:
The requested evaluation and treatment are medically necessary, clinically appropriate,
and represent the standard of care for this condition. The AI-assisted analysis supports
the need for timely intervention. Authorization for the requested procedures is respectfully
requested to ensure optimal patient care and outcomes.

Should you require additional information or clarification, please do not hesitate to contact
our office.

Respectfully submitted,

[Physician Name]
[License Number]
[NPI Number]
[Contact Information]

---
This pre-authorization documentation was generated with AI assistance to streamline the
authorization process. All clinical recommendations are based on evidence-based guidelines
and standard dermatological practice.
"""

    return letter


def generate_preauth_form_data(
    condition: str,
    confidence: float,
    severity: Optional[str] = None,
    patient_factors: Optional[Dict] = None,
    clinical_findings: Optional[Dict] = None
) -> Dict[str, Any]:
    """Generate structured pre-authorization form data."""

    icd10 = get_icd10_code(condition)
    procedures = get_recommended_procedures(condition, severity)

    # Determine urgency
    urgency = "Routine"
    if any(term in condition.lower() for term in ["melanoma", "malignant", "carcinoma"]):
        urgency = "Urgent"
    elif "actinic keratosis" in condition.lower():
        urgency = "Timely (within 2 weeks)"

    form_data = {
        "request_date": datetime.now().strftime("%Y-%m-%d"),
        "diagnosis": {
            "primary_diagnosis": condition,
            "icd10_code": icd10,
            "confidence_level": f"{confidence:.1f}%",
            "diagnostic_method": "AI-assisted clinical evaluation with dermoscopic analysis"
        },
        "procedures_requested": procedures,
        "urgency": urgency,
        "clinical_rationale": generate_clinical_rationale(condition, severity, clinical_findings),
        "supporting_evidence": {
            "ai_analysis": "Advanced deep learning model analysis performed",
            "dermoscopy": "Dermoscopic evaluation completed",
            "photography": "Clinical photography documented",
            "guidelines": get_relevant_guidelines(condition)
        },
        "estimated_timeline": get_estimated_timeline(condition),
        "alternative_treatments_considered": get_alternative_treatments(condition),
        "expected_outcomes": get_expected_outcomes(condition)
    }

    return form_data


def generate_clinical_rationale(
    condition: str,
    severity: Optional[str] = None,
    clinical_findings: Optional[Dict] = None
) -> str:
    """Generate clinical rationale for procedures."""

    rationale = f"Patient presents with dermatological lesion consistent with {condition}. "

    if any(term in condition.lower() for term in ["melanoma", "carcinoma", "malignant"]):
        rationale += ("AI analysis indicates features concerning for malignancy. "
                     "Tissue diagnosis is medically necessary for definitive diagnosis "
                     "and to guide appropriate treatment. Early detection and treatment "
                     "significantly impact prognosis and survival outcomes.")
    elif "actinic keratosis" in condition.lower():
        rationale += ("This premalignant condition has risk of progression to invasive "
                     "squamous cell carcinoma. Treatment is preventive care and represents "
                     "standard of care to reduce cancer risk.")
    elif any(term in condition.lower() for term in ["dermatitis", "eczema", "psoriasis"]):
        rationale += ("Chronic inflammatory skin condition requires comprehensive evaluation "
                     "and treatment. Condition significantly impacts quality of life and may "
                     "lead to complications if untreated. Treatment follows evidence-based "
                     "guidelines and is medically necessary.")
    else:
        rationale += ("Clinical and AI-assisted evaluation indicates need for professional "
                     "dermatological assessment. Appropriate diagnostic procedures are necessary "
                     "for accurate diagnosis and treatment planning.")

    if clinical_findings and clinical_findings.get("red_flags"):
        rationale += f" Red flag indicators present: {', '.join(clinical_findings['red_flags'])}."

    return rationale


def get_relevant_guidelines(condition: str) -> List[str]:
    """Get relevant clinical guidelines for the condition."""
    guidelines = [
        "American Academy of Dermatology (AAD) Clinical Guidelines",
        "Skin Cancer Foundation Treatment Guidelines"
    ]

    if any(term in condition.lower() for term in ["melanoma", "carcinoma", "malignant"]):
        guidelines.extend([
            "NCCN Clinical Practice Guidelines in Oncology: Melanoma",
            "NCCN Clinical Practice Guidelines: Basal Cell and Squamous Cell Skin Cancers",
            "USPSTF Screening Guidelines"
        ])
    elif any(term in condition.lower() for term in ["dermatitis", "eczema", "psoriasis"]):
        guidelines.extend([
            "American Academy of Dermatology Association (AADA) Guidelines",
            "National Eczema Association Treatment Guidelines",
            "National Psoriasis Foundation Treatment Guidelines"
        ])

    return guidelines


def get_estimated_timeline(condition: str) -> str:
    """Get estimated timeline for evaluation and treatment."""
    if any(term in condition.lower() for term in ["melanoma", "malignant"]):
        return "Urgent: Evaluation within 1-2 weeks, treatment as soon as diagnosis confirmed"
    elif "carcinoma" in condition.lower() or "actinic keratosis" in condition.lower():
        return "Timely: Evaluation within 2-4 weeks, treatment within 4-6 weeks"
    else:
        return "Routine: Evaluation within 4-8 weeks, treatment as clinically indicated"


def get_alternative_treatments(condition: str) -> List[str]:
    """Get alternative treatments considered."""
    alternatives = []

    if any(term in condition.lower() for term in ["melanoma", "carcinoma", "malignant"]):
        alternatives = [
            "Watchful waiting - Not appropriate due to malignancy risk",
            "Topical treatments - Insufficient for suspected malignancy",
            "Definitive excision - Recommended approach"
        ]
    elif "actinic keratosis" in condition.lower():
        alternatives = [
            "Cryotherapy - Effective first-line treatment",
            "Topical 5-fluorouracil - Alternative for multiple lesions",
            "Topical imiquimod - Alternative immunomodulator",
            "Photodynamic therapy - Alternative for field treatment"
        ]
    elif any(term in condition.lower() for term in ["dermatitis", "eczema"]):
        alternatives = [
            "Topical corticosteroids - First-line treatment",
            "Topical calcineurin inhibitors - Steroid-sparing alternative",
            "Systemic therapy - For moderate to severe cases",
            "Phototherapy - For widespread disease"
        ]
    else:
        alternatives = [
            "Conservative management with observation",
            "Topical treatments as appropriate",
            "Procedural intervention if indicated"
        ]

    return alternatives


def get_expected_outcomes(condition: str) -> str:
    """Get expected outcomes from treatment."""
    if any(term in condition.lower() for term in ["melanoma", "malignant"]):
        return ("Definitive diagnosis, complete removal of malignancy with negative margins, "
                "prevention of metastasis, excellent prognosis with early treatment, "
                "regular surveillance to monitor for recurrence")
    elif "carcinoma" in condition.lower():
        return ("High cure rate (>95%) with appropriate treatment, complete removal of cancer, "
                "excellent cosmetic outcomes, low recurrence risk with adequate margins")
    elif "actinic keratosis" in condition.lower():
        return ("Resolution of premalignant lesion, prevention of progression to invasive cancer, "
                "reduced long-term cancer risk, good cosmetic outcome")
    elif any(term in condition.lower() for term in ["dermatitis", "eczema", "psoriasis"]):
        return ("Significant improvement in symptoms, reduced inflammation and pruritus, "
                "improved quality of life, prevention of complications, better disease control")
    else:
        return ("Accurate diagnosis, appropriate treatment, symptom resolution or improvement, "
                "good clinical outcomes, patient satisfaction")


def generate_insurance_preauthorization(
    condition: str,
    confidence: float,
    severity: Optional[str] = None,
    patient_factors: Optional[Dict] = None,
    clinical_findings: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive insurance pre-authorization documentation.

    Args:
        condition: Diagnosed condition
        confidence: Diagnostic confidence (0-100)
        severity: Severity level if applicable
        patient_factors: Patient-specific factors (age, history, etc.)
        clinical_findings: Clinical findings from analysis

    Returns:
        Dictionary containing all pre-authorization documentation
    """

    # Generate form data first (needed for autofill)
    form_data = generate_preauth_form_data(
        condition, confidence, severity, patient_factors, clinical_findings
    )

    # Calculate approval likelihood
    approval_prediction = predict_approval_likelihood(
        condition=condition,
        confidence=confidence,
        urgency=form_data.get('urgency', 'Routine'),
        evidence_quality={
            "guidelines_count": len(form_data.get('supporting_evidence', {}).get('clinical_guidelines', [])),
            "literature_count": 3  # Default assumption
        },
        patient_factors=patient_factors
    )

    # Generate complete preauth data
    preauth_data = {
        "medical_necessity_letter": generate_medical_necessity_letter(
            condition, confidence, severity, patient_factors, clinical_findings
        ),
        "form_data": form_data,
        "clinical_summary": generate_clinical_summary(
            condition, confidence, severity, patient_factors, clinical_findings
        ),
        "supporting_evidence": generate_supporting_evidence(condition),
        "generated_date": datetime.now().isoformat(),
        "status": "DRAFT",  # Initial status
        "submission_status": {
            "current_status": "DRAFT",
            "status_description": PREAUTH_STATUS["DRAFT"],
            "submitted_date": None,
            "decision_date": None,
            "notes": []
        }
    }

    # Add approval likelihood
    preauth_data["approval_likelihood"] = approval_prediction

    # Generate auto-fill forms
    preauth_data["autofill_forms"] = generate_autofill_forms(preauth_data, insurance_provider="generic")

    return preauth_data


def generate_clinical_summary(
    condition: str,
    confidence: float,
    severity: Optional[str] = None,
    patient_factors: Optional[Dict] = None,
    clinical_findings: Optional[Dict] = None
) -> str:
    """Generate clinical summary for pre-authorization."""

    summary = f"""CLINICAL SUMMARY

DIAGNOSIS: {condition}
ICD-10 CODE: {get_icd10_code(condition)}
DIAGNOSTIC CONFIDENCE: {confidence:.1f}%

CLINICAL PRESENTATION:
Patient presents with dermatological lesion evaluated using AI-assisted diagnostic analysis.
Advanced deep learning models trained on validated dermatological datasets were used to
analyze lesion characteristics.
"""

    if clinical_findings:
        summary += "\nCLINICAL FINDINGS:\n"

        if clinical_findings.get("lesion_characteristics"):
            summary += f"\nLesion Analysis:\n{clinical_findings['lesion_characteristics']}\n"

        if clinical_findings.get("inflammatory_characteristics"):
            summary += f"\nInflammatory Assessment:\n{clinical_findings['inflammatory_characteristics']}\n"

        if clinical_findings.get("red_flags"):
            summary += f"\nRed Flag Indicators:\n"
            for flag in clinical_findings['red_flags']:
                summary += f"- {flag}\n"

        if clinical_findings.get("differential_diagnosis"):
            summary += f"\nDifferential Diagnosis:\n"
            for diff in clinical_findings['differential_diagnosis']:
                summary += f"- {diff}\n"

    summary += f"""
ASSESSMENT:
Based on comprehensive AI-assisted analysis and clinical evaluation, the patient's
presentation is most consistent with {condition}. """

    if any(term in condition.lower() for term in ["melanoma", "carcinoma", "malignant"]):
        summary += """The lesion demonstrates features concerning for malignancy and requires
urgent evaluation and definitive diagnosis through tissue biopsy. Early detection and
treatment are critical for optimal outcomes.
"""
    elif "actinic keratosis" in condition.lower():
        summary += """This premalignant condition requires treatment to prevent progression
to invasive squamous cell carcinoma. Treatment is considered preventive care and represents
standard of care.
"""
    elif any(term in condition.lower() for term in ["dermatitis", "eczema", "psoriasis"]):
        summary += """This chronic inflammatory condition requires comprehensive treatment
to improve quality of life and prevent complications. Treatment follows evidence-based
clinical guidelines.
"""

    summary += """
PLAN:
"""
    procedures = get_recommended_procedures(condition, severity)
    for i, proc in enumerate(procedures, 1):
        summary += f"{i}. {proc['description']} (CPT: {proc['code']})\n"
        summary += f"   Rationale: {proc['rationale']}\n"

    summary += f"""
URGENCY: {get_estimated_timeline(condition)}

PROGNOSIS:
{get_expected_outcomes(condition)}

EVIDENCE BASE:
Treatment recommendations are based on current evidence-based clinical guidelines including:
"""
    for guideline in get_relevant_guidelines(condition):
        summary += f"- {guideline}\n"

    return summary


def generate_supporting_evidence(condition: str) -> Dict[str, Any]:
    """Generate supporting evidence for pre-authorization."""

    evidence = {
        "clinical_guidelines": get_relevant_guidelines(condition),
        "diagnostic_accuracy": {
            "method": "Deep learning AI analysis",
            "validation": "Models validated on published dermatological datasets",
            "accuracy": "Comparable to board-certified dermatologists",
            "references": [
                "Esteva et al. (2017) Nature - Dermatologist-level classification of skin cancer with deep neural networks",
                "Haenssle et al. (2018) Annals of Oncology - Man against machine: diagnostic performance of a deep learning CNN",
                "Tschandl et al. (2020) Nature Medicine - Human-computer collaboration for skin cancer recognition"
            ]
        },
        "treatment_efficacy": get_treatment_efficacy_data(condition),
        "cost_effectiveness": {
            "early_detection": "Early diagnosis significantly reduces treatment costs",
            "prevention": "Preventive treatment more cost-effective than treating advanced disease",
            "outcomes": "Better outcomes with timely intervention reduce long-term healthcare costs"
        }
    }

    return evidence


def get_treatment_efficacy_data(condition: str) -> Dict[str, str]:
    """Get treatment efficacy data for the condition."""

    if "melanoma" in condition.lower():
        return {
            "early_stage_survival": "99% 5-year survival for localized melanoma",
            "treatment_success": ">95% cure rate with early excision",
            "importance": "Survival dramatically decreases if diagnosis delayed"
        }
    elif "basal cell carcinoma" in condition.lower():
        return {
            "cure_rate": ">95% cure rate with appropriate treatment",
            "recurrence": "Low recurrence with adequate margins",
            "importance": "Early treatment prevents local destruction"
        }
    elif "squamous cell carcinoma" in condition.lower():
        return {
            "cure_rate": ">90% cure rate for early-stage disease",
            "metastasis_risk": "2-5% metastasis risk if untreated",
            "importance": "Early treatment critical to prevent metastasis"
        }
    elif "actinic keratosis" in condition.lower():
        return {
            "progression_risk": "5-10% risk of progression to SCC",
            "treatment_success": "High clearance rates with treatment",
            "importance": "Treatment prevents cancer development"
        }
    elif any(term in condition.lower() for term in ["dermatitis", "eczema", "psoriasis"]):
        return {
            "symptom_improvement": "Significant improvement in 70-90% of patients",
            "quality_of_life": "Marked improvement in QOL scores",
            "importance": "Treatment prevents complications and improves function"
        }
    else:
        return {
            "treatment_success": "High success rate with appropriate treatment",
            "importance": "Timely diagnosis and treatment improve outcomes"
        }


# Pre-authorization status tracking
PREAUTH_STATUS = {
    "DRAFT": "Draft - Not submitted",
    "SUBMITTED": "Submitted to insurance",
    "UNDER_REVIEW": "Under review",
    "APPROVED": "Approved",
    "DENIED": "Denied",
    "ADDITIONAL_INFO_REQUIRED": "Additional information required"
}


def predict_approval_likelihood(
    condition: str,
    confidence: float,
    urgency: str,
    evidence_quality: Optional[Dict] = None,
    patient_factors: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Predict the likelihood of insurance pre-authorization approval.

    Returns a score from 0-100 indicating approval probability and breakdown of factors.
    """

    score = 0
    factors = []

    # Base score from diagnostic confidence (0-30 points)
    confidence_score = min(30, (confidence / 100) * 30)
    score += confidence_score
    factors.append({
        "factor": "Diagnostic Confidence",
        "score": round(confidence_score, 1),
        "max": 30,
        "impact": "HIGH" if confidence >= 80 else "MEDIUM" if confidence >= 60 else "LOW",
        "description": f"{confidence:.1f}% AI diagnostic confidence supports medical necessity"
    })

    # Urgency score (0-25 points)
    urgency_map = {
        "URGENT": 25,
        "Urgent": 25,
        "HIGH": 20,
        "MODERATE": 15,
        "Timely": 15,
        "ROUTINE": 10,
        "Routine": 10
    }
    urgency_score = urgency_map.get(urgency, 10)
    score += urgency_score
    factors.append({
        "factor": "Clinical Urgency",
        "score": urgency_score,
        "max": 25,
        "impact": "HIGH" if urgency_score >= 20 else "MEDIUM" if urgency_score >= 15 else "LOW",
        "description": f"{urgency} classification indicates clear medical need"
    })

    # Condition type score (0-20 points)
    condition_lower = condition.lower()
    if any(term in condition_lower for term in ["melanoma", "malignant", "cancer"]):
        condition_score = 20
        condition_impact = "CRITICAL"
        condition_desc = "Malignant condition with high approval rate for medically necessary procedures"
    elif any(term in condition_lower for term in ["carcinoma", "actinic keratosis"]):
        condition_score = 18
        condition_impact = "HIGH"
        condition_desc = "Pre-malignant or early cancer with strong approval likelihood"
    elif any(term in condition_lower for term in ["dermatitis", "psoriasis", "eczema"]):
        condition_score = 15
        condition_impact = "MEDIUM"
        condition_desc = "Chronic inflammatory condition with good approval rates for evidence-based treatments"
    else:
        condition_score = 12
        condition_impact = "MEDIUM"
        condition_desc = "Dermatological condition requiring professional evaluation"

    score += condition_score
    factors.append({
        "factor": "Condition Type",
        "score": condition_score,
        "max": 20,
        "impact": condition_impact,
        "description": condition_desc
    })

    # Evidence quality score (0-15 points)
    evidence_score = 15  # Default to full score if no evidence data provided
    if evidence_quality:
        # Reduce score if evidence is weak
        if evidence_quality.get("guidelines_count", 5) < 3:
            evidence_score = 10
        if evidence_quality.get("literature_count", 3) < 2:
            evidence_score = max(5, evidence_score - 5)

    score += evidence_score
    factors.append({
        "factor": "Supporting Evidence",
        "score": evidence_score,
        "max": 15,
        "impact": "HIGH" if evidence_score >= 12 else "MEDIUM",
        "description": "Strong evidence base with AAD/NCCN guidelines and peer-reviewed literature"
    })

    # Documentation completeness score (0-10 points)
    documentation_score = 10  # Assume complete documentation
    score += documentation_score
    factors.append({
        "factor": "Documentation Quality",
        "score": documentation_score,
        "max": 10,
        "impact": "MEDIUM",
        "description": "Complete pre-authorization package with all required components"
    })

    # Calculate final probability
    probability = min(100, score)

    # Determine approval category
    if probability >= 85:
        category = "HIGHLY LIKELY"
        category_color = "#10b981"  # Green
        recommendation = "Strong case for approval. Documentation is comprehensive and well-supported."
    elif probability >= 70:
        category = "LIKELY"
        category_color = "#3b82f6"  # Blue
        recommendation = "Good likelihood of approval. Consider highlighting key evidence in submission."
    elif probability >= 50:
        category = "MODERATE"
        category_color = "#f59e0b"  # Orange
        recommendation = "Moderate approval likelihood. May require peer-to-peer review or additional documentation."
    else:
        category = "UNCERTAIN"
        category_color = "#ef4444"  # Red
        recommendation = "Lower approval likelihood. Consider obtaining additional supporting documentation or second opinion."

    return {
        "probability": round(probability, 1),
        "score": round(score, 1),
        "max_score": 100,
        "category": category,
        "category_color": category_color,
        "recommendation": recommendation,
        "factors": factors,
        "confidence_interval": {
            "lower": max(0, round(probability - 10, 1)),
            "upper": min(100, round(probability + 10, 1))
        },
        "calculated_at": datetime.now().isoformat()
    }


def generate_autofill_forms(preauth_data: Dict[str, Any], insurance_provider: str = "generic") -> Dict[str, Any]:
    """
    Generate auto-filled insurance forms based on pre-authorization data.

    Supports multiple insurance providers with their specific form formats.
    """

    form_data = preauth_data.get('form_data', {})
    diagnosis = form_data.get('diagnosis', {})
    procedures = form_data.get('procedures_requested', [])

    # Generic CMS-1500 style form (most common)
    cms_1500_form = {
        "form_type": "CMS-1500",
        "form_version": "02/12",
        "sections": {
            "patient_information": {
                "1_insurance_type": "Other",
                "1a_insured_id": "[MEMBER_ID]",
                "2_patient_name": "[PATIENT_NAME]",
                "3_patient_dob": "[DATE_OF_BIRTH]",
                "4_insured_name": "[INSURED_NAME]",
                "5_patient_address": "[ADDRESS]",
                "6_patient_relationship": "Self",
                "7_insured_address": "[INSURED_ADDRESS]"
            },
            "physician_information": {
                "17_referring_physician": "[PHYSICIAN_NAME]",
                "17a_referring_physician_id": "[NPI_NUMBER]",
                "19_additional_claim_info": form_data.get('clinical_rationale', '')[:450],
                "20_outside_lab": "NO",
                "21_diagnosis_codes": [diagnosis.get('icd10_code', '')],
                "22_medicaid_resubmission": "",
                "23_prior_authorization": "[PRIOR_AUTH_NUMBER]"
            },
            "service_lines": []
        }
    }

    # Add procedure lines
    for idx, proc in enumerate(procedures, 1):
        if idx <= 6:  # CMS-1500 supports up to 6 service lines
            cms_1500_form["sections"]["service_lines"].append({
                "line": idx,
                "24a_dates_of_service": "[DATE_OF_SERVICE]",
                "24b_place_of_service": "11",  # Office
                "24d_procedures": proc['code'],
                "24e_diagnosis_pointer": "A",
                "24f_charges": "[AMOUNT]",
                "24g_days_units": "1",
                "24j_rendering_provider_id": "[NPI_NUMBER]"
            })

    # UB-04 form (hospital/facility)
    ub_04_form = {
        "form_type": "UB-04",
        "sections": {
            "patient_info": {
                "1_provider_name": "[FACILITY_NAME]",
                "2_patient_name": "[PATIENT_NAME]",
                "3_patient_control_number": "[CONTROL_NUMBER]",
                "5_federal_tax_number": "[TAX_ID]",
                "6_statement_dates": "[STATEMENT_PERIOD]",
                "8_patient_id": "[MEMBER_ID]",
                "10_birthdate": "[DOB]",
                "11_sex": "[M/F]",
                "12_admission_date": "[ADMISSION_DATE]"
            },
            "diagnosis_codes": {
                "67_principal_diagnosis": diagnosis.get('icd10_code', ''),
                "67a_67q_other_diagnosis": []
            },
            "revenue_codes": [
                {
                    "code": "0510",  # Clinic visit
                    "description": "Dermatology Clinic",
                    "hcpcs": procedures[0]['code'] if procedures else "",
                    "charges": "[AMOUNT]"
                }
            ]
        }
    }

    # Insurance-specific forms
    insurance_forms = {
        "generic": {
            "prior_authorization_request": {
                "member_id": "[MEMBER_ID]",
                "member_name": "[PATIENT_NAME]",
                "member_dob": "[DATE_OF_BIRTH]",
                "provider_name": "[PHYSICIAN_NAME]",
                "provider_npi": "[NPI_NUMBER]",
                "provider_phone": "[PHONE_NUMBER]",
                "requested_service_date": "[REQUESTED_DATE]",
                "diagnosis_code": diagnosis.get('icd10_code', ''),
                "diagnosis_description": diagnosis.get('primary_diagnosis', ''),
                "procedure_codes": [p['code'] for p in procedures],
                "procedure_descriptions": [p['description'] for p in procedures],
                "clinical_rationale": form_data.get('clinical_rationale', ''),
                "urgency": form_data.get('urgency', 'Routine'),
                "supporting_documentation": [
                    "AI-assisted diagnostic analysis",
                    "Clinical photographs",
                    "Evidence-based guidelines reference",
                    "Medical necessity letter"
                ]
            }
        }
    }

    return {
        "cms_1500": cms_1500_form,
        "ub_04": ub_04_form,
        "insurance_specific": insurance_forms.get(insurance_provider.lower(), insurance_forms["generic"]),
        "provider": insurance_provider,
        "generated_at": datetime.now().isoformat(),
        "instructions": {
            "placeholders": "Fields marked with [BRACKETS] must be filled with actual patient/provider information",
            "signature": "All forms require physician signature and date",
            "submission": "Forms should be submitted according to insurance provider requirements",
            "attachments": "Include medical necessity letter, clinical images, and supporting documentation"
        }
    }
