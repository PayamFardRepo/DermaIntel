"""
Appeal Letter Generator Module

Generates professional appeal letters for insurance claim denials.
Supports multiple levels of appeals and various denial reasons.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
import re


class DenialReason(str, Enum):
    """Common insurance denial reasons"""
    MEDICAL_NECESSITY = "medical_necessity"
    NOT_COVERED = "not_covered"
    PRIOR_AUTH_REQUIRED = "prior_auth_required"
    OUT_OF_NETWORK = "out_of_network"
    DUPLICATE_CLAIM = "duplicate_claim"
    TIMELY_FILING = "timely_filing"
    INCOMPLETE_INFO = "incomplete_info"
    EXPERIMENTAL = "experimental"
    COSMETIC = "cosmetic"
    PRE_EXISTING = "pre_existing"
    CODING_ERROR = "coding_error"
    BENEFIT_EXHAUSTED = "benefit_exhausted"
    OTHER = "other"


class AppealLevel(str, Enum):
    """Appeal escalation levels"""
    FIRST_LEVEL = "first_level"
    SECOND_LEVEL = "second_level"
    EXTERNAL_REVIEW = "external_review"
    STATE_INSURANCE = "state_insurance"


class AppealStatus(str, Enum):
    """Appeal tracking status"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    ADDITIONAL_INFO_REQUESTED = "additional_info_requested"
    APPROVED = "approved"
    DENIED = "denied"
    ESCALATED = "escalated"


class AppealRequest(BaseModel):
    """Request model for generating an appeal letter"""
    patient_name: str
    patient_dob: str
    patient_id: str
    insurance_company: str
    policy_number: str
    claim_number: str
    date_of_service: str
    denial_date: str
    denial_reason: DenialReason
    denial_reason_text: Optional[str] = None
    diagnosis: str
    icd10_code: str
    procedure: str
    cpt_code: str
    claim_amount: float
    provider_name: str
    provider_npi: str
    provider_address: str
    appeal_level: AppealLevel = AppealLevel.FIRST_LEVEL
    clinical_notes: Optional[str] = None
    supporting_evidence: Optional[List[str]] = None
    previous_treatments: Optional[List[str]] = None


class AppealLetter(BaseModel):
    """Generated appeal letter"""
    appeal_id: str
    generated_at: datetime
    appeal_level: AppealLevel
    letter_content: str
    subject_line: str
    summary: str
    key_arguments: List[str]
    supporting_documents_needed: List[str]
    deadline: datetime
    recommended_next_steps: List[str]
    success_likelihood: int  # 0-100


# Medical necessity justifications by condition
MEDICAL_NECESSITY_ARGUMENTS = {
    "melanoma": [
        "Melanoma is a life-threatening malignancy with high mortality rates if left untreated",
        "Early detection and treatment significantly improves 5-year survival rates",
        "The American Academy of Dermatology guidelines recommend immediate intervention",
        "Delayed treatment increases risk of metastasis and reduces treatment efficacy",
        "This procedure is the standard of care per NCCN guidelines"
    ],
    "basal_cell_carcinoma": [
        "Basal cell carcinoma, while rarely metastatic, causes significant local tissue destruction",
        "Without treatment, the lesion will continue to grow and invade surrounding tissue",
        "Mohs surgery is the gold standard with highest cure rate (99%)",
        "Treatment now prevents more extensive and costly interventions later",
        "This is a medically necessary procedure, not cosmetic"
    ],
    "squamous_cell_carcinoma": [
        "Squamous cell carcinoma has potential for metastasis and requires prompt treatment",
        "High-risk features present necessitate aggressive treatment approach",
        "Delayed treatment increases risk of lymph node involvement",
        "Per AAD guidelines, excision is the recommended treatment",
        "This procedure is essential to prevent disease progression"
    ],
    "actinic_keratosis": [
        "Actinic keratoses are pre-cancerous lesions with 10-15% progression to SCC",
        "Treatment prevents development of invasive squamous cell carcinoma",
        "Multiple lesions indicate field cancerization requiring treatment",
        "This is preventive care that reduces future cancer treatment costs",
        "Per AAD guidelines, treatment of AKs is medically indicated"
    ],
    "psoriasis": [
        "Psoriasis is a chronic inflammatory condition requiring ongoing management",
        "Untreated psoriasis leads to significant quality of life impairment",
        "Patient has failed conservative treatments, escalation is necessary",
        "Biologic therapy is indicated per AAD/NPF guidelines after conventional failure",
        "Cardiovascular and metabolic comorbidities necessitate disease control"
    ],
    "eczema": [
        "Atopic dermatitis causes significant morbidity and quality of life impact",
        "Patient has moderate-to-severe disease uncontrolled with topical therapy",
        "Systemic therapy is indicated per AAD guidelines",
        "Risk of skin infections and complications without adequate treatment",
        "This therapy is FDA-approved for this indication"
    ],
    "default": [
        "This condition requires medical intervention to prevent progression",
        "The proposed treatment is the standard of care for this diagnosis",
        "Conservative measures have been attempted and failed",
        "Delay in treatment will result in worsening outcomes",
        "Treatment is medically necessary, not elective or cosmetic"
    ]
}

# Denial-specific rebuttal templates
DENIAL_REBUTTALS = {
    DenialReason.MEDICAL_NECESSITY: """
The denial stating lack of medical necessity is incorrect based on the following clinical evidence:

1. CLINICAL PRESENTATION: {diagnosis} ({icd10_code}) is a serious condition requiring intervention.

2. STANDARD OF CARE: The requested procedure ({procedure}, CPT {cpt_code}) is the established standard of care per:
   - American Academy of Dermatology (AAD) guidelines
   - National Comprehensive Cancer Network (NCCN) recommendations
   - Peer-reviewed medical literature

3. CLINICAL DOCUMENTATION: The medical records clearly demonstrate:
   {clinical_evidence}

4. CONSEQUENCES OF DENIAL: Without this treatment:
   - Disease progression is likely
   - Patient outcomes will be negatively impacted
   - More costly interventions may be required in the future

5. SUPPORTING LITERATURE: Multiple peer-reviewed studies support the medical necessity of this treatment for this condition (citations available upon request).
""",

    DenialReason.EXPERIMENTAL: """
The denial categorizing this treatment as experimental/investigational is factually incorrect:

1. FDA APPROVAL: This treatment has FDA approval for the indicated condition since [approval date].

2. ESTABLISHED EFFICACY:
   - Multiple randomized controlled trials demonstrate efficacy
   - Treatment is included in major clinical guidelines (AAD, NCCN)
   - Widely used in clinical practice with proven outcomes

3. COVERAGE PRECEDENT: This treatment is routinely covered by major insurers including Medicare and other commercial plans.

4. GUIDELINE SUPPORT: Professional society guidelines recommend this treatment as:
   - First-line therapy for this condition
   - Medically appropriate based on clinical presentation

This is not experimental - it is evidence-based medicine meeting established standards of care.
""",

    DenialReason.COSMETIC: """
The denial categorizing this procedure as cosmetic is medically incorrect:

1. MEDICAL DIAGNOSIS: The patient has a diagnosed medical condition ({diagnosis}, ICD-10: {icd10_code}), not a cosmetic concern.

2. CLINICAL INDICATION: This procedure is being performed for:
   - Treatment of a pathological condition
   - Prevention of disease progression
   - Medical necessity, NOT aesthetic improvement

3. FUNCTIONAL IMPAIRMENT: This condition causes:
   - Physical symptoms (pain, bleeding, infection risk)
   - Functional limitations
   - Potential for serious medical complications

4. STANDARD OF CARE: The procedure ({procedure}) is the medically indicated treatment for this diagnosis per clinical guidelines.

5. DOCUMENTATION: Medical records clearly document the pathological nature of this condition, including biopsy results where applicable.

The cosmetic denial classification is inappropriate and should be reversed.
""",

    DenialReason.PRIOR_AUTH_REQUIRED: """
Regarding the denial for lack of prior authorization:

1. EMERGENCY/URGENT CARE: The clinical situation required immediate intervention. Per your policy:
   - Emergency/urgent services may be provided without prior authorization
   - Retrospective authorization is appropriate in these circumstances

2. MEDICAL NECESSITY: The treatment was medically necessary regardless of authorization status. Denial based on administrative technicality should not override clinical need.

3. PROVIDER GOOD FAITH: The provider acted in good faith to deliver necessary care to the patient.

4. ALTERNATIVE REQUEST: If prior authorization is required, we hereby request retrospective authorization based on the enclosed clinical documentation demonstrating medical necessity.

5. PATIENT PROTECTION: Denying coverage for necessary care based on administrative issues penalizes the patient for circumstances beyond their control.

We request coverage of this claim or expedited retrospective authorization review.
""",

    DenialReason.NOT_COVERED: """
The denial stating this service is not covered requires reconsideration:

1. POLICY REVIEW: Please provide the specific policy language excluding this service. Our review indicates:
   - The diagnosis ({diagnosis}) is a covered condition
   - The procedure ({procedure}) is a recognized medical treatment
   - No specific exclusion applies to this service

2. MEDICAL NECESSITY OVERRIDE: Even if typically excluded, medical necessity should warrant coverage:
   - This treatment is essential for the patient's health
   - No adequate covered alternative exists
   - Denial would result in harm to the patient

3. APPEAL TO COVERAGE: We request:
   - Specific citation of exclusion language
   - Medical director review of clinical circumstances
   - Exception consideration based on medical necessity

4. ALTERNATIVE BENEFITS: If excluded under one benefit category, please review under:
   - Medical benefits
   - Preventive care benefits
   - Disease management provisions

This service should be covered based on medical necessity and policy provisions.
""",

    DenialReason.CODING_ERROR: """
Regarding the denial due to coding issues:

1. CODE VERIFICATION: We have verified the codes submitted are correct:
   - Diagnosis: {icd10_code} - {diagnosis}
   - Procedure: {cpt_code} - {procedure}
   - Codes are appropriate for the services rendered

2. DOCUMENTATION SUPPORT: Medical records support the codes billed:
   - Clinical documentation matches code descriptions
   - Medical necessity is established for the codes used
   - No unbundling or upcoding has occurred

3. CORRECTION IF NEEDED: If specific coding errors are identified, please specify:
   - Which codes are incorrect
   - What codes you would accept
   - Specific documentation requirements

4. RESUBMISSION: We are resubmitting with:
   - Corrected claim (if errors identified)
   - Additional documentation supporting code selection
   - Request for specific feedback on coding concerns

Please process this corrected/clarified claim for payment.
"""
}


def get_condition_key(diagnosis: str) -> str:
    """Map diagnosis to condition key for argument lookup"""
    diagnosis_lower = diagnosis.lower()

    if "melanoma" in diagnosis_lower:
        return "melanoma"
    elif "basal cell" in diagnosis_lower or "bcc" in diagnosis_lower:
        return "basal_cell_carcinoma"
    elif "squamous cell" in diagnosis_lower or "scc" in diagnosis_lower:
        return "squamous_cell_carcinoma"
    elif "actinic" in diagnosis_lower or "keratosis" in diagnosis_lower:
        return "actinic_keratosis"
    elif "psoriasis" in diagnosis_lower:
        return "psoriasis"
    elif "eczema" in diagnosis_lower or "dermatitis" in diagnosis_lower:
        return "eczema"
    else:
        return "default"


def calculate_appeal_deadline(denial_date: str, appeal_level: AppealLevel) -> datetime:
    """Calculate appeal deadline based on denial date and level"""
    try:
        denial_dt = datetime.strptime(denial_date, "%Y-%m-%d")
    except:
        denial_dt = datetime.now()

    # Standard deadlines by level
    deadlines = {
        AppealLevel.FIRST_LEVEL: 180,  # 180 days for first appeal
        AppealLevel.SECOND_LEVEL: 60,   # 60 days for second level
        AppealLevel.EXTERNAL_REVIEW: 120,  # 4 months for external
        AppealLevel.STATE_INSURANCE: 180  # 6 months for state
    }

    days = deadlines.get(appeal_level, 180)
    return denial_dt + timedelta(days=days)


def estimate_success_likelihood(
    denial_reason: DenialReason,
    appeal_level: AppealLevel,
    has_clinical_notes: bool,
    has_supporting_evidence: bool,
    diagnosis: str
) -> int:
    """Estimate likelihood of appeal success (0-100)"""

    base_scores = {
        DenialReason.MEDICAL_NECESSITY: 65,
        DenialReason.CODING_ERROR: 80,
        DenialReason.PRIOR_AUTH_REQUIRED: 70,
        DenialReason.INCOMPLETE_INFO: 75,
        DenialReason.COSMETIC: 55,
        DenialReason.EXPERIMENTAL: 45,
        DenialReason.NOT_COVERED: 40,
        DenialReason.OUT_OF_NETWORK: 35,
        DenialReason.TIMELY_FILING: 30,
        DenialReason.DUPLICATE_CLAIM: 85,
        DenialReason.BENEFIT_EXHAUSTED: 25,
        DenialReason.PRE_EXISTING: 50,
        DenialReason.OTHER: 50
    }

    score = base_scores.get(denial_reason, 50)

    # Adjust for documentation
    if has_clinical_notes:
        score += 10
    if has_supporting_evidence:
        score += 10

    # Adjust for appeal level (success decreases with escalation)
    level_adjustments = {
        AppealLevel.FIRST_LEVEL: 0,
        AppealLevel.SECOND_LEVEL: -5,
        AppealLevel.EXTERNAL_REVIEW: -10,
        AppealLevel.STATE_INSURANCE: -15
    }
    score += level_adjustments.get(appeal_level, 0)

    # Adjust for diagnosis (cancer diagnoses have higher success)
    diagnosis_lower = diagnosis.lower()
    if any(term in diagnosis_lower for term in ["melanoma", "carcinoma", "cancer"]):
        score += 10

    return max(0, min(100, score))


def generate_appeal_letter(request: AppealRequest) -> AppealLetter:
    """Generate a complete appeal letter for an insurance denial"""

    import uuid
    appeal_id = f"APL-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    # Get condition-specific arguments
    condition_key = get_condition_key(request.diagnosis)
    medical_arguments = MEDICAL_NECESSITY_ARGUMENTS.get(condition_key, MEDICAL_NECESSITY_ARGUMENTS["default"])

    # Get denial-specific rebuttal
    rebuttal_template = DENIAL_REBUTTALS.get(
        request.denial_reason,
        DENIAL_REBUTTALS[DenialReason.MEDICAL_NECESSITY]
    )

    # Build clinical evidence section
    clinical_evidence_items = []
    if request.clinical_notes:
        clinical_evidence_items.append(f"   - Clinical findings: {request.clinical_notes}")
    if request.previous_treatments:
        clinical_evidence_items.append(f"   - Previous treatments attempted: {', '.join(request.previous_treatments)}")
    if request.supporting_evidence:
        for evidence in request.supporting_evidence:
            clinical_evidence_items.append(f"   - {evidence}")

    clinical_evidence = "\n".join(clinical_evidence_items) if clinical_evidence_items else "   - See attached medical records"

    # Format the rebuttal
    rebuttal = rebuttal_template.format(
        diagnosis=request.diagnosis,
        icd10_code=request.icd10_code,
        procedure=request.procedure,
        cpt_code=request.cpt_code,
        clinical_evidence=clinical_evidence
    )

    # Build the complete letter
    level_text = {
        AppealLevel.FIRST_LEVEL: "First Level Appeal",
        AppealLevel.SECOND_LEVEL: "Second Level Appeal",
        AppealLevel.EXTERNAL_REVIEW: "Request for External Review",
        AppealLevel.STATE_INSURANCE: "State Insurance Commissioner Complaint"
    }

    letter_content = f"""
{datetime.now().strftime("%B %d, %Y")}

{request.insurance_company}
Claims Appeal Department
[Insurance Company Address]

RE: {level_text.get(request.appeal_level, "Appeal")} - URGENT
    Patient: {request.patient_name}
    Date of Birth: {request.patient_dob}
    Policy Number: {request.policy_number}
    Claim Number: {request.claim_number}
    Date of Service: {request.date_of_service}
    Amount: ${request.claim_amount:,.2f}

Dear Appeals Review Committee:

I am writing to formally appeal the denial of coverage for {request.patient_name}'s claim referenced above. The claim was denied on {request.denial_date} with the stated reason of "{request.denial_reason.value.replace('_', ' ').title()}."

This denial is inappropriate and should be overturned for the following reasons:

PATIENT CLINICAL SUMMARY:
------------------------
The patient presented with {request.diagnosis} (ICD-10: {request.icd10_code}), requiring {request.procedure} (CPT: {request.cpt_code}). This treatment is medically necessary and represents the standard of care for this condition.

APPEAL ARGUMENTS:
----------------
{rebuttal}

MEDICAL NECESSITY JUSTIFICATION:
--------------------------------
{"".join([chr(10) + "• " + arg for arg in medical_arguments])}

REQUESTED ACTION:
----------------
Based on the clinical evidence and arguments presented above, we respectfully request that you:

1. Reverse the denial of claim {request.claim_number}
2. Authorize payment of ${request.claim_amount:,.2f} for the services rendered
3. Provide written confirmation of this appeal decision within 30 days

DOCUMENTATION ENCLOSED:
----------------------
• Complete medical records for dates of service
• Clinical notes and diagnostic findings
• Relevant imaging/pathology reports
• Peer-reviewed literature supporting medical necessity
• Provider credentials and certifications

Please note that under applicable state and federal regulations, you are required to respond to this appeal within the statutory timeframe. Failure to respond may be reported to the State Insurance Commissioner.

If you require any additional information, please contact our office immediately at the number below. We are prepared to participate in a peer-to-peer review if requested.

Thank you for your prompt attention to this matter.

Sincerely,

{request.provider_name}
NPI: {request.provider_npi}
{request.provider_address}

cc: Patient file
    State Insurance Commissioner (if applicable)
"""

    # Generate subject line
    subject_line = f"URGENT: {level_text.get(request.appeal_level)} - Claim #{request.claim_number} - {request.patient_name}"

    # Summary for tracking
    summary = f"Appeal for {request.diagnosis} treatment denial. Claim amount: ${request.claim_amount:,.2f}. Denial reason: {request.denial_reason.value}."

    # Key arguments list
    key_arguments = medical_arguments[:3] + [
        f"Standard of care procedure for {request.diagnosis}",
        "Clinical documentation supports medical necessity",
        "Denial is inconsistent with established guidelines"
    ]

    # Documents needed
    supporting_docs = [
        "Complete medical records",
        "Clinical notes and findings",
        "Pathology/biopsy reports (if applicable)",
        "Prior authorization documentation (if applicable)",
        "Relevant imaging studies",
        "Peer-reviewed literature citations"
    ]

    # Calculate deadline and success likelihood
    deadline = calculate_appeal_deadline(request.denial_date, request.appeal_level)
    success_likelihood = estimate_success_likelihood(
        request.denial_reason,
        request.appeal_level,
        bool(request.clinical_notes),
        bool(request.supporting_evidence),
        request.diagnosis
    )

    # Recommended next steps
    next_steps = [
        "Submit appeal via certified mail with return receipt",
        "Keep copies of all submitted documents",
        "Follow up within 30 days if no response",
        "Request peer-to-peer review if initial appeal denied",
        "Consider external review if second level appeal fails"
    ]

    if request.appeal_level == AppealLevel.SECOND_LEVEL:
        next_steps.insert(0, "Request expedited review citing patient harm from delay")
    elif request.appeal_level == AppealLevel.EXTERNAL_REVIEW:
        next_steps.insert(0, "File complaint with State Insurance Commissioner")

    return AppealLetter(
        appeal_id=appeal_id,
        generated_at=datetime.now(),
        appeal_level=request.appeal_level,
        letter_content=letter_content.strip(),
        subject_line=subject_line,
        summary=summary,
        key_arguments=key_arguments,
        supporting_documents_needed=supporting_docs,
        deadline=deadline,
        recommended_next_steps=next_steps,
        success_likelihood=success_likelihood
    )


def generate_appeal_for_analysis(
    analysis_data: Dict[str, Any],
    denial_info: Dict[str, Any],
    patient_info: Dict[str, Any],
    provider_info: Dict[str, Any]
) -> AppealLetter:
    """Generate appeal letter from analysis and denial data"""

    # Map denial reason text to enum
    denial_reason_text = denial_info.get("reason", "").lower()
    denial_reason = DenialReason.OTHER

    for reason in DenialReason:
        if reason.value in denial_reason_text or reason.value.replace("_", " ") in denial_reason_text:
            denial_reason = reason
            break

    # Build request from data
    request = AppealRequest(
        patient_name=patient_info.get("name", "Patient"),
        patient_dob=patient_info.get("dob", ""),
        patient_id=patient_info.get("id", ""),
        insurance_company=denial_info.get("insurance_company", "Insurance Company"),
        policy_number=denial_info.get("policy_number", ""),
        claim_number=denial_info.get("claim_number", ""),
        date_of_service=analysis_data.get("date", datetime.now().strftime("%Y-%m-%d")),
        denial_date=denial_info.get("denial_date", datetime.now().strftime("%Y-%m-%d")),
        denial_reason=denial_reason,
        denial_reason_text=denial_info.get("reason"),
        diagnosis=analysis_data.get("diagnosis", analysis_data.get("predicted_condition", "")),
        icd10_code=analysis_data.get("icd10_code", ""),
        procedure=analysis_data.get("procedure", ""),
        cpt_code=analysis_data.get("cpt_code", ""),
        claim_amount=float(denial_info.get("amount", 0)),
        provider_name=provider_info.get("name", "Provider"),
        provider_npi=provider_info.get("npi", ""),
        provider_address=provider_info.get("address", ""),
        appeal_level=AppealLevel(denial_info.get("appeal_level", "first_level")),
        clinical_notes=analysis_data.get("clinical_notes"),
        supporting_evidence=analysis_data.get("supporting_evidence", []),
        previous_treatments=analysis_data.get("previous_treatments", [])
    )

    return generate_appeal_letter(request)


# Utility function to get denial reasons list
def get_denial_reasons() -> List[Dict[str, str]]:
    """Get list of denial reasons for UI dropdown"""
    return [
        {"value": reason.value, "label": reason.value.replace("_", " ").title()}
        for reason in DenialReason
    ]


def get_appeal_levels() -> List[Dict[str, str]]:
    """Get list of appeal levels for UI dropdown"""
    return [
        {"value": level.value, "label": level.value.replace("_", " ").title()}
        for level in AppealLevel
    ]
