"""
Clinical Decision Support System
Provides evidence-based treatment recommendations, drug protocols, and management plans
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Drug interaction database (simplified - in production, use DrugBank API or similar)
DRUG_INTERACTIONS = {
    "isotretinoin": {
        "major": ["tetracycline", "minocycline", "doxycycline", "vitamin_a"],
        "moderate": ["alcohol", "corticosteroids"],
        "contraindications": ["pregnancy", "vitamin_a_supplements"]
    },
    "methotrexate": {
        "major": ["nsaids", "trimethoprim", "probenecid"],
        "moderate": ["alcohol", "folic_acid"],
        "contraindications": ["pregnancy", "liver_disease", "immunodeficiency"]
    },
    "biologics": {
        "major": ["live_vaccines", "immunosuppressants"],
        "moderate": ["nsaids"],
        "contraindications": ["active_infection", "tuberculosis", "hepatitis"]
    }
}

# Clinical protocols based on AAD, NCCN, and other guidelines
CLINICAL_PROTOCOLS = {
    "Melanoma": {
        "urgency": "URGENT",
        "timeline": "Refer within 2 weeks",
        "first_line": [
            {
                "action": "Immediate dermatology referral",
                "timeframe": "Within 2 weeks",
                "priority": "critical",
                "rationale": "Early intervention critical for survival"
            },
            {
                "action": "Wide local excision",
                "timeframe": "Within 4 weeks of diagnosis",
                "priority": "critical",
                "rationale": "Surgical margins depend on Breslow depth"
            },
            {
                "action": "Sentinel lymph node biopsy",
                "timeframe": "If depth >0.8mm or ulcerated",
                "priority": "high",
                "rationale": "Staging and prognosis determination"
            }
        ],
        "staging_required": True,
        "staging_tests": [
            "Complete skin exam",
            "Lymph node palpation",
            "CT chest/abdomen/pelvis (if stage IIB or higher)",
            "Brain MRI (if symptomatic or stage IV)",
            "LDH level",
            "CBC"
        ],
        "medications": [],  # Systemic therapy only after staging
        "follow_up": {
            "initial": "Every 3 months for 2 years",
            "long_term": "Every 6-12 months for 5 years, then annually",
            "imaging": "Based on stage"
        },
        "patient_education": [
            "Sun protection (SPF 30+)",
            "Monthly self-skin exams",
            "Avoid tanning beds",
            "Watch for new or changing lesions",
            "Family screening recommended"
        ],
        "red_flags": [
            "Rapid growth",
            "Bleeding",
            "New satellite lesions",
            "Palpable lymph nodes",
            "Systemic symptoms"
        ],
        "insurance_codes": {
            "icd10": "C43.9",
            "cpt": ["11600-11646", "38500", "38525"],
            "pre_auth_required": True,
            "documentation": "Photos, biopsy report, staging workup"
        }
    },

    "Basal Cell Carcinoma": {
        "urgency": "MODERATE",
        "timeline": "Refer within 4-6 weeks",
        "first_line": [
            {
                "action": "Dermatology consultation",
                "timeframe": "Within 4-6 weeks",
                "priority": "high",
                "rationale": "Determine optimal treatment modality"
            },
            {
                "action": "Excisional surgery or Mohs",
                "timeframe": "Within 8-12 weeks",
                "priority": "high",
                "rationale": "Cure rate >95% with complete excision"
            }
        ],
        "staging_required": False,
        "medications": [
            {
                "name": "Imiquimod 5% cream",
                "indication": "Superficial BCC only",
                "dosage": "Apply 5x/week for 6 weeks",
                "duration": "6 weeks",
                "response_rate": "75-80%",
                "contraindications": ["pregnancy", "autoimmune disease"],
                "side_effects": ["Local irritation", "redness", "crusting"],
                "cost_range": "$200-400",
                "insurance_coverage": "Usually covered"
            },
            {
                "name": "5-Fluorouracil 5% cream",
                "indication": "Superficial BCC, alternative",
                "dosage": "Apply 2x/day for 3-6 weeks",
                "duration": "3-6 weeks",
                "response_rate": "70-80%",
                "contraindications": ["pregnancy", "DPD deficiency"],
                "side_effects": ["Severe inflammation", "pain", "crusting"],
                "cost_range": "$50-150",
                "insurance_coverage": "Usually covered"
            }
        ],
        "surgical_options": [
            {
                "procedure": "Mohs micrographic surgery",
                "indication": "High-risk areas (face, ears, nose)",
                "cure_rate": "99%",
                "advantages": ["Tissue sparing", "real-time margin assessment"],
                "cost": "$1,500-3,000"
            },
            {
                "procedure": "Standard excision",
                "indication": "Low-risk areas",
                "cure_rate": "95%",
                "advantages": ["Faster", "less expensive"],
                "cost": "$500-1,500"
            },
            {
                "procedure": "Electrodesiccation & curettage",
                "indication": "Small, low-risk BCCs",
                "cure_rate": "90-92%",
                "advantages": ["Quick", "low cost"],
                "cost": "$300-800"
            }
        ],
        "follow_up": {
            "initial": "Every 6 months for 5 years",
            "long_term": "Annually thereafter",
            "imaging": "Not routinely required"
        },
        "patient_education": [
            "Sun protection critical",
            "20-50% risk of new BCC within 5 years",
            "Self-skin exams monthly",
            "Avoid tanning beds"
        ],
        "insurance_codes": {
            "icd10": "C44.91",
            "cpt": ["11640-11646", "17311-17315"],
            "pre_auth_required": False,
            "documentation": "Photos, clinical description"
        }
    },

    "Squamous Cell Carcinoma": {
        "urgency": "HIGH",
        "timeline": "Refer within 2-4 weeks",
        "first_line": [
            {
                "action": "Dermatology urgent referral",
                "timeframe": "Within 2-4 weeks",
                "priority": "high",
                "rationale": "Higher metastatic potential than BCC"
            },
            {
                "action": "Surgical excision",
                "timeframe": "Within 4-6 weeks",
                "priority": "high",
                "rationale": "Primary treatment modality"
            },
            {
                "action": "Risk stratification",
                "timeframe": "At diagnosis",
                "priority": "high",
                "rationale": "Identify high-risk features"
            }
        ],
        "risk_factors_high": [
            "Size >2cm",
            "Depth >4mm",
            "Perineural invasion",
            "Poor differentiation",
            "Immunosuppression",
            "Location on ear or lip"
        ],
        "staging_required": True,
        "staging_tests": [
            "Physical exam with lymph node palpation",
            "CT or MRI if high risk",
            "PET scan if metastasis suspected"
        ],
        "medications": [
            {
                "name": "Cemiplimab (Libtayo)",
                "indication": "Advanced/metastatic SCC",
                "dosage": "350mg IV every 3 weeks",
                "duration": "Until progression",
                "response_rate": "45-50%",
                "contraindications": ["autoimmune disease", "pregnancy"],
                "side_effects": ["Immune-related adverse events", "fatigue", "rash"],
                "cost_range": "$15,000+ per infusion",
                "insurance_coverage": "Pre-authorization required"
            }
        ],
        "follow_up": {
            "initial": "Every 3-6 months for 2 years",
            "long_term": "Every 6-12 months for years 3-5",
            "imaging": "If high-risk features present"
        },
        "patient_education": [
            "Aggressive sun protection",
            "Regular dermatology follow-up critical",
            "Watch for lymph node enlargement",
            "Immunosuppressed patients at higher risk"
        ],
        "insurance_codes": {
            "icd10": "C44.92",
            "cpt": ["11640-11646", "96567"],
            "pre_auth_required": True,
            "documentation": "Biopsy report, staging, photos"
        }
    },

    "Actinic Keratoses": {
        "urgency": "ROUTINE",
        "timeline": "Refer within 3 months",
        "first_line": [
            {
                "action": "Dermatology consultation",
                "timeframe": "Within 3 months",
                "priority": "moderate",
                "rationale": "Prevent progression to SCC (8-10% risk)"
            },
            {
                "action": "Field therapy vs lesion-directed",
                "timeframe": "Based on number/distribution",
                "priority": "moderate",
                "rationale": "Treat visible and subclinical lesions"
            }
        ],
        "medications": [
            {
                "name": "5-Fluorouracil 5% cream",
                "indication": "Multiple AKs (field therapy)",
                "dosage": "Apply 2x/day for 2-4 weeks",
                "duration": "2-4 weeks",
                "response_rate": "85-90%",
                "contraindications": ["pregnancy", "DPD deficiency"],
                "side_effects": ["Severe inflammation", "crusting", "pain"],
                "cost_range": "$50-150",
                "insurance_coverage": "Usually covered"
            },
            {
                "name": "Imiquimod 5% cream",
                "indication": "Face/scalp AKs",
                "dosage": "Apply 2x/week for 16 weeks OR 3x/week for 4 weeks",
                "duration": "4-16 weeks",
                "response_rate": "75-85%",
                "contraindications": ["autoimmune disease", "pregnancy"],
                "side_effects": ["Redness", "flaking", "flu-like symptoms"],
                "cost_range": "$200-400",
                "insurance_coverage": "Usually covered"
            },
            {
                "name": "Ingenol mebutate gel",
                "indication": "Localized AKs",
                "dosage": "0.015% (face) or 0.05% (trunk) for 2-3 days",
                "duration": "2-3 days",
                "response_rate": "75-80%",
                "contraindications": ["pregnancy"],
                "side_effects": ["Severe local reaction", "pain", "swelling"],
                "cost_range": "$300-500",
                "insurance_coverage": "Pre-authorization often required"
            },
            {
                "name": "Diclofenac 3% gel",
                "indication": "Mild AKs",
                "dosage": "Apply 2x/day for 60-90 days",
                "duration": "60-90 days",
                "response_rate": "50-60%",
                "contraindications": ["NSAID allergy", "aspirin-sensitive asthma"],
                "side_effects": ["Mild irritation", "dryness"],
                "cost_range": "$100-200",
                "insurance_coverage": "Usually covered"
            }
        ],
        "procedures": [
            {
                "name": "Cryotherapy (liquid nitrogen)",
                "indication": "Individual lesions",
                "cure_rate": "67-99%",
                "advantages": ["Quick", "office-based", "no anesthesia"],
                "disadvantages": ["Hypopigmentation", "pain", "scarring"],
                "cost": "$100-300 per session"
            },
            {
                "name": "Photodynamic therapy (PDT)",
                "indication": "Multiple AKs on face/scalp",
                "cure_rate": "85-90%",
                "advantages": ["Excellent cosmetic outcome", "field therapy"],
                "disadvantages": ["Pain during treatment", "expensive"],
                "cost": "$1,000-2,000"
            },
            {
                "name": "Chemical peel",
                "indication": "Multiple facial AKs",
                "cure_rate": "75-80%",
                "advantages": ["Improves photoaging", "field therapy"],
                "disadvantages": ["Downtime", "risk of scarring"],
                "cost": "$500-1,500"
            }
        ],
        "follow_up": {
            "initial": "Every 6-12 months",
            "long_term": "Annually if stable",
            "imaging": "Not required"
        },
        "patient_education": [
            "Daily sunscreen SPF 30+",
            "Avoid peak sun hours",
            "Protective clothing",
            "Regular skin checks",
            "Report new/changing lesions"
        ],
        "insurance_codes": {
            "icd10": "L57.0",
            "cpt": ["17000", "17003-17004", "96567"],
            "pre_auth_required": False,
            "documentation": "Photos, clinical description"
        }
    },

    "Atopic Dermatitis (Eczema)": {
        "urgency": "ROUTINE",
        "timeline": "Manage in primary care or refer if severe",
        "severity_assessment": {
            "mild": "BSA <10%, minimal impact on life",
            "moderate": "BSA 10-50%, some impact on life",
            "severe": "BSA >50%, significant impact on life"
        },
        "first_line": [
            {
                "action": "Skin barrier repair",
                "timeframe": "Immediate",
                "priority": "high",
                "rationale": "Foundation of all eczema treatment"
            },
            {
                "action": "Topical corticosteroids",
                "timeframe": "For flares",
                "priority": "high",
                "rationale": "Anti-inflammatory first-line"
            },
            {
                "action": "Trigger identification",
                "timeframe": "Ongoing",
                "priority": "moderate",
                "rationale": "Prevent flares"
            }
        ],
        "medications": [
            {
                "name": "Emollients (Cerave, Cetaphil, Vanicream)",
                "indication": "All severity levels",
                "dosage": "Apply liberally 2-3x daily",
                "duration": "Daily, indefinitely",
                "response_rate": "Essential baseline",
                "contraindications": [],
                "side_effects": ["Rare: contact dermatitis"],
                "cost_range": "$10-30/month",
                "insurance_coverage": "OTC - not covered"
            },
            {
                "name": "Hydrocortisone 1-2.5%",
                "indication": "Mild eczema, face",
                "dosage": "Apply 1-2x daily to affected areas",
                "duration": "Up to 2 weeks per flare",
                "response_rate": "60-70%",
                "contraindications": ["skin infection", "rosacea"],
                "side_effects": ["Skin atrophy (with prolonged use)", "telangiectasia"],
                "cost_range": "$5-20",
                "insurance_coverage": "OTC - not covered"
            },
            {
                "name": "Triamcinolone 0.1% cream",
                "indication": "Moderate eczema, body",
                "dosage": "Apply 1-2x daily",
                "duration": "2-4 weeks",
                "response_rate": "75-85%",
                "contraindications": ["infection", "face (avoid)"],
                "side_effects": ["Atrophy", "striae", "purpura"],
                "cost_range": "$15-40",
                "insurance_coverage": "Usually covered"
            },
            {
                "name": "Clobetasol 0.05% cream",
                "indication": "Severe eczema flares",
                "dosage": "Apply 1-2x daily",
                "duration": "Max 2 weeks",
                "response_rate": "85-90%",
                "contraindications": ["face", "genital area", "infection"],
                "side_effects": ["Significant atrophy risk", "HPA suppression"],
                "cost_range": "$30-80",
                "insurance_coverage": "Usually covered"
            },
            {
                "name": "Tacrolimus 0.1% ointment",
                "indication": "Steroid-sparing, face/genital areas",
                "dosage": "Apply 2x daily",
                "duration": "Long-term safe",
                "response_rate": "70-80%",
                "contraindications": ["immunodeficiency", "active infection"],
                "side_effects": ["Burning (temporary)", "increased infection risk"],
                "cost_range": "$300-500",
                "insurance_coverage": "Often requires pre-auth"
            },
            {
                "name": "Dupilumab (Dupixent)",
                "indication": "Moderate-severe eczema uncontrolled with topicals",
                "dosage": "300mg SC every 2 weeks",
                "duration": "Long-term",
                "response_rate": "60-70% achieve clear/almost clear",
                "contraindications": ["hypersensitivity"],
                "side_effects": ["Injection site reactions", "conjunctivitis", "oral herpes"],
                "cost_range": "$3,000+/month",
                "insurance_coverage": "Pre-authorization required, step therapy"
            }
        ],
        "adjunct_therapies": [
            "Bleach baths (1/4-1/2 cup per full tub, 2x/week)",
            "Antihistamines for itch (hydroxyzine, doxepin)",
            "Wet wrap therapy for severe flares",
            "Phototherapy (UVB) for refractory cases"
        ],
        "follow_up": {
            "initial": "2-4 weeks after starting treatment",
            "long_term": "Every 3-6 months if stable",
            "imaging": "Not required"
        },
        "patient_education": [
            "Moisturize immediately after bathing",
            "Use fragrance-free products",
            "Avoid hot water",
            "Identify and avoid triggers",
            "Don't scratch - use cool compress",
            "Cotton clothing preferred"
        ],
        "insurance_codes": {
            "icd10": "L20.9",
            "cpt": ["99213", "96567"],
            "pre_auth_required": True,
            "documentation": "SCORAD or EASI severity score, photos, failed treatments"
        }
    },

    "Psoriasis": {
        "urgency": "ROUTINE",
        "timeline": "Refer to dermatology within 1-3 months",
        "severity_assessment": {
            "mild": "BSA <3%",
            "moderate": "BSA 3-10%",
            "severe": "BSA >10% or significant impact"
        },
        "first_line": [
            {
                "action": "Dermatology consultation",
                "timeframe": "Within 1-3 months",
                "priority": "moderate",
                "rationale": "Confirm diagnosis, assess severity"
            },
            {
                "action": "Topical therapy",
                "timeframe": "Immediate for mild cases",
                "priority": "high",
                "rationale": "First-line for limited disease"
            },
            {
                "action": "Psoriatic arthritis screening",
                "timeframe": "At diagnosis",
                "priority": "high",
                "rationale": "30% develop joint disease"
            }
        ],
        "medications": [
            {
                "name": "Calcipotriene 0.005% cream",
                "indication": "Mild-moderate plaque psoriasis",
                "dosage": "Apply 2x daily",
                "duration": "Long-term safe",
                "response_rate": "60-70%",
                "contraindications": ["hypercalcemia"],
                "side_effects": ["Irritation", "hypercalcemia (if >100g/week)"],
                "cost_range": "$200-400",
                "insurance_coverage": "Usually covered"
            },
            {
                "name": "Betamethasone + Calcipotriene (Taclonex)",
                "indication": "Moderate plaque psoriasis",
                "dosage": "Apply once daily",
                "duration": "Up to 4 weeks",
                "response_rate": "75-80%",
                "contraindications": ["hypercalcemia", "severe renal/hepatic impairment"],
                "side_effects": ["Atrophy with prolonged use", "irritation"],
                "cost_range": "$400-600",
                "insurance_coverage": "Usually covered with prior auth"
            },
            {
                "name": "Methotrexate",
                "indication": "Moderate-severe psoriasis",
                "dosage": "7.5-25mg PO weekly + folic acid 1mg daily",
                "duration": "Long-term",
                "response_rate": "60-70%",
                "contraindications": ["pregnancy", "liver disease", "alcohol abuse"],
                "side_effects": ["Hepatotoxicity", "myelosuppression", "nausea"],
                "monitoring": "CBC, CMP every 3 months",
                "cost_range": "$10-50/month",
                "insurance_coverage": "Covered"
            },
            {
                "name": "Apremilast (Otezla)",
                "indication": "Moderate psoriasis",
                "dosage": "30mg PO twice daily",
                "duration": "Long-term",
                "response_rate": "30-40% PASI-75",
                "contraindications": ["depression risk"],
                "side_effects": ["Diarrhea", "nausea", "depression", "weight loss"],
                "cost_range": "$4,000+/month",
                "insurance_coverage": "Pre-auth required, step therapy"
            },
            {
                "name": "Adalimumab (Humira)",
                "indication": "Moderate-severe psoriasis",
                "dosage": "80mg SC initial, then 40mg every other week",
                "duration": "Long-term",
                "response_rate": "70-80% PASI-75",
                "contraindications": ["active TB", "CHF", "demyelinating disease"],
                "side_effects": ["Infection risk", "injection site reactions", "lymphoma risk"],
                "monitoring": "TB test before starting, annual flu vaccine",
                "cost_range": "$6,000+/month",
                "insurance_coverage": "Pre-auth required, step therapy"
            },
            {
                "name": "Guselkumab (Tremfya)",
                "indication": "Moderate-severe plaque psoriasis",
                "dosage": "100mg SC at weeks 0, 4, then every 8 weeks",
                "duration": "Long-term",
                "response_rate": "85-90% PASI-75",
                "contraindications": ["active infection"],
                "side_effects": ["Infection risk", "headache", "arthralgia"],
                "cost_range": "$7,000+/injection",
                "insurance_coverage": "Pre-auth required, step therapy"
            }
        ],
        "step_therapy_pathway": [
            "Step 1: Topical steroids + vitamin D analogs (6-8 weeks)",
            "Step 2: Phototherapy (UVB) if inadequate response",
            "Step 3: Methotrexate or apremilast",
            "Step 4: Biologics (TNF-inhibitors first, then IL-17/IL-23 inhibitors)"
        ],
        "follow_up": {
            "initial": "Every 4-8 weeks during treatment adjustment",
            "long_term": "Every 3-6 months if stable on systemic",
            "imaging": "Not required"
        },
        "patient_education": [
            "Not contagious",
            "Stress management helps",
            "Avoid triggers (smoking, alcohol, stress)",
            "Moisturize daily",
            "Sun exposure helps (but use sunscreen)",
            "Screen for psoriatic arthritis symptoms"
        ],
        "insurance_codes": {
            "icd10": "L40.0",
            "cpt": ["99213", "96567", "96372"],
            "pre_auth_required": True,
            "documentation": "BSA %, PASI score, photos, failed treatments for biologics"
        }
    },

    "Acne": {
        "urgency": "ROUTINE",
        "timeline": "Can manage in primary care",
        "severity_assessment": {
            "mild": "Few comedones, <20 papules",
            "moderate": "Comedones + 20-100 papules/pustules",
            "severe": ">100 lesions or nodules/cysts"
        },
        "first_line": [
            {
                "action": "Assess severity and type",
                "timeframe": "At presentation",
                "priority": "high",
                "rationale": "Determines treatment approach"
            },
            {
                "action": "Topical therapy",
                "timeframe": "Immediate",
                "priority": "high",
                "rationale": "Foundation of acne treatment"
            }
        ],
        "medications": [
            {
                "name": "Benzoyl peroxide 2.5-5%",
                "indication": "Mild-moderate acne",
                "dosage": "Apply once daily (evening), increase to 2x if tolerated",
                "duration": "Ongoing",
                "response_rate": "60-70%",
                "contraindications": [],
                "side_effects": ["Dryness", "irritation", "bleaching of fabrics"],
                "cost_range": "$5-20",
                "insurance_coverage": "OTC - not covered"
            },
            {
                "name": "Tretinoin 0.025-0.1% cream",
                "indication": "Comedonal and inflammatory acne",
                "dosage": "Apply nightly, start low strength",
                "duration": "Ongoing (maintenance)",
                "response_rate": "70-80%",
                "contraindications": ["pregnancy"],
                "side_effects": ["Dryness", "peeling", "photosensitivity", "purge initially"],
                "cost_range": "$75-200",
                "insurance_coverage": "Usually covered"
            },
            {
                "name": "Adapalene 0.1% gel",
                "indication": "Acne (better tolerated than tretinoin)",
                "dosage": "Apply nightly",
                "duration": "Ongoing",
                "response_rate": "65-75%",
                "contraindications": ["pregnancy"],
                "side_effects": ["Less irritating than tretinoin", "dryness"],
                "cost_range": "$15 (OTC 0.1%) - $200 (Rx 0.3%)",
                "insurance_coverage": "0.1% OTC, 0.3% usually covered"
            },
            {
                "name": "Clindamycin 1% lotion",
                "indication": "Inflammatory acne",
                "dosage": "Apply 2x daily",
                "duration": "Use with benzoyl peroxide to prevent resistance",
                "response_rate": "60-70%",
                "contraindications": ["history of C. diff"],
                "side_effects": ["Dryness", "antibiotic resistance if used alone"],
                "cost_range": "$50-150",
                "insurance_coverage": "Usually covered"
            },
            {
                "name": "Doxycycline 50-100mg",
                "indication": "Moderate-severe inflammatory acne",
                "dosage": "100mg PO daily or twice daily",
                "duration": "3-6 months, then reassess",
                "response_rate": "70-80%",
                "contraindications": ["pregnancy", "children <8 years"],
                "side_effects": ["Photosensitivity", "GI upset", "vaginal yeast"],
                "cost_range": "$10-30/month",
                "insurance_coverage": "Covered"
            },
            {
                "name": "Spironolactone 50-200mg",
                "indication": "Hormonal acne in women",
                "dosage": "Start 25-50mg daily, titrate to 100-200mg",
                "duration": "Long-term",
                "response_rate": "70-80% in hormonal acne",
                "contraindications": ["pregnancy", "hyperkalemia", "renal impairment"],
                "side_effects": ["Irregular periods", "breast tenderness", "hyperkalemia"],
                "monitoring": "K+ and BP at baseline and 4 weeks",
                "cost_range": "$10-30/month",
                "insurance_coverage": "Covered (off-label)"
            },
            {
                "name": "Isotretinoin 0.5-1mg/kg/day",
                "indication": "Severe nodulocystic acne or failed conventional therapy",
                "dosage": "Start 20-40mg daily, adjust based on weight",
                "duration": "5-6 months (cumulative dose 120-150mg/kg)",
                "response_rate": "85-95%",
                "contraindications": ["pregnancy", "liver disease", "hyperlipidemia"],
                "side_effects": ["Teratogenicity", "dry skin/lips", "myalgias", "depression"],
                "monitoring": "iPledge enrollment, monthly pregnancy tests, lipids, LFTs",
                "cost_range": "$200-500/month",
                "insurance_coverage": "Pre-auth required, strict regulations"
            }
        ],
        "treatment_algorithm": [
            "Mild comedonal: Retinoid alone",
            "Mild inflammatory: Retinoid + benzoyl peroxide or topical antibiotic",
            "Moderate: Retinoid + oral antibiotic + benzoyl peroxide",
            "Severe or scarring: Consider isotretinoin early",
            "Hormonal (women): Spironolactone ± OCPs"
        ],
        "follow_up": {
            "initial": "6-8 weeks",
            "long_term": "Every 3 months if on oral therapy",
            "imaging": "Not required"
        },
        "patient_education": [
            "Takes 6-12 weeks to see improvement",
            "Don't pick or squeeze lesions",
            "Use non-comedogenic products",
            "Gentle cleansing twice daily",
            "Sunscreen (many treatments increase photosensitivity)",
            "Diet: low glycemic may help, dairy may worsen"
        ],
        "insurance_codes": {
            "icd10": "L70.0",
            "cpt": ["99213"],
            "pre_auth_required": False,
            "documentation": "Severity, prior treatments, photos"
        }
    }
}


def get_clinical_decision_support(condition: str, severity: Optional[str] = None,
                                   patient_factors: Optional[Dict] = None) -> Dict:
    """
    Get comprehensive clinical decision support for a given condition

    Args:
        condition: The diagnosed skin condition
        severity: Optional severity level (mild, moderate, severe)
        patient_factors: Dict with patient-specific factors (age, pregnancy, medications, etc.)

    Returns:
        Dict with treatment recommendations, protocols, and decision support
    """
    if condition not in CLINICAL_PROTOCOLS:
        return {
            "error": "No protocol available for this condition",
            "recommendation": "Refer to dermatology for evaluation"
        }

    protocol = CLINICAL_PROTOCOLS[condition].copy()

    # Add drug interaction checks if patient is on other medications
    if patient_factors and 'current_medications' in patient_factors:
        protocol['drug_interactions'] = check_drug_interactions(
            protocol.get('medications', []),
            patient_factors['current_medications']
        )

    # Customize based on patient factors
    if patient_factors:
        protocol['customized_recommendations'] = customize_recommendations(
            protocol, patient_factors
        )

    return protocol


def check_drug_interactions(recommended_drugs: List[Dict],
                            current_medications: List[str]) -> List[Dict]:
    """
    Check for drug interactions between recommended and current medications

    Returns:
        List of interaction warnings
    """
    interactions = []

    for drug in recommended_drugs:
        drug_name = drug.get('name', '').lower()

        # Check against known interactions
        for med in current_medications:
            med_lower = med.lower()

            # Check each drug in our interaction database
            for check_drug, interaction_data in DRUG_INTERACTIONS.items():
                if check_drug in drug_name:
                    if any(interact in med_lower for interact in interaction_data.get('major', [])):
                        interactions.append({
                            "severity": "MAJOR",
                            "drug1": drug.get('name'),
                            "drug2": med,
                            "warning": f"Major interaction between {drug.get('name')} and {med}. Consult pharmacist/physician.",
                            "action": "Consider alternative or monitor closely"
                        })
                    elif any(interact in med_lower for interact in interaction_data.get('moderate', [])):
                        interactions.append({
                            "severity": "MODERATE",
                            "drug1": drug.get('name'),
                            "drug2": med,
                            "warning": f"Moderate interaction possible. Monitor patient.",
                            "action": "Monitor for adverse effects"
                        })

    return interactions


def customize_recommendations(protocol: Dict, patient_factors: Dict) -> Dict:
    """
    Customize treatment recommendations based on patient-specific factors
    """
    customized = {
        "warnings": [],
        "adjusted_medications": [],
        "additional_considerations": []
    }

    # Check pregnancy
    if patient_factors.get('pregnant') or patient_factors.get('planning_pregnancy'):
        customized['warnings'].append({
            "type": "CRITICAL",
            "message": "PREGNANCY: Avoid all retinoids, methotrexate, and certain other medications"
        })
        # Filter out contraindicated medications
        safe_meds = []
        for med in protocol.get('medications', []):
            if 'pregnancy' not in [c.lower() for c in med.get('contraindications', [])]:
                safe_meds.append(med)
            else:
                customized['warnings'].append({
                    "type": "WARNING",
                    "message": f"{med.get('name')} is contraindicated in pregnancy"
                })
        customized['adjusted_medications'] = safe_meds

    # Check age
    age = patient_factors.get('age')
    if age and age < 18:
        customized['additional_considerations'].append(
            "Pediatric dosing may differ. Consider weight-based dosing."
        )
    if age and age > 65:
        customized['additional_considerations'].append(
            "Elderly patients may have increased sensitivity to medications. Start low, go slow."
        )

    # Check renal/hepatic function
    if patient_factors.get('renal_impairment'):
        customized['warnings'].append({
            "type": "WARNING",
            "message": "Renal impairment: Adjust doses for renally-cleared medications"
        })

    if patient_factors.get('liver_disease'):
        customized['warnings'].append({
            "type": "CRITICAL",
            "message": "Liver disease: Avoid methotrexate and other hepatotoxic agents"
        })

    # Check immunosuppression
    if patient_factors.get('immunosuppressed'):
        customized['warnings'].append({
            "type": "CRITICAL",
            "message": "Immunosuppression: Avoid live vaccines and additional immunosuppressants. Increased infection risk with biologics."
        })

    return customized


def generate_insurance_pre_authorization(condition: str, treatment: str,
                                         patient_data: Dict) -> Dict:
    """
    Generate pre-authorization documentation for insurance

    Returns:
        Dict with all necessary documentation for pre-auth
    """
    protocol = CLINICAL_PROTOCOLS.get(condition, {})
    insurance_info = protocol.get('insurance_codes', {})

    pre_auth = {
        "required": insurance_info.get('pre_auth_required', False),
        "diagnosis_codes": {
            "primary": insurance_info.get('icd10'),
            "description": condition
        },
        "procedure_codes": insurance_info.get('cpt', []),
        "medical_necessity": {
            "diagnosis": condition,
            "severity": patient_data.get('severity', 'moderate'),
            "failed_treatments": patient_data.get('prior_treatments', []),
            "justification": f"Evidence-based treatment per AAD/NCCN guidelines for {condition}"
        },
        "required_documentation": insurance_info.get('documentation', ''),
        "estimated_cost": patient_data.get('treatment_cost'),
        "prior_authorization_template": generate_prior_auth_letter(condition, treatment, patient_data)
    }

    return pre_auth


def generate_prior_auth_letter(condition: str, treatment: str, patient_data: Dict) -> str:
    """
    Generate a template letter for insurance pre-authorization
    """
    template = f"""
PRIOR AUTHORIZATION REQUEST

Patient: {patient_data.get('name', '[Patient Name]')}
DOB: {patient_data.get('dob', '[DOB]')}
Insurance ID: {patient_data.get('insurance_id', '[Insurance ID]')}

DIAGNOSIS: {condition} (ICD-10: {CLINICAL_PROTOCOLS.get(condition, {}).get('insurance_codes', {}).get('icd10')})

REQUESTED TREATMENT: {treatment}

MEDICAL NECESSITY:
This patient has been diagnosed with {condition} based on clinical examination and [biopsy/diagnostic testing].

Severity: {patient_data.get('severity', 'Moderate')}

Prior Treatments Attempted:
{chr(10).join(['- ' + tx for tx in patient_data.get('prior_treatments', ['Standard topical therapies'])])}

The requested treatment is evidence-based per American Academy of Dermatology guidelines and is medically necessary for this patient's condition.

Supporting Documentation:
- Clinical photographs
- Pathology report (if applicable)
- Treatment history
- Severity assessment

Prescribing Physician: [Physician Name]
NPI: [NPI Number]
Date: {datetime.now().strftime('%Y-%m-%d')}
    """

    return template.strip()
