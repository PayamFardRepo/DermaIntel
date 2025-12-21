"""
Over-the-Counter (OTC) Medication Recommendations Module

Provides educational OTC medication suggestions for appropriate skin conditions.
This module includes safety guardrails to ensure recommendations are only
provided for conditions where OTC treatment is appropriate.

IMPORTANT REGULATORY NOTES:
- These are educational suggestions, NOT medical prescriptions
- Recommendations are suppressed for malignancies, severe conditions, and infections
- All output includes appropriate disclaimers
- This does not constitute medical advice
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ConditionCategory(str, Enum):
    """Categories of skin conditions for OTC recommendation purposes"""
    MALIGNANT = "malignant"  # No OTC - needs professional care
    POTENTIALLY_MALIGNANT = "potentially_malignant"  # No OTC - needs evaluation
    INFECTIOUS = "infectious"  # Limited OTC - may need Rx
    INFLAMMATORY_MILD = "inflammatory_mild"  # OTC appropriate
    INFLAMMATORY_MODERATE = "inflammatory_moderate"  # Limited OTC + see doctor
    BURN_MINOR = "burn_minor"  # OTC appropriate
    BURN_SEVERE = "burn_severe"  # No OTC - needs professional care
    BENIGN = "benign"  # OTC may help with symptoms
    UNKNOWN = "unknown"  # Conservative approach


@dataclass
class OTCRecommendation:
    """Single OTC recommendation"""
    category: str  # Drug category (e.g., "Topical Corticosteroids")
    examples: List[str]  # Generic examples (not brand names)
    usage: str  # How to use
    duration: str  # How long to use
    warnings: List[str]  # Important warnings
    contraindications: List[str]  # When NOT to use


# =============================================================================
# OTC RECOMMENDATION DATABASE
# Organized by condition type with appropriate safety information
# =============================================================================

OTC_DATABASE: Dict[str, List[Dict[str, Any]]] = {
    # -------------------------------------------------------------------------
    # INFLAMMATORY CONDITIONS - Mild
    # -------------------------------------------------------------------------
    "eczema_mild": [
        {
            "category": "Topical Corticosteroids (Low Potency)",
            "examples": ["Hydrocortisone 1% cream or ointment"],
            "usage": "Apply a thin layer to affected areas 1-2 times daily",
            "duration": "Up to 7 days. If no improvement, consult a healthcare provider",
            "warnings": [
                "Avoid use on face, groin, or armpits for more than a few days",
                "Do not use on broken or infected skin",
                "Stop use if irritation or burning occurs"
            ],
            "contraindications": [
                "Known allergy to hydrocortisone",
                "Skin infections (bacterial, viral, fungal)",
                "Children under 2 years without medical advice"
            ]
        },
        {
            "category": "Moisturizers/Emollients",
            "examples": [
                "Ceramide-containing creams",
                "Petrolatum-based ointments",
                "Colloidal oatmeal lotions"
            ],
            "usage": "Apply liberally and frequently, especially after bathing while skin is still damp",
            "duration": "Ongoing daily use recommended for eczema-prone skin",
            "warnings": [
                "Choose fragrance-free products",
                "Patch test new products on small area first"
            ],
            "contraindications": [
                "Known allergy to any ingredients"
            ]
        },
        {
            "category": "Anti-itch Treatments",
            "examples": [
                "Pramoxine-containing lotions",
                "Colloidal oatmeal baths",
                "Calamine lotion"
            ],
            "usage": "Apply to itchy areas as needed, up to 4 times daily",
            "duration": "As needed for symptom relief",
            "warnings": [
                "Oral antihistamines may cause drowsiness",
                "Do not apply to broken skin"
            ],
            "contraindications": []
        }
    ],

    "contact_dermatitis": [
        {
            "category": "Topical Corticosteroids (Low Potency)",
            "examples": ["Hydrocortisone 1% cream"],
            "usage": "Apply to affected area 2-3 times daily",
            "duration": "Up to 7 days",
            "warnings": [
                "Identify and avoid the triggering substance",
                "Do not use on face for extended periods"
            ],
            "contraindications": [
                "Infected skin",
                "Known corticosteroid allergy"
            ]
        },
        {
            "category": "Barrier Creams",
            "examples": [
                "Dimethicone-based barrier creams",
                "Zinc oxide ointments"
            ],
            "usage": "Apply before potential exposure to irritants",
            "duration": "As needed for prevention",
            "warnings": [],
            "contraindications": []
        },
        {
            "category": "Oral Antihistamines",
            "examples": [
                "Diphenhydramine (sedating)",
                "Cetirizine (non-sedating)",
                "Loratadine (non-sedating)"
            ],
            "usage": "Take as directed on package for itch relief",
            "duration": "As needed, typically up to 7 days",
            "warnings": [
                "Sedating antihistamines may cause drowsiness - avoid driving",
                "Check for drug interactions with other medications"
            ],
            "contraindications": [
                "Glaucoma (for some antihistamines)",
                "Prostate enlargement (for some antihistamines)"
            ]
        }
    ],

    "psoriasis_mild": [
        {
            "category": "Coal Tar Preparations",
            "examples": [
                "Coal tar shampoo (for scalp)",
                "Coal tar cream or ointment"
            ],
            "usage": "Apply to affected areas once daily, typically at bedtime",
            "duration": "Ongoing as directed; improvement may take several weeks",
            "warnings": [
                "May stain clothing and bedding",
                "Avoid sun exposure on treated areas (photosensitivity)",
                "Has distinctive odor"
            ],
            "contraindications": [
                "Infected or broken skin",
                "Use on face or genitals"
            ]
        },
        {
            "category": "Salicylic Acid",
            "examples": [
                "Salicylic acid 2-3% cream or lotion",
                "Salicylic acid shampoo"
            ],
            "usage": "Apply to scaly areas to help remove scales",
            "duration": "As directed; helps prepare skin for other treatments",
            "warnings": [
                "May cause skin irritation",
                "Do not use on large body areas"
            ],
            "contraindications": [
                "Aspirin allergy",
                "Use in children without medical advice"
            ]
        },
        {
            "category": "Moisturizers",
            "examples": [
                "Thick creams or ointments",
                "Urea-containing moisturizers (for thick scales)"
            ],
            "usage": "Apply liberally after bathing",
            "duration": "Ongoing daily use",
            "warnings": [],
            "contraindications": []
        }
    ],

    "seborrheic_dermatitis": [
        {
            "category": "Antifungal Shampoos",
            "examples": [
                "Ketoconazole 1% shampoo",
                "Selenium sulfide shampoo",
                "Zinc pyrithione shampoo"
            ],
            "usage": "Use 2-3 times weekly; leave on scalp for 3-5 minutes before rinsing",
            "duration": "Ongoing for control; condition often recurs",
            "warnings": [
                "May cause dryness or irritation",
                "Selenium sulfide may discolor light-colored hair"
            ],
            "contraindications": [
                "Known allergy to ingredients"
            ]
        },
        {
            "category": "Topical Corticosteroids (Low Potency)",
            "examples": ["Hydrocortisone 1% cream for face/body patches"],
            "usage": "Apply thin layer to affected areas 1-2 times daily",
            "duration": "Short-term use only (up to 7 days on face)",
            "warnings": [
                "Use lower potency on face",
                "Long-term use can thin skin"
            ],
            "contraindications": []
        }
    ],

    "acne_mild": [
        {
            "category": "Benzoyl Peroxide",
            "examples": [
                "Benzoyl peroxide 2.5% gel or cream",
                "Benzoyl peroxide 5% wash"
            ],
            "usage": "Apply once daily, increasing to twice daily as tolerated",
            "duration": "Ongoing; improvement may take 4-6 weeks",
            "warnings": [
                "Start with lower concentration to minimize irritation",
                "Can bleach fabrics and hair",
                "May cause dryness and peeling initially"
            ],
            "contraindications": [
                "Known allergy to benzoyl peroxide"
            ]
        },
        {
            "category": "Salicylic Acid",
            "examples": [
                "Salicylic acid 0.5-2% cleanser",
                "Salicylic acid treatment pads"
            ],
            "usage": "Use 1-2 times daily as part of cleansing routine",
            "duration": "Ongoing for maintenance",
            "warnings": [
                "May cause mild stinging",
                "Use sunscreen as may increase sun sensitivity"
            ],
            "contraindications": [
                "Aspirin allergy"
            ]
        },
        {
            "category": "Adapalene (Retinoid)",
            "examples": ["Adapalene 0.1% gel (now available OTC)"],
            "usage": "Apply thin layer to entire face once daily at bedtime",
            "duration": "Ongoing; may take 8-12 weeks to see full benefit",
            "warnings": [
                "Causes sun sensitivity - use sunscreen daily",
                "Initial irritation, dryness, and peeling are common",
                "Avoid waxing treated areas"
            ],
            "contraindications": [
                "Pregnancy or planning pregnancy",
                "Eczema or very sensitive skin"
            ]
        }
    ],

    "dry_skin": [
        {
            "category": "Emollient Moisturizers",
            "examples": [
                "Petrolatum-based ointments",
                "Ceramide-containing creams",
                "Glycerin-based lotions"
            ],
            "usage": "Apply liberally after bathing while skin is still damp",
            "duration": "Daily ongoing use",
            "warnings": [
                "Choose fragrance-free for sensitive skin"
            ],
            "contraindications": []
        },
        {
            "category": "Humectants",
            "examples": [
                "Urea 10-20% cream (for very dry/rough skin)",
                "Lactic acid lotion",
                "Hyaluronic acid serum"
            ],
            "usage": "Apply to dry areas 1-2 times daily",
            "duration": "Ongoing as needed",
            "warnings": [
                "Urea and lactic acid may sting on cracked skin"
            ],
            "contraindications": []
        }
    ],

    "urticaria_hives": [
        {
            "category": "Oral Antihistamines (Non-sedating)",
            "examples": [
                "Cetirizine 10mg",
                "Loratadine 10mg",
                "Fexofenadine 180mg"
            ],
            "usage": "Take once daily as directed",
            "duration": "As needed while hives persist; if lasting >6 weeks, see doctor",
            "warnings": [
                "Do not exceed recommended dose",
                "Check for interactions with other medications"
            ],
            "contraindications": [
                "Severe kidney disease (dose adjustment needed)"
            ]
        },
        {
            "category": "Topical Anti-itch",
            "examples": [
                "Calamine lotion",
                "Pramoxine lotion",
                "Menthol-containing lotions"
            ],
            "usage": "Apply to itchy areas as needed",
            "duration": "As needed for symptom relief",
            "warnings": [
                "Provides temporary relief only"
            ],
            "contraindications": []
        }
    ],

    # -------------------------------------------------------------------------
    # MINOR BURNS
    # -------------------------------------------------------------------------
    "burn_first_degree": [
        {
            "category": "Cooling and Pain Relief",
            "examples": [
                "Cool (not cold) running water",
                "Aloe vera gel (pure, without alcohol)"
            ],
            "usage": "Cool burn under running water for 10-20 minutes immediately after injury",
            "duration": "Immediate first aid; aloe can be applied ongoing",
            "warnings": [
                "Do NOT use ice directly on burn",
                "Do NOT apply butter or oil",
                "Do NOT break blisters if they form"
            ],
            "contraindications": []
        },
        {
            "category": "Topical Antibiotics",
            "examples": [
                "Bacitracin ointment",
                "Polymyxin B/Bacitracin combination"
            ],
            "usage": "Apply thin layer to burn 1-2 times daily",
            "duration": "Until healed, typically 7-10 days",
            "warnings": [
                "Stop if rash or reaction develops"
            ],
            "contraindications": [
                "Known allergy to these antibiotics"
            ]
        },
        {
            "category": "Oral Pain Relievers",
            "examples": [
                "Acetaminophen (Tylenol)",
                "Ibuprofen (Advil, Motrin)"
            ],
            "usage": "Take as directed on package for pain relief",
            "duration": "As needed for pain, typically first few days",
            "warnings": [
                "Do not exceed maximum daily dose",
                "Ibuprofen: take with food"
            ],
            "contraindications": [
                "Acetaminophen: liver disease",
                "Ibuprofen: kidney disease, stomach ulcers, bleeding disorders"
            ]
        },
        {
            "category": "Wound Dressings",
            "examples": [
                "Non-stick sterile gauze pads",
                "Hydrogel dressings"
            ],
            "usage": "Cover burn with sterile dressing if needed for protection",
            "duration": "Change daily or as needed",
            "warnings": [
                "Keep wound clean and dry"
            ],
            "contraindications": []
        }
    ],

    "sunburn": [
        {
            "category": "Cooling Agents",
            "examples": [
                "Aloe vera gel",
                "After-sun lotions with aloe"
            ],
            "usage": "Apply liberally to sunburned areas as needed",
            "duration": "As needed until healed",
            "warnings": [
                "Choose alcohol-free products",
                "Avoid products with benzocaine (can cause allergic reaction)"
            ],
            "contraindications": []
        },
        {
            "category": "Anti-inflammatory",
            "examples": [
                "Ibuprofen",
                "Aspirin (adults only)"
            ],
            "usage": "Take as directed for pain and inflammation",
            "duration": "First 24-48 hours when most effective",
            "warnings": [
                "Take with food",
                "Stay well hydrated"
            ],
            "contraindications": [
                "Children under 18: no aspirin",
                "Stomach ulcers",
                "Bleeding disorders"
            ]
        },
        {
            "category": "Hydration",
            "examples": [
                "Oral rehydration solutions",
                "Water and electrolyte drinks"
            ],
            "usage": "Drink plenty of fluids",
            "duration": "Until symptoms resolve",
            "warnings": [
                "Sunburn draws fluid to skin surface"
            ],
            "contraindications": []
        }
    ],

    # -------------------------------------------------------------------------
    # BENIGN CONDITIONS
    # -------------------------------------------------------------------------
    "benign_nevus": [
        {
            "category": "Sun Protection",
            "examples": [
                "Broad-spectrum SPF 30+ sunscreen",
                "Zinc oxide or titanium dioxide physical blockers"
            ],
            "usage": "Apply to all exposed skin 15-30 minutes before sun exposure; reapply every 2 hours",
            "duration": "Ongoing whenever outdoors",
            "warnings": [
                "No OTC treatment changes moles",
                "Protecting moles from sun helps prevent changes"
            ],
            "contraindications": []
        }
    ],

    "seborrheic_keratosis": [
        {
            "category": "No OTC Treatment Recommended",
            "examples": [],
            "usage": "Seborrheic keratoses do not require treatment unless bothered by them",
            "duration": "N/A",
            "warnings": [
                "Do NOT try to remove at home",
                "See dermatologist for removal if desired",
                "Monitor for any changes"
            ],
            "contraindications": []
        }
    ],

    "dermatofibroma": [
        {
            "category": "Symptomatic Relief (if itchy or irritated)",
            "examples": [
                "Hydrocortisone 1% cream",
                "Anti-itch lotions with pramoxine"
            ],
            "usage": "Apply to the area if experiencing itching or minor irritation",
            "duration": "As needed for symptom relief; short-term use only",
            "warnings": [
                "Dermatofibromas are benign and do not require treatment",
                "OTC products will not remove or shrink the lesion",
                "Do NOT attempt to remove at home"
            ],
            "contraindications": []
        },
        {
            "category": "Sun Protection",
            "examples": [
                "Broad-spectrum SPF 30+ sunscreen"
            ],
            "usage": "Protect the area from sun exposure",
            "duration": "Ongoing when outdoors",
            "warnings": [
                "Sun protection helps prevent skin changes"
            ],
            "contraindications": []
        },
        {
            "category": "Cosmetic Coverage (if desired)",
            "examples": [
                "Concealer or foundation makeup",
                "Waterproof cover-up products"
            ],
            "usage": "Apply to camouflage appearance if desired",
            "duration": "As desired for cosmetic purposes",
            "warnings": [
                "Choose non-comedogenic products",
                "This is purely cosmetic - does not treat the lesion"
            ],
            "contraindications": []
        }
    ],

    "vascular_lesion": [
        {
            "category": "Sun Protection",
            "examples": [
                "Broad-spectrum SPF 30+ sunscreen",
                "Zinc oxide physical blocker"
            ],
            "usage": "Protect the area from sun exposure",
            "duration": "Ongoing when outdoors",
            "warnings": [
                "Most vascular lesions (cherry angiomas, etc.) are benign",
                "OTC products will not remove vascular lesions",
                "See dermatologist if removal is desired"
            ],
            "contraindications": []
        },
        {
            "category": "Cosmetic Coverage (if desired)",
            "examples": [
                "Green-tinted color-correcting concealer",
                "Full-coverage foundation"
            ],
            "usage": "Green tint helps neutralize redness before applying concealer",
            "duration": "As desired for cosmetic purposes",
            "warnings": [
                "This is purely cosmetic"
            ],
            "contraindications": []
        }
    ],

    # -------------------------------------------------------------------------
    # FUNGAL INFECTIONS (Limited OTC)
    # -------------------------------------------------------------------------
    "tinea_fungal_mild": [
        {
            "category": "Topical Antifungals",
            "examples": [
                "Clotrimazole 1% cream",
                "Miconazole 2% cream",
                "Terbinafine 1% cream"
            ],
            "usage": "Apply to affected area and surrounding skin 1-2 times daily",
            "duration": "Continue for 1-2 weeks after symptoms clear (usually 2-4 weeks total)",
            "warnings": [
                "Complete full course even if symptoms improve",
                "Keep area clean and dry",
                "See doctor if no improvement in 2 weeks"
            ],
            "contraindications": [
                "Do not use for scalp ringworm (needs oral treatment)",
                "Do not use for nail fungus (needs oral treatment)"
            ]
        }
    ]
}


# =============================================================================
# CONDITION MAPPING
# Maps classification results to OTC recommendation categories
# =============================================================================

CONDITION_TO_OTC_MAP: Dict[str, Dict[str, Any]] = {
    # Malignant - NO OTC
    "melanoma": {"category": ConditionCategory.MALIGNANT, "otc_key": None},
    "basal cell carcinoma": {"category": ConditionCategory.MALIGNANT, "otc_key": None},
    "squamous cell carcinoma": {"category": ConditionCategory.MALIGNANT, "otc_key": None},
    "bcc": {"category": ConditionCategory.MALIGNANT, "otc_key": None},
    "scc": {"category": ConditionCategory.MALIGNANT, "otc_key": None},
    "carcinoma": {"category": ConditionCategory.MALIGNANT, "otc_key": None},

    # Potentially malignant - NO OTC
    "actinic keratosis": {"category": ConditionCategory.POTENTIALLY_MALIGNANT, "otc_key": None},
    "dysplastic nevus": {"category": ConditionCategory.POTENTIALLY_MALIGNANT, "otc_key": None},
    "atypical mole": {"category": ConditionCategory.POTENTIALLY_MALIGNANT, "otc_key": None},

    # Inflammatory - OTC appropriate
    "eczema": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "eczema_mild"},
    "atopic dermatitis": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "eczema_mild"},
    "dermatitis": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "contact_dermatitis"},
    "contact dermatitis": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "contact_dermatitis"},
    "psoriasis": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "psoriasis_mild"},
    "seborrheic dermatitis": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "seborrheic_dermatitis"},
    "acne": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "acne_mild"},
    "acne vulgaris": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "acne_mild"},
    "rosacea": {"category": ConditionCategory.INFLAMMATORY_MODERATE, "otc_key": None},  # Needs Rx
    "urticaria": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "urticaria_hives"},
    "hives": {"category": ConditionCategory.INFLAMMATORY_MILD, "otc_key": "urticaria_hives"},

    # Benign lesions
    "nevus": {"category": ConditionCategory.BENIGN, "otc_key": "benign_nevus"},
    "nevi": {"category": ConditionCategory.BENIGN, "otc_key": "benign_nevus"},
    "melanocytic nevus": {"category": ConditionCategory.BENIGN, "otc_key": "benign_nevus"},
    "melanocytic nevi": {"category": ConditionCategory.BENIGN, "otc_key": "benign_nevus"},
    "mole": {"category": ConditionCategory.BENIGN, "otc_key": "benign_nevus"},
    "benign keratosis": {"category": ConditionCategory.BENIGN, "otc_key": "seborrheic_keratosis"},
    "seborrheic keratosis": {"category": ConditionCategory.BENIGN, "otc_key": "seborrheic_keratosis"},
    "dermatofibroma": {"category": ConditionCategory.BENIGN, "otc_key": "dermatofibroma"},
    "vascular lesion": {"category": ConditionCategory.BENIGN, "otc_key": "vascular_lesion"},
    "cherry angioma": {"category": ConditionCategory.BENIGN, "otc_key": "vascular_lesion"},
    "hemangioma": {"category": ConditionCategory.BENIGN, "otc_key": "vascular_lesion"},

    # Infectious - Limited OTC
    "ringworm": {"category": ConditionCategory.INFECTIOUS, "otc_key": "tinea_fungal_mild"},
    "tinea": {"category": ConditionCategory.INFECTIOUS, "otc_key": "tinea_fungal_mild"},
    "athlete's foot": {"category": ConditionCategory.INFECTIOUS, "otc_key": "tinea_fungal_mild"},
    "jock itch": {"category": ConditionCategory.INFECTIOUS, "otc_key": "tinea_fungal_mild"},
    "impetigo": {"category": ConditionCategory.INFECTIOUS, "otc_key": None},  # Needs Rx
    "cellulitis": {"category": ConditionCategory.INFECTIOUS, "otc_key": None},  # Needs Rx
    "herpes": {"category": ConditionCategory.INFECTIOUS, "otc_key": None},  # Needs Rx
    "shingles": {"category": ConditionCategory.INFECTIOUS, "otc_key": None},  # Needs Rx
    "warts": {"category": ConditionCategory.INFECTIOUS, "otc_key": None},  # See doctor
    "molluscum": {"category": ConditionCategory.INFECTIOUS, "otc_key": None},  # See doctor

    # Burns
    "first degree burn": {"category": ConditionCategory.BURN_MINOR, "otc_key": "burn_first_degree"},
    "sunburn": {"category": ConditionCategory.BURN_MINOR, "otc_key": "sunburn"},
    "second degree burn": {"category": ConditionCategory.BURN_SEVERE, "otc_key": None},
    "third degree burn": {"category": ConditionCategory.BURN_SEVERE, "otc_key": None},

    # Dry skin
    "xerosis": {"category": ConditionCategory.BENIGN, "otc_key": "dry_skin"},
    "dry skin": {"category": ConditionCategory.BENIGN, "otc_key": "dry_skin"},
}


def get_otc_recommendations(
    classification_result: Optional[str],
    classification_confidence: Optional[float],
    condition_type: Optional[str] = None,
    burn_severity: Optional[int] = None,
    is_infectious: bool = False,
    infectious_disease: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get OTC medication recommendations for a skin condition.

    Includes safety guardrails to suppress recommendations for conditions
    that require professional medical care.

    Args:
        classification_result: The predicted skin condition
        classification_confidence: Confidence score (0-1)
        condition_type: Type of condition (lesion, inflammatory, burn, infectious)
        burn_severity: Burn severity level (1, 2, or 3)
        is_infectious: Whether an infectious condition was detected
        infectious_disease: Name of infectious disease if detected

    Returns:
        Dictionary with OTC recommendations or explanation of why none provided
    """

    # Base response structure
    response = {
        "applicable": False,
        "recommendations": [],
        "reason_not_applicable": None,
        "see_doctor_reasons": [],
        "when_to_seek_care": [],
        "disclaimer": (
            "These suggestions are for educational purposes only and do not constitute "
            "medical advice. Always read and follow product labels. Consult a healthcare "
            "provider before starting any treatment, especially if you have other medical "
            "conditions, take other medications, are pregnant, or if symptoms persist or worsen."
        )
    }

    # Safety check: Severe burns
    if burn_severity is not None and burn_severity >= 2:
        response["reason_not_applicable"] = "Severe burns require professional medical care"
        response["see_doctor_reasons"] = [
            "Second and third degree burns need professional medical treatment",
            "Risk of infection and scarring without proper care",
            "May require prescription medications or wound care"
        ]
        return response

    # Safety check: High confidence malignancy
    if classification_result:
        result_lower = classification_result.lower()

        malignant_terms = ["melanoma", "carcinoma", "malignant", "bcc", "scc"]
        if any(term in result_lower for term in malignant_terms):
            response["reason_not_applicable"] = "Suspected malignancy requires professional evaluation"
            response["see_doctor_reasons"] = [
                f"AI detected possible {classification_result}",
                "This type of lesion requires professional dermatology evaluation",
                "Early diagnosis and treatment are important",
                "Do NOT attempt to treat with over-the-counter products"
            ]
            return response

        potentially_malignant = ["actinic keratosis", "dysplastic", "atypical"]
        if any(term in result_lower for term in potentially_malignant):
            response["reason_not_applicable"] = "Potentially pre-cancerous lesion requires evaluation"
            response["see_doctor_reasons"] = [
                "This type of lesion should be evaluated by a dermatologist",
                "May require professional treatment or monitoring"
            ]
            return response

    # Look up condition in mapping
    condition_info = None
    otc_key = None

    if classification_result:
        result_lower = classification_result.lower()

        # Try exact match first
        if result_lower in CONDITION_TO_OTC_MAP:
            condition_info = CONDITION_TO_OTC_MAP[result_lower]
        else:
            # Try partial match
            for key, info in CONDITION_TO_OTC_MAP.items():
                if key in result_lower or result_lower in key:
                    condition_info = info
                    break

    # Check infectious diseases
    if is_infectious and infectious_disease:
        disease_lower = infectious_disease.lower()
        if disease_lower in CONDITION_TO_OTC_MAP:
            condition_info = CONDITION_TO_OTC_MAP[disease_lower]
        else:
            # Most infectious conditions need Rx
            response["reason_not_applicable"] = "Infectious conditions often require prescription treatment"
            response["see_doctor_reasons"] = [
                f"Suspected {infectious_disease} should be evaluated by a healthcare provider",
                "May require prescription antibiotics or antifungals",
                "Proper diagnosis is important for effective treatment"
            ]
            return response

    # Check burn (minor)
    if burn_severity == 1:
        condition_info = {"category": ConditionCategory.BURN_MINOR, "otc_key": "burn_first_degree"}

    # If no condition info found
    if not condition_info:
        response["reason_not_applicable"] = "No specific OTC recommendations available for this condition"
        response["see_doctor_reasons"] = [
            "Consult a healthcare provider for proper diagnosis and treatment recommendations"
        ]
        return response

    # Check if OTC is appropriate for this category
    category = condition_info.get("category")
    otc_key = condition_info.get("otc_key")

    if category in [ConditionCategory.MALIGNANT, ConditionCategory.POTENTIALLY_MALIGNANT]:
        response["reason_not_applicable"] = "This condition requires professional medical evaluation"
        response["see_doctor_reasons"] = [
            "Professional evaluation and treatment required",
            "Do not attempt to treat with OTC products"
        ]
        return response

    if category == ConditionCategory.BURN_SEVERE:
        response["reason_not_applicable"] = "Severe burns require professional medical care"
        response["see_doctor_reasons"] = [
            "Seek immediate medical attention",
            "Professional wound care needed"
        ]
        return response

    if not otc_key:
        response["reason_not_applicable"] = "No OTC treatment recommended for this specific condition"
        response["see_doctor_reasons"] = [
            "Consult a healthcare provider for treatment options"
        ]
        return response

    # Get OTC recommendations
    if otc_key in OTC_DATABASE:
        response["applicable"] = True
        response["condition_category"] = category.value
        response["recommendations"] = OTC_DATABASE[otc_key]

        # Add general when_to_seek_care
        response["when_to_seek_care"] = [
            "Symptoms worsen or do not improve after 7 days of treatment",
            "Signs of infection develop (increasing redness, warmth, swelling, pus, fever)",
            "Rash or irritation spreads to new areas",
            "You develop new symptoms",
            "You have concerns about your condition"
        ]

        # Add condition-specific notes
        if category == ConditionCategory.INFLAMMATORY_MILD:
            response["general_advice"] = [
                "Avoid known triggers and irritants",
                "Keep skin well moisturized",
                "Avoid scratching to prevent infection"
            ]
        elif category == ConditionCategory.BURN_MINOR:
            response["general_advice"] = [
                "Keep the burn clean and protected",
                "Do not pop blisters if they form",
                "Seek medical care if burn is larger than 3 inches or on face/joints/genitals"
            ]
        elif category == ConditionCategory.INFECTIOUS:
            response["general_advice"] = [
                "Complete the full course of treatment",
                "Keep affected area clean and dry",
                "Avoid sharing personal items",
                "See doctor if no improvement in 2 weeks"
            ]

    return response


def format_otc_for_response(otc_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format OTC recommendations for API response.
    Ensures consistent structure and includes all necessary disclaimers.
    """
    return {
        "otc_recommendations": otc_data,
        "regulatory_notes": {
            "is_medical_advice": False,
            "requires_professional_diagnosis": True,
            "educational_purposes_only": True
        }
    }
