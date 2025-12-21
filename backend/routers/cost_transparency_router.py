"""
Cost Transparency Router

Endpoints for:
- Estimated costs before booking
- Dermatologist price comparison
- GoodRx medication price integration
- Insurance coverage estimates
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import os
import httpx

from database import get_db, User
from auth import get_current_active_user

router = APIRouter(prefix="/costs", tags=["Cost Transparency"])


# =============================================================================
# DATA MODELS
# =============================================================================

class ProcedureCostEstimate(BaseModel):
    """Cost estimate for a dermatology procedure."""
    procedure_code: str
    procedure_name: str
    description: str
    average_cost: float
    cost_range_low: float
    cost_range_high: float
    medicare_rate: float
    typical_insurance_coverage: float
    out_of_pocket_estimate: float
    factors_affecting_cost: List[str]


class ProviderPricing(BaseModel):
    """Provider pricing information."""
    provider_id: str
    provider_name: str
    specialty: str
    location: str
    distance_miles: float
    consultation_fee: float
    average_procedure_cost: float
    accepts_insurance: bool
    insurance_networks: List[str]
    rating: float
    review_count: int
    wait_time_days: int
    telemedicine_available: bool
    telemedicine_fee: float


class MedicationPrice(BaseModel):
    """GoodRx-style medication pricing."""
    medication_name: str
    generic_name: str
    dosage: str
    quantity: int
    pharmacy_name: str
    pharmacy_address: str
    price: float
    original_price: float
    savings_percent: float
    coupon_code: Optional[str]
    requires_membership: bool


# =============================================================================
# PROCEDURE COST DATABASE
# =============================================================================

DERMATOLOGY_PROCEDURE_COSTS = {
    "consultation": {
        "code": "99213",
        "name": "Office Visit - Established Patient",
        "description": "Standard dermatology consultation for established patients",
        "average_cost": 150.00,
        "cost_range": (100, 250),
        "medicare_rate": 110.00,
        "insurance_coverage": 0.80
    },
    "new_patient_visit": {
        "code": "99203",
        "name": "Office Visit - New Patient",
        "description": "Initial comprehensive dermatology consultation",
        "average_cost": 225.00,
        "cost_range": (175, 350),
        "medicare_rate": 165.00,
        "insurance_coverage": 0.80
    },
    "skin_biopsy": {
        "code": "11102",
        "name": "Skin Biopsy - Tangential",
        "description": "Tangential biopsy of skin lesion for pathological examination",
        "average_cost": 250.00,
        "cost_range": (150, 400),
        "medicare_rate": 175.00,
        "insurance_coverage": 0.80
    },
    "excision_benign": {
        "code": "11400",
        "name": "Excision of Benign Lesion",
        "description": "Surgical removal of benign skin lesion with margin",
        "average_cost": 350.00,
        "cost_range": (200, 600),
        "medicare_rate": 250.00,
        "insurance_coverage": 0.75
    },
    "excision_malignant": {
        "code": "11600",
        "name": "Excision of Malignant Lesion",
        "description": "Surgical removal of malignant skin lesion with appropriate margins",
        "average_cost": 550.00,
        "cost_range": (350, 900),
        "medicare_rate": 400.00,
        "insurance_coverage": 0.85
    },
    "cryotherapy": {
        "code": "17000",
        "name": "Cryotherapy - First Lesion",
        "description": "Destruction of premalignant lesion using liquid nitrogen",
        "average_cost": 150.00,
        "cost_range": (75, 250),
        "medicare_rate": 85.00,
        "insurance_coverage": 0.80
    },
    "mohs_surgery": {
        "code": "17311",
        "name": "Mohs Micrographic Surgery",
        "description": "Specialized skin cancer surgery with microscopic margin analysis",
        "average_cost": 1500.00,
        "cost_range": (1000, 3000),
        "medicare_rate": 950.00,
        "insurance_coverage": 0.85
    },
    "phototherapy": {
        "code": "96910",
        "name": "Phototherapy - Full Body",
        "description": "UV light treatment for psoriasis, eczema, and other conditions",
        "average_cost": 175.00,
        "cost_range": (100, 300),
        "medicare_rate": 125.00,
        "insurance_coverage": 0.70
    },
    "dermoscopy": {
        "code": "96902",
        "name": "Dermoscopic Examination",
        "description": "Microscopic examination of skin lesions using dermoscope",
        "average_cost": 85.00,
        "cost_range": (50, 150),
        "medicare_rate": 60.00,
        "insurance_coverage": 0.80
    },
    "chemical_peel": {
        "code": "15788",
        "name": "Chemical Peel - Light",
        "description": "Chemical exfoliation for acne scars, sun damage, or rejuvenation",
        "average_cost": 200.00,
        "cost_range": (150, 400),
        "medicare_rate": 0.00,  # Typically cosmetic
        "insurance_coverage": 0.00
    },
    "laser_treatment": {
        "code": "17106",
        "name": "Laser Skin Treatment",
        "description": "Laser therapy for vascular lesions, pigmentation, or resurfacing",
        "average_cost": 450.00,
        "cost_range": (200, 1000),
        "medicare_rate": 200.00,
        "insurance_coverage": 0.50
    },
    "patch_testing": {
        "code": "95044",
        "name": "Allergy Patch Testing",
        "description": "Contact dermatitis evaluation with multiple allergen patches",
        "average_cost": 350.00,
        "cost_range": (200, 600),
        "medicare_rate": 275.00,
        "insurance_coverage": 0.80
    },
    "telemedicine": {
        "code": "99441",
        "name": "Telemedicine Consultation",
        "description": "Virtual dermatology consultation via video call",
        "average_cost": 100.00,
        "cost_range": (50, 175),
        "medicare_rate": 75.00,
        "insurance_coverage": 0.80
    }
}


# =============================================================================
# DEMO PROVIDER DATABASE
# =============================================================================

DEMO_PROVIDERS = [
    {
        "provider_id": "dr_smith_001",
        "provider_name": "Dr. Sarah Smith, MD",
        "specialty": "Board-Certified Dermatologist",
        "location": "123 Medical Center Dr, Suite 100",
        "city": "San Francisco",
        "distance_miles": 2.3,
        "consultation_fee": 175.00,
        "average_procedure_cost": 350.00,
        "accepts_insurance": True,
        "insurance_networks": ["Blue Cross", "Aetna", "United Healthcare", "Cigna"],
        "rating": 4.8,
        "review_count": 342,
        "wait_time_days": 7,
        "telemedicine_available": True,
        "telemedicine_fee": 95.00,
        "specializations": ["Skin Cancer", "Mohs Surgery", "Melanoma"]
    },
    {
        "provider_id": "dr_johnson_002",
        "provider_name": "Dr. Michael Johnson, DO",
        "specialty": "Dermatologist",
        "location": "456 Health Plaza, Floor 3",
        "city": "Oakland",
        "distance_miles": 5.8,
        "consultation_fee": 150.00,
        "average_procedure_cost": 300.00,
        "accepts_insurance": True,
        "insurance_networks": ["Kaiser", "Blue Shield", "Medicare"],
        "rating": 4.6,
        "review_count": 215,
        "wait_time_days": 14,
        "telemedicine_available": True,
        "telemedicine_fee": 85.00,
        "specializations": ["Psoriasis", "Eczema", "Acne"]
    },
    {
        "provider_id": "dr_chen_003",
        "provider_name": "Dr. Lisa Chen, MD",
        "specialty": "Dermatopathologist",
        "location": "789 University Medical Center",
        "city": "Palo Alto",
        "distance_miles": 12.4,
        "consultation_fee": 225.00,
        "average_procedure_cost": 450.00,
        "accepts_insurance": True,
        "insurance_networks": ["Stanford Health", "Aetna", "Blue Cross", "United Healthcare"],
        "rating": 4.9,
        "review_count": 567,
        "wait_time_days": 21,
        "telemedicine_available": False,
        "telemedicine_fee": 0.00,
        "specializations": ["Skin Cancer Diagnosis", "Melanoma", "Rare Skin Conditions"]
    },
    {
        "provider_id": "dr_patel_004",
        "provider_name": "Dr. Raj Patel, MD",
        "specialty": "Board-Certified Dermatologist",
        "location": "321 Clinic Way",
        "city": "San Jose",
        "distance_miles": 15.2,
        "consultation_fee": 125.00,
        "average_procedure_cost": 275.00,
        "accepts_insurance": True,
        "insurance_networks": ["Blue Cross", "Aetna", "Cigna", "Humana", "Medicare", "Medicaid"],
        "rating": 4.5,
        "review_count": 189,
        "wait_time_days": 5,
        "telemedicine_available": True,
        "telemedicine_fee": 75.00,
        "specializations": ["General Dermatology", "Cosmetic Procedures", "Acne"]
    },
    {
        "provider_id": "dr_williams_005",
        "provider_name": "Dr. Emily Williams, MD",
        "specialty": "Pediatric Dermatologist",
        "location": "555 Children's Hospital Blvd",
        "city": "San Francisco",
        "distance_miles": 3.7,
        "consultation_fee": 200.00,
        "average_procedure_cost": 375.00,
        "accepts_insurance": True,
        "insurance_networks": ["Blue Cross", "Aetna", "United Healthcare", "Kaiser"],
        "rating": 4.9,
        "review_count": 423,
        "wait_time_days": 10,
        "telemedicine_available": True,
        "telemedicine_fee": 110.00,
        "specializations": ["Pediatric Skin Conditions", "Birthmarks", "Eczema"]
    }
]


# =============================================================================
# DEMO MEDICATION PRICES (GoodRx-style)
# =============================================================================

DEMO_MEDICATION_PRICES = {
    "tretinoin": {
        "brand_name": "Retin-A",
        "generic_name": "Tretinoin",
        "drug_class": "Retinoid",
        "dosages": ["0.025%", "0.05%", "0.1%"],
        "forms": ["Cream", "Gel"],
        "prices": [
            {"pharmacy": "CVS Pharmacy", "address": "123 Main St", "price": 45.00, "original": 180.00},
            {"pharmacy": "Walgreens", "address": "456 Oak Ave", "price": 52.00, "original": 185.00},
            {"pharmacy": "Costco", "address": "789 Retail Blvd", "price": 28.00, "original": 175.00},
            {"pharmacy": "Walmart", "address": "321 Shop Way", "price": 35.00, "original": 180.00},
            {"pharmacy": "Rite Aid", "address": "555 Drug Dr", "price": 48.00, "original": 182.00}
        ]
    },
    "hydrocortisone": {
        "brand_name": "Cortaid",
        "generic_name": "Hydrocortisone",
        "drug_class": "Corticosteroid",
        "dosages": ["0.5%", "1%", "2.5%"],
        "forms": ["Cream", "Ointment", "Lotion"],
        "prices": [
            {"pharmacy": "CVS Pharmacy", "address": "123 Main St", "price": 8.00, "original": 15.00},
            {"pharmacy": "Walgreens", "address": "456 Oak Ave", "price": 9.50, "original": 16.00},
            {"pharmacy": "Costco", "address": "789 Retail Blvd", "price": 5.50, "original": 14.00},
            {"pharmacy": "Walmart", "address": "321 Shop Way", "price": 6.00, "original": 14.50},
            {"pharmacy": "Amazon Pharmacy", "address": "Online", "price": 7.00, "original": 15.00}
        ]
    },
    "clobetasol": {
        "brand_name": "Temovate",
        "generic_name": "Clobetasol Propionate",
        "drug_class": "High-Potency Corticosteroid",
        "dosages": ["0.05%"],
        "forms": ["Cream", "Ointment", "Foam", "Solution"],
        "prices": [
            {"pharmacy": "CVS Pharmacy", "address": "123 Main St", "price": 35.00, "original": 250.00},
            {"pharmacy": "Walgreens", "address": "456 Oak Ave", "price": 42.00, "original": 255.00},
            {"pharmacy": "Costco", "address": "789 Retail Blvd", "price": 22.00, "original": 245.00},
            {"pharmacy": "Walmart", "address": "321 Shop Way", "price": 28.00, "original": 248.00},
            {"pharmacy": "Rite Aid", "address": "555 Drug Dr", "price": 38.00, "original": 252.00}
        ]
    },
    "tacrolimus": {
        "brand_name": "Protopic",
        "generic_name": "Tacrolimus",
        "drug_class": "Calcineurin Inhibitor",
        "dosages": ["0.03%", "0.1%"],
        "forms": ["Ointment"],
        "prices": [
            {"pharmacy": "CVS Pharmacy", "address": "123 Main St", "price": 125.00, "original": 450.00},
            {"pharmacy": "Walgreens", "address": "456 Oak Ave", "price": 135.00, "original": 455.00},
            {"pharmacy": "Costco", "address": "789 Retail Blvd", "price": 95.00, "original": 440.00},
            {"pharmacy": "Walmart", "address": "321 Shop Way", "price": 110.00, "original": 445.00},
            {"pharmacy": "Amazon Pharmacy", "address": "Online", "price": 105.00, "original": 448.00}
        ]
    },
    "methotrexate": {
        "brand_name": "Trexall",
        "generic_name": "Methotrexate",
        "drug_class": "Immunosuppressant",
        "dosages": ["2.5mg", "5mg", "7.5mg", "10mg", "15mg"],
        "forms": ["Tablets", "Injectable"],
        "prices": [
            {"pharmacy": "CVS Pharmacy", "address": "123 Main St", "price": 18.00, "original": 85.00},
            {"pharmacy": "Walgreens", "address": "456 Oak Ave", "price": 22.00, "original": 88.00},
            {"pharmacy": "Costco", "address": "789 Retail Blvd", "price": 12.00, "original": 80.00},
            {"pharmacy": "Walmart", "address": "321 Shop Way", "price": 15.00, "original": 82.00},
            {"pharmacy": "Rite Aid", "address": "555 Drug Dr", "price": 20.00, "original": 86.00}
        ]
    },
    "doxycycline": {
        "brand_name": "Vibramycin",
        "generic_name": "Doxycycline",
        "drug_class": "Tetracycline Antibiotic",
        "dosages": ["50mg", "100mg"],
        "forms": ["Capsules", "Tablets"],
        "prices": [
            {"pharmacy": "CVS Pharmacy", "address": "123 Main St", "price": 15.00, "original": 120.00},
            {"pharmacy": "Walgreens", "address": "456 Oak Ave", "price": 18.00, "original": 125.00},
            {"pharmacy": "Costco", "address": "789 Retail Blvd", "price": 8.00, "original": 115.00},
            {"pharmacy": "Walmart", "address": "321 Shop Way", "price": 10.00, "original": 118.00},
            {"pharmacy": "Amazon Pharmacy", "address": "Online", "price": 12.00, "original": 120.00}
        ]
    },
    "adapalene": {
        "brand_name": "Differin",
        "generic_name": "Adapalene",
        "drug_class": "Retinoid",
        "dosages": ["0.1%", "0.3%"],
        "forms": ["Gel", "Cream"],
        "prices": [
            {"pharmacy": "CVS Pharmacy", "address": "123 Main St", "price": 32.00, "original": 150.00},
            {"pharmacy": "Walgreens", "address": "456 Oak Ave", "price": 38.00, "original": 155.00},
            {"pharmacy": "Costco", "address": "789 Retail Blvd", "price": 22.00, "original": 145.00},
            {"pharmacy": "Walmart", "address": "321 Shop Way", "price": 28.00, "original": 148.00},
            {"pharmacy": "Target", "address": "777 Target Way", "price": 30.00, "original": 150.00}
        ]
    },
    "isotretinoin": {
        "brand_name": "Accutane",
        "generic_name": "Isotretinoin",
        "drug_class": "Retinoid",
        "dosages": ["10mg", "20mg", "30mg", "40mg"],
        "forms": ["Capsules"],
        "prices": [
            {"pharmacy": "CVS Pharmacy", "address": "123 Main St", "price": 285.00, "original": 650.00},
            {"pharmacy": "Walgreens", "address": "456 Oak Ave", "price": 310.00, "original": 660.00},
            {"pharmacy": "Costco", "address": "789 Retail Blvd", "price": 225.00, "original": 640.00},
            {"pharmacy": "Walmart", "address": "321 Shop Way", "price": 255.00, "original": 645.00},
            {"pharmacy": "Rite Aid", "address": "555 Drug Dr", "price": 295.00, "original": 655.00}
        ]
    }
}


# =============================================================================
# COST ESTIMATE ENDPOINTS
# =============================================================================

@router.get("/procedures")
async def get_procedure_cost_estimates(
    procedure_type: Optional[str] = None,
    insurance_coverage: float = Query(default=0.8, ge=0, le=1),
    current_user: User = Depends(get_current_active_user)
):
    """Get estimated costs for dermatology procedures."""
    try:
        estimates = []

        procedures_to_return = DERMATOLOGY_PROCEDURE_COSTS
        if procedure_type:
            procedures_to_return = {
                k: v for k, v in DERMATOLOGY_PROCEDURE_COSTS.items()
                if procedure_type.lower() in k.lower() or procedure_type.lower() in v["name"].lower()
            }

        for key, proc in procedures_to_return.items():
            out_of_pocket = proc["average_cost"] * (1 - min(insurance_coverage, proc["insurance_coverage"]))

            estimates.append({
                "procedure_key": key,
                "procedure_code": proc["code"],
                "procedure_name": proc["name"],
                "description": proc["description"],
                "average_cost": proc["average_cost"],
                "cost_range_low": proc["cost_range"][0],
                "cost_range_high": proc["cost_range"][1],
                "medicare_rate": proc["medicare_rate"],
                "typical_insurance_coverage": proc["insurance_coverage"],
                "estimated_out_of_pocket": round(out_of_pocket, 2),
                "factors_affecting_cost": [
                    "Geographic location",
                    "Provider experience",
                    "Facility type (hospital vs. office)",
                    "Complexity of case",
                    "Insurance plan deductible status"
                ]
            })

        return {
            "estimates": estimates,
            "disclaimer": "These are estimated costs and may vary. Contact your insurance provider and healthcare facility for exact pricing.",
            "insurance_coverage_used": insurance_coverage
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cost estimates: {str(e)}")


@router.get("/procedures/{procedure_key}")
async def get_specific_procedure_cost(
    procedure_key: str,
    insurance_type: Optional[str] = None,
    deductible_met: bool = True,
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed cost breakdown for a specific procedure."""
    try:
        if procedure_key not in DERMATOLOGY_PROCEDURE_COSTS:
            raise HTTPException(status_code=404, detail="Procedure not found")

        proc = DERMATOLOGY_PROCEDURE_COSTS[procedure_key]

        # Calculate coverage based on insurance type
        coverage_rates = {
            "medicare": 0.80,
            "medicaid": 0.90,
            "private": proc["insurance_coverage"],
            "hmo": proc["insurance_coverage"] - 0.05,
            "ppo": proc["insurance_coverage"],
            "none": 0.0
        }

        insurance_type = (insurance_type or "private").lower()
        coverage = coverage_rates.get(insurance_type, proc["insurance_coverage"])

        # If deductible not met, patient pays full amount up to deductible
        if not deductible_met:
            out_of_pocket = proc["average_cost"]
            coverage_applied = 0
        else:
            coverage_applied = proc["average_cost"] * coverage
            out_of_pocket = proc["average_cost"] - coverage_applied

        return {
            "procedure_key": procedure_key,
            "procedure_code": proc["code"],
            "procedure_name": proc["name"],
            "description": proc["description"],
            "cost_breakdown": {
                "total_procedure_cost": proc["average_cost"],
                "cost_range": {
                    "low": proc["cost_range"][0],
                    "high": proc["cost_range"][1]
                },
                "medicare_rate": proc["medicare_rate"],
                "insurance_type": insurance_type,
                "coverage_percentage": coverage * 100,
                "insurance_pays": round(coverage_applied, 2),
                "your_estimated_cost": round(out_of_pocket, 2),
                "deductible_met": deductible_met
            },
            "additional_costs": [
                {"item": "Facility fee", "estimated_range": "$50-$200"},
                {"item": "Pathology (if biopsy)", "estimated_range": "$75-$300"},
                {"item": "Follow-up visit", "estimated_range": "$100-$175"},
                {"item": "Prescription medications", "estimated_range": "Varies"}
            ],
            "cost_saving_tips": [
                "Ask about cash-pay discounts if uninsured",
                "Compare prices between providers using our comparison tool",
                "Check if procedure can be done in-office vs. hospital for lower facility fees",
                "Verify your deductible status before scheduling",
                "Ask about payment plans for larger procedures"
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting procedure cost: {str(e)}")


# =============================================================================
# PROVIDER COMPARISON ENDPOINTS
# =============================================================================

@router.get("/providers")
async def compare_provider_prices(
    procedure_type: Optional[str] = None,
    max_distance: float = Query(default=25.0, ge=0),
    insurance_network: Optional[str] = None,
    telemedicine_only: bool = False,
    sort_by: str = Query(default="price", regex="^(price|distance|rating|wait_time)$"),
    current_user: User = Depends(get_current_active_user)
):
    """Compare dermatologist prices and availability."""
    try:
        providers = DEMO_PROVIDERS.copy()

        # Filter by distance
        providers = [p for p in providers if p["distance_miles"] <= max_distance]

        # Filter by insurance network
        if insurance_network:
            providers = [
                p for p in providers
                if insurance_network.lower() in [n.lower() for n in p["insurance_networks"]]
            ]

        # Filter by telemedicine
        if telemedicine_only:
            providers = [p for p in providers if p["telemedicine_available"]]

        # Sort providers
        sort_keys = {
            "price": lambda x: x["consultation_fee"],
            "distance": lambda x: x["distance_miles"],
            "rating": lambda x: -x["rating"],  # Negative for descending
            "wait_time": lambda x: x["wait_time_days"]
        }
        providers.sort(key=sort_keys.get(sort_by, sort_keys["price"]))

        return {
            "providers": providers,
            "total_count": len(providers),
            "filters_applied": {
                "procedure_type": procedure_type,
                "max_distance": max_distance,
                "insurance_network": insurance_network,
                "telemedicine_only": telemedicine_only,
                "sorted_by": sort_by
            },
            "price_summary": {
                "lowest_consultation": min(p["consultation_fee"] for p in providers) if providers else 0,
                "highest_consultation": max(p["consultation_fee"] for p in providers) if providers else 0,
                "average_consultation": sum(p["consultation_fee"] for p in providers) / len(providers) if providers else 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing providers: {str(e)}")


@router.get("/providers/{provider_id}")
async def get_provider_details(
    provider_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed pricing information for a specific provider."""
    try:
        provider = next((p for p in DEMO_PROVIDERS if p["provider_id"] == provider_id), None)

        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")

        # Generate procedure-specific pricing for this provider
        procedure_pricing = []
        base_multiplier = provider["consultation_fee"] / 150.0  # Relative to average

        for key, proc in DERMATOLOGY_PROCEDURE_COSTS.items():
            adjusted_cost = round(proc["average_cost"] * base_multiplier, 2)
            procedure_pricing.append({
                "procedure_key": key,
                "procedure_name": proc["name"],
                "provider_price": adjusted_cost,
                "average_market_price": proc["average_cost"],
                "savings": round(proc["average_cost"] - adjusted_cost, 2) if adjusted_cost < proc["average_cost"] else 0,
                "premium": round(adjusted_cost - proc["average_cost"], 2) if adjusted_cost > proc["average_cost"] else 0
            })

        return {
            "provider": provider,
            "procedure_pricing": procedure_pricing,
            "payment_options": [
                "Major credit cards accepted",
                "HSA/FSA cards accepted",
                "Payment plans available for procedures over $500",
                "CareCredit financing available"
            ],
            "cancellation_policy": "24-hour notice required for cancellations to avoid $50 fee"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting provider details: {str(e)}")


# =============================================================================
# OPENFDA INTEGRATION & DRUG PRICING
# =============================================================================

# Drug class to estimated price ranges (based on typical market prices)
DRUG_CLASS_PRICING = {
    # Dermatology-specific
    "retinoid": {"low": 25, "mid": 75, "high": 300, "unit": "tube/month"},
    "corticosteroid": {"low": 5, "mid": 25, "high": 80, "unit": "tube"},
    "corticosteroid, topical": {"low": 5, "mid": 25, "high": 80, "unit": "tube"},
    "immunosuppressant": {"low": 50, "mid": 150, "high": 500, "unit": "tube"},
    "calcineurin inhibitor": {"low": 80, "mid": 150, "high": 300, "unit": "tube"},
    "antibiotic": {"low": 8, "mid": 30, "high": 100, "unit": "course"},
    "antibiotic, tetracycline": {"low": 10, "mid": 25, "high": 60, "unit": "month"},
    "antifungal": {"low": 10, "mid": 40, "high": 150, "unit": "treatment"},
    "antiviral": {"low": 15, "mid": 80, "high": 300, "unit": "course"},
    "antihistamine": {"low": 5, "mid": 20, "high": 50, "unit": "month"},
    "biologic": {"low": 1500, "mid": 4000, "high": 7000, "unit": "month"},
    "immunomodulator": {"low": 100, "mid": 400, "high": 1000, "unit": "month"},
    "keratolytic": {"low": 10, "mid": 30, "high": 80, "unit": "tube"},
    "photosensitizing agent": {"low": 50, "mid": 150, "high": 400, "unit": "treatment"},
    "vitamin d analog": {"low": 100, "mid": 300, "high": 600, "unit": "tube"},
    "pde4 inhibitor": {"low": 400, "mid": 800, "high": 1200, "unit": "month"},
    # General categories
    "nsaid": {"low": 5, "mid": 15, "high": 40, "unit": "month"},
    "analgesic": {"low": 5, "mid": 20, "high": 60, "unit": "month"},
    "antipruritic": {"low": 8, "mid": 25, "high": 60, "unit": "tube"},
    "emollient": {"low": 10, "mid": 25, "high": 50, "unit": "bottle"},
    "sunscreen": {"low": 10, "mid": 20, "high": 40, "unit": "bottle"},
    # Default
    "default": {"low": 15, "mid": 50, "high": 200, "unit": "prescription"},
}

# Common dermatology drug mappings for better classification
DERMATOLOGY_DRUG_CLASSES = {
    # Retinoids
    "tretinoin": "retinoid",
    "adapalene": "retinoid",
    "tazarotene": "retinoid",
    "isotretinoin": "retinoid",
    "retinol": "retinoid",
    "trifarotene": "retinoid",
    # Corticosteroids
    "hydrocortisone": "corticosteroid",
    "betamethasone": "corticosteroid",
    "clobetasol": "corticosteroid",
    "triamcinolone": "corticosteroid",
    "fluocinonide": "corticosteroid",
    "mometasone": "corticosteroid",
    "desonide": "corticosteroid",
    "fluticasone": "corticosteroid",
    "halobetasol": "corticosteroid",
    "desoximetasone": "corticosteroid",
    "prednicarbate": "corticosteroid",
    "diflorasone": "corticosteroid",
    # Calcineurin inhibitors
    "tacrolimus": "calcineurin inhibitor",
    "pimecrolimus": "calcineurin inhibitor",
    # Antibiotics
    "doxycycline": "antibiotic, tetracycline",
    "minocycline": "antibiotic, tetracycline",
    "clindamycin": "antibiotic",
    "erythromycin": "antibiotic",
    "metronidazole": "antibiotic",
    "mupirocin": "antibiotic",
    "bacitracin": "antibiotic",
    "neomycin": "antibiotic",
    "sulfacetamide": "antibiotic",
    "gentamicin": "antibiotic",
    # Antifungals
    "ketoconazole": "antifungal",
    "terbinafine": "antifungal",
    "fluconazole": "antifungal",
    "itraconazole": "antifungal",
    "clotrimazole": "antifungal",
    "miconazole": "antifungal",
    "nystatin": "antifungal",
    "ciclopirox": "antifungal",
    "econazole": "antifungal",
    "sertaconazole": "antifungal",
    # Antivirals
    "acyclovir": "antiviral",
    "valacyclovir": "antiviral",
    "famciclovir": "antiviral",
    "penciclovir": "antiviral",
    "docosanol": "antiviral",
    # Biologics
    "dupilumab": "biologic",
    "adalimumab": "biologic",
    "etanercept": "biologic",
    "secukinumab": "biologic",
    "ixekizumab": "biologic",
    "ustekinumab": "biologic",
    "guselkumab": "biologic",
    "risankizumab": "biologic",
    "tildrakizumab": "biologic",
    "brodalumab": "biologic",
    "certolizumab": "biologic",
    "infliximab": "biologic",
    "tralokinumab": "biologic",
    # Immunomodulators
    "methotrexate": "immunomodulator",
    "cyclosporine": "immunomodulator",
    "azathioprine": "immunomodulator",
    "mycophenolate": "immunomodulator",
    # PDE4 inhibitors
    "apremilast": "pde4 inhibitor",
    "crisaborole": "pde4 inhibitor",
    "roflumilast": "pde4 inhibitor",
    # Vitamin D analogs
    "calcipotriene": "vitamin d analog",
    "calcitriol": "vitamin d analog",
    # Keratolytics
    "salicylic acid": "keratolytic",
    "urea": "keratolytic",
    "lactic acid": "keratolytic",
    # Other dermatology drugs
    "benzoyl peroxide": "keratolytic",
    "azelaic acid": "keratolytic",
    "ivermectin": "antiparasitic",
    "permethrin": "antiparasitic",
    "hydroxychloroquine": "immunomodulator",
    "dapsone": "antibiotic",
    "finasteride": "hormone modulator",
    "spironolactone": "hormone modulator",
    "minoxidil": "vasodilator",
}


async def fetch_openfda_drug(drug_name: str) -> Optional[dict]:
    """Fetch drug information from OpenFDA API."""
    try:
        # Clean the drug name
        clean_name = drug_name.strip().lower()

        # OpenFDA drug label API
        url = f"https://api.fda.gov/drug/label.json?search=(openfda.brand_name:{clean_name}+OR+openfda.generic_name:{clean_name})&limit=5"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)

            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    return data["results"]

            # Try alternative search with substance name
            url2 = f"https://api.fda.gov/drug/label.json?search=openfda.substance_name:{clean_name}&limit=5"
            response2 = await client.get(url2)

            if response2.status_code == 200:
                data2 = response2.json()
                if data2.get("results"):
                    return data2["results"]

        return None
    except Exception as e:
        print(f"OpenFDA API error: {str(e)}")
        return None


async def fetch_openfda_ndc(drug_name: str) -> Optional[dict]:
    """Fetch NDC (National Drug Code) information for dosage forms."""
    try:
        clean_name = drug_name.strip().lower()
        url = f"https://api.fda.gov/drug/ndc.json?search=(brand_name:{clean_name}+OR+generic_name:{clean_name})&limit=20"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)

            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    return data["results"]
        return None
    except Exception as e:
        print(f"OpenFDA NDC API error: {str(e)}")
        return None


def get_drug_class(drug_name: str, openfda_data: Optional[List] = None) -> str:
    """Determine drug class from name or OpenFDA data."""
    drug_lower = drug_name.lower().strip()

    # First check our dermatology-specific mappings
    for drug_key, drug_class in DERMATOLOGY_DRUG_CLASSES.items():
        if drug_key in drug_lower:
            return drug_class

    # Check OpenFDA data for pharmacologic class
    if openfda_data:
        for result in openfda_data:
            openfda = result.get("openfda", {})
            pharm_class = openfda.get("pharm_class_epc", [])
            if pharm_class:
                class_lower = pharm_class[0].lower()
                for class_key in DRUG_CLASS_PRICING.keys():
                    if class_key in class_lower:
                        return class_key

    return "default"


def estimate_price(drug_class: str, quantity: int = 30) -> dict:
    """Estimate price based on drug class."""
    pricing = DRUG_CLASS_PRICING.get(drug_class, DRUG_CLASS_PRICING["default"])

    # Adjust for quantity (base is 30-day supply)
    multiplier = quantity / 30

    return {
        "estimated_low": round(pricing["low"] * multiplier, 2),
        "estimated_mid": round(pricing["mid"] * multiplier, 2),
        "estimated_high": round(pricing["high"] * multiplier, 2),
        "unit": pricing["unit"],
        "drug_class": drug_class
    }


def generate_external_links(drug_name: str, zip_code: str = "94102") -> dict:
    """Generate links to external pricing services."""
    encoded_name = drug_name.replace(" ", "-").lower()

    return {
        "goodrx": f"https://www.goodrx.com/{encoded_name}",
        "rxsaver": f"https://www.rxsaver.com/drugs/{encoded_name}",
        "costco": f"https://www.costco.com/Pharmacy/drug-results-details-price?storeId=10301&drugId={encoded_name}&drugName={drug_name}",
        "blink_health": f"https://www.blinkhealth.com/search?query={drug_name}",
        "amazon_pharmacy": f"https://pharmacy.amazon.com/search?query={drug_name}",
        "singlecare": f"https://www.singlecare.com/prescription/{encoded_name}",
        "manufacturer_coupon": f"https://www.drugs.com/price-guide/{encoded_name}",
    }


def generate_estimated_pharmacy_prices(base_price: float, drug_name: str) -> List[dict]:
    """Generate estimated pharmacy prices based on typical market variations."""
    import random

    pharmacies = [
        {"name": "Costco Pharmacy", "address": "Membership warehouse", "multiplier": 0.70, "membership": True},
        {"name": "Walmart Pharmacy", "address": "Retail location", "multiplier": 0.85, "membership": False},
        {"name": "Amazon Pharmacy", "address": "Online delivery", "multiplier": 0.80, "membership": False},
        {"name": "CVS Pharmacy", "address": "Retail location", "multiplier": 1.0, "membership": False},
        {"name": "Walgreens", "address": "Retail location", "multiplier": 1.05, "membership": False},
        {"name": "Rite Aid", "address": "Retail location", "multiplier": 1.02, "membership": False},
    ]

    # Use drug name hash for consistent "random" variation
    name_hash = sum(ord(c) for c in drug_name.lower())

    prices = []
    for i, pharmacy in enumerate(pharmacies):
        # Add some variation but keep it deterministic based on drug name
        variation = 1 + ((name_hash + i) % 10 - 5) / 100
        estimated_price = round(base_price * pharmacy["multiplier"] * variation, 2)
        retail_price = round(base_price * 1.5, 2)

        prices.append({
            "pharmacy_name": pharmacy["name"],
            "pharmacy_address": pharmacy["address"],
            "price": estimated_price,
            "original_price": retail_price,
            "savings": round(retail_price - estimated_price, 2),
            "savings_percent": round((1 - estimated_price / retail_price) * 100, 1),
            "coupon_available": True,
            "estimated": True,
            "requires_membership": pharmacy["membership"]
        })

    # Sort by price
    prices.sort(key=lambda x: x["price"])
    return prices


@router.get("/medications")
async def search_medication_prices(
    medication_name: str,
    dosage: Optional[str] = None,
    quantity: int = Query(default=30, ge=1, le=180),
    zip_code: Optional[str] = "94102",
    current_user: User = Depends(get_current_active_user)
):
    """
    Search for medication prices using OpenFDA for drug data with estimated pricing.
    Provides links to GoodRx, RxSaver, and other services for exact local prices.
    """
    try:
        # First check if it's in our demo database for exact data
        medication_key = medication_name.lower().replace(" ", "").replace("-", "")
        demo_medication = None

        for key, med in DEMO_MEDICATION_PRICES.items():
            if (medication_key in key or
                medication_key in med["generic_name"].lower().replace(" ", "") or
                medication_key in med["brand_name"].lower().replace(" ", "")):
                demo_medication = med
                demo_medication["key"] = key
                break

        # Fetch real drug data from OpenFDA
        openfda_results = await fetch_openfda_drug(medication_name)
        ndc_results = await fetch_openfda_ndc(medication_name)

        if not openfda_results and not demo_medication:
            # No results found - provide suggestions
            return {
                "medication_found": False,
                "search_term": medication_name,
                "message": "Medication not found in FDA database. Please check the spelling or try the generic name.",
                "suggestions": list(DERMATOLOGY_DRUG_CLASSES.keys())[:10],
                "common_dermatology_drugs": [
                    "tretinoin", "hydrocortisone", "clobetasol", "tacrolimus",
                    "doxycycline", "ketoconazole", "acyclovir", "methotrexate"
                ],
                "prices": [],
                "external_links": generate_external_links(medication_name, zip_code)
            }

        # Extract drug information
        drug_info = {}

        if openfda_results:
            result = openfda_results[0]
            openfda = result.get("openfda", {})

            drug_info = {
                "brand_name": openfda.get("brand_name", [medication_name.title()])[0] if openfda.get("brand_name") else medication_name.title(),
                "generic_name": openfda.get("generic_name", [medication_name.lower()])[0] if openfda.get("generic_name") else medication_name.lower(),
                "manufacturer": openfda.get("manufacturer_name", ["Unknown"])[0] if openfda.get("manufacturer_name") else "Unknown",
                "substance_name": openfda.get("substance_name", []),
                "product_type": openfda.get("product_type", ["HUMAN PRESCRIPTION DRUG"])[0] if openfda.get("product_type") else "HUMAN PRESCRIPTION DRUG",
                "route": openfda.get("route", ["TOPICAL"])[0] if openfda.get("route") else "Unknown",
                "pharm_class": openfda.get("pharm_class_epc", []),
                "indications": result.get("indications_and_usage", [""])[0][:500] if result.get("indications_and_usage") else "",
                "warnings": result.get("warnings", [""])[0][:300] if result.get("warnings") else "",
                "dosage_forms": [],
                "available_dosages": [],
                "ndc_codes": openfda.get("product_ndc", [])[:5],
            }
        elif demo_medication:
            drug_info = {
                "brand_name": demo_medication["brand_name"],
                "generic_name": demo_medication["generic_name"],
                "manufacturer": "Various",
                "substance_name": [demo_medication["generic_name"]],
                "product_type": "HUMAN PRESCRIPTION DRUG",
                "route": "TOPICAL",
                "pharm_class": [demo_medication["drug_class"]],
                "indications": "",
                "warnings": "",
                "dosage_forms": demo_medication["forms"],
                "available_dosages": demo_medication["dosages"],
                "ndc_codes": [],
            }

        # Extract dosage forms from NDC data
        if ndc_results:
            forms = set()
            dosages = set()
            for ndc in ndc_results:
                if ndc.get("dosage_form"):
                    forms.add(ndc["dosage_form"])
                if ndc.get("active_ingredients"):
                    for ing in ndc["active_ingredients"]:
                        if ing.get("strength"):
                            dosages.add(ing["strength"])
            drug_info["dosage_forms"] = list(forms)[:5]
            drug_info["available_dosages"] = list(dosages)[:8]

        # Determine drug class and estimate pricing
        drug_class = get_drug_class(medication_name, openfda_results)
        price_estimate = estimate_price(drug_class, quantity)

        # Generate estimated pharmacy prices
        estimated_prices = generate_estimated_pharmacy_prices(
            price_estimate["estimated_mid"],
            medication_name
        )

        # If we have demo data, merge it for more accurate pricing
        if demo_medication:
            quantity_multiplier = quantity / 30
            demo_prices = []
            for pharmacy in demo_medication["prices"]:
                adjusted_price = round(pharmacy["price"] * quantity_multiplier, 2)
                original_price = round(pharmacy["original"] * quantity_multiplier, 2)
                demo_prices.append({
                    "pharmacy_name": pharmacy["pharmacy"],
                    "pharmacy_address": pharmacy["address"],
                    "price": adjusted_price,
                    "original_price": original_price,
                    "savings": round(original_price - adjusted_price, 2),
                    "savings_percent": round((1 - adjusted_price / original_price) * 100, 1),
                    "coupon_available": True,
                    "coupon_code": f"SKIN{demo_medication['key'].upper()[:4]}",
                    "estimated": False,
                    "requires_membership": pharmacy["pharmacy"] == "Costco"
                })
            demo_prices.sort(key=lambda x: x["price"])
            estimated_prices = demo_prices  # Use demo prices as they're more accurate

        # Generate external links for exact pricing
        external_links = generate_external_links(drug_info["generic_name"], zip_code)

        return {
            "medication_found": True,
            "data_source": "demo" if demo_medication else "openfda",
            "medication": {
                "brand_name": drug_info["brand_name"],
                "generic_name": drug_info["generic_name"],
                "manufacturer": drug_info["manufacturer"],
                "drug_class": drug_class,
                "product_type": drug_info["product_type"],
                "route": drug_info["route"],
                "available_dosages": drug_info["available_dosages"] or ["Various strengths available"],
                "available_forms": drug_info["dosage_forms"] or ["Cream", "Gel", "Ointment"],
                "selected_dosage": dosage or (drug_info["available_dosages"][0] if drug_info["available_dosages"] else "Standard"),
                "quantity": quantity,
                "indications": drug_info["indications"],
                "ndc_codes": drug_info["ndc_codes"],
            },
            "price_estimate": price_estimate,
            "prices": estimated_prices,
            "lowest_price": estimated_prices[0]["price"] if estimated_prices else 0,
            "highest_price": estimated_prices[-1]["price"] if estimated_prices else 0,
            "average_savings": round(sum(p["savings_percent"] for p in estimated_prices) / len(estimated_prices), 1) if estimated_prices else 0,
            "external_links": external_links,
            "disclaimer": "Prices shown are estimates based on national averages. Click 'Get Exact Prices' for real-time local pharmacy pricing.",
            "tips": [
                "Click the GoodRx or RxSaver link for exact prices at pharmacies near you",
                "Generic medications are typically 80-85% cheaper than brand names",
                "Warehouse clubs like Costco often have the lowest prices",
                "Ask your doctor if a 90-day supply is appropriate for additional savings",
                "Some manufacturers offer patient assistance programs for expensive medications",
                "Check if your insurance has a mail-order pharmacy option for better prices"
            ]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error searching medication prices: {str(e)}")


@router.get("/medications/categories")
async def get_medication_categories(
    current_user: User = Depends(get_current_active_user)
):
    """Get common dermatology medication categories with searchable drugs."""
    return {
        "categories": [
            {
                "name": "Retinoids",
                "description": "For acne, anti-aging, and sun damage",
                "examples": ["Tretinoin (Retin-A)", "Adapalene (Differin)", "Isotretinoin (Accutane)", "Tazarotene"],
                "searchable_drugs": ["tretinoin", "adapalene", "isotretinoin", "tazarotene", "trifarotene"],
                "typical_price_range": "$25-$300/month"
            },
            {
                "name": "Corticosteroids",
                "description": "For eczema, psoriasis, and inflammatory conditions",
                "examples": ["Hydrocortisone", "Clobetasol (Temovate)", "Triamcinolone", "Betamethasone"],
                "searchable_drugs": ["hydrocortisone", "clobetasol", "triamcinolone", "betamethasone", "mometasone", "fluocinonide", "desonide"],
                "typical_price_range": "$5-$80/tube"
            },
            {
                "name": "Calcineurin Inhibitors",
                "description": "For eczema and psoriasis without steroids",
                "examples": ["Tacrolimus (Protopic)", "Pimecrolimus (Elidel)"],
                "searchable_drugs": ["tacrolimus", "pimecrolimus"],
                "typical_price_range": "$80-$300/tube"
            },
            {
                "name": "Antibiotics",
                "description": "For acne, rosacea, and skin infections",
                "examples": ["Doxycycline", "Minocycline", "Clindamycin", "Mupirocin"],
                "searchable_drugs": ["doxycycline", "minocycline", "clindamycin", "erythromycin", "metronidazole", "mupirocin"],
                "typical_price_range": "$8-$100/course"
            },
            {
                "name": "Antifungals",
                "description": "For fungal skin infections",
                "examples": ["Ketoconazole", "Terbinafine (Lamisil)", "Fluconazole", "Clotrimazole"],
                "searchable_drugs": ["ketoconazole", "terbinafine", "fluconazole", "clotrimazole", "miconazole", "ciclopirox"],
                "typical_price_range": "$10-$150/treatment"
            },
            {
                "name": "Antivirals",
                "description": "For herpes, cold sores, and shingles",
                "examples": ["Acyclovir (Zovirax)", "Valacyclovir (Valtrex)", "Famciclovir"],
                "searchable_drugs": ["acyclovir", "valacyclovir", "famciclovir", "penciclovir"],
                "typical_price_range": "$15-$300/course"
            },
            {
                "name": "Biologics",
                "description": "For severe psoriasis and eczema",
                "examples": ["Dupixent", "Humira", "Cosentyx", "Skyrizi"],
                "searchable_drugs": ["dupilumab", "adalimumab", "secukinumab", "risankizumab", "ustekinumab", "ixekizumab"],
                "typical_price_range": "$1,500-$7,000/month (before insurance)"
            },
            {
                "name": "Immunomodulators",
                "description": "For severe inflammatory skin diseases",
                "examples": ["Methotrexate", "Cyclosporine", "Azathioprine"],
                "searchable_drugs": ["methotrexate", "cyclosporine", "azathioprine", "mycophenolate"],
                "typical_price_range": "$100-$1,000/month"
            },
            {
                "name": "PDE4 Inhibitors",
                "description": "For psoriasis and eczema",
                "examples": ["Otezla (Apremilast)", "Eucrisa (Crisaborole)"],
                "searchable_drugs": ["apremilast", "crisaborole"],
                "typical_price_range": "$400-$1,200/month"
            },
            {
                "name": "Vitamin D Analogs",
                "description": "For psoriasis",
                "examples": ["Calcipotriene (Dovonex)", "Calcitriol"],
                "searchable_drugs": ["calcipotriene", "calcitriol"],
                "typical_price_range": "$100-$600/tube"
            }
        ],
        "all_searchable_drugs": list(DERMATOLOGY_DRUG_CLASSES.keys()),
        "common_dermatology_medications": list(DEMO_MEDICATION_PRICES.keys()),
        "external_pricing_services": {
            "goodrx": "https://www.goodrx.com",
            "rxsaver": "https://www.rxsaver.com",
            "singlecare": "https://www.singlecare.com",
            "blink_health": "https://www.blinkhealth.com"
        }
    }


# =============================================================================
# COST CALCULATOR
# =============================================================================

@router.post("/calculate")
async def calculate_total_cost(
    request: dict,
    current_user: User = Depends(get_current_active_user)
):
    """Calculate total estimated cost for a treatment plan."""
    try:
        procedures = request.get("procedures", [])
        medications = request.get("medications", [])
        insurance_type = request.get("insurance_type", "private")
        deductible_remaining = request.get("deductible_remaining", 0)

        total_procedure_cost = 0
        total_medication_cost = 0
        total_insurance_pays = 0
        breakdown = []

        # Calculate procedure costs
        for proc_key in procedures:
            if proc_key in DERMATOLOGY_PROCEDURE_COSTS:
                proc = DERMATOLOGY_PROCEDURE_COSTS[proc_key]
                cost = proc["average_cost"]
                coverage = proc["insurance_coverage"]
                insurance_pays = cost * coverage

                total_procedure_cost += cost
                total_insurance_pays += insurance_pays

                breakdown.append({
                    "type": "procedure",
                    "name": proc["name"],
                    "cost": cost,
                    "insurance_pays": round(insurance_pays, 2),
                    "your_cost": round(cost - insurance_pays, 2)
                })

        # Calculate medication costs (using lowest price)
        for med_name in medications:
            med_key = med_name.lower().replace(" ", "").replace("-", "")
            for key, med in DEMO_MEDICATION_PRICES.items():
                if med_key in key or med_key in med["generic_name"].lower():
                    lowest_price = min(p["price"] for p in med["prices"])
                    total_medication_cost += lowest_price

                    breakdown.append({
                        "type": "medication",
                        "name": f"{med['brand_name']} ({med['generic_name']})",
                        "cost": lowest_price,
                        "insurance_pays": 0,  # Medications vary by plan
                        "your_cost": lowest_price
                    })
                    break

        # Apply deductible
        out_of_pocket = (total_procedure_cost - total_insurance_pays) + total_medication_cost
        if deductible_remaining > 0:
            deductible_applied = min(deductible_remaining, out_of_pocket)
            out_of_pocket = out_of_pocket  # Still pay but counts toward deductible

        return {
            "summary": {
                "total_procedure_cost": round(total_procedure_cost, 2),
                "total_medication_cost": round(total_medication_cost, 2),
                "total_cost": round(total_procedure_cost + total_medication_cost, 2),
                "insurance_pays": round(total_insurance_pays, 2),
                "your_estimated_total": round(out_of_pocket, 2),
                "deductible_remaining": deductible_remaining
            },
            "breakdown": breakdown,
            "payment_options": [
                "Pay in full at time of service",
                "Payment plan: 3-12 months interest-free for amounts over $500",
                "HSA/FSA eligible",
                "CareCredit financing available"
            ],
            "disclaimer": "These are estimates only. Actual costs may vary based on your specific insurance plan, provider, and treatment complexity."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating costs: {str(e)}")


# =============================================================================
# INSURANCE COVERAGE CHECK
# =============================================================================

@router.get("/insurance/coverage")
async def check_insurance_coverage(
    procedure_code: str,
    insurance_type: str = "private",
    current_user: User = Depends(get_current_active_user)
):
    """Check typical insurance coverage for a procedure."""
    try:
        # Find procedure by code
        procedure = None
        for key, proc in DERMATOLOGY_PROCEDURE_COSTS.items():
            if proc["code"] == procedure_code:
                procedure = proc
                procedure["key"] = key
                break

        if not procedure:
            raise HTTPException(status_code=404, detail="Procedure code not found")

        # Coverage varies by insurance type
        coverage_info = {
            "medicare": {
                "covered": True,
                "coverage_percent": 80,
                "prior_auth_required": procedure["average_cost"] > 500,
                "notes": "Medicare Part B covers medically necessary dermatology services"
            },
            "medicaid": {
                "covered": True,
                "coverage_percent": 90,
                "prior_auth_required": True,
                "notes": "Coverage varies by state. Prior authorization often required."
            },
            "private": {
                "covered": True,
                "coverage_percent": int(procedure["insurance_coverage"] * 100),
                "prior_auth_required": procedure["average_cost"] > 1000,
                "notes": "Coverage depends on your specific plan. Check with your insurer."
            }
        }

        insurance_type = insurance_type.lower()
        coverage = coverage_info.get(insurance_type, coverage_info["private"])

        return {
            "procedure": {
                "code": procedure["code"],
                "name": procedure["name"],
                "average_cost": procedure["average_cost"]
            },
            "insurance_type": insurance_type,
            "coverage": coverage,
            "tips_for_coverage": [
                "Get a referral from your PCP if required by your plan",
                "Request prior authorization before scheduling if needed",
                "Ask for a cost estimate from the provider's billing department",
                "Keep all documentation for potential appeals"
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking coverage: {str(e)}")


# =============================================================================
# HSA/FSA INTEGRATION
# =============================================================================

# HSA/FSA Eligibility Categories (IRS Publication 502)
HSA_FSA_ELIGIBLE_CATEGORIES = {
    "medical_diagnosis": {
        "name": "Medical Diagnosis & Screening",
        "eligible": True,
        "description": "Diagnostic procedures to detect or monitor medical conditions",
        "examples": ["Skin cancer screening", "Biopsy", "Dermoscopy", "Allergy testing"]
    },
    "medical_treatment": {
        "name": "Medical Treatment",
        "eligible": True,
        "description": "Treatment of diagnosed medical conditions",
        "examples": ["Excision of lesions", "Cryotherapy", "Phototherapy", "Mohs surgery"]
    },
    "prescription_medication": {
        "name": "Prescription Medications",
        "eligible": True,
        "description": "Medications prescribed by a licensed healthcare provider",
        "examples": ["Tretinoin", "Antibiotics", "Corticosteroids", "Biologics"]
    },
    "preventive_care": {
        "name": "Preventive Care",
        "eligible": True,
        "description": "Preventive screenings and care",
        "examples": ["Full body skin exams", "Mole mapping", "Sun damage assessment"]
    },
    "cosmetic": {
        "name": "Cosmetic Procedures",
        "eligible": False,
        "description": "Procedures purely for cosmetic purposes without medical necessity",
        "examples": ["Chemical peels (cosmetic)", "Botox (cosmetic)", "Laser resurfacing (cosmetic)"]
    },
    "otc_with_prescription": {
        "name": "OTC with Prescription",
        "eligible": True,
        "description": "Over-the-counter items with a prescription (CARES Act)",
        "examples": ["Sunscreen (SPF 15+)", "Hydrocortisone cream", "First aid supplies"]
    }
}

# HSA/FSA Eligibility for Procedures
HSA_FSA_PROCEDURE_ELIGIBILITY = {
    "consultation": {"eligible": True, "category": "medical_diagnosis", "notes": "Medical evaluation"},
    "new_patient_visit": {"eligible": True, "category": "medical_diagnosis", "notes": "Initial medical consultation"},
    "skin_biopsy": {"eligible": True, "category": "medical_diagnosis", "notes": "Diagnostic procedure"},
    "excision_benign": {"eligible": True, "category": "medical_treatment", "notes": "Removal of abnormal growth"},
    "excision_malignant": {"eligible": True, "category": "medical_treatment", "notes": "Cancer treatment"},
    "cryotherapy": {"eligible": True, "category": "medical_treatment", "notes": "Treatment of skin lesions"},
    "mohs_surgery": {"eligible": True, "category": "medical_treatment", "notes": "Skin cancer surgery"},
    "phototherapy": {"eligible": True, "category": "medical_treatment", "notes": "Treatment for psoriasis/eczema"},
    "dermoscopy": {"eligible": True, "category": "medical_diagnosis", "notes": "Diagnostic imaging"},
    "chemical_peel": {"eligible": False, "category": "cosmetic", "notes": "Cosmetic unless medically necessary"},
    "laser_treatment": {"eligible": "conditional", "category": "medical_treatment", "notes": "Eligible if medically necessary (e.g., vascular lesions)"},
    "patch_testing": {"eligible": True, "category": "medical_diagnosis", "notes": "Allergy diagnosis"},
    "telemedicine": {"eligible": True, "category": "medical_diagnosis", "notes": "Virtual medical consultation"},
}


class HSAExpenseRecord(BaseModel):
    """HSA/FSA expense record."""
    id: Optional[str] = None
    date: str
    description: str
    category: str
    provider: str
    amount: float
    eligible: bool
    notes: Optional[str] = None


@router.get("/hsa-fsa/eligibility")
async def get_hsa_fsa_eligibility(
    current_user: User = Depends(get_current_active_user)
):
    """Get HSA/FSA eligibility information for all procedures."""
    try:
        eligible_procedures = []
        ineligible_procedures = []
        conditional_procedures = []

        for proc_key, proc_data in DERMATOLOGY_PROCEDURE_COSTS.items():
            eligibility = HSA_FSA_PROCEDURE_ELIGIBILITY.get(proc_key, {
                "eligible": True,
                "category": "medical_treatment",
                "notes": "Generally eligible as medical expense"
            })

            procedure_info = {
                "procedure_key": proc_key,
                "procedure_name": proc_data["name"],
                "procedure_code": proc_data["code"],
                "average_cost": proc_data["average_cost"],
                "eligible": eligibility["eligible"],
                "category": eligibility["category"],
                "category_name": HSA_FSA_ELIGIBLE_CATEGORIES.get(eligibility["category"], {}).get("name", "Medical"),
                "notes": eligibility["notes"]
            }

            if eligibility["eligible"] == True:
                eligible_procedures.append(procedure_info)
            elif eligibility["eligible"] == False:
                ineligible_procedures.append(procedure_info)
            else:
                conditional_procedures.append(procedure_info)

        return {
            "eligible_procedures": eligible_procedures,
            "conditional_procedures": conditional_procedures,
            "ineligible_procedures": ineligible_procedures,
            "categories": HSA_FSA_ELIGIBLE_CATEGORIES,
            "tips": [
                "Keep all receipts and Explanation of Benefits (EOB) statements",
                "Get a Letter of Medical Necessity for conditional items",
                "Prescription required for OTC medications to be eligible",
                "Cosmetic procedures may be eligible if medically necessary (e.g., skin cancer reconstruction)"
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting eligibility: {str(e)}")


@router.get("/hsa-fsa/expenses")
async def get_hsa_fsa_expenses(
    year: int = Query(default=2024),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get HSA/FSA eligible expenses for the user."""
    try:
        from database import AnalysisHistory

        # Get analyses for the year
        from datetime import datetime
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)

        analyses = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == current_user.id,
            AnalysisHistory.created_at >= start_date,
            AnalysisHistory.created_at <= end_date
        ).order_by(AnalysisHistory.created_at.desc()).all()

        expenses = []
        total_eligible = 0
        total_ineligible = 0

        for analysis in analyses:
            # Determine procedure type based on analysis
            proc_key = "dermoscopy" if analysis.dermoscopy_data else "consultation"
            if analysis.risk_level in ["high", "very_high"]:
                proc_key = "skin_biopsy"

            proc_data = DERMATOLOGY_PROCEDURE_COSTS.get(proc_key, DERMATOLOGY_PROCEDURE_COSTS["consultation"])
            eligibility = HSA_FSA_PROCEDURE_ELIGIBILITY.get(proc_key, {"eligible": True, "category": "medical_diagnosis"})

            amount = proc_data["average_cost"]
            is_eligible = eligibility["eligible"] == True

            if is_eligible:
                total_eligible += amount
            else:
                total_ineligible += amount

            expenses.append({
                "id": str(analysis.id),
                "date": analysis.created_at.strftime("%Y-%m-%d"),
                "description": f"{proc_data['name']} - {analysis.predicted_class or 'Skin Analysis'}",
                "category": eligibility["category"],
                "category_name": HSA_FSA_ELIGIBLE_CATEGORIES.get(eligibility["category"], {}).get("name", "Medical"),
                "provider": "Skin Classifier Health",
                "amount": amount,
                "eligible": is_eligible,
                "procedure_code": proc_data["code"],
                "diagnosis": analysis.predicted_class,
                "notes": eligibility.get("notes", "")
            })

        return {
            "year": year,
            "expenses": expenses,
            "summary": {
                "total_eligible": round(total_eligible, 2),
                "total_ineligible": round(total_ineligible, 2),
                "total_expenses": round(total_eligible + total_ineligible, 2),
                "expense_count": len(expenses),
                "eligible_count": len([e for e in expenses if e["eligible"]])
            },
            "category_breakdown": {}  # Could add category-wise breakdown
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting expenses: {str(e)}")


@router.get("/hsa-fsa/receipt/{expense_id}")
async def generate_hsa_fsa_receipt(
    expense_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate HSA/FSA-compliant receipt for an expense."""
    try:
        from database import AnalysisHistory

        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == expense_id,
            AnalysisHistory.user_id == current_user.id
        ).first()

        if not analysis:
            raise HTTPException(status_code=404, detail="Expense not found")

        # Determine procedure
        proc_key = "dermoscopy" if analysis.dermoscopy_data else "consultation"
        if analysis.risk_level in ["high", "very_high"]:
            proc_key = "skin_biopsy"

        proc_data = DERMATOLOGY_PROCEDURE_COSTS.get(proc_key, DERMATOLOGY_PROCEDURE_COSTS["consultation"])
        eligibility = HSA_FSA_PROCEDURE_ELIGIBILITY.get(proc_key, {"eligible": True, "category": "medical_diagnosis"})

        # Generate receipt data
        receipt_number = f"HSA-{analysis.created_at.strftime('%Y%m%d')}-{analysis.id:05d}"

        receipt = {
            "receipt_number": receipt_number,
            "generated_at": datetime.now().isoformat(),

            # Provider Information
            "provider": {
                "name": "Skin Classifier Health Services",
                "address": "123 Medical Center Drive, Suite 100",
                "city_state_zip": "San Francisco, CA 94102",
                "phone": "(555) 123-4567",
                "tax_id": "XX-XXXXXXX",
                "npi": "1234567890"
            },

            # Patient Information
            "patient": {
                "name": current_user.full_name or current_user.username,
                "patient_id": str(current_user.id),
                "date_of_birth": ""  # Would come from user profile
            },

            # Service Information
            "service": {
                "date_of_service": analysis.created_at.strftime("%m/%d/%Y"),
                "procedure_code": proc_data["code"],
                "procedure_name": proc_data["name"],
                "description": proc_data["description"],
                "diagnosis": analysis.predicted_class or "Skin Lesion Evaluation",
                "diagnosis_code": "L98.9"  # Default; would use actual ICD-10
            },

            # Financial Information
            "financial": {
                "amount_charged": proc_data["average_cost"],
                "amount_paid": proc_data["average_cost"],
                "payment_method": "HSA/FSA Card",
                "payment_date": analysis.created_at.strftime("%m/%d/%Y")
            },

            # HSA/FSA Eligibility
            "hsa_fsa": {
                "eligible": eligibility["eligible"] == True,
                "category": HSA_FSA_ELIGIBLE_CATEGORIES.get(eligibility["category"], {}).get("name", "Medical Expense"),
                "eligibility_statement": "This expense qualifies as a medical expense under IRS Publication 502." if eligibility["eligible"] == True else "This expense may not qualify for HSA/FSA reimbursement.",
                "notes": eligibility.get("notes", "")
            },

            # Receipt HTML for PDF generation
            "receipt_html": f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; border-bottom: 2px solid #2563eb; padding-bottom: 20px; margin-bottom: 20px; }}
        .header h1 {{ color: #2563eb; margin: 0; }}
        .header p {{ color: #6b7280; margin: 5px 0; }}
        .section {{ margin-bottom: 20px; }}
        .section-title {{ font-weight: bold; color: #1f2937; border-bottom: 1px solid #e5e7eb; padding-bottom: 5px; margin-bottom: 10px; }}
        .row {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
        .label {{ color: #6b7280; }}
        .value {{ font-weight: 500; }}
        .eligible-badge {{ background: #10b981; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }}
        .ineligible-badge {{ background: #ef4444; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }}
        .total {{ font-size: 24px; color: #2563eb; font-weight: bold; }}
        .footer {{ text-align: center; color: #9ca3af; font-size: 12px; margin-top: 40px; border-top: 1px solid #e5e7eb; padding-top: 20px; }}
        .hsa-statement {{ background: #f0fdf4; border: 1px solid #10b981; padding: 15px; border-radius: 8px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HSA/FSA ELIGIBLE RECEIPT</h1>
        <p>Receipt #: {receipt_number}</p>
        <p>Date Generated: {datetime.now().strftime("%m/%d/%Y")}</p>
    </div>

    <div class="section">
        <div class="section-title">Provider Information</div>
        <div class="row"><span class="label">Provider:</span><span class="value">Skin Classifier Health Services</span></div>
        <div class="row"><span class="label">Address:</span><span class="value">123 Medical Center Drive, Suite 100, San Francisco, CA 94102</span></div>
        <div class="row"><span class="label">Phone:</span><span class="value">(555) 123-4567</span></div>
        <div class="row"><span class="label">Tax ID:</span><span class="value">XX-XXXXXXX</span></div>
    </div>

    <div class="section">
        <div class="section-title">Patient Information</div>
        <div class="row"><span class="label">Patient Name:</span><span class="value">{current_user.full_name or current_user.username}</span></div>
        <div class="row"><span class="label">Patient ID:</span><span class="value">{current_user.id}</span></div>
    </div>

    <div class="section">
        <div class="section-title">Service Details</div>
        <div class="row"><span class="label">Date of Service:</span><span class="value">{analysis.created_at.strftime("%m/%d/%Y")}</span></div>
        <div class="row"><span class="label">Procedure Code (CPT):</span><span class="value">{proc_data["code"]}</span></div>
        <div class="row"><span class="label">Procedure:</span><span class="value">{proc_data["name"]}</span></div>
        <div class="row"><span class="label">Description:</span><span class="value">{proc_data["description"]}</span></div>
        <div class="row"><span class="label">Diagnosis:</span><span class="value">{analysis.predicted_class or "Skin Lesion Evaluation"}</span></div>
    </div>

    <div class="section">
        <div class="section-title">Payment Information</div>
        <div class="row"><span class="label">Amount Charged:</span><span class="value">${proc_data["average_cost"]:.2f}</span></div>
        <div class="row"><span class="label">Amount Paid:</span><span class="total">${proc_data["average_cost"]:.2f}</span></div>
        <div class="row"><span class="label">Payment Method:</span><span class="value">HSA/FSA Card</span></div>
        <div class="row"><span class="label">Payment Date:</span><span class="value">{analysis.created_at.strftime("%m/%d/%Y")}</span></div>
    </div>

    <div class="section">
        <div class="section-title">HSA/FSA Eligibility</div>
        <div class="row">
            <span class="label">Status:</span>
            <span class="{'eligible-badge' if eligibility['eligible'] == True else 'ineligible-badge'}">
                {'ELIGIBLE' if eligibility['eligible'] == True else 'NOT ELIGIBLE'}
            </span>
        </div>
        <div class="row"><span class="label">Category:</span><span class="value">{HSA_FSA_ELIGIBLE_CATEGORIES.get(eligibility['category'], {}).get('name', 'Medical Expense')}</span></div>
    </div>

    <div class="hsa-statement">
        <strong>HSA/FSA Eligibility Statement:</strong><br>
        {'This expense qualifies as a medical expense under IRS Publication 502 and is eligible for HSA/FSA reimbursement.' if eligibility['eligible'] == True else 'This expense may not qualify for HSA/FSA reimbursement. Please consult IRS Publication 502 or your plan administrator.'}
    </div>

    <div class="footer">
        <p>This receipt is provided for HSA/FSA reimbursement purposes.</p>
        <p>Please retain for your tax records.</p>
        <p>Questions? Contact support@skinclassifier.com</p>
    </div>
</body>
</html>
            """
        }

        return receipt

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating receipt: {str(e)}")


@router.get("/hsa-fsa/year-summary/{year}")
async def get_hsa_fsa_year_summary(
    year: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get year-end HSA/FSA summary for tax purposes."""
    try:
        from database import AnalysisHistory

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)

        analyses = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == current_user.id,
            AnalysisHistory.created_at >= start_date,
            AnalysisHistory.created_at <= end_date
        ).order_by(AnalysisHistory.created_at).all()

        # Calculate monthly breakdown
        monthly_expenses = {i: {"eligible": 0, "ineligible": 0, "count": 0} for i in range(1, 13)}
        category_totals = {}
        total_eligible = 0
        total_ineligible = 0

        for analysis in analyses:
            month = analysis.created_at.month

            proc_key = "dermoscopy" if analysis.dermoscopy_data else "consultation"
            if analysis.risk_level in ["high", "very_high"]:
                proc_key = "skin_biopsy"

            proc_data = DERMATOLOGY_PROCEDURE_COSTS.get(proc_key, DERMATOLOGY_PROCEDURE_COSTS["consultation"])
            eligibility = HSA_FSA_PROCEDURE_ELIGIBILITY.get(proc_key, {"eligible": True, "category": "medical_diagnosis"})

            amount = proc_data["average_cost"]
            category = eligibility["category"]
            is_eligible = eligibility["eligible"] == True

            monthly_expenses[month]["count"] += 1
            if is_eligible:
                monthly_expenses[month]["eligible"] += amount
                total_eligible += amount
            else:
                monthly_expenses[month]["ineligible"] += amount
                total_ineligible += amount

            if category not in category_totals:
                category_totals[category] = {"name": HSA_FSA_ELIGIBLE_CATEGORIES.get(category, {}).get("name", category), "total": 0, "count": 0}
            category_totals[category]["total"] += amount
            category_totals[category]["count"] += 1

        return {
            "year": year,
            "patient": {
                "name": current_user.full_name or current_user.username,
                "id": str(current_user.id)
            },
            "summary": {
                "total_eligible_expenses": round(total_eligible, 2),
                "total_ineligible_expenses": round(total_ineligible, 2),
                "total_all_expenses": round(total_eligible + total_ineligible, 2),
                "total_expense_count": len(analyses),
                "eligible_expense_count": len([a for a in analyses if HSA_FSA_PROCEDURE_ELIGIBILITY.get("consultation", {}).get("eligible", True)])
            },
            "monthly_breakdown": [
                {
                    "month": month,
                    "month_name": datetime(year, month, 1).strftime("%B"),
                    "eligible_amount": round(data["eligible"], 2),
                    "ineligible_amount": round(data["ineligible"], 2),
                    "total_amount": round(data["eligible"] + data["ineligible"], 2),
                    "expense_count": data["count"]
                }
                for month, data in monthly_expenses.items()
            ],
            "category_breakdown": [
                {
                    "category": cat,
                    "category_name": data["name"],
                    "total_amount": round(data["total"], 2),
                    "expense_count": data["count"]
                }
                for cat, data in category_totals.items()
            ],
            "tax_notes": [
                f"Total HSA/FSA eligible medical expenses for {year}: ${total_eligible:.2f}",
                "Keep all receipts and documentation for IRS records",
                "Medical expenses exceeding 7.5% of AGI may be tax-deductible (Schedule A)",
                "Consult a tax professional for personalized advice"
            ],
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating year summary: {str(e)}")
