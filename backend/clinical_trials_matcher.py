"""
Clinical Trials Matching Algorithm

Matches users to clinical trials based on:
- Diagnosis history (35% weight - exact match)
- Related conditions (15% weight)
- Genetic/biomarker matching (15% weight) - NEW
- Location proximity (12% weight)
- Age eligibility (10% weight)
- Gender eligibility (8% weight)
- Trial phase preference (5% weight)

Priority: Diagnosis matching is critical per user requirements.
Genetic matching provides precision medicine enhancement.
"""

import math
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from sqlalchemy.orm import Session

from database import (
    User, UserProfile, AnalysisHistory, ClinicalTrial,
    TrialMatch, GeneticVariant, SessionLocal
)

# =============================================================================
# MATCHING WEIGHTS (Total: 100)
# =============================================================================

WEIGHTS = {
    "diagnosis_exact": 35,      # Exact diagnosis match
    "diagnosis_related": 15,    # Related condition match
    "genetic_match": 15,        # Genetic/biomarker matching (NEW)
    "location_nearby": 12,      # Within preferred distance
    "age_eligible": 10,         # Meets age criteria
    "gender_eligible": 8,       # Meets gender criteria
    "phase_preferred": 5,       # Phase 2/3 trials preferred
}

# =============================================================================
# CONDITION MAPPINGS
# =============================================================================

# Map our predicted classes to trial condition terms
CONDITION_MAPPINGS = {
    # Malignant conditions
    "melanoma": ["melanoma", "cutaneous melanoma", "skin cancer", "malignant melanoma"],
    "mel": ["melanoma", "cutaneous melanoma", "skin cancer"],
    "basal_cell_carcinoma": ["basal cell carcinoma", "basal cell", "bcc", "skin cancer", "nonmelanoma skin cancer"],
    "bcc": ["basal cell carcinoma", "skin cancer", "nonmelanoma skin cancer"],
    "squamous_cell_carcinoma": ["squamous cell carcinoma", "squamous cell", "scc", "skin cancer", "nonmelanoma skin cancer"],
    "akiec": ["actinic keratosis", "actinic keratoses", "solar keratosis", "precancerous"],

    # Benign lesions
    "melanocytic_nevus": ["nevus", "mole", "melanocytic nevus", "dysplastic nevus"],
    "nv": ["nevus", "mole", "melanocytic nevus"],
    "seborrheic_keratosis": ["seborrheic keratosis", "seborrheic keratoses"],
    "bkl": ["benign keratosis", "seborrheic keratosis"],
    "dermatofibroma": ["dermatofibroma", "benign skin lesion"],
    "df": ["dermatofibroma"],
    "vascular_lesion": ["vascular lesion", "hemangioma", "angioma"],
    "vasc": ["vascular lesion"],

    # Inflammatory conditions
    "eczema": ["eczema", "atopic dermatitis", "dermatitis"],
    "psoriasis": ["psoriasis", "plaque psoriasis"],
    "contact_dermatitis": ["contact dermatitis", "dermatitis"],
    "seborrheic_dermatitis": ["seborrheic dermatitis", "dermatitis"],
    "urticaria": ["urticaria", "hives"],
    "rosacea": ["rosacea"],
    "acne": ["acne", "acne vulgaris"],

    # Infectious conditions
    "fungal_infection": ["fungal infection", "tinea", "ringworm", "dermatomycosis"],
    "bacterial_infection": ["bacterial infection", "cellulitis", "impetigo"],
    "viral_infection": ["viral infection", "herpes", "warts", "molluscum"],
}

# Related conditions (for partial matching)
RELATED_CONDITIONS = {
    "melanoma": {"skin cancer", "nevus", "dysplastic nevus"},
    "basal_cell_carcinoma": {"skin cancer", "actinic keratosis", "nonmelanoma"},
    "squamous_cell_carcinoma": {"skin cancer", "actinic keratosis", "nonmelanoma"},
    "actinic_keratosis": {"skin cancer", "precancerous"},
    "eczema": {"dermatitis", "atopic dermatitis"},
    "psoriasis": {"autoimmune", "inflammatory skin"},
}

# =============================================================================
# BIOMARKER / GENETIC MAPPINGS
# =============================================================================

# Map gene symbols to their common aliases and variant patterns
BIOMARKER_GENE_MAPPING = {
    "BRAF": {
        "aliases": ["BRAF", "B-RAF", "B-Raf"],
        "common_variants": ["V600E", "V600K", "V600D", "V600R", "K601E"],
        "variant_patterns": [r"V600[EKDR]?", r"K601[ENQ]?"],
    },
    "NRAS": {
        "aliases": ["NRAS", "N-RAS", "N-Ras"],
        "common_variants": ["Q61K", "Q61R", "Q61L", "Q61H", "G12D", "G13R"],
        "variant_patterns": [r"Q61[KRLH]?", r"G12[DVCRAS]?", r"G13[RDC]?"],
    },
    "KIT": {
        "aliases": ["KIT", "C-KIT", "c-Kit", "CD117"],
        "common_variants": ["D816V", "D816H", "L576P", "K642E"],
        "variant_patterns": [r"D816[VHY]?", r"L576P", r"K642E"],
    },
    "CDKN2A": {
        "aliases": ["CDKN2A", "p16", "INK4A", "p16INK4a"],
        "common_variants": ["deletion", "loss", "inactivation"],
        "variant_patterns": [],
    },
    "PTEN": {
        "aliases": ["PTEN"],
        "common_variants": ["loss", "deletion", "mutation"],
        "variant_patterns": [],
    },
    "NF1": {
        "aliases": ["NF1", "neurofibromin"],
        "common_variants": ["loss", "mutation", "inactivation"],
        "variant_patterns": [],
    },
    "TP53": {
        "aliases": ["TP53", "p53"],
        "common_variants": ["mutation", "loss"],
        "variant_patterns": [],
    },
}

# Biomarker status mappings (for PD-L1, TMB, MSI, etc.)
BIOMARKER_STATUS_MAPPING = {
    "PD-L1": {
        "positive_terms": ["PD-L1 positive", "PD-L1+", "PD-L1 >= 1%", "PD-L1 high", "TPS >= 1%", "CPS >= 1"],
        "negative_terms": ["PD-L1 negative", "PD-L1-", "PD-L1 < 1%"],
        "score_threshold": 1,  # TPS percentage
    },
    "TMB": {
        "positive_terms": ["TMB-high", "TMB-H", "TMB >= 10", "high TMB", "tumor mutational burden high"],
        "negative_terms": ["TMB-low", "TMB-L", "TMB < 10", "low TMB"],
        "score_threshold": 10,  # mutations/Mb
    },
    "MSI": {
        "positive_terms": ["MSI-high", "MSI-H", "microsatellite instability high", "dMMR"],
        "negative_terms": ["MSI-stable", "MSS", "microsatellite stable", "pMMR"],
    },
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in miles.
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Earth's radius in miles
    r = 3956

    return c * r


def get_user_conditions(db: Session, user_id: int) -> Set[str]:
    """
    Extract all diagnosed conditions from user's analysis history.
    Returns normalized condition names.
    """
    conditions = set()

    # Get all analyses for user
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == user_id
    ).all()

    for analysis in analyses:
        # Primary diagnosis
        if analysis.predicted_class:
            conditions.add(analysis.predicted_class.lower())

        # Biopsy-confirmed diagnosis (most reliable)
        if analysis.biopsy_result:
            conditions.add(analysis.biopsy_result.lower())

        # Inflammatory conditions
        if analysis.inflammatory_condition:
            conditions.add(analysis.inflammatory_condition.lower())

        # Infectious disease
        if analysis.infectious_disease:
            conditions.add(analysis.infectious_disease.lower())

    return conditions


def get_user_genetic_variants(db: Session, user_id: int) -> Dict[str, List[Dict]]:
    """
    Retrieve user's genetic variants organized by gene symbol.

    Returns:
        Dict mapping gene symbol to list of variant details
    """
    variants_by_gene = {}

    variants = db.query(GeneticVariant).filter(
        GeneticVariant.user_id == user_id,
        GeneticVariant.classification.in_(["pathogenic", "likely_pathogenic", "vus"])
    ).all()

    for variant in variants:
        gene = variant.gene_symbol.upper() if variant.gene_symbol else None
        if not gene:
            continue

        if gene not in variants_by_gene:
            variants_by_gene[gene] = []

        # Extract variant name from HGVS protein notation (e.g., "p.Val600Glu" -> "V600E")
        variant_name = None
        if variant.hgvs_p:
            # Parse HGVS protein notation
            hgvs = variant.hgvs_p.replace("p.", "")
            # Common amino acid abbreviations
            aa_map = {
                "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
                "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
                "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
                "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
            }
            # Try to convert 3-letter to 1-letter (e.g., Val600Glu -> V600E)
            for three, one in aa_map.items():
                hgvs = hgvs.replace(three, one)
            variant_name = hgvs

        variants_by_gene[gene].append({
            "variant_name": variant_name,
            "hgvs_p": variant.hgvs_p,
            "hgvs_c": variant.hgvs_c,
            "classification": variant.classification,
            "clinvar_significance": variant.clinvar_significance,
            "consequence": variant.consequence,
            "zygosity": variant.zygosity,
        })

    return variants_by_gene


def calculate_genetic_score(
    user_variants: Dict[str, List[Dict]],
    trial: ClinicalTrial
) -> Dict:
    """
    Calculate genetic/biomarker match score for a user-trial pair.

    Returns dict with:
        - genetic_score (0-15)
        - matched_biomarkers: List of biomarkers user has that trial requires
        - missing_biomarkers: List of required biomarkers user doesn't have
        - excluded_biomarkers_found: List of excluded biomarkers user has
        - genetic_eligible: Boolean - False if user has an excluded biomarker
        - genetic_match_type: "exact", "partial", "none", "excluded"
    """
    result = {
        "genetic_score": 0,
        "matched_biomarkers": [],
        "missing_biomarkers": [],
        "excluded_biomarkers_found": [],
        "genetic_eligible": True,
        "genetic_match_type": "none",
    }

    # Get trial biomarker requirements
    required_biomarkers = trial.required_biomarkers or []
    excluded_biomarkers = trial.excluded_biomarkers or []
    genetic_requirements = trial.genetic_requirements or []

    # If no biomarker requirements, give neutral score
    if not required_biomarkers and not excluded_biomarkers and not genetic_requirements:
        result["genetic_score"] = WEIGHTS["genetic_match"] * 0.3  # Small bonus for non-biomarker trials
        result["genetic_match_type"] = "not_applicable"
        return result

    # Check for excluded biomarkers first (disqualifying)
    user_genes = set(user_variants.keys())
    for excluded in excluded_biomarkers:
        excluded_upper = excluded.upper()
        # Check if user has this gene/mutation
        for gene in user_genes:
            if gene == excluded_upper or excluded_upper in gene:
                # User has an excluded biomarker
                result["excluded_biomarkers_found"].append(excluded)
                result["genetic_eligible"] = False
                result["genetic_match_type"] = "excluded"

    if not result["genetic_eligible"]:
        result["genetic_score"] = 0
        return result

    # Check for required biomarkers
    total_required = len(required_biomarkers)
    matched_count = 0

    for required in required_biomarkers:
        required_upper = required.upper()
        matched = False

        # Check if user has this gene
        for gene, variants in user_variants.items():
            # Check gene match
            gene_info = BIOMARKER_GENE_MAPPING.get(gene, {})
            aliases = gene_info.get("aliases", [gene])

            if any(alias.upper() == required_upper or required_upper in alias.upper()
                   for alias in aliases):
                # Gene match - check if specific variant required
                matched = True
                result["matched_biomarkers"].append(f"{gene} ({variants[0].get('variant_name', 'mutation')})")
                break

            # Check for specific variant match (e.g., "BRAF V600E")
            if gene in required_upper:
                for variant in variants:
                    variant_name = variant.get("variant_name", "")
                    if variant_name and variant_name in required_upper:
                        matched = True
                        result["matched_biomarkers"].append(f"{gene} {variant_name}")
                        break

        if matched:
            matched_count += 1
        else:
            result["missing_biomarkers"].append(required)

    # Calculate score based on match ratio
    if total_required > 0:
        match_ratio = matched_count / total_required

        if match_ratio == 1.0:
            result["genetic_score"] = WEIGHTS["genetic_match"]
            result["genetic_match_type"] = "exact"
        elif match_ratio >= 0.5:
            result["genetic_score"] = WEIGHTS["genetic_match"] * 0.7
            result["genetic_match_type"] = "partial"
        elif match_ratio > 0:
            result["genetic_score"] = WEIGHTS["genetic_match"] * 0.4
            result["genetic_match_type"] = "partial"
        else:
            result["genetic_score"] = 0
            result["genetic_match_type"] = "none"

    # Bonus for targeted therapy trials if user has relevant mutations
    if trial.targeted_therapy_trial and user_variants:
        result["genetic_score"] = min(result["genetic_score"] * 1.2, WEIGHTS["genetic_match"])

    return result


def match_conditions(
    user_conditions: Set[str],
    trial_conditions: List[str]
) -> Tuple[float, List[str], bool]:
    """
    Match user conditions against trial conditions.

    Returns:
        - score (0-60): Combined diagnosis score
        - matched_conditions: List of matched condition names
        - exact_match: Whether there was an exact match
    """
    if not trial_conditions or not user_conditions:
        return 0, [], False

    score = 0
    matched = []
    exact_match = False

    # Normalize trial conditions
    trial_conditions_lower = [c.lower() for c in trial_conditions]

    for user_condition in user_conditions:
        # Get possible matching terms for this condition
        matching_terms = CONDITION_MAPPINGS.get(user_condition, [user_condition])

        # Check for exact match
        for term in matching_terms:
            for trial_condition in trial_conditions_lower:
                if term.lower() in trial_condition or trial_condition in term.lower():
                    if not exact_match:
                        score += WEIGHTS["diagnosis_exact"]
                        exact_match = True
                    matched.append(trial_condition)
                    break

        # Check for related condition match (only if no exact match yet)
        if not exact_match:
            related = RELATED_CONDITIONS.get(user_condition, set())
            for related_term in related:
                for trial_condition in trial_conditions_lower:
                    if related_term.lower() in trial_condition:
                        score += WEIGHTS["diagnosis_related"]
                        matched.append(f"{trial_condition} (related)")
                        break

    return min(score, 60), list(set(matched)), exact_match


def calculate_location_score(
    user_profile: Optional[UserProfile],
    trial_locations: List[Dict]
) -> Tuple[float, float, Optional[Dict]]:
    """
    Calculate location score based on proximity.

    Returns:
        - score (0-15)
        - distance_miles to nearest location
        - nearest_location dict
    """
    if not user_profile or not user_profile.latitude or not user_profile.longitude:
        return 0, None, None

    if not trial_locations:
        return 0, None, None

    user_lat = user_profile.latitude
    user_lng = user_profile.longitude
    preferred_distance = user_profile.preferred_distance_miles or 50

    min_distance = float('inf')
    nearest_location = None

    for location in trial_locations:
        # Skip if no coordinates (we could geocode, but that's expensive)
        loc_lat = location.get("lat")
        loc_lng = location.get("lng")

        # If no coordinates, use state matching as fallback
        if not loc_lat or not loc_lng:
            # Simple state match gives partial score
            if user_profile.state and location.get("state"):
                if user_profile.state.lower() == location.get("state", "").lower():
                    return WEIGHTS["location_nearby"] * 0.5, None, location
            continue

        distance = haversine_distance(user_lat, user_lng, loc_lat, loc_lng)

        if distance < min_distance:
            min_distance = distance
            nearest_location = location

    if nearest_location is None:
        return 0, None, None

    # Score based on distance
    if min_distance <= preferred_distance:
        score = WEIGHTS["location_nearby"]
    elif min_distance <= preferred_distance * 2:
        score = WEIGHTS["location_nearby"] * 0.7
    elif min_distance <= preferred_distance * 4:
        score = WEIGHTS["location_nearby"] * 0.4
    else:
        score = WEIGHTS["location_nearby"] * 0.1

    return score, min_distance, nearest_location


def calculate_age_score(user: User, trial: ClinicalTrial) -> Tuple[float, bool]:
    """
    Check if user meets age eligibility.

    Returns:
        - score (0-10)
        - eligible (bool)
    """
    user_age = user.age

    if not user_age:
        # Can't determine eligibility
        return 0, True  # Assume eligible if unknown

    min_age = trial.min_age or 0
    max_age = trial.max_age or 999

    if min_age <= user_age <= max_age:
        return WEIGHTS["age_eligible"], True
    else:
        return 0, False


def calculate_gender_score(user: User, trial: ClinicalTrial) -> Tuple[float, bool]:
    """
    Check if user meets gender eligibility.

    Returns:
        - score (0-10)
        - eligible (bool)
    """
    trial_gender = trial.gender

    if not trial_gender or trial_gender.lower() == "all":
        return WEIGHTS["gender_eligible"], True

    user_gender = user.gender
    if not user_gender:
        return WEIGHTS["gender_eligible"] * 0.5, True  # Assume might be eligible

    # Normalize gender values
    trial_gender_lower = trial_gender.lower()
    user_gender_lower = user_gender.lower()

    if trial_gender_lower in ["male", "men", "m"]:
        eligible = user_gender_lower in ["male", "m", "man"]
    elif trial_gender_lower in ["female", "women", "f"]:
        eligible = user_gender_lower in ["female", "f", "woman"]
    else:
        eligible = True

    return WEIGHTS["gender_eligible"] if eligible else 0, eligible


def calculate_phase_score(trial: ClinicalTrial) -> float:
    """
    Give bonus for Phase 2/3 trials (more likely to be effective).
    """
    phase = trial.phase or ""
    phase_lower = phase.lower()

    if "phase 2" in phase_lower or "phase 3" in phase_lower:
        return WEIGHTS["phase_preferred"]
    elif "phase 4" in phase_lower:
        return WEIGHTS["phase_preferred"] * 0.8
    elif "phase 1" in phase_lower:
        return WEIGHTS["phase_preferred"] * 0.3
    else:
        return WEIGHTS["phase_preferred"] * 0.5


def calculate_match(
    user: User,
    user_profile: Optional[UserProfile],
    user_conditions: Set[str],
    user_variants: Dict[str, List[Dict]],
    trial: ClinicalTrial
) -> Dict:
    """
    Calculate comprehensive match score for a user-trial pair.

    Returns dict with:
        - match_score (0-100)
        - match_reasons
        - unmet_criteria
        - diagnosis_score
        - matched_conditions
        - distance_miles
        - nearest_location
        - age_eligible
        - gender_eligible
        - genetic_score (NEW)
        - matched_biomarkers (NEW)
        - missing_biomarkers (NEW)
        - excluded_biomarkers_found (NEW)
        - genetic_eligible (NEW)
        - genetic_match_type (NEW)
    """
    reasons = []
    unmet = []

    # 1. Diagnosis matching (40% exact + 20% related = 60% max)
    diagnosis_score, matched_conditions, exact_match = match_conditions(
        user_conditions, trial.conditions or []
    )
    if exact_match:
        reasons.append("diagnosis_exact_match")
    elif matched_conditions:
        reasons.append("diagnosis_related_match")
    else:
        unmet.append("no_diagnosis_match")

    # 2. Location score (15%)
    location_score, distance_miles, nearest_location = calculate_location_score(
        user_profile, trial.locations or []
    )
    if location_score > 0:
        reasons.append("location_nearby")
    else:
        unmet.append("location_unknown_or_far")

    # 3. Age eligibility (10%)
    age_score, age_eligible = calculate_age_score(user, trial)
    if age_eligible:
        if age_score > 0:
            reasons.append("age_eligible")
    else:
        unmet.append("age_ineligible")

    # 4. Gender eligibility (8%)
    gender_score, gender_eligible = calculate_gender_score(user, trial)
    if gender_eligible:
        if gender_score > 0:
            reasons.append("gender_eligible")
    else:
        unmet.append("gender_ineligible")

    # 5. Genetic/biomarker matching (15%) - NEW
    genetic_result = calculate_genetic_score(user_variants, trial)
    genetic_score = genetic_result["genetic_score"]
    genetic_eligible = genetic_result["genetic_eligible"]

    if genetic_result["genetic_match_type"] == "exact":
        reasons.append("genetic_exact_match")
    elif genetic_result["genetic_match_type"] == "partial":
        reasons.append("genetic_partial_match")
    elif genetic_result["genetic_match_type"] == "excluded":
        unmet.append("genetic_excluded")
    elif genetic_result["genetic_match_type"] == "none" and trial.required_biomarkers:
        unmet.append("genetic_not_matched")

    # 6. Phase preference (5%)
    phase_score = calculate_phase_score(trial)
    if phase_score >= WEIGHTS["phase_preferred"]:
        reasons.append("preferred_phase")

    # Calculate total score
    total_score = (
        diagnosis_score + location_score + age_score +
        gender_score + genetic_score + phase_score
    )

    return {
        "match_score": round(total_score, 2),
        "match_reasons": reasons,
        "unmet_criteria": unmet,
        "diagnosis_score": diagnosis_score,
        "matched_conditions": matched_conditions,
        "distance_miles": round(distance_miles, 1) if distance_miles else None,
        "nearest_location": nearest_location,
        "age_eligible": age_eligible,
        "gender_eligible": gender_eligible,
        # Genetic matching fields (NEW)
        "genetic_score": round(genetic_score, 2),
        "matched_biomarkers": genetic_result["matched_biomarkers"],
        "missing_biomarkers": genetic_result["missing_biomarkers"],
        "excluded_biomarkers_found": genetic_result["excluded_biomarkers_found"],
        "genetic_eligible": genetic_eligible,
        "genetic_match_type": genetic_result["genetic_match_type"],
    }


def find_matches_for_user(
    db: Session,
    user_id: int,
    min_score: float = 20,
    limit: int = 50
) -> List[Dict]:
    """
    Find and score all trial matches for a user.

    Args:
        db: Database session
        user_id: User ID to match
        min_score: Minimum match score to include (default 20)
        limit: Maximum matches to return

    Returns:
        List of match dicts sorted by score descending
    """
    # Get user and profile
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return []

    user_profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

    # Get user's conditions
    user_conditions = get_user_conditions(db, user_id)

    if not user_conditions:
        # No diagnosis history - can't match effectively
        return []

    # Get user's genetic variants (NEW)
    user_variants = get_user_genetic_variants(db, user_id)

    # Get all recruiting trials
    trials = db.query(ClinicalTrial).filter(
        ClinicalTrial.status.in_(["Recruiting", "RECRUITING", "Active, not recruiting"])
    ).all()

    matches = []

    for trial in trials:
        match_result = calculate_match(
            user, user_profile, user_conditions, user_variants, trial
        )

        # Only include if meets minimum score and is eligible
        # Now also checks genetic_eligible (NEW)
        if match_result["match_score"] >= min_score:
            if (match_result["age_eligible"] and
                match_result["gender_eligible"] and
                match_result["genetic_eligible"]):
                match_result["trial_id"] = trial.id
                match_result["trial"] = {
                    "id": trial.id,
                    "nct_id": trial.nct_id,
                    "title": trial.title,
                    "phase": trial.phase,
                    "status": trial.status,
                    "conditions": trial.conditions,
                    "sponsor": trial.sponsor,
                    "url": trial.url,
                    # Biomarker info (NEW)
                    "required_biomarkers": trial.required_biomarkers,
                    "excluded_biomarkers": trial.excluded_biomarkers,
                    "targeted_therapy_trial": trial.targeted_therapy_trial,
                    "requires_genetic_testing": trial.requires_genetic_testing,
                }
                matches.append(match_result)

    # Sort by score descending
    matches.sort(key=lambda x: x["match_score"], reverse=True)

    return matches[:limit]


def create_or_update_matches(db: Session, user_id: int) -> int:
    """
    Create or update TrialMatch records for a user.

    Returns number of matches created/updated.
    """
    matches = find_matches_for_user(db, user_id)

    count = 0
    for match_data in matches:
        trial_id = match_data["trial_id"]

        # Check if match exists
        existing = db.query(TrialMatch).filter(
            TrialMatch.user_id == user_id,
            TrialMatch.trial_id == trial_id
        ).first()

        if existing:
            # Update existing match
            existing.match_score = match_data["match_score"]
            existing.match_reasons = match_data["match_reasons"]
            existing.unmet_criteria = match_data["unmet_criteria"]
            existing.matched_conditions = match_data["matched_conditions"]
            existing.diagnosis_score = match_data["diagnosis_score"]
            existing.distance_miles = match_data["distance_miles"]
            existing.nearest_location = match_data["nearest_location"]
            existing.age_eligible = match_data["age_eligible"]
            existing.gender_eligible = match_data["gender_eligible"]
            # Genetic match fields (NEW)
            existing.genetic_score = match_data.get("genetic_score", 0)
            existing.matched_biomarkers = match_data.get("matched_biomarkers", [])
            existing.missing_biomarkers = match_data.get("missing_biomarkers", [])
            existing.excluded_biomarkers_found = match_data.get("excluded_biomarkers_found", [])
            existing.genetic_eligible = match_data.get("genetic_eligible", True)
            existing.genetic_match_type = match_data.get("genetic_match_type", "none")
            existing.updated_at = datetime.utcnow()
        else:
            # Create new match
            trial_match = TrialMatch(
                user_id=user_id,
                trial_id=trial_id,
                match_score=match_data["match_score"],
                match_reasons=match_data["match_reasons"],
                unmet_criteria=match_data["unmet_criteria"],
                matched_conditions=match_data["matched_conditions"],
                diagnosis_score=match_data["diagnosis_score"],
                distance_miles=match_data["distance_miles"],
                nearest_location=match_data["nearest_location"],
                age_eligible=match_data["age_eligible"],
                gender_eligible=match_data["gender_eligible"],
                # Genetic match fields (NEW)
                genetic_score=match_data.get("genetic_score", 0),
                matched_biomarkers=match_data.get("matched_biomarkers", []),
                missing_biomarkers=match_data.get("missing_biomarkers", []),
                excluded_biomarkers_found=match_data.get("excluded_biomarkers_found", []),
                genetic_eligible=match_data.get("genetic_eligible", True),
                genetic_match_type=match_data.get("genetic_match_type", "none"),
                status="matched",
            )
            db.add(trial_match)

        count += 1

    db.commit()
    return count
