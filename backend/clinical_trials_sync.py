"""
Clinical Trials Sync Service

Syncs dermatology-related clinical trials from ClinicalTrials.gov API.
API Documentation: https://clinicaltrials.gov/data-api/api

This service:
- Fetches recruiting trials for skin conditions
- Parses and normalizes trial data
- Upserts trials into the database
- Tracks sync timestamps for incremental updates
"""

import os
import re
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Set
from sqlalchemy.orm import Session

from database import ClinicalTrial, SessionLocal


# =============================================================================
# BIOMARKER/GENETIC EXTRACTION
# =============================================================================

# Known biomarkers and genetic variants relevant to dermatology/skin cancer trials
BIOMARKER_PATTERNS = {
    # BRAF mutations (very common in melanoma)
    "BRAF": {
        "gene": "BRAF",
        "patterns": [
            r"BRAF[\s\-]*V600[EKR]?",
            r"BRAF[\s\-]*mutation",
            r"BRAF[\s\-]*positive",
            r"BRAF[\s\-]*mutant",
            r"BRAF[\s\-]*wild[\s\-]*type",
            r"BRAF[\s\-]*WT",
        ],
        "variants": ["V600E", "V600K", "V600R", "V600"],
    },
    # NRAS mutations
    "NRAS": {
        "gene": "NRAS",
        "patterns": [
            r"NRAS[\s\-]*mutation",
            r"NRAS[\s\-]*Q61[KLHR]?",
            r"NRAS[\s\-]*positive",
            r"NRAS[\s\-]*mutant",
        ],
        "variants": ["Q61K", "Q61L", "Q61H", "Q61R", "Q61"],
    },
    # c-KIT mutations
    "KIT": {
        "gene": "KIT",
        "patterns": [
            r"c[\s\-]*KIT[\s\-]*mutation",
            r"KIT[\s\-]*mutation",
            r"KIT[\s\-]*positive",
            r"KIT[\s\-]*mutant",
        ],
        "variants": [],
    },
    # PD-L1 expression
    "PD-L1": {
        "gene": "CD274",
        "patterns": [
            r"PD[\s\-]*L1[\s\-]*positive",
            r"PD[\s\-]*L1[\s\-]*expression",
            r"PD[\s\-]*L1[\s\-]*[>≥]\s*\d+%?",
            r"PD[\s\-]*L1[\s\-]*TPS",
            r"PD[\s\-]*L1[\s\-]*CPS",
        ],
        "variants": [],
    },
    # Tumor Mutational Burden
    "TMB": {
        "gene": None,
        "patterns": [
            r"TMB[\s\-]*high",
            r"tumor[\s\-]*mutational[\s\-]*burden",
            r"TMB[\s\-]*[>≥]\s*\d+",
        ],
        "variants": [],
    },
    # Microsatellite instability
    "MSI": {
        "gene": None,
        "patterns": [
            r"MSI[\s\-]*H",
            r"MSI[\s\-]*high",
            r"microsatellite[\s\-]*instability",
            r"mismatch[\s\-]*repair[\s\-]*deficient",
            r"dMMR",
        ],
        "variants": [],
    },
    # CDKN2A (familial melanoma)
    "CDKN2A": {
        "gene": "CDKN2A",
        "patterns": [
            r"CDKN2A[\s\-]*mutation",
            r"p16[\s\-]*mutation",
            r"CDKN2A[\s\-]*deletion",
        ],
        "variants": [],
    },
    # Other melanoma-relevant genes
    "PTEN": {
        "gene": "PTEN",
        "patterns": [r"PTEN[\s\-]*loss", r"PTEN[\s\-]*mutation", r"PTEN[\s\-]*deletion"],
        "variants": [],
    },
    "NF1": {
        "gene": "NF1",
        "patterns": [r"NF1[\s\-]*mutation", r"NF1[\s\-]*loss"],
        "variants": [],
    },
    # HLA types (immunotherapy)
    "HLA": {
        "gene": None,
        "patterns": [r"HLA[\s\-]*A\*?\d+", r"HLA[\s\-]*typing"],
        "variants": [],
    },
}

# Keywords that indicate biomarker requirement vs exclusion
INCLUSION_KEYWORDS = [
    "must have", "required", "positive for", "with", "harboring",
    "documented", "confirmed", "presence of", "eligible if"
]
EXCLUSION_KEYWORDS = [
    "must not have", "excluded", "negative for", "without", "wild-type",
    "wild type", "no known", "absence of", "not eligible if", "ineligible"
]


def extract_biomarker_requirements(eligibility_text: str) -> Dict[str, Any]:
    """
    Extract biomarker/genetic requirements from eligibility criteria text.

    Returns:
        Dict with:
        - required_biomarkers: List of required biomarkers/mutations
        - excluded_biomarkers: List of excluded biomarkers/mutations
        - genetic_requirements: Structured list of requirements
        - biomarker_keywords: All biomarker-related keywords found
        - requires_genetic_testing: Boolean if trial requires genetic testing
        - targeted_therapy_trial: Boolean if this appears to be a targeted therapy trial
    """
    if not eligibility_text:
        return {
            "required_biomarkers": [],
            "excluded_biomarkers": [],
            "genetic_requirements": [],
            "biomarker_keywords": [],
            "requires_genetic_testing": False,
            "targeted_therapy_trial": False,
        }

    text = eligibility_text.lower()
    text_original = eligibility_text  # Keep original for case-sensitive matching

    required = []
    excluded = []
    requirements = []
    keywords_found = set()

    # Split into inclusion and exclusion sections if possible
    inclusion_section = text
    exclusion_section = ""

    # Common patterns for exclusion criteria section
    exclusion_markers = [
        "exclusion criteria", "exclusion:", "not eligible",
        "ineligible if", "patients must not"
    ]
    for marker in exclusion_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) == 2:
                inclusion_section = parts[0]
                exclusion_section = parts[1]
                break

    # Search for biomarker patterns
    for biomarker_name, biomarker_info in BIOMARKER_PATTERNS.items():
        gene = biomarker_info["gene"]
        patterns = biomarker_info["patterns"]
        variants = biomarker_info["variants"]

        for pattern in patterns:
            # Check inclusion section
            inclusion_matches = re.findall(pattern, text_original, re.IGNORECASE)
            exclusion_matches = re.findall(pattern, exclusion_section, re.IGNORECASE) if exclusion_section else []

            if inclusion_matches:
                keywords_found.add(biomarker_name)

                # Determine if it's a requirement or exclusion based on context
                for match in inclusion_matches:
                    match_text = match if isinstance(match, str) else match[0]

                    # Check surrounding context
                    match_pos = text.find(match_text.lower())
                    context_start = max(0, match_pos - 100)
                    context_end = min(len(text), match_pos + len(match_text) + 100)
                    context = text[context_start:context_end]

                    # Check if in exclusion section or has exclusion keywords nearby
                    is_excluded = any(kw in context for kw in EXCLUSION_KEYWORDS)
                    is_required = any(kw in context for kw in INCLUSION_KEYWORDS)

                    # Detect specific variants
                    detected_variants = []
                    for variant in variants:
                        if variant.lower() in match_text.lower():
                            detected_variants.append(variant)

                    biomarker_entry = f"{biomarker_name}"
                    if detected_variants:
                        biomarker_entry = f"{biomarker_name} {detected_variants[0]}"

                    if is_excluded or match_text.lower() in exclusion_section:
                        if biomarker_entry not in excluded:
                            excluded.append(biomarker_entry)
                    elif is_required or (not is_excluded and match_pos < len(text) // 2):
                        # If in first half and not clearly excluded, likely a requirement
                        if biomarker_entry not in required:
                            required.append(biomarker_entry)

                    # Build structured requirement
                    req = {
                        "gene": gene or biomarker_name,
                        "biomarker": biomarker_name,
                        "variants": detected_variants,
                        "required": not is_excluded,
                        "context": context.strip()[:200],
                    }
                    if req not in requirements:
                        requirements.append(req)

    # Determine if this is a targeted therapy trial
    targeted_therapy_indicators = [
        "targeted therapy", "targeted treatment", "tyrosine kinase inhibitor",
        "tki", "braf inhibitor", "mek inhibitor", "checkpoint inhibitor",
        "immunotherapy", "anti-pd-1", "anti-pd-l1", "pembrolizumab",
        "nivolumab", "ipilimumab", "vemurafenib", "dabrafenib", "trametinib",
        "encorafenib", "binimetinib", "cobimetinib"
    ]
    is_targeted = any(indicator in text for indicator in targeted_therapy_indicators)

    # Determine if genetic testing is required
    genetic_testing_indicators = [
        "genetic testing", "molecular testing", "tumor testing",
        "biomarker testing", "next-generation sequencing", "ngs",
        "mutation analysis", "molecular profiling"
    ]
    requires_testing = any(indicator in text for indicator in genetic_testing_indicators)
    requires_testing = requires_testing or len(required) > 0

    return {
        "required_biomarkers": required,
        "excluded_biomarkers": excluded,
        "genetic_requirements": requirements,
        "biomarker_keywords": list(keywords_found),
        "requires_genetic_testing": requires_testing,
        "targeted_therapy_trial": is_targeted or len(required) > 0,
    }

# ClinicalTrials.gov API v2 endpoint
CLINICALTRIALS_API_URL = "https://clinicaltrials.gov/api/v2/studies"

# Dermatology-related conditions to search for
SKIN_CONDITIONS = [
    "melanoma",
    "basal cell carcinoma",
    "squamous cell carcinoma",
    "skin cancer",
    "actinic keratosis",
    "dysplastic nevus",
    "cutaneous melanoma",
    "merkel cell carcinoma",
    "psoriasis",
    "eczema",
    "atopic dermatitis",
    "dermatitis",
    "skin lesion",
    "nevus",
    "mole",
    "seborrheic keratosis",
]

# Fields to request from the API
API_FIELDS = [
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "BriefSummary",
    "DetailedDescription",
    "OverallStatus",
    "Phase",
    "StudyType",
    "Condition",
    "InterventionName",
    "InterventionType",
    "EligibilityCriteria",
    "MinimumAge",
    "MaximumAge",
    "Gender",
    "LocationFacility",
    "LocationCity",
    "LocationState",
    "LocationCountry",
    "LocationZip",
    "CentralContactName",
    "CentralContactEMail",
    "CentralContactPhone",
    "LeadSponsorName",
    "CollaboratorName",
    "EnrollmentCount",
    "StartDate",
    "CompletionDate",
    "PrimaryCompletionDate",
    "LastUpdatePostDate",
]


def parse_age(age_str: Optional[str]) -> Optional[int]:
    """Parse age string like '18 Years' into integer."""
    if not age_str:
        return None
    try:
        # Handle formats like "18 Years", "6 Months", etc.
        parts = age_str.strip().split()
        if len(parts) >= 2:
            value = int(parts[0])
            unit = parts[1].lower()
            if "year" in unit:
                return value
            elif "month" in unit:
                return max(0, value // 12)
            elif "day" in unit or "week" in unit:
                return 0
        return int(parts[0])
    except (ValueError, IndexError):
        return None


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string from API into datetime."""
    if not date_str:
        return None
    try:
        # Handle various formats
        for fmt in ["%Y-%m-%d", "%B %d, %Y", "%B %Y", "%Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None


def extract_locations(study_data: Dict) -> List[Dict]:
    """Extract and structure location data from study."""
    locations = []

    # Get location arrays from protocol section
    protocol = study_data.get("protocolSection", {})
    contacts_locations = protocol.get("contactsLocationsModule", {})
    location_list = contacts_locations.get("locations", [])

    for loc in location_list:
        location = {
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "country": loc.get("country"),
            "zip": loc.get("zip"),
            "status": loc.get("status"),
            "lat": None,
            "lng": None,
        }
        # Only add if we have at least city or facility
        if location["city"] or location["facility"]:
            locations.append(location)

    return locations


def extract_interventions(study_data: Dict) -> List[Dict]:
    """Extract intervention data from study."""
    interventions = []

    protocol = study_data.get("protocolSection", {})
    arms_module = protocol.get("armsInterventionsModule", {})
    intervention_list = arms_module.get("interventions", [])

    for intervention in intervention_list:
        interventions.append({
            "type": intervention.get("type"),
            "name": intervention.get("name"),
            "description": intervention.get("description"),
        })

    return interventions


def extract_conditions(study_data: Dict) -> List[str]:
    """Extract condition list from study."""
    protocol = study_data.get("protocolSection", {})
    conditions_module = protocol.get("conditionsModule", {})
    return conditions_module.get("conditions", [])


def parse_study(study_data: Dict) -> Dict:
    """Parse a single study from the API response into our model format."""
    protocol = study_data.get("protocolSection", {})

    # Identification
    id_module = protocol.get("identificationModule", {})
    nct_id = id_module.get("nctId")

    # Description
    desc_module = protocol.get("descriptionModule", {})

    # Status
    status_module = protocol.get("statusModule", {})

    # Design
    design_module = protocol.get("designModule", {})

    # Eligibility
    eligibility_module = protocol.get("eligibilityModule", {})
    eligibility_criteria = eligibility_module.get("eligibilityCriteria")

    # Extract biomarker/genetic requirements from eligibility text
    biomarker_data = extract_biomarker_requirements(eligibility_criteria)

    # Contacts
    contacts_module = protocol.get("contactsLocationsModule", {})
    central_contacts = contacts_module.get("centralContacts", [])
    contact = central_contacts[0] if central_contacts else {}

    # Sponsor
    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    lead_sponsor = sponsor_module.get("leadSponsor", {})
    collaborators = sponsor_module.get("collaborators", [])

    return {
        "nct_id": nct_id,
        "title": id_module.get("briefTitle") or id_module.get("officialTitle"),
        "brief_summary": desc_module.get("briefSummary"),
        "detailed_description": desc_module.get("detailedDescription"),
        "phase": ", ".join(design_module.get("phases", [])) or None,
        "status": status_module.get("overallStatus"),
        "study_type": design_module.get("studyType"),
        "conditions": extract_conditions(study_data),
        "interventions": extract_interventions(study_data),
        "eligibility_criteria": eligibility_criteria,
        "min_age": parse_age(eligibility_module.get("minimumAge")),
        "max_age": parse_age(eligibility_module.get("maximumAge")),
        "gender": eligibility_module.get("sex"),
        "locations": extract_locations(study_data),
        "contact_name": contact.get("name"),
        "contact_email": contact.get("email"),
        "contact_phone": contact.get("phone"),
        "principal_investigator": None,  # Would need to extract from officials
        "target_enrollment": design_module.get("enrollmentInfo", {}).get("count"),
        "sponsor": lead_sponsor.get("name"),
        "collaborators": [c.get("name") for c in collaborators],
        "start_date": parse_date(status_module.get("startDateStruct", {}).get("date")),
        "completion_date": parse_date(status_module.get("completionDateStruct", {}).get("date")),
        "primary_completion_date": parse_date(status_module.get("primaryCompletionDateStruct", {}).get("date")),
        "last_update_posted": parse_date(status_module.get("lastUpdatePostDateStruct", {}).get("date")),
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
        # Biomarker/Genetic requirements
        "required_biomarkers": biomarker_data["required_biomarkers"],
        "excluded_biomarkers": biomarker_data["excluded_biomarkers"],
        "genetic_requirements": biomarker_data["genetic_requirements"],
        "biomarker_keywords": biomarker_data["biomarker_keywords"],
        "requires_genetic_testing": biomarker_data["requires_genetic_testing"],
        "targeted_therapy_trial": biomarker_data["targeted_therapy_trial"],
    }


async def fetch_trials_for_condition(
    session: aiohttp.ClientSession,
    condition: str,
    status: str = "RECRUITING",
    page_size: int = 100,
    max_pages: int = 5
) -> List[Dict]:
    """Fetch trials for a specific condition from ClinicalTrials.gov API."""
    all_studies = []
    page_token = None
    pages_fetched = 0

    while pages_fetched < max_pages:
        params = {
            "query.cond": condition,
            "filter.overallStatus": status,
            "pageSize": page_size,
            "format": "json",
        }

        if page_token:
            params["pageToken"] = page_token

        try:
            async with session.get(CLINICALTRIALS_API_URL, params=params) as response:
                if response.status != 200:
                    print(f"Error fetching trials for {condition}: HTTP {response.status}")
                    break

                data = await response.json()
                studies = data.get("studies", [])
                all_studies.extend(studies)

                # Check for more pages
                page_token = data.get("nextPageToken")
                if not page_token:
                    break

                pages_fetched += 1

        except Exception as e:
            print(f"Error fetching trials for {condition}: {e}")
            break

    return all_studies


def upsert_trial(db: Session, trial_data: Dict) -> Optional[ClinicalTrial]:
    """Insert or update a trial in the database."""
    try:
        nct_id = trial_data.get("nct_id")
        if not nct_id:
            return None

        # Check if trial exists
        existing = db.query(ClinicalTrial).filter(ClinicalTrial.nct_id == nct_id).first()

        if existing:
            # Update existing trial
            for key, value in trial_data.items():
                if hasattr(existing, key) and value is not None:
                    setattr(existing, key, value)
            existing.synced_at = datetime.utcnow()
            existing.updated_at = datetime.utcnow()
            return existing
        else:
            # Create new trial
            trial = ClinicalTrial(
                nct_id=trial_data.get("nct_id"),
                title=trial_data.get("title"),
                brief_summary=trial_data.get("brief_summary"),
                detailed_description=trial_data.get("detailed_description"),
                phase=trial_data.get("phase"),
                status=trial_data.get("status"),
                study_type=trial_data.get("study_type"),
                conditions=trial_data.get("conditions"),
                interventions=trial_data.get("interventions"),
                eligibility_criteria=trial_data.get("eligibility_criteria"),
                min_age=trial_data.get("min_age"),
                max_age=trial_data.get("max_age"),
                gender=trial_data.get("gender"),
                locations=trial_data.get("locations"),
                contact_name=trial_data.get("contact_name"),
                contact_email=trial_data.get("contact_email"),
                contact_phone=trial_data.get("contact_phone"),
                principal_investigator=trial_data.get("principal_investigator"),
                target_enrollment=trial_data.get("target_enrollment"),
                sponsor=trial_data.get("sponsor"),
                collaborators=trial_data.get("collaborators"),
                start_date=trial_data.get("start_date"),
                completion_date=trial_data.get("completion_date"),
                primary_completion_date=trial_data.get("primary_completion_date"),
                last_update_posted=trial_data.get("last_update_posted"),
                url=trial_data.get("url"),
                # Biomarker/Genetic requirements
                required_biomarkers=trial_data.get("required_biomarkers"),
                excluded_biomarkers=trial_data.get("excluded_biomarkers"),
                genetic_requirements=trial_data.get("genetic_requirements"),
                biomarker_keywords=trial_data.get("biomarker_keywords"),
                requires_genetic_testing=trial_data.get("requires_genetic_testing", False),
                targeted_therapy_trial=trial_data.get("targeted_therapy_trial", False),
                synced_at=datetime.utcnow(),
            )
            db.add(trial)
            return trial

    except Exception as e:
        print(f"Error upserting trial {trial_data.get('nct_id')}: {e}")
        return None


async def sync_dermatology_trials(
    conditions: List[str] = None,
    status: str = "RECRUITING"
) -> Dict[str, Any]:
    """
    Main sync function - fetches and stores dermatology trials.

    Args:
        conditions: List of conditions to search for (defaults to SKIN_CONDITIONS)
        status: Trial status filter (default: RECRUITING)

    Returns:
        Dict with sync statistics
    """
    conditions = conditions or SKIN_CONDITIONS

    stats = {
        "started_at": datetime.utcnow().isoformat(),
        "conditions_searched": len(conditions),
        "trials_fetched": 0,
        "trials_created": 0,
        "trials_updated": 0,
        "errors": [],
    }

    # Track NCT IDs to avoid duplicates across conditions
    seen_nct_ids = set()
    all_trials = []

    # Fetch trials for all conditions
    async with aiohttp.ClientSession() as session:
        for condition in conditions:
            print(f"Fetching trials for: {condition}")
            try:
                studies = await fetch_trials_for_condition(session, condition, status)

                for study in studies:
                    parsed = parse_study(study)
                    nct_id = parsed.get("nct_id")

                    if nct_id and nct_id not in seen_nct_ids:
                        seen_nct_ids.add(nct_id)
                        all_trials.append(parsed)

                # Rate limiting - be nice to the API
                await asyncio.sleep(0.5)

            except Exception as e:
                error_msg = f"Error fetching {condition}: {str(e)}"
                print(error_msg)
                stats["errors"].append(error_msg)

    stats["trials_fetched"] = len(all_trials)
    print(f"Fetched {len(all_trials)} unique trials")

    # Store in database
    db = SessionLocal()
    try:
        for trial_data in all_trials:
            nct_id = trial_data.get("nct_id")

            # Check if exists
            existing = db.query(ClinicalTrial).filter(ClinicalTrial.nct_id == nct_id).first()

            result = upsert_trial(db, trial_data)
            if result:
                if existing:
                    stats["trials_updated"] += 1
                else:
                    stats["trials_created"] += 1

        db.commit()
        print(f"Sync complete: {stats['trials_created']} created, {stats['trials_updated']} updated")

    except Exception as e:
        db.rollback()
        error_msg = f"Database error: {str(e)}"
        print(error_msg)
        stats["errors"].append(error_msg)
    finally:
        db.close()

    stats["completed_at"] = datetime.utcnow().isoformat()
    return stats


def run_sync():
    """Synchronous wrapper for running the sync."""
    return asyncio.run(sync_dermatology_trials())


# For manual testing
if __name__ == "__main__":
    print("Starting clinical trials sync...")
    stats = run_sync()
    print(f"\nSync Statistics:")
    print(f"  Trials fetched: {stats['trials_fetched']}")
    print(f"  Trials created: {stats['trials_created']}")
    print(f"  Trials updated: {stats['trials_updated']}")
    if stats['errors']:
        print(f"  Errors: {len(stats['errors'])}")
        for error in stats['errors'][:5]:
            print(f"    - {error}")
