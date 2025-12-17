"""
Genetics Data Service - Dynamic gene/variant data from external sources

Integrates with:
- ClinVar (NCBI) - Clinical variant significance
- OMIM - Gene-disease associations (via ClinVar links)
- gnomAD - Population allele frequencies
"""

import httpx
import asyncio
import json
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from functools import lru_cache
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

class GeneticsCache:
    """Simple in-memory cache with TTL for genetics data."""

    def __init__(self, default_ttl_hours: int = 24):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = timedelta(hours=default_ttl_hours)

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() < entry["expires"]:
                return entry["data"]
            else:
                del self._cache[key]
        return None

    def set(self, key: str, data: Any, ttl_hours: Optional[int] = None):
        ttl = timedelta(hours=ttl_hours) if ttl_hours else self._default_ttl
        self._cache[key] = {
            "data": data,
            "expires": datetime.now() + ttl,
            "cached_at": datetime.now().isoformat()
        }

    def clear(self):
        self._cache.clear()


# Global cache instance
_genetics_cache = GeneticsCache(default_ttl_hours=24)


# =============================================================================
# NCBI E-UTILITIES API (ClinVar)
# =============================================================================

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINVAR_BASE_URL = "https://www.ncbi.nlm.nih.gov/clinvar"

# Dermatology-relevant search terms for ClinVar
DERMATOLOGY_SEARCH_TERMS = [
    # Melanoma genes
    '"melanoma"[disease] AND "pathogenic"[clinical significance]',
    '"familial melanoma"[disease]',
    '"dysplastic nevus syndrome"[disease]',

    # Skin cancer genes
    '"basal cell carcinoma"[disease] AND "pathogenic"[clinical significance]',
    '"squamous cell carcinoma of skin"[disease]',
    '"Gorlin syndrome"[disease]',
    '"nevoid basal cell carcinoma syndrome"[disease]',

    # Photosensitivity disorders
    '"xeroderma pigmentosum"[disease]',
    '"Cockayne syndrome"[disease]',

    # Pigmentation disorders
    '"oculocutaneous albinism"[disease]',
    '"Hermansky-Pudlak syndrome"[disease]',

    # Pharmacogenomics relevant to dermatology
    '"TPMT"[gene] AND "pathogenic"[clinical significance]',
    '"DPYD"[gene] AND "pathogenic"[clinical significance]',
]

# Key dermatology genes to always include
DERMATOLOGY_GENE_SYMBOLS = [
    # Melanoma susceptibility
    "CDKN2A", "CDK4", "BAP1", "MITF", "POT1", "TERT", "ACD", "TERF2IP",
    "MC1R",  # Pigmentation/melanoma risk

    # Somatic (tumor profiling)
    "BRAF", "NRAS", "KIT", "NF1", "PTEN",

    # BCC/SCC genes
    "PTCH1", "PTCH2", "SUFU", "SMO",  # Hedgehog pathway / Gorlin
    "TP53",  # Various skin cancers

    # Xeroderma pigmentosum (DNA repair)
    "XPA", "XPB", "XPC", "XPD", "XPE", "XPF", "XPG", "POLH",

    # Pharmacogenomics
    "TPMT", "DPYD", "NUDT15",

    # Other skin conditions
    "TYR", "OCA2", "TYRP1", "SLC45A2",  # Albinism/pigmentation
    "COL7A1", "KRT5", "KRT14",  # Epidermolysis bullosa
]


async def fetch_clinvar_gene_info(gene_symbol: str) -> Optional[Dict]:
    """
    Fetch gene information from ClinVar/NCBI for a specific gene.

    Returns gene details including:
    - Associated conditions
    - Pathogenic variants
    - Clinical significance
    """
    cache_key = f"clinvar_gene_{gene_symbol}"
    cached = _genetics_cache.get(cache_key)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Search ClinVar for gene
            search_url = f"{NCBI_BASE_URL}/esearch.fcgi"
            search_params = {
                "db": "clinvar",
                "term": f'"{gene_symbol}"[gene] AND "pathogenic"[clinical significance]',
                "retmax": 100,
                "retmode": "json",
                "usehistory": "y"
            }

            search_response = await client.get(search_url, params=search_params)
            search_data = search_response.json()

            result = search_data.get("esearchresult", {})
            id_list = result.get("idlist", [])
            count = int(result.get("count", 0))

            if not id_list:
                return None

            # Fetch summary for variants
            summary_url = f"{NCBI_BASE_URL}/esummary.fcgi"
            summary_params = {
                "db": "clinvar",
                "id": ",".join(id_list[:20]),  # Limit to first 20
                "retmode": "json"
            }

            summary_response = await client.get(summary_url, params=summary_params)
            summary_data = summary_response.json()

            # Parse variant summaries
            variants = []
            conditions = set()

            doc_sums = summary_data.get("result", {})
            for uid in id_list[:20]:
                if uid in doc_sums and uid != "uids":
                    variant_info = doc_sums[uid]

                    # Extract clinical significance
                    clin_sig = variant_info.get("clinical_significance", {})
                    if isinstance(clin_sig, dict):
                        significance = clin_sig.get("description", "unknown")
                    else:
                        significance = str(clin_sig)

                    # Extract conditions/traits
                    trait_set = variant_info.get("trait_set", [])
                    for trait in trait_set:
                        if isinstance(trait, dict):
                            trait_name = trait.get("trait_name", "")
                            if trait_name:
                                conditions.add(trait_name)

                    variants.append({
                        "variation_id": variant_info.get("uid"),
                        "title": variant_info.get("title", ""),
                        "clinical_significance": significance,
                        "review_status": variant_info.get("review_status", ""),
                        "variation_type": variant_info.get("variation_type", ""),
                    })

            gene_info = {
                "gene_symbol": gene_symbol,
                "total_pathogenic_variants": count,
                "variants": variants,
                "associated_conditions": list(conditions),
                "source": "ClinVar",
                "last_updated": datetime.now().isoformat(),
            }

            _genetics_cache.set(cache_key, gene_info)
            return gene_info

    except Exception as e:
        logger.error(f"Error fetching ClinVar data for {gene_symbol}: {e}")
        return None


async def fetch_gene_details_from_ncbi(gene_symbol: str) -> Optional[Dict]:
    """
    Fetch detailed gene information from NCBI Gene database.
    """
    cache_key = f"ncbi_gene_{gene_symbol}"
    cached = _genetics_cache.get(cache_key)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Search for gene ID
            search_url = f"{NCBI_BASE_URL}/esearch.fcgi"
            search_params = {
                "db": "gene",
                "term": f'"{gene_symbol}"[sym] AND "Homo sapiens"[orgn]',
                "retmode": "json"
            }

            search_response = await client.get(search_url, params=search_params)
            search_data = search_response.json()

            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return None

            gene_id = id_list[0]

            # Fetch gene summary
            summary_url = f"{NCBI_BASE_URL}/esummary.fcgi"
            summary_params = {
                "db": "gene",
                "id": gene_id,
                "retmode": "json"
            }

            summary_response = await client.get(summary_url, params=summary_params)
            summary_data = summary_response.json()

            gene_data = summary_data.get("result", {}).get(gene_id, {})

            gene_details = {
                "gene_id": gene_id,
                "gene_symbol": gene_symbol,
                "name": gene_data.get("description", ""),
                "chromosome": gene_data.get("chromosome", ""),
                "summary": gene_data.get("summary", ""),
                "aliases": gene_data.get("otheraliases", "").split(", ") if gene_data.get("otheraliases") else [],
                "source": "NCBI Gene",
            }

            _genetics_cache.set(cache_key, gene_details)
            return gene_details

    except Exception as e:
        logger.error(f"Error fetching NCBI Gene data for {gene_symbol}: {e}")
        return None


# =============================================================================
# DERMATOLOGY-SPECIFIC ANNOTATIONS
# =============================================================================

# These are dermatology-specific risk annotations that supplement external data
DERMATOLOGY_ANNOTATIONS = {
    "CDKN2A": {
        "category": "melanoma",
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "60-90% lifetime melanoma risk",
        "melanoma_risk_multiplier": 10.0,
        "screening": "Annual full-body skin exam starting at age 10, ophthalmologic exam",
        "acmg_actionable": True,
    },
    "CDK4": {
        "category": "melanoma",
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "Similar to CDKN2A",
        "melanoma_risk_multiplier": 8.0,
        "screening": "Annual full-body skin exam",
        "acmg_actionable": True,
    },
    "BAP1": {
        "category": "melanoma",
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "Increased risk of uveal and cutaneous melanoma",
        "melanoma_risk_multiplier": 5.0,
        "screening": "Annual skin exam, ophthalmologic exam every 6-12 months",
        "acmg_actionable": True,
    },
    "MC1R": {
        "category": "pigmentation",
        "inheritance": "complex",
        "penetrance": "variable",
        "risk_increase": "2-4x increased melanoma risk per variant allele",
        "melanoma_risk_multiplier": 2.5,
        "screening": "Enhanced sun protection, regular skin exams",
        "acmg_actionable": False,
    },
    "MITF": {
        "category": "melanoma",
        "inheritance": "AD",
        "penetrance": "moderate",
        "risk_increase": "2-5x increased melanoma risk",
        "melanoma_risk_multiplier": 3.0,
        "screening": "Regular skin and kidney surveillance",
        "acmg_actionable": False,
    },
    "POT1": {
        "category": "melanoma",
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "High melanoma risk in families",
        "melanoma_risk_multiplier": 6.0,
        "screening": "Annual full-body skin exam",
        "acmg_actionable": True,
    },
    "TERT": {
        "category": "melanoma",
        "inheritance": "AD",
        "penetrance": "moderate",
        "risk_increase": "Increased melanoma susceptibility",
        "melanoma_risk_multiplier": 2.0,
        "screening": "Regular skin surveillance",
        "acmg_actionable": False,
    },
    "BRAF": {
        "category": "melanoma_somatic",
        "inheritance": "somatic",
        "penetrance": "N/A",
        "risk_increase": "Present in ~50% of melanomas - therapeutic target",
        "melanoma_risk_multiplier": None,
        "screening": "Tumor testing for targeted therapy selection",
        "treatment_implications": "BRAF inhibitors (vemurafenib, dabrafenib) + MEK inhibitors",
        "acmg_actionable": False,
    },
    "NRAS": {
        "category": "melanoma_somatic",
        "inheritance": "somatic",
        "penetrance": "N/A",
        "risk_increase": "Present in ~20% of melanomas",
        "melanoma_risk_multiplier": None,
        "screening": "Tumor testing for treatment planning",
        "treatment_implications": "MEK inhibitors may be considered",
        "acmg_actionable": False,
    },
    "KIT": {
        "category": "melanoma_somatic",
        "inheritance": "somatic",
        "penetrance": "N/A",
        "risk_increase": "Present in acral and mucosal melanomas",
        "melanoma_risk_multiplier": None,
        "screening": "Tumor testing",
        "treatment_implications": "Imatinib for KIT-mutant melanoma",
        "acmg_actionable": False,
    },
    "PTCH1": {
        "category": "keratinocyte_cancer",
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "90% develop BCCs by age 35 (Gorlin syndrome)",
        "bcc_risk_multiplier": 20.0,
        "screening": "Skin exams every 3-6 months, avoid radiation, dental surveillance",
        "acmg_actionable": True,
    },
    "PTCH2": {
        "category": "keratinocyte_cancer",
        "inheritance": "AD",
        "penetrance": "moderate",
        "risk_increase": "Increased BCC risk",
        "bcc_risk_multiplier": 5.0,
        "screening": "Regular skin exams",
        "acmg_actionable": False,
    },
    "SUFU": {
        "category": "keratinocyte_cancer",
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "Associated with Gorlin-like syndrome",
        "bcc_risk_multiplier": 10.0,
        "screening": "Skin exams every 3-6 months",
        "acmg_actionable": True,
    },
    "XPA": {
        "category": "photosensitivity",
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "1000x increased skin cancer risk (Xeroderma Pigmentosum)",
        "melanoma_risk_multiplier": 100.0,
        "screening": "Strict UV avoidance, skin exams every 3 months, neurological monitoring",
        "acmg_actionable": True,
    },
    "XPC": {
        "category": "photosensitivity",
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "High skin cancer risk (XP-C)",
        "melanoma_risk_multiplier": 50.0,
        "screening": "Strict UV avoidance, frequent skin exams",
        "acmg_actionable": True,
    },
    "XPD": {
        "category": "photosensitivity",
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "Xeroderma Pigmentosum / Trichothiodystrophy",
        "melanoma_risk_multiplier": 50.0,
        "screening": "Strict UV avoidance, skin and neurological monitoring",
        "acmg_actionable": True,
    },
    "POLH": {
        "category": "photosensitivity",
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "XP variant - increased skin cancer risk",
        "melanoma_risk_multiplier": 30.0,
        "screening": "UV protection, regular skin exams",
        "acmg_actionable": True,
    },
    "TPMT": {
        "category": "pharmacogenomics",
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "Myelosuppression risk with azathioprine",
        "drug_implications": "Reduce azathioprine dose by 50-90% in poor metabolizers",
        "screening": "Test before azathioprine therapy",
        "acmg_actionable": True,
    },
    "DPYD": {
        "category": "pharmacogenomics",
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "Severe/fatal toxicity with 5-FU",
        "drug_implications": "Avoid 5-FU or reduce dose by 50% in heterozygotes",
        "screening": "Test before 5-FU therapy for skin cancer",
        "acmg_actionable": True,
    },
    "NUDT15": {
        "category": "pharmacogenomics",
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "Thiopurine toxicity risk",
        "drug_implications": "Dose adjustment for thiopurines",
        "screening": "Test before thiopurine therapy",
        "acmg_actionable": True,
    },
    "TYR": {
        "category": "pigmentation",
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "Oculocutaneous albinism type 1",
        "melanoma_risk_multiplier": 1.0,  # Actually protective for melanoma but high SCC/BCC
        "bcc_risk_multiplier": 5.0,
        "screening": "Regular skin exams, UV protection, eye exams",
        "acmg_actionable": False,
    },
    "OCA2": {
        "category": "pigmentation",
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "Oculocutaneous albinism type 2",
        "bcc_risk_multiplier": 5.0,
        "screening": "Regular skin exams, UV protection",
        "acmg_actionable": False,
    },
    "TP53": {
        "category": "multiple_cancers",
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "Li-Fraumeni syndrome - multiple cancer types including skin",
        "melanoma_risk_multiplier": 2.0,
        "screening": "Comprehensive cancer surveillance",
        "acmg_actionable": True,
    },
}


# =============================================================================
# MAIN SERVICE FUNCTIONS
# =============================================================================

async def get_dermatology_gene(gene_symbol: str) -> Optional[Dict]:
    """
    Get comprehensive gene information combining external APIs and dermatology annotations.
    """
    gene_symbol = gene_symbol.upper()
    cache_key = f"derm_gene_{gene_symbol}"

    cached = _genetics_cache.get(cache_key)
    if cached:
        return cached

    # Fetch from external sources in parallel
    clinvar_task = fetch_clinvar_gene_info(gene_symbol)
    ncbi_task = fetch_gene_details_from_ncbi(gene_symbol)

    clinvar_data, ncbi_data = await asyncio.gather(clinvar_task, ncbi_task)

    # Start with dermatology annotations if available
    derm_annotations = DERMATOLOGY_ANNOTATIONS.get(gene_symbol, {})

    # Build comprehensive gene record
    gene_info = {
        "gene_symbol": gene_symbol,
        "gene_name": ncbi_data.get("name", "") if ncbi_data else derm_annotations.get("name", gene_symbol),
        "chromosome": ncbi_data.get("chromosome", "") if ncbi_data else "",
        "summary": ncbi_data.get("summary", "") if ncbi_data else "",
        "aliases": ncbi_data.get("aliases", []) if ncbi_data else [],

        # Dermatology-specific annotations
        "category": derm_annotations.get("category", "other"),
        "inheritance": derm_annotations.get("inheritance", "unknown"),
        "penetrance": derm_annotations.get("penetrance", "unknown"),
        "risk_increase": derm_annotations.get("risk_increase", ""),
        "melanoma_risk_multiplier": derm_annotations.get("melanoma_risk_multiplier", 1.0),
        "bcc_risk_multiplier": derm_annotations.get("bcc_risk_multiplier", 1.0),
        "screening": derm_annotations.get("screening", "Consult genetic counselor"),
        "acmg_actionable": derm_annotations.get("acmg_actionable", False),
        "drug_implications": derm_annotations.get("drug_implications"),
        "treatment_implications": derm_annotations.get("treatment_implications"),

        # ClinVar data
        "clinvar_pathogenic_count": clinvar_data.get("total_pathogenic_variants", 0) if clinvar_data else 0,
        "associated_conditions": clinvar_data.get("associated_conditions", []) if clinvar_data else [],
        "key_variants": clinvar_data.get("variants", [])[:5] if clinvar_data else [],  # Top 5 variants

        # Metadata
        "sources": ["NCBI Gene", "ClinVar"] if clinvar_data and ncbi_data else
                   ["NCBI Gene"] if ncbi_data else
                   ["ClinVar"] if clinvar_data else
                   ["Local annotations"],
        "last_updated": datetime.now().isoformat(),
    }

    _genetics_cache.set(cache_key, gene_info, ttl_hours=24)
    return gene_info


async def get_all_dermatology_genes() -> List[Dict]:
    """
    Get information for all dermatology-relevant genes.
    """
    cache_key = "all_derm_genes"
    cached = _genetics_cache.get(cache_key)
    if cached:
        return cached

    # Fetch all genes in parallel (with rate limiting)
    genes = []
    batch_size = 5  # Process 5 genes at a time to avoid rate limiting

    for i in range(0, len(DERMATOLOGY_GENE_SYMBOLS), batch_size):
        batch = DERMATOLOGY_GENE_SYMBOLS[i:i + batch_size]
        tasks = [get_dermatology_gene(gene) for gene in batch]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result:
                genes.append(result)

        # Small delay between batches to respect rate limits
        if i + batch_size < len(DERMATOLOGY_GENE_SYMBOLS):
            await asyncio.sleep(0.5)

    # Sort by category and then by risk multiplier
    category_order = {
        "melanoma": 0,
        "keratinocyte_cancer": 1,
        "photosensitivity": 2,
        "pigmentation": 3,
        "pharmacogenomics": 4,
        "melanoma_somatic": 5,
        "multiple_cancers": 6,
        "other": 7,
    }

    genes.sort(key=lambda g: (
        category_order.get(g.get("category", "other"), 7),
        -(g.get("melanoma_risk_multiplier") or 0)
    ))

    _genetics_cache.set(cache_key, genes, ttl_hours=12)
    return genes


async def search_clinvar_for_condition(condition: str, limit: int = 50) -> List[Dict]:
    """
    Search ClinVar for genes associated with a specific condition.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            search_url = f"{NCBI_BASE_URL}/esearch.fcgi"
            search_params = {
                "db": "clinvar",
                "term": f'"{condition}"[disease] AND "pathogenic"[clinical significance]',
                "retmax": limit,
                "retmode": "json"
            }

            response = await client.get(search_url, params=search_params)
            data = response.json()

            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # Fetch summaries
            summary_url = f"{NCBI_BASE_URL}/esummary.fcgi"
            summary_params = {
                "db": "clinvar",
                "id": ",".join(id_list),
                "retmode": "json"
            }

            summary_response = await client.get(summary_url, params=summary_params)
            summary_data = summary_response.json()

            results = []
            genes_seen = set()

            doc_sums = summary_data.get("result", {})
            for uid in id_list:
                if uid in doc_sums and uid != "uids":
                    variant = doc_sums[uid]
                    gene_list = variant.get("genes", [])

                    for gene in gene_list:
                        if isinstance(gene, dict):
                            gene_symbol = gene.get("symbol", "")
                            if gene_symbol and gene_symbol not in genes_seen:
                                genes_seen.add(gene_symbol)
                                results.append({
                                    "gene_symbol": gene_symbol,
                                    "gene_id": gene.get("geneid"),
                                    "condition": condition,
                                    "variant_title": variant.get("title", ""),
                                })

            return results

    except Exception as e:
        logger.error(f"Error searching ClinVar for condition {condition}: {e}")
        return []


def clear_genetics_cache():
    """Clear the genetics data cache."""
    _genetics_cache.clear()
    return {"status": "success", "message": "Cache cleared"}


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    return {
        "entries": len(_genetics_cache._cache),
        "keys": list(_genetics_cache._cache.keys())[:20],  # First 20 keys
    }
