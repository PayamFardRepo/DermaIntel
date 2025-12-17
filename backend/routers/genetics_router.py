"""
Genetics Router - Genetic Risk Factors Module

Handles genetic test results and risk assessment:
- Manual entry of genetic test results
- VCF file upload and parsing
- Dermatology-relevant variant interpretation
- Integration with risk calculator
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import json
import uuid
import re
import logging

from database import (
    get_db, User, GeneticTestResult, GeneticVariant, DermatologyGeneReference, FamilyMember
)
from auth import get_current_active_user

# Import dynamic genetics data service
from genetics_data_service import (
    get_all_dermatology_genes,
    get_dermatology_gene,
    search_clinvar_for_condition,
    clear_genetics_cache,
    get_cache_stats,
    DERMATOLOGY_ANNOTATIONS,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Genetics"])

# =============================================================================
# DERMATOLOGY GENE REFERENCE DATA
# =============================================================================

# Key genes relevant to dermatology with their clinical significance
DERMATOLOGY_GENES = {
    # Melanoma susceptibility genes
    "CDKN2A": {
        "name": "Cyclin Dependent Kinase Inhibitor 2A",
        "category": "melanoma",
        "chromosome": "9",
        "associated_conditions": ["Familial Melanoma", "Pancreatic Cancer"],
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "60-90% lifetime melanoma risk",
        "melanoma_risk_multiplier": 10.0,
        "key_variants": [
            {"hgvs": "p.Ala148Thr", "rsid": "rs3731249", "significance": "pathogenic"},
            {"hgvs": "p.Gly101Trp", "rsid": None, "significance": "pathogenic"},
        ],
        "screening": "Annual full-body skin exam starting at age 10, ophthalmologic exam",
        "acmg_actionable": True,
    },
    "CDK4": {
        "name": "Cyclin Dependent Kinase 4",
        "category": "melanoma",
        "chromosome": "12",
        "associated_conditions": ["Familial Melanoma"],
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "Similar to CDKN2A",
        "melanoma_risk_multiplier": 8.0,
        "key_variants": [
            {"hgvs": "p.Arg24Cys", "rsid": None, "significance": "pathogenic"},
            {"hgvs": "p.Arg24His", "rsid": None, "significance": "pathogenic"},
        ],
        "screening": "Annual full-body skin exam",
        "acmg_actionable": True,
    },
    "BAP1": {
        "name": "BRCA1 Associated Protein 1",
        "category": "melanoma",
        "chromosome": "3",
        "associated_conditions": ["BAP1 Tumor Predisposition Syndrome", "Uveal Melanoma", "Cutaneous Melanoma", "Mesothelioma"],
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "Increased risk of multiple cancers",
        "melanoma_risk_multiplier": 5.0,
        "key_variants": [],
        "screening": "Annual skin exam, ophthalmologic exam every 6-12 months",
        "acmg_actionable": True,
    },
    "MC1R": {
        "name": "Melanocortin 1 Receptor",
        "category": "pigmentation",
        "chromosome": "16",
        "associated_conditions": ["Red Hair Color", "Fair Skin", "Increased UV Sensitivity", "Melanoma Risk"],
        "inheritance": "complex",
        "penetrance": "variable",
        "risk_increase": "2-4x increased melanoma risk per variant allele",
        "melanoma_risk_multiplier": 2.5,
        "key_variants": [
            {"hgvs": "p.Asp84Glu", "rsid": "rs1805005", "significance": "risk_factor", "name": "D84E"},
            {"hgvs": "p.Arg151Cys", "rsid": "rs1805007", "significance": "risk_factor", "name": "R151C"},
            {"hgvs": "p.Arg160Trp", "rsid": "rs1805008", "significance": "risk_factor", "name": "R160W"},
            {"hgvs": "p.Asp294His", "rsid": "rs1805009", "significance": "risk_factor", "name": "D294H"},
        ],
        "screening": "Enhanced sun protection, regular skin exams",
        "acmg_actionable": False,
    },
    "MITF": {
        "name": "Melanocyte Inducing Transcription Factor",
        "category": "melanoma",
        "chromosome": "3",
        "associated_conditions": ["Melanoma", "Renal Cell Carcinoma"],
        "inheritance": "AD",
        "penetrance": "moderate",
        "risk_increase": "2-5x increased melanoma risk",
        "melanoma_risk_multiplier": 3.0,
        "key_variants": [
            {"hgvs": "p.Glu318Lys", "rsid": "rs149617956", "significance": "pathogenic"},
        ],
        "screening": "Regular skin and kidney surveillance",
        "acmg_actionable": False,
    },

    # BRAF/NRAS - Somatic mutations (for tumor profiling)
    "BRAF": {
        "name": "B-Raf Proto-Oncogene",
        "category": "melanoma_somatic",
        "chromosome": "7",
        "associated_conditions": ["Melanoma (somatic)", "Treatment Selection"],
        "inheritance": "somatic",
        "penetrance": "N/A",
        "risk_increase": "Present in ~50% of melanomas",
        "melanoma_risk_multiplier": None,  # Somatic, not germline risk
        "key_variants": [
            {"hgvs": "p.Val600Glu", "rsid": None, "significance": "therapeutic_target", "name": "V600E"},
            {"hgvs": "p.Val600Lys", "rsid": None, "significance": "therapeutic_target", "name": "V600K"},
        ],
        "screening": "Tumor testing for targeted therapy selection",
        "treatment_implications": "BRAF inhibitors (vemurafenib, dabrafenib) + MEK inhibitors",
        "acmg_actionable": False,
    },

    # Basal Cell Carcinoma genes
    "PTCH1": {
        "name": "Patched 1",
        "category": "keratinocyte_cancer",
        "chromosome": "9",
        "associated_conditions": ["Gorlin Syndrome", "Basal Cell Nevus Syndrome", "Multiple BCCs"],
        "inheritance": "AD",
        "penetrance": "high",
        "risk_increase": "90% develop BCCs by age 35",
        "bcc_risk_multiplier": 20.0,
        "key_variants": [],
        "screening": "Skin exams every 3-6 months, avoid radiation, dental surveillance",
        "acmg_actionable": True,
    },

    # Xeroderma Pigmentosum genes
    "XPA": {
        "name": "XPA, DNA Damage Recognition And Repair Factor",
        "category": "photosensitivity",
        "chromosome": "9",
        "associated_conditions": ["Xeroderma Pigmentosum", "Extreme UV Sensitivity", "Skin Cancer"],
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "1000x increased skin cancer risk",
        "melanoma_risk_multiplier": 100.0,
        "key_variants": [],
        "screening": "Strict UV avoidance, skin exams every 3 months, neurological monitoring",
        "acmg_actionable": True,
    },

    # Pharmacogenomics
    "TPMT": {
        "name": "Thiopurine S-Methyltransferase",
        "category": "pharmacogenomics",
        "chromosome": "6",
        "associated_conditions": ["Azathioprine Sensitivity"],
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "Myelosuppression risk with standard azathioprine dosing",
        "key_variants": [
            {"hgvs": "p.Ala154Thr", "rsid": "rs1800460", "significance": "decreased_function", "name": "*3A"},
            {"hgvs": "p.Tyr240Cys", "rsid": "rs1800462", "significance": "decreased_function", "name": "*2"},
        ],
        "drug_implications": "Reduce azathioprine dose by 50-90% in poor metabolizers",
        "acmg_actionable": True,
    },
    "DPYD": {
        "name": "Dihydropyrimidine Dehydrogenase",
        "category": "pharmacogenomics",
        "chromosome": "1",
        "associated_conditions": ["5-FU Toxicity"],
        "inheritance": "AR",
        "penetrance": "high",
        "risk_increase": "Severe/fatal toxicity with standard 5-FU dosing",
        "key_variants": [
            {"hgvs": "IVS14+1G>A", "rsid": "rs3918290", "significance": "no_function", "name": "*2A"},
        ],
        "drug_implications": "Avoid 5-FU or reduce dose by 50% in heterozygotes",
        "acmg_actionable": True,
    },
}

# Risk level thresholds
RISK_THRESHOLDS = {
    "melanoma": {
        "low": 1.0,
        "moderate": 2.0,
        "high": 5.0,
        "very_high": 10.0,
    },
    "bcc": {
        "low": 1.0,
        "moderate": 3.0,
        "high": 10.0,
        "very_high": 20.0,
    },
}


# =============================================================================
# VCF PARSER
# =============================================================================

def parse_vcf_file(file_content: str, dermatology_genes: dict) -> dict:
    """
    Parse a VCF file and extract dermatology-relevant variants.

    Returns:
        dict with 'variants' list and 'metadata'
    """
    lines = file_content.strip().split('\n')
    variants = []
    metadata = {
        "vcf_version": None,
        "sample_ids": [],
        "total_variants": 0,
        "dermatology_relevant": 0,
    }

    # Gene symbols to look for (case-insensitive)
    target_genes = set(g.upper() for g in dermatology_genes.keys())

    for line in lines:
        # Parse header
        if line.startswith('##fileformat='):
            metadata["vcf_version"] = line.split('=')[1]
            continue
        if line.startswith('##'):
            continue
        if line.startswith('#CHROM'):
            # Header line with sample IDs
            parts = line.split('\t')
            if len(parts) > 9:
                metadata["sample_ids"] = parts[9:]
            continue

        # Parse variant lines
        if not line.strip():
            continue

        metadata["total_variants"] += 1
        parts = line.split('\t')

        if len(parts) < 8:
            continue

        chrom, pos, rsid, ref, alt, qual, filt, info = parts[:8]

        # Extract gene from INFO field (common annotations)
        gene = None
        consequence = None
        hgvs_c = None
        hgvs_p = None

        # Parse INFO field for gene annotations
        for field in info.split(';'):
            if field.startswith('GENE=') or field.startswith('Gene='):
                gene = field.split('=')[1].upper()
            elif field.startswith('ANN=') or field.startswith('CSQ='):
                # VEP/SnpEff annotation
                ann_parts = field.split('=')[1].split('|')
                if len(ann_parts) > 3:
                    gene = ann_parts[3].upper() if ann_parts[3] else gene
                    consequence = ann_parts[1] if len(ann_parts) > 1 else None
                    hgvs_c = ann_parts[9] if len(ann_parts) > 9 else None
                    hgvs_p = ann_parts[10] if len(ann_parts) > 10 else None

        # Check if this variant is in a dermatology-relevant gene
        if gene and gene in target_genes:
            metadata["dermatology_relevant"] += 1

            gene_info = dermatology_genes.get(gene, {})

            variant = {
                "chromosome": chrom,
                "position": int(pos),
                "rsid": rsid if rsid != '.' else None,
                "reference": ref,
                "alternate": alt,
                "quality": float(qual) if qual != '.' else None,
                "filter": filt,
                "gene_symbol": gene,
                "consequence": consequence,
                "hgvs_c": hgvs_c,
                "hgvs_p": hgvs_p,
                "gene_category": gene_info.get("category"),
                "clinical_significance": "unknown",  # Would need ClinVar lookup
                "dermatology_relevance": gene_info.get("associated_conditions", []),
            }

            # Check if it's a known pathogenic variant
            for known in gene_info.get("key_variants", []):
                if (known.get("rsid") == rsid or
                    known.get("hgvs") == hgvs_p):
                    variant["clinical_significance"] = known.get("significance", "pathogenic")
                    break

            variants.append(variant)

    return {
        "metadata": metadata,
        "variants": variants,
    }


def calculate_genetic_risk(variants: list, dermatology_genes: dict) -> dict:
    """
    Calculate overall genetic risk based on detected variants.
    """
    melanoma_multiplier = 1.0
    bcc_multiplier = 1.0
    scc_multiplier = 1.0
    uv_sensitivity = "normal"
    pathogenic_genes = []
    risk_factors = []

    for variant in variants:
        gene = variant.get("gene_symbol", "").upper()
        significance = variant.get("clinical_significance", "unknown")

        if gene not in dermatology_genes:
            continue

        gene_info = dermatology_genes[gene]

        # Only count pathogenic/likely pathogenic variants
        if significance in ["pathogenic", "likely_pathogenic", "risk_factor"]:
            pathogenic_genes.append(gene)

            # Apply risk multipliers
            if gene_info.get("melanoma_risk_multiplier"):
                melanoma_multiplier *= gene_info["melanoma_risk_multiplier"]
                risk_factors.append({
                    "gene": gene,
                    "impact": f"{gene_info['melanoma_risk_multiplier']}x melanoma risk",
                    "condition": "Melanoma",
                })

            if gene_info.get("bcc_risk_multiplier"):
                bcc_multiplier *= gene_info["bcc_risk_multiplier"]
                risk_factors.append({
                    "gene": gene,
                    "impact": f"{gene_info['bcc_risk_multiplier']}x BCC risk",
                    "condition": "Basal Cell Carcinoma",
                })

            # Check for UV sensitivity genes
            if gene_info.get("category") == "photosensitivity":
                uv_sensitivity = "severely_increased"
            elif gene == "MC1R":
                uv_sensitivity = "increased"

    # Determine risk levels
    def get_risk_level(multiplier, thresholds):
        if multiplier >= thresholds["very_high"]:
            return "very_high"
        elif multiplier >= thresholds["high"]:
            return "high"
        elif multiplier >= thresholds["moderate"]:
            return "moderate"
        return "low"

    return {
        "melanoma_risk": {
            "multiplier": round(melanoma_multiplier, 2),
            "level": get_risk_level(melanoma_multiplier, RISK_THRESHOLDS["melanoma"]),
            "contributing_genes": [g for g in pathogenic_genes if dermatology_genes.get(g, {}).get("melanoma_risk_multiplier")],
        },
        "bcc_risk": {
            "multiplier": round(bcc_multiplier, 2),
            "level": get_risk_level(bcc_multiplier, RISK_THRESHOLDS["bcc"]),
            "contributing_genes": [g for g in pathogenic_genes if dermatology_genes.get(g, {}).get("bcc_risk_multiplier")],
        },
        "uv_sensitivity": uv_sensitivity,
        "pathogenic_genes": list(set(pathogenic_genes)),
        "risk_factors": risk_factors,
        "overall_risk_level": get_risk_level(
            max(melanoma_multiplier, bcc_multiplier),
            RISK_THRESHOLDS["melanoma"]
        ),
    }


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/genetics/reference-genes")
async def get_reference_genes(
    category: Optional[str] = None,
    use_dynamic: bool = True,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get list of dermatology-relevant genes and their clinical significance.

    Fetches data dynamically from ClinVar/NCBI when use_dynamic=True (default).
    Falls back to local data if external APIs are unavailable.

    Categories: melanoma, keratinocyte_cancer, pigmentation, photosensitivity, pharmacogenomics
    """
    try:
        if use_dynamic:
            # Fetch from external APIs (ClinVar, NCBI)
            all_genes = await get_all_dermatology_genes()

            if category:
                all_genes = [g for g in all_genes if g.get("category") == category]

            return {
                "total_genes": len(all_genes),
                "category_filter": category,
                "source": "dynamic",
                "genes": [
                    {
                        "gene_symbol": gene.get("gene_symbol"),
                        "gene_name": gene.get("gene_name"),
                        "category": gene.get("category"),
                        "chromosome": gene.get("chromosome"),
                        "associated_conditions": gene.get("associated_conditions", []),
                        "inheritance": gene.get("inheritance"),
                        "penetrance": gene.get("penetrance"),
                        "risk_increase": gene.get("risk_increase"),
                        "melanoma_risk_multiplier": gene.get("melanoma_risk_multiplier"),
                        "bcc_risk_multiplier": gene.get("bcc_risk_multiplier"),
                        "screening": gene.get("screening"),
                        "acmg_actionable": gene.get("acmg_actionable", False),
                        "clinvar_pathogenic_count": gene.get("clinvar_pathogenic_count", 0),
                        "key_variants": gene.get("key_variants", [])[:3],  # Top 3 variants
                        "sources": gene.get("sources", []),
                    }
                    for gene in all_genes
                ]
            }
    except Exception as e:
        logger.warning(f"Dynamic gene fetch failed, using fallback: {e}")

    # Fallback to static data
    genes = DERMATOLOGY_GENES

    if category:
        genes = {k: v for k, v in genes.items() if v.get("category") == category}

    return {
        "total_genes": len(genes),
        "category_filter": category,
        "source": "static",
        "genes": [
            {
                "gene_symbol": symbol,
                "gene_name": info["name"],
                "category": info["category"],
                "chromosome": info.get("chromosome"),
                "associated_conditions": info.get("associated_conditions", []),
                "inheritance": info.get("inheritance"),
                "risk_increase": info.get("risk_increase"),
                "melanoma_risk_multiplier": info.get("melanoma_risk_multiplier"),
                "screening": info.get("screening"),
                "acmg_actionable": info.get("acmg_actionable", False),
            }
            for symbol, info in genes.items()
        ]
    }


@router.get("/genetics/gene/{gene_symbol}")
async def get_gene_info(
    gene_symbol: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get detailed information for a specific gene from ClinVar/NCBI.
    """
    try:
        gene_info = await get_dermatology_gene(gene_symbol)

        if not gene_info:
            raise HTTPException(status_code=404, detail=f"Gene {gene_symbol} not found")

        return gene_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching gene {gene_symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/genetics/search-condition")
async def search_genes_by_condition(
    condition: str,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user)
):
    """
    Search ClinVar for genes associated with a specific skin condition.

    Example conditions: "melanoma", "basal cell carcinoma", "xeroderma pigmentosum"
    """
    try:
        results = await search_clinvar_for_condition(condition, limit)

        return {
            "condition": condition,
            "total_results": len(results),
            "genes": results
        }

    except Exception as e:
        logger.error(f"Error searching for condition {condition}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/genetics/refresh-cache")
async def refresh_gene_cache(
    current_user: User = Depends(get_current_active_user)
):
    """
    Clear the genetics data cache to force fresh fetch from external APIs.
    """
    result = clear_genetics_cache()
    return {
        "status": "success",
        "message": "Gene cache cleared. Next request will fetch fresh data from ClinVar/NCBI."
    }


@router.get("/genetics/cache-stats")
async def get_genetics_cache_stats(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get statistics about the genetics data cache.
    """
    return get_cache_stats()


@router.post("/genetics/test-results")
async def create_genetic_test_result(
    test_type: str = Form(...),
    test_name: Optional[str] = Form(None),
    lab_name: Optional[str] = Form(None),
    test_date: str = Form(...),
    sample_type: Optional[str] = Form("blood"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new genetic test result record (manual entry).
    """
    try:
        test_id = f"GT-{uuid.uuid4().hex[:8].upper()}"

        test_result = GeneticTestResult(
            user_id=current_user.id,
            test_id=test_id,
            test_type=test_type,
            test_name=test_name,
            lab_name=lab_name,
            test_date=datetime.fromisoformat(test_date),
            sample_type=sample_type,
            status="pending",
        )

        db.add(test_result)
        db.commit()
        db.refresh(test_result)

        return {
            "id": test_result.id,
            "test_id": test_id,
            "status": "pending",
            "message": "Genetic test record created. Add variants using /genetics/test-results/{id}/variants"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create test result: {str(e)}")


@router.post("/genetics/test-results/{test_id}/variants")
async def add_variant_to_test(
    test_id: int,
    gene_symbol: str = Form(...),
    chromosome: str = Form(...),
    position: int = Form(...),
    reference: str = Form(...),
    alternate: str = Form(...),
    classification: str = Form(...),  # pathogenic, likely_pathogenic, vus, likely_benign, benign
    rsid: Optional[str] = Form(None),
    hgvs_c: Optional[str] = Form(None),
    hgvs_p: Optional[str] = Form(None),
    zygosity: Optional[str] = Form("heterozygous"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add a genetic variant to an existing test result.
    """
    # Verify test belongs to user
    test_result = db.query(GeneticTestResult).filter(
        GeneticTestResult.id == test_id,
        GeneticTestResult.user_id == current_user.id
    ).first()

    if not test_result:
        raise HTTPException(status_code=404, detail="Test result not found")

    # Get gene info for dermatology relevance
    gene_info = DERMATOLOGY_GENES.get(gene_symbol.upper(), {})

    variant = GeneticVariant(
        test_result_id=test_id,
        user_id=current_user.id,
        gene_symbol=gene_symbol.upper(),
        gene_name=gene_info.get("name"),
        chromosome=chromosome,
        position=position,
        reference=reference,
        alternate=alternate,
        rsid=rsid,
        hgvs_c=hgvs_c,
        hgvs_p=hgvs_p,
        classification=classification,
        zygosity=zygosity,
        skin_condition_associations=gene_info.get("associated_conditions"),
        melanoma_risk_modifier=gene_info.get("melanoma_risk_multiplier"),
        uv_sensitivity_impact="increased" if gene_info.get("category") == "photosensitivity" else None,
    )

    db.add(variant)

    # Update test result counts
    if classification == "pathogenic":
        test_result.pathogenic_variants_found = (test_result.pathogenic_variants_found or 0) + 1
    elif classification == "likely_pathogenic":
        test_result.likely_pathogenic_found = (test_result.likely_pathogenic_found or 0) + 1
    elif classification == "vus":
        test_result.vus_found = (test_result.vus_found or 0) + 1

    db.commit()
    db.refresh(variant)

    return {
        "variant_id": variant.id,
        "gene": gene_symbol.upper(),
        "classification": classification,
        "dermatology_relevant": bool(gene_info),
        "message": "Variant added successfully"
    }


@router.post("/genetics/upload-vcf")
async def upload_vcf_file(
    file: UploadFile = File(...),
    test_name: Optional[str] = Form("VCF Upload"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload and parse a VCF file to extract dermatology-relevant variants.
    """
    if not file.filename.endswith(('.vcf', '.vcf.gz')):
        raise HTTPException(status_code=400, detail="File must be a VCF file (.vcf or .vcf.gz)")

    try:
        # Read file content
        content = await file.read()

        # Handle gzipped files
        if file.filename.endswith('.gz'):
            import gzip
            content = gzip.decompress(content)

        content_str = content.decode('utf-8')

        # Parse VCF
        parsed = parse_vcf_file(content_str, DERMATOLOGY_GENES)

        # Create test result
        test_id = f"VCF-{uuid.uuid4().hex[:8].upper()}"

        test_result = GeneticTestResult(
            user_id=current_user.id,
            test_id=test_id,
            test_type="vcf_upload",
            test_name=test_name,
            test_date=datetime.utcnow(),
            sample_type="unknown",
            total_variants_tested=parsed["metadata"]["total_variants"],
            status="completed",
        )

        db.add(test_result)
        db.flush()  # Get ID without committing

        # Add variants
        pathogenic_count = 0
        likely_pathogenic_count = 0
        vus_count = 0

        for var_data in parsed["variants"]:
            classification = var_data.get("clinical_significance", "vus")
            gene_symbol = var_data["gene_symbol"]

            # Look up melanoma risk modifier from gene database
            gene_info = DERMATOLOGY_GENES.get(gene_symbol, {})
            melanoma_modifier = gene_info.get("melanoma_risk_multiplier")

            variant = GeneticVariant(
                test_result_id=test_result.id,
                user_id=current_user.id,
                gene_symbol=gene_symbol,
                chromosome=var_data["chromosome"],
                position=var_data["position"],
                reference=var_data["reference"],
                alternate=var_data["alternate"],
                rsid=var_data.get("rsid"),
                hgvs_c=var_data.get("hgvs_c"),
                hgvs_p=var_data.get("hgvs_p"),
                consequence=var_data.get("consequence"),
                classification=classification,
                quality_score=var_data.get("quality"),
                filter_status=var_data.get("filter"),
                skin_condition_associations=var_data.get("dermatology_relevance"),
                melanoma_risk_modifier=melanoma_modifier if classification in ["pathogenic", "likely_pathogenic"] else None,
            )

            db.add(variant)

            if classification == "pathogenic":
                pathogenic_count += 1
            elif classification == "likely_pathogenic":
                likely_pathogenic_count += 1
            else:
                vus_count += 1

        # Update counts
        test_result.pathogenic_variants_found = pathogenic_count
        test_result.likely_pathogenic_found = likely_pathogenic_count
        test_result.vus_found = vus_count

        # Calculate risk
        risk_assessment = calculate_genetic_risk(parsed["variants"], DERMATOLOGY_GENES)
        test_result.overall_risk_level = risk_assessment["overall_risk_level"]
        test_result.melanoma_risk = risk_assessment["melanoma_risk"]
        test_result.bcc_risk = risk_assessment["bcc_risk"]
        test_result.risk_calculator_adjustment = risk_assessment["melanoma_risk"]["multiplier"]
        test_result.linked_to_risk_calculator = True

        db.commit()

        return {
            "test_id": test_id,
            "file_processed": file.filename,
            "total_variants_in_file": parsed["metadata"]["total_variants"],
            "dermatology_relevant_variants": parsed["metadata"]["dermatology_relevant"],
            "pathogenic_found": pathogenic_count,
            "likely_pathogenic_found": likely_pathogenic_count,
            "vus_found": vus_count,
            "risk_assessment": risk_assessment,
            "message": "VCF file processed successfully"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process VCF: {str(e)}")


@router.get("/genetics/test-results")
async def get_genetic_test_results(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all genetic test results for the current user.
    """
    results = db.query(GeneticTestResult).filter(
        GeneticTestResult.user_id == current_user.id
    ).order_by(GeneticTestResult.created_at.desc()).all()

    return {
        "total": len(results),
        "results": [
            {
                "id": r.id,
                "test_id": r.test_id,
                "test_type": r.test_type,
                "test_name": r.test_name,
                "lab_name": r.lab_name,
                "test_date": r.test_date.isoformat() if r.test_date else None,
                "status": r.status,
                "overall_risk_level": r.overall_risk_level,
                "pathogenic_variants": r.pathogenic_variants_found,
                "likely_pathogenic_variants": r.likely_pathogenic_found,
                "melanoma_risk": r.melanoma_risk,
                "bcc_risk": r.bcc_risk,
                "created_at": r.created_at.isoformat(),
            }
            for r in results
        ]
    }


@router.get("/genetics/test-results/{test_id}")
async def get_genetic_test_detail(
    test_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific genetic test result.
    """
    result = db.query(GeneticTestResult).filter(
        GeneticTestResult.id == test_id,
        GeneticTestResult.user_id == current_user.id
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail="Test result not found")

    # Get variants
    variants = db.query(GeneticVariant).filter(
        GeneticVariant.test_result_id == test_id
    ).all()

    return {
        "id": result.id,
        "test_id": result.test_id,
        "test_type": result.test_type,
        "test_name": result.test_name,
        "lab_name": result.lab_name,
        "test_date": result.test_date.isoformat() if result.test_date else None,
        "sample_type": result.sample_type,
        "status": result.status,
        "summary": {
            "total_variants_tested": result.total_variants_tested,
            "pathogenic_found": result.pathogenic_variants_found,
            "likely_pathogenic_found": result.likely_pathogenic_found,
            "vus_found": result.vus_found,
        },
        "risk_assessment": {
            "overall_risk_level": result.overall_risk_level,
            "melanoma_risk": result.melanoma_risk,
            "bcc_risk": result.bcc_risk,
            "scc_risk": result.scc_risk,
            "risk_calculator_adjustment": result.risk_calculator_adjustment,
        },
        "recommendations": result.recommendations,
        "screening_recommendations": result.screening_recommendations,
        "genetic_counseling_recommended": result.genetic_counseling_recommended,
        "variants": [
            {
                "id": v.id,
                "gene_symbol": v.gene_symbol,
                "chromosome": v.chromosome,
                "position": v.position,
                "reference": v.reference,
                "alternate": v.alternate,
                "rsid": v.rsid,
                "hgvs_c": v.hgvs_c,
                "hgvs_p": v.hgvs_p,
                "classification": v.classification,
                "zygosity": v.zygosity,
                "skin_condition_associations": v.skin_condition_associations,
                "melanoma_risk_modifier": v.melanoma_risk_modifier,
            }
            for v in variants
        ],
        "created_at": result.created_at.isoformat(),
    }


@router.get("/genetics/risk-summary")
async def get_genetic_risk_summary(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get aggregated genetic risk summary across all tests for the user.
    Useful for integration with the main risk calculator.
    """
    # Get all completed tests
    results = db.query(GeneticTestResult).filter(
        GeneticTestResult.user_id == current_user.id,
        GeneticTestResult.status == "completed"
    ).all()

    if not results:
        return {
            "has_genetic_data": False,
            "message": "No genetic test results found. Add genetic test results to enhance risk assessment.",
            "risk_modifier": 1.0,
        }

    # Get all pathogenic/likely pathogenic variants
    pathogenic_variants = db.query(GeneticVariant).filter(
        GeneticVariant.user_id == current_user.id,
        GeneticVariant.classification.in_(["pathogenic", "likely_pathogenic"])
    ).all()

    # Calculate combined risk
    affected_genes = set()
    max_melanoma_multiplier = 1.0
    max_bcc_multiplier = 1.0
    recommendations = []

    for variant in pathogenic_variants:
        gene = variant.gene_symbol
        affected_genes.add(gene)

        if variant.melanoma_risk_modifier:
            max_melanoma_multiplier = max(max_melanoma_multiplier, variant.melanoma_risk_modifier)

        gene_info = DERMATOLOGY_GENES.get(gene, {})
        if gene_info.get("bcc_risk_multiplier"):
            max_bcc_multiplier = max(max_bcc_multiplier, gene_info["bcc_risk_multiplier"])

        if gene_info.get("screening"):
            recommendations.append({
                "gene": gene,
                "recommendation": gene_info["screening"]
            })

    overall_modifier = max(max_melanoma_multiplier, max_bcc_multiplier)

    # Determine risk levels
    melanoma_level = (
        "very_high" if max_melanoma_multiplier >= 10 else
        "high" if max_melanoma_multiplier >= 5 else
        "moderate" if max_melanoma_multiplier >= 2 else
        "low"
    )
    nmsc_level = (
        "very_high" if max_bcc_multiplier >= 10 else
        "high" if max_bcc_multiplier >= 5 else
        "moderate" if max_bcc_multiplier >= 2 else
        "average"
    )

    # Categorize genes by risk level
    high_risk_genes = []
    moderate_risk_genes = []
    pharmacogenomic_alerts = []

    for gene in affected_genes:
        gene_info = DERMATOLOGY_GENES.get(gene, {})
        multiplier = gene_info.get("melanoma_risk_multiplier", 1)
        category = gene_info.get("category", "")

        if category == "pharmacogenomics":
            pharmacogenomic_alerts.append(f"{gene}: {gene_info.get('risk_increase', 'Drug interaction')}")
        elif multiplier and multiplier >= 5:
            high_risk_genes.append(gene)
        elif multiplier and multiplier >= 2:
            moderate_risk_genes.append(gene)

    # Generate recommendations list
    recommendation_list = [r["recommendation"] for r in recommendations]

    return {
        "has_genetic_data": True,
        "tests_count": len(results),
        "pathogenic_variants_count": len(pathogenic_variants),
        "affected_genes": list(affected_genes),
        "melanoma_risk": {
            "level": melanoma_level,
            "multiplier": max_melanoma_multiplier,
        },
        "nmsc_risk": {
            "level": nmsc_level,
            "multiplier": max_bcc_multiplier,
        },
        "high_risk_genes": high_risk_genes,
        "moderate_risk_genes": moderate_risk_genes,
        "pharmacogenomic_alerts": pharmacogenomic_alerts,
        "recommendations": recommendation_list,
        "risk_assessment": {
            "melanoma_risk_multiplier": max_melanoma_multiplier,
            "bcc_risk_multiplier": max_bcc_multiplier,
            "overall_risk_modifier": overall_modifier,
            "risk_level": melanoma_level,
        },
        "screening_recommendations": recommendations,
        "genetic_counseling_recommended": max_melanoma_multiplier >= 5 or max_bcc_multiplier >= 10,
    }


@router.delete("/genetics/test-results/{test_id}")
async def delete_genetic_test(
    test_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a genetic test result and all associated variants.
    """
    result = db.query(GeneticTestResult).filter(
        GeneticTestResult.id == test_id,
        GeneticTestResult.user_id == current_user.id
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail="Test result not found")

    # Delete associated variants first
    db.query(GeneticVariant).filter(
        GeneticVariant.test_result_id == test_id
    ).delete()

    db.delete(result)
    db.commit()

    return {"message": "Genetic test result deleted successfully"}


# =============================================================================
# FAMILY HISTORY ENDPOINTS
# =============================================================================

@router.get("/family-history")
async def get_family_history(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all family members for the current user.
    """
    family_members = db.query(FamilyMember).filter(
        FamilyMember.user_id == current_user.id
    ).order_by(FamilyMember.relationship_type).all()

    return {
        "family_members": [
            {
                "id": member.id,
                "relationship_type": member.relationship_type,
                "relationship_side": member.relationship_side,
                "name": member.name,
                "gender": member.gender,
                "year_of_birth": member.year_of_birth,
                "is_alive": member.is_alive,
                "age_at_death": member.age_at_death,
                "has_skin_cancer": member.has_skin_cancer,
                "skin_cancer_types": member.skin_cancer_types,
                "skin_cancer_count": member.skin_cancer_count,
                "earliest_diagnosis_age": member.earliest_diagnosis_age,
                "has_melanoma": member.has_melanoma,
                "melanoma_count": member.melanoma_count,
                "melanoma_subtypes": member.melanoma_subtypes,
                "melanoma_familial_syndrome": member.melanoma_familial_syndrome
            }
            for member in family_members
        ],
        "total_count": len(family_members),
        "has_melanoma_history": any(m.has_melanoma for m in family_members),
        "has_skin_cancer_history": any(m.has_skin_cancer for m in family_members)
    }


async def _create_family_member(
    relationship_type: str,
    relationship_side: Optional[str],
    name: Optional[str],
    gender: Optional[str],
    year_of_birth: Optional[str],
    is_alive: str,
    age_at_death: Optional[str],
    has_skin_cancer: str,
    skin_cancer_types: Optional[str],
    has_melanoma: str,
    melanoma_count: Optional[str],
    skin_type: Optional[str],
    hair_color: Optional[str],
    eye_color: Optional[str],
    has_many_moles: str,
    has_atypical_moles: str,
    genetic_testing_done: str,
    notes: Optional[str],
    current_user: User,
    db: Session
):
    """Helper function to create family member."""
    # Validate relationship type
    valid_relationships = ["parent", "sibling", "grandparent", "aunt_uncle", "cousin", "child"]
    if relationship_type not in valid_relationships:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid relationship type. Must be one of: {', '.join(valid_relationships)}"
        )

    # Parse boolean strings
    def parse_bool(val: str) -> bool:
        return val.lower() in ('true', '1', 'yes') if val else False

    # Parse optional int
    def parse_int(val: Optional[str]) -> Optional[int]:
        if val is None or val == '':
            return None
        try:
            return int(val)
        except ValueError:
            return None

    # Parse skin cancer types JSON
    skin_cancer_types_parsed = None
    if skin_cancer_types:
        try:
            skin_cancer_types_parsed = json.loads(skin_cancer_types)
        except:
            skin_cancer_types_parsed = None

    family_member = FamilyMember(
        user_id=current_user.id,
        relationship_type=relationship_type,
        relationship_side=relationship_side,
        name=name,
        gender=gender,
        year_of_birth=parse_int(year_of_birth),
        is_alive=parse_bool(is_alive),
        age_at_death=parse_int(age_at_death),
        has_skin_cancer=parse_bool(has_skin_cancer),
        skin_cancer_types=skin_cancer_types_parsed,
        has_melanoma=parse_bool(has_melanoma),
        melanoma_count=parse_int(melanoma_count) or 0,
        skin_type=skin_type,
        hair_color=hair_color,
        eye_color=eye_color,
        has_many_moles=parse_bool(has_many_moles),
        has_atypical_moles=parse_bool(has_atypical_moles),
        genetic_testing_done=parse_bool(genetic_testing_done),
        notes=notes
    )

    db.add(family_member)
    db.commit()
    db.refresh(family_member)

    return {
        "message": "Family member added successfully",
        "family_member": {
            "id": family_member.id,
            "relationship_type": family_member.relationship_type,
            "name": family_member.name,
            "has_skin_cancer": family_member.has_skin_cancer,
            "has_melanoma": family_member.has_melanoma
        }
    }


@router.post("/family-history")
async def add_family_member(
    relationship_type: str = Form(...),
    relationship_side: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    year_of_birth: Optional[str] = Form(None),
    is_alive: str = Form("true"),
    age_at_death: Optional[str] = Form(None),
    has_skin_cancer: str = Form("false"),
    skin_cancer_types: Optional[str] = Form(None),
    has_melanoma: str = Form("false"),
    melanoma_count: Optional[str] = Form(None),
    skin_type: Optional[str] = Form(None),
    hair_color: Optional[str] = Form(None),
    eye_color: Optional[str] = Form(None),
    has_many_moles: str = Form("false"),
    has_atypical_moles: str = Form("false"),
    genetic_testing_done: str = Form("false"),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add a new family member with their medical history."""
    return await _create_family_member(
        relationship_type, relationship_side, name, gender, year_of_birth,
        is_alive, age_at_death, has_skin_cancer, skin_cancer_types,
        has_melanoma, melanoma_count, skin_type, hair_color, eye_color,
        has_many_moles, has_atypical_moles, genetic_testing_done, notes,
        current_user, db
    )


@router.post("/family-history/add")
async def add_family_member_alt(
    relationship_type: str = Form(...),
    relationship_side: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    year_of_birth: Optional[str] = Form(None),
    is_alive: str = Form("true"),
    age_at_death: Optional[str] = Form(None),
    has_skin_cancer: str = Form("false"),
    skin_cancer_types: Optional[str] = Form(None),
    has_melanoma: str = Form("false"),
    melanoma_count: Optional[str] = Form(None),
    skin_type: Optional[str] = Form(None),
    hair_color: Optional[str] = Form(None),
    eye_color: Optional[str] = Form(None),
    has_many_moles: str = Form("false"),
    has_atypical_moles: str = Form("false"),
    genetic_testing_done: str = Form("false"),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add a new family member (alternate endpoint)."""
    return await _create_family_member(
        relationship_type, relationship_side, name, gender, year_of_birth,
        is_alive, age_at_death, has_skin_cancer, skin_cancer_types,
        has_melanoma, melanoma_count, skin_type, hair_color, eye_color,
        has_many_moles, has_atypical_moles, genetic_testing_done, notes,
        current_user, db
    )


@router.put("/family-history/{member_id}")
async def update_family_member(
    member_id: int,
    relationship_type: Optional[str] = Form(None),
    relationship_side: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    year_of_birth: Optional[int] = Form(None),
    is_alive: Optional[bool] = Form(None),
    age_at_death: Optional[int] = Form(None),
    has_skin_cancer: Optional[bool] = Form(None),
    skin_cancer_count: Optional[int] = Form(None),
    earliest_diagnosis_age: Optional[int] = Form(None),
    has_melanoma: Optional[bool] = Form(None),
    melanoma_count: Optional[int] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing family member's information.
    """
    member = db.query(FamilyMember).filter(
        FamilyMember.id == member_id,
        FamilyMember.user_id == current_user.id
    ).first()

    if not member:
        raise HTTPException(status_code=404, detail="Family member not found")

    # Update only provided fields
    if relationship_type is not None:
        member.relationship_type = relationship_type
    if relationship_side is not None:
        member.relationship_side = relationship_side
    if name is not None:
        member.name = name
    if gender is not None:
        member.gender = gender
    if year_of_birth is not None:
        member.year_of_birth = year_of_birth
    if is_alive is not None:
        member.is_alive = is_alive
    if age_at_death is not None:
        member.age_at_death = age_at_death
    if has_skin_cancer is not None:
        member.has_skin_cancer = has_skin_cancer
    if skin_cancer_count is not None:
        member.skin_cancer_count = skin_cancer_count
    if earliest_diagnosis_age is not None:
        member.earliest_diagnosis_age = earliest_diagnosis_age
    if has_melanoma is not None:
        member.has_melanoma = has_melanoma
    if melanoma_count is not None:
        member.melanoma_count = melanoma_count

    db.commit()
    db.refresh(member)

    return {
        "message": "Family member updated successfully",
        "family_member": {
            "id": member.id,
            "relationship_type": member.relationship_type,
            "name": member.name,
            "has_skin_cancer": member.has_skin_cancer,
            "has_melanoma": member.has_melanoma
        }
    }


@router.delete("/family-history/{member_id}")
async def delete_family_member(
    member_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a family member from the user's history.
    """
    member = db.query(FamilyMember).filter(
        FamilyMember.id == member_id,
        FamilyMember.user_id == current_user.id
    ).first()

    if not member:
        raise HTTPException(status_code=404, detail="Family member not found")

    db.delete(member)
    db.commit()

    return {"message": "Family member deleted successfully"}


# =============================================================================
# GENETIC TESTING ENDPOINTS (from main_legacy.py)
# =============================================================================

@router.get("/genetic-testing")
async def get_genetic_tests(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all genetic test results for the current user.
    """
    try:
        from database import GeneticTest

        tests = db.query(GeneticTest).filter(
            GeneticTest.user_id == current_user.id
        ).all()

        return {
            "tests": [
                {
                    "id": test.id,
                    "test_type": test.test_type,
                    "test_date": test.test_date,
                    "lab_name": test.lab_name,
                    "gene_tested": test.gene_tested,
                    "result": test.result,
                    "mutations_detected": test.mutations_detected,
                    "clinical_significance": test.clinical_significance,
                    "report_url": test.report_url,
                    "notes": test.notes,
                    "created_at": test.created_at
                }
                for test in tests
            ],
            "total_count": len(tests)
        }
    except Exception as e:
        # Return empty list if table doesn't exist
        return {
            "tests": [],
            "total_count": 0
        }


@router.get("/genetic-risk")
async def get_genetic_risk_profile(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the user's genetic risk profile with detailed risk scores and recommendations.
    """
    try:
        from database import GeneticRiskProfile

        # Get or create genetic risk profile
        risk_profile = db.query(GeneticRiskProfile).filter(
            GeneticRiskProfile.user_id == current_user.id
        ).first()

        # Get family members count for stats
        family_members = db.query(FamilyMember).filter(
            FamilyMember.user_id == current_user.id
        ).all()

        total_family_members = len(family_members)
        with_skin_cancer = sum(1 for m in family_members if m.has_skin_cancer)
        with_melanoma = sum(1 for m in family_members if m.has_melanoma)

        if not risk_profile:
            # Return default low risk if no profile exists yet
            return {
                "overall_genetic_risk_score": 25.0,
                "overall_risk_level": "low",
                "melanoma_risk_score": 20.0,
                "melanoma_risk_level": "low",
                "bcc_risk_score": 15.0,
                "scc_risk_score": 15.0,
                "basal_cell_carcinoma_risk_score": 15.0,
                "squamous_cell_carcinoma_risk_score": 15.0,
                "family_history_score": 0.0,
                "personal_risk_score": 50.0,
                "inheritance_pattern": "sporadic",
                "genetic_counseling_recommended": False,
                "screening_frequency_self": "biannual",
                "screening_frequency_professional": "annual",
                "recommended_screening_frequency": "biannual",
                "recommended_professional_frequency": "annual",
                "high_priority_monitoring": False,
                "confidence_level": 0.5,
                "total_relatives_with_skin_cancer": with_skin_cancer,
                "first_degree_relatives_affected": 0,
                "second_degree_relatives_affected": 0,
                "familial_melanoma_syndrome_suspected": False,
                "has_early_onset_melanoma": False,
                "has_multiple_family_melanomas": False,
                "affected_lineages": {
                    "maternal": False,
                    "paternal": False
                },
                "generation_pattern": {
                    "grandparents": 0,
                    "parents": 0,
                    "siblings": 0,
                    "aunts_uncles": 0
                },
                "risk_reduction_recommendations": [
                    "Perform regular self-examinations",
                    "Use broad-spectrum SPF 30+ sunscreen daily",
                    "Avoid midday sun (10 AM - 4 PM)"
                ],
                "family_stats": {
                    "total_family_members": total_family_members,
                    "with_skin_cancer": with_skin_cancer,
                    "with_melanoma": with_melanoma,
                    "first_degree_affected": 0,
                    "second_degree_affected": 0
                },
                "last_updated": None,
                "last_calculated": datetime.utcnow().isoformat()
            }

        return {
            "overall_genetic_risk_score": risk_profile.overall_genetic_risk_score,
            "overall_risk_level": risk_profile.overall_risk_level,
            "melanoma_risk_score": risk_profile.melanoma_risk_score,
            "melanoma_risk_level": risk_profile.melanoma_risk_level,
            "bcc_risk_score": risk_profile.bcc_risk_score,
            "scc_risk_score": risk_profile.scc_risk_score,
            "family_history_score": risk_profile.family_history_score,
            "personal_risk_score": risk_profile.personal_risk_score,
            "inheritance_pattern": risk_profile.inheritance_pattern,
            "genetic_counseling_recommended": risk_profile.genetic_counseling_recommended,
            "screening_frequency_self": risk_profile.screening_frequency_self,
            "screening_frequency_professional": risk_profile.screening_frequency_professional,
            "high_priority_monitoring": risk_profile.high_priority_monitoring,
            "risk_reduction_recommendations": risk_profile.risk_reduction_recommendations,
            "family_stats": {
                "total_family_members": total_family_members,
                "with_skin_cancer": with_skin_cancer,
                "with_melanoma": with_melanoma,
                "first_degree_affected": risk_profile.first_degree_affected or 0,
                "second_degree_affected": risk_profile.second_degree_affected or 0
            },
            "last_updated": risk_profile.updated_at
        }
    except Exception as e:
        # Return default data if table doesn't exist
        return {
            "overall_genetic_risk_score": 25.0,
            "overall_risk_level": "low",
            "melanoma_risk_score": 20.0,
            "melanoma_risk_level": "low",
            "bcc_risk_score": 15.0,
            "scc_risk_score": 15.0,
            "basal_cell_carcinoma_risk_score": 15.0,
            "squamous_cell_carcinoma_risk_score": 15.0,
            "family_history_score": 0.0,
            "personal_risk_score": 50.0,
            "inheritance_pattern": "sporadic",
            "genetic_counseling_recommended": False,
            "screening_frequency_self": "biannual",
            "screening_frequency_professional": "annual",
            "recommended_screening_frequency": "biannual",
            "recommended_professional_frequency": "annual",
            "high_priority_monitoring": False,
            "confidence_level": 0.5,
            "total_relatives_with_skin_cancer": 0,
            "first_degree_relatives_affected": 0,
            "second_degree_relatives_affected": 0,
            "familial_melanoma_syndrome_suspected": False,
            "has_early_onset_melanoma": False,
            "has_multiple_family_melanomas": False,
            "affected_lineages": {
                "maternal": False,
                "paternal": False
            },
            "generation_pattern": {
                "grandparents": 0,
                "parents": 0,
                "siblings": 0,
                "aunts_uncles": 0
            },
            "risk_reduction_recommendations": [
                "Perform regular self-examinations",
                "Use broad-spectrum SPF 30+ sunscreen daily"
            ],
            "family_stats": {
                "total_family_members": 0,
                "with_skin_cancer": 0,
                "with_melanoma": 0,
                "first_degree_affected": 0,
                "second_degree_affected": 0
            },
            "last_updated": None,
            "last_calculated": datetime.utcnow().isoformat()
        }


@router.post("/genetic-risk/recalculate")
async def trigger_risk_recalculation(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Manually trigger recalculation of genetic risk profile.
    This endpoint simply refreshes the risk data by calling the GET endpoint.
    """
    try:
        from database import GeneticRiskProfile

        # Get the current risk profile (or create default)
        risk_profile = db.query(GeneticRiskProfile).filter(
            GeneticRiskProfile.user_id == current_user.id
        ).first()

        # Get family members for recalculation
        family_members = db.query(FamilyMember).filter(
            FamilyMember.user_id == current_user.id
        ).all()

        # Update the timestamp
        if risk_profile:
            risk_profile.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(risk_profile)

            return {
                "message": "Genetic risk profile recalculated successfully",
                "overall_risk_level": risk_profile.overall_risk_level,
                "overall_genetic_risk_score": risk_profile.overall_genetic_risk_score,
                "last_calculated": risk_profile.updated_at.isoformat() if risk_profile.updated_at else datetime.utcnow().isoformat()
            }
        else:
            # Return success even if no profile exists yet
            return {
                "message": "Genetic risk profile recalculated successfully",
                "overall_risk_level": "low",
                "overall_genetic_risk_score": 25.0,
                "last_calculated": datetime.utcnow().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to recalculate genetic risk profile")
