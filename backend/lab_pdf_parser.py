"""
Lab Results PDF Parser

Parses lab result PDFs to extract values and populate the lab results form.
Uses PyMuPDF for text extraction and regex patterns for value matching.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import io


@dataclass
class LabValue:
    """Represents a parsed lab value."""
    name: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    is_abnormal: Optional[bool] = None


# Common lab test patterns with regex for value extraction
# Format: (display_name, field_name, patterns, unit_patterns)
LAB_PATTERNS = [
    # ============================================
    # COMPLETE BLOOD COUNT (CBC)
    # ============================================
    ("WBC", "wbc", [
        r"(?:WHITE\s*BLOOD\s*CELL\s*COUNT|WBC|White\s*Blood\s*Cell[s]?)\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:WBC|White\s*Blood\s*Cell[s]?)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"(?:Thousand|K)/uL"),

    ("RBC", "rbc", [
        r"(?:RED\s*BLOOD\s*CELL\s*COUNT|RBC|Red\s*Blood\s*Cell[s]?)\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:RBC|Red\s*Blood\s*Cell[s]?)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"(?:Million|M)/uL"),

    ("Hemoglobin", "hemoglobin", [
        r"HEMOGLOBIN\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Hemoglobin|Hgb|Hb)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"g/dL"),

    ("Hematocrit", "hematocrit", [
        r"HEMATOCRIT\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Hematocrit|Hct|HCT)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"%"),

    ("Platelets", "platelets", [
        r"PLATELET\s*COUNT\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Platelet[s]?|PLT)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"(?:Thousand|K)/uL"),

    ("MCV", "mcv", [
        r"MCV\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:MCV|Mean\s*Corpuscular\s*Volume)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"fL"),

    ("MCH", "mch", [
        r"MCH\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:MCH|Mean\s*Corpuscular\s*Hemoglobin)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"pg"),

    ("MCHC", "mchc", [
        r"MCHC\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:MCHC)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"g/dL"),

    ("RDW", "rdw", [
        r"RDW\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:RDW|Red\s*Cell\s*Distribution)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"%"),

    ("MPV", "mpv", [
        r"MPV\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:MPV|Mean\s*Platelet\s*Volume)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"fL"),

    # WBC Differential (%)
    ("Neutrophils", "neutrophils", [
        r"NEUTROPHILS\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Neutrophil[s]?|NEUT)\s*[:\s]+([0-9]+\.?[0-9]*)\s*%",
    ], r"%"),

    ("Lymphocytes", "lymphocytes", [
        r"LYMPHOCYTES\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Lymphocyte[s]?|LYMPH)\s*[:\s]+([0-9]+\.?[0-9]*)\s*%",
    ], r"%"),

    ("Monocytes", "monocytes", [
        r"MONOCYTES\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Monocyte[s]?|MONO)\s*[:\s]+([0-9]+\.?[0-9]*)\s*%",
    ], r"%"),

    ("Eosinophils", "eosinophils", [
        r"EOSINOPHILS\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Eosinophil[s]?|EOS)\s*[:\s]+([0-9]+\.?[0-9]*)\s*%",
    ], r"%"),

    ("Basophils", "basophils", [
        r"BASOPHILS\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Basophil[s]?|BASO)\s*[:\s]+([0-9]+\.?[0-9]*)\s*%",
    ], r"%"),

    # Absolute WBC Counts
    ("Abs Neutrophils", "neutrophils_abs", [
        r"ABSOLUTE\s*NEUTROPHILS\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Absolute\s*Neutrophil[s]?|ANC)\s*[:\s]+([0-9]+)",
    ], r"cells/uL"),

    ("Abs Lymphocytes", "lymphocytes_abs", [
        r"ABSOLUTE\s*LYMPHOCYTES\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Absolute\s*Lymphocyte[s]?|ALC)\s*[:\s]+([0-9]+)",
    ], r"cells/uL"),

    ("Abs Monocytes", "monocytes_abs", [
        r"ABSOLUTE\s*MONOCYTES\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Absolute\s*Monocyte[s]?|AMC)\s*[:\s]+([0-9]+)",
    ], r"cells/uL"),

    ("Abs Eosinophils", "eosinophils_abs", [
        r"ABSOLUTE\s*EOSINOPHILS\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Absolute\s*Eosinophil[s]?|AEC)\s*[:\s]+([0-9]+)",
    ], r"cells/uL"),

    ("Abs Basophils", "basophils_abs", [
        r"ABSOLUTE\s*BASOPHILS\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Absolute\s*Basophil[s]?)\s*[:\s]+([0-9]+)",
    ], r"cells/uL"),

    # ============================================
    # COMPREHENSIVE METABOLIC PANEL
    # ============================================
    ("Glucose", "glucose_fasting", [
        r"GLUCOSE\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Glucose|Blood\s*Sugar|Fasting\s*Glucose)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    ("HbA1c", "hba1c", [
        r"HEMOGLOBIN\s*A1c\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:HbA1c|Hemoglobin\s*A1c|A1C)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"%"),

    ("eAG", "eag", [
        r"eAG\s*\(mg/dL\)\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:eAG|Estimated\s*Average\s*Glucose)\s*[:\s]+([0-9]+)",
    ], r"mg/dL"),

    ("BUN", "bun", [
        r"UREA\s*NITROGEN\s*\(BUN\)\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:BUN|Blood\s*Urea\s*Nitrogen|Urea\s*Nitrogen)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    ("Creatinine", "creatinine", [
        r"CREATININE\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Creatinine|CREAT)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    ("BUN/Creatinine Ratio", "bun_creatinine_ratio", [
        r"BUN/CREATININE\s*RATIO\s+(?:NOT\s*APPLICABLE|([0-9]+\.?[0-9]*))\s+(?:NORMAL|HIGH|LOW)",
    ], r""),

    ("eGFR", "egfr", [
        r"eGFR\s*NON-?AFR\.?\s*AMERICAN\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:eGFR|GFR|Estimated\s*GFR)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mL/min/1.73m2"),

    ("eGFR African American", "egfr_african_american", [
        r"eGFR\s*AFRICAN\s*AMERICAN\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
    ], r"mL/min/1.73m2"),

    ("Sodium", "sodium", [
        r"SODIUM\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Sodium|Na)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mmol/L"),

    ("Potassium", "potassium", [
        r"POTASSIUM\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Potassium|K)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mmol/L"),

    ("Chloride", "chloride", [
        r"CHLORIDE\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Chloride|Cl)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mmol/L"),

    ("CO2", "co2", [
        r"CARBON\s*DIOXIDE\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:CO2|Carbon\s*Dioxide|Bicarbonate)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mmol/L"),

    ("Calcium", "calcium", [
        r"CALCIUM\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Calcium|Ca)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    ("Magnesium", "magnesium", [
        r"MAGNESIUM(?:,\s*RBC)?\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Magnesium|Mg)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    # ============================================
    # LIVER FUNCTION
    # ============================================
    ("ALT", "alt", [
        r"ALT\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:ALT|SGPT|Alanine\s*(?:Amino)?transferase)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"U/L"),

    ("AST", "ast", [
        r"AST\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:AST|SGOT|Aspartate\s*(?:Amino)?transferase)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"U/L"),

    ("Alkaline Phosphatase", "alp", [
        r"ALKALINE\s*PHOSPHATASE\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Alkaline\s*Phosphatase|ALP|Alk\s*Phos)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"U/L"),

    ("Bilirubin Total", "bilirubin_total", [
        r"BILIRUBIN,?\s*TOTAL\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Bilirubin\s*(?:Total)?|Total\s*Bilirubin)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    ("Total Protein", "total_protein", [
        r"PROTEIN,?\s*TOTAL\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Total\s*Protein|Protein\s*Total)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"g/dL"),

    ("Albumin", "albumin", [
        r"ALBUMIN\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Albumin)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"g/dL"),

    ("Globulin", "globulin", [
        r"GLOBULIN\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Globulin)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"g/dL"),

    ("A/G Ratio", "albumin_globulin_ratio", [
        r"ALBUMIN/GLOBULIN\s*RATIO\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:A/G\s*Ratio|Albumin/Globulin)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r""),

    # ============================================
    # LIPID PANEL
    # ============================================
    ("Cholesterol Total", "cholesterol_total", [
        r"CHOLESTEROL,?\s*TOTAL\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Total\s*Cholesterol|Cholesterol\s*Total|Cholesterol)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    ("HDL", "hdl", [
        r"HDL\s*CHOLESTEROL\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:HDL|HDL-?C|HDL\s*Cholesterol)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    ("LDL", "ldl", [
        r"LDL-?CHOLESTEROL\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:LDL|LDL-?C|LDL\s*Cholesterol)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    ("Triglycerides", "triglycerides", [
        r"TRIGLYCERIDES\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Triglycerides|TG|Trigs)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    ("Chol/HDL Ratio", "chol_hdl_ratio", [
        r"CHOL/HDLC\s*RATIO\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Chol/HDL|Cholesterol/HDL)\s*Ratio\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r""),

    ("Non-HDL Cholesterol", "non_hdl_cholesterol", [
        r"NON\s*HDL\s*CHOLESTEROL\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Non-?HDL\s*Cholesterol)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/dL"),

    # ============================================
    # THYROID PANEL
    # ============================================
    ("TSH", "tsh", [
        r"TSH\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:TSH|Thyroid\s*Stimulating\s*Hormone)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mIU/L"),

    ("T3 Uptake", "t3_uptake", [
        r"T3\s*UPTAKE\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:T3\s*Uptake)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"%"),

    ("T4 Total", "t4_total", [
        r"T4\s*\(THYROXINE\),?\s*TOTAL\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:T4\s*Total|Total\s*T4|Thyroxine)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mcg/dL"),

    ("Free T4 Index", "free_t4_index", [
        r"FREE\s*T4\s*INDEX\s*\(T7\)\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Free\s*T4\s*Index|T7|FT4I)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r""),

    ("Free T4", "t4_free", [
        r"(?:FREE\s*T4|FT4)\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Free\s*T4|FT4|T4\s*Free)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"ng/dL"),

    # ============================================
    # IRON STUDIES
    # ============================================
    ("Iron", "iron", [
        r"(?:IRON|SERUM\s*IRON)\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Iron|Serum\s*Iron|Fe)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"µg/dL"),

    ("Ferritin", "ferritin", [
        r"FERRITIN\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Ferritin)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"ng/mL"),

    ("TIBC", "tibc", [
        r"TIBC\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:TIBC|Total\s*Iron\s*Binding\s*Capacity)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"µg/dL"),

    # ============================================
    # VITAMINS
    # ============================================
    ("Vitamin D", "vitamin_d", [
        r"VITAMIN\s*D,?25-?OH,?TOTAL,?IA\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Vitamin\s*D|25-?(?:OH)?-?Vitamin\s*D|Vit\s*D)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"ng/mL"),

    ("Vitamin B12", "vitamin_b12", [
        r"VITAMIN\s*B12\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Vitamin\s*B12|B12|Cobalamin)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"pg/mL"),

    ("Folate", "folate", [
        r"FOLATE\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Folate|Folic\s*Acid)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"ng/mL"),

    # ============================================
    # INFLAMMATORY MARKERS
    # ============================================
    ("CRP", "crp", [
        r"C-?REACTIVE\s*PROTEIN\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:CRP|C-?Reactive\s*Protein|hs-?CRP)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mg/L"),

    ("ESR", "esr", [
        r"SED\s*RATE\s*BY\s*MODIFIED\s*WESTERGREN\s+([0-9]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:ESR|Sed\s*Rate|Sedimentation\s*Rate)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"mm/hr"),

    # ============================================
    # ALLERGY
    # ============================================
    ("Total IgE", "ige_total", [
        r"(?:TOTAL\s*)?IgE\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Total\s*IgE|IgE\s*Total|IgE)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], r"IU/mL"),
]

# Urinalysis patterns
URINE_PATTERNS = [
    ("Urine Specific Gravity", "urine_specific_gravity", [
        r"SPECIFIC\s*GRAVITY\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Specific\s*Gravity|Sp\.?\s*Gr\.?|SG)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], ""),

    ("Urine pH", "urine_ph", [
        r"(?:URINE\s*)?PH\s+([0-9]+\.?[0-9]*)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:pH|Urine\s*pH)\s*[:\s]+([0-9]+\.?[0-9]*)",
    ], ""),
]

# Qualitative patterns (positive/negative, trace, etc.)
QUALITATIVE_PATTERNS = [
    ("Urine Color", "urine_color", [
        r"COLOR\s+(YELLOW|AMBER|STRAW|PALE|DARK|RED|BROWN|CLOUDY|CLEAR)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Color|Urine\s*Color)\s*[:\s]*(yellow|amber|straw|pale|dark|red|brown|cloudy|clear)",
    ]),

    ("Urine Appearance", "urine_appearance", [
        r"APPEARANCE\s+(CLEAR|CLOUDY|TURBID|HAZY)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Appearance|Clarity)\s*[:\s]*(clear|cloudy|turbid|hazy)",
    ]),

    ("Urine Protein", "urine_protein", [
        r"(?:URINE\s*)?PROTEIN\s+(NEGATIVE|TRACE|1\+|2\+|3\+|4\+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Protein|Urine\s*Protein)\s*[:\s]*(negative|trace|1\+|2\+|3\+|4\+)",
    ]),

    ("Urine Glucose", "urine_glucose", [
        r"(?:URINE\s*)?GLUCOSE\s+(NEGATIVE|TRACE|1\+|2\+|3\+|4\+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Glucose|Urine\s*Glucose)\s*[:\s]*(negative|trace|1\+|2\+|3\+|4\+)",
    ]),

    ("Urine Ketones", "urine_ketones", [
        r"KETONES\s+(NEGATIVE|TRACE|SMALL|MODERATE|LARGE)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Ketones|Urine\s*Ketones)\s*[:\s]*(negative|trace|small|moderate|large)",
    ]),

    ("Urine Blood", "urine_blood", [
        r"(?:OCCULT\s*)?BLOOD\s+(NEGATIVE|TRACE|SMALL|MODERATE|LARGE|1\+|2\+|3\+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Blood|Occult\s*Blood|Urine\s*Blood)\s*[:\s]*(negative|trace|small|moderate|large|1\+|2\+|3\+)",
    ]),

    ("Urine Bilirubin", "urine_bilirubin", [
        r"BILIRUBIN\s+(NEGATIVE|1\+|2\+|3\+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Bilirubin|Urine\s*Bilirubin)\s*[:\s]*(negative|1\+|2\+|3\+)",
    ]),

    ("Urine Nitrite", "urine_nitrite", [
        r"NITRITE\s+(NEGATIVE|POSITIVE)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Nitrite|Urine\s*Nitrite)\s*[:\s]*(negative|positive)",
    ]),

    ("Leukocyte Esterase", "urine_leukocyte_esterase", [
        r"LEUKOCYTE\s*ESTERASE\s+(NEGATIVE|TRACE|SMALL|MODERATE|LARGE|1\+|2\+|3\+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Leukocyte\s*Esterase|LE)\s*[:\s]*(negative|trace|small|moderate|large|1\+|2\+|3\+)",
    ]),

    ("Urine WBC", "urine_wbc", [
        r"(?:URINE\s*)?WBC\s+(NONE\s*SEEN|[0-9\-]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:WBC|Urine\s*WBC)\s*[:\s]*(none\s*seen|[0-9\-]+)",
    ]),

    ("Urine RBC", "urine_rbc", [
        r"(?:URINE\s*)?RBC\s+(NONE\s*SEEN|[0-9\-]+)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:RBC|Urine\s*RBC)\s*[:\s]*(none\s*seen|[0-9\-]+)",
    ]),

    ("Urine Bacteria", "urine_bacteria", [
        r"BACTERIA\s+(NONE\s*SEEN|FEW|MODERATE|MANY)\s+(?:NORMAL|HIGH|LOW)",
        r"(?:Bacteria|Urine\s*Bacteria)\s*[:\s]*(none\s*seen|few|moderate|many)",
    ]),

    ("Squamous Epithelial", "urine_squamous_epithelial", [
        r"SQUAMOUS\s*EPITHELIAL\s*CELLS\s+(NONE\s*SEEN|FEW|[0-9\-]+)\s+(?:NORMAL|HIGH|LOW)",
    ]),

    ("Hyaline Cast", "urine_hyaline_cast", [
        r"HYALINE\s*CAST\s+(NONE\s*SEEN|FEW|[0-9\-]+)\s+(?:NORMAL|HIGH|LOW)",
    ]),

    ("ANA", "ana_positive", [
        r"(?:ANA|Antinuclear\s*Antibod(?:y|ies))\s*[:\s]*(positive|negative|reactive|non-?reactive)",
    ]),

    ("Stool Occult Blood", "stool_occult_blood", [
        r"(?:OCCULT\s*BLOOD|STOOL\s*(?:OCCULT\s*)?BLOOD|FOBT|FIT)\s*[:\s]*(positive|negative)",
    ]),

    ("Stool Parasites", "stool_parasites", [
        r"(?:PARASITES?|OVA\s*(?:AND|&)\s*PARASITES?|O&P)\s*[:\s]*(positive|negative|detected|not\s*detected|none)",
    ]),
]


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        text_content = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            text_content.append(text)

        doc.close()
        return "\n".join(text_content)

    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    """Extract text using OCR for scanned PDFs."""
    try:
        import fitz
        import pytesseract
        from PIL import Image

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_content = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # First try regular text extraction
            text = page.get_text()

            # If no text found, try OCR
            if not text.strip():
                # Render page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")

                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_bytes))

                # Run OCR
                text = pytesseract.image_to_string(img)

            text_content.append(text)

        doc.close()
        return "\n".join(text_content)

    except ImportError:
        # Fall back to regular extraction if pytesseract not available
        return extract_text_from_pdf(pdf_bytes)
    except Exception as e:
        # Fall back to regular extraction on error
        return extract_text_from_pdf(pdf_bytes)


def parse_numeric_value(text: str, patterns: List[str]) -> Optional[float]:
    """Try to extract a numeric value using multiple regex patterns."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                captured = match.group(1)
                if captured is None:
                    continue
                value = float(captured)
                return value
            except (ValueError, IndexError, TypeError):
                continue
    return None


def parse_qualitative_value(text: str, patterns: List[str]) -> Optional[str]:
    """Try to extract a qualitative value using regex patterns."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                captured = match.group(1)
                if captured is not None:
                    return captured.lower()
            except (IndexError, AttributeError):
                continue
    return None


def extract_lab_name(text: str) -> Optional[str]:
    """Try to extract the lab/facility name from the PDF."""
    lab_patterns = [
        r"(Quest\s*Diagnostics)",
        r"(LabCorp|Laboratory\s*Corporation)",
        r"(Mayo\s*(?:Clinic\s*)?Laboratories)",
        r"(ARUP\s*Laboratories)",
        r"(BioReference)",
        r"(Sonic\s*Healthcare)",
        r"(Eurofins)",
        r"Lab(?:oratory)?:\s*([^\n]+)",
        r"Performed\s*(?:at|by):\s*([^\n]+)",
        r"Facility:\s*([^\n]+)",
    ]

    for pattern in lab_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def extract_test_date(text: str) -> Optional[str]:
    """Try to extract the test/collection date from the PDF."""
    date_patterns = [
        # YYYY/MM/DD format (Quest Diagnostics style)
        r"COLLECTED:\s*(\d{4}/\d{2}/\d{2})",
        # MM/DD/YYYY or MM-DD-YYYY
        r"(?:Collection|Collected|Test|Report|Specimen)\s*Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        # YYYY-MM-DD
        r"(?:Collection|Collected|Test|Report|Specimen)\s*Date[:\s]+(\d{4}[/-]\d{1,2}[/-]\d{1,2})",
        # Month DD, YYYY
        r"(?:Collection|Collected|Test|Report)\s*Date[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1).strip()
            # Try to normalize to YYYY-MM-DD
            return normalize_date(date_str)

    return None


def normalize_date(date_str: str) -> str:
    """Normalize various date formats to YYYY-MM-DD."""
    from datetime import datetime

    formats = [
        "%Y/%m/%d", "%Y-%m-%d",
        "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y",
        "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return date_str


def parse_lab_pdf(pdf_bytes: bytes, use_ocr: bool = False) -> Dict[str, Any]:
    """
    Parse a lab results PDF and extract values.

    Args:
        pdf_bytes: The PDF file content as bytes
        use_ocr: Whether to use OCR for scanned PDFs

    Returns:
        Dictionary with extracted lab values and metadata
    """
    # Extract text from PDF
    if use_ocr:
        text = extract_text_with_ocr(pdf_bytes)
    else:
        text = extract_text_from_pdf(pdf_bytes)

    if not text.strip():
        raise Exception("Could not extract text from PDF. Try enabling OCR for scanned documents.")

    result = {
        "extracted_values": {},
        "raw_text_preview": text[:2000],  # First 2000 chars for debugging
        "lab_name": extract_lab_name(text),
        "test_date": extract_test_date(text),
        "parse_confidence": "high",
        "values_found": 0,
        "parsing_notes": [],
    }

    # Extract numeric lab values
    for display_name, field_name, patterns, unit_pattern in LAB_PATTERNS:
        value = parse_numeric_value(text, patterns)
        if value is not None:
            result["extracted_values"][field_name] = value
            result["values_found"] += 1

    # Extract urine numeric values
    for display_name, field_name, patterns, unit_pattern in URINE_PATTERNS:
        value = parse_numeric_value(text, patterns)
        if value is not None:
            result["extracted_values"][field_name] = value
            result["values_found"] += 1

    # Extract qualitative values
    for display_name, field_name, patterns in QUALITATIVE_PATTERNS:
        value = parse_qualitative_value(text, patterns)
        if value is not None:
            # Convert to appropriate format
            if field_name == "ana_positive":
                result["extracted_values"][field_name] = value in ["positive", "reactive"]
            else:
                result["extracted_values"][field_name] = value
            result["values_found"] += 1

    # Determine confidence based on values found
    if result["values_found"] == 0:
        result["parse_confidence"] = "low"
        result["parsing_notes"].append("No lab values could be extracted. The PDF may be a scanned image - try enabling OCR.")
    elif result["values_found"] < 5:
        result["parse_confidence"] = "medium"
        result["parsing_notes"].append("Only a few values were extracted. Some values may need manual entry.")
    else:
        result["parsing_notes"].append(f"Successfully extracted {result['values_found']} lab values.")

    return result


def validate_extracted_values(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate extracted values are within reasonable ranges.
    Returns the values with any obvious errors flagged.
    """
    # Define reasonable ranges for common tests
    reasonable_ranges = {
        "wbc": (1.0, 50.0),
        "rbc": (2.0, 8.0),
        "hemoglobin": (5.0, 25.0),
        "hematocrit": (20.0, 65.0),
        "platelets": (50.0, 1000.0),
        "mcv": (50.0, 150.0),
        "mch": (15.0, 45.0),
        "mchc": (25.0, 40.0),
        "rdw": (8.0, 25.0),
        "mpv": (5.0, 20.0),
        "glucose_fasting": (30.0, 500.0),
        "hba1c": (3.0, 20.0),
        "creatinine": (0.1, 15.0),
        "egfr": (5.0, 150.0),
        "alt": (1.0, 1000.0),
        "ast": (1.0, 1000.0),
        "tsh": (0.01, 100.0),
        "vitamin_d": (1.0, 200.0),
        "vitamin_b12": (50.0, 5000.0),
        "crp": (0.0, 500.0),
        "esr": (0.0, 150.0),
        "cholesterol_total": (50.0, 500.0),
        "hdl": (10.0, 150.0),
        "ldl": (10.0, 400.0),
        "triglycerides": (20.0, 1000.0),
    }

    validated = {}
    warnings = []

    for field, value in extracted.items():
        if isinstance(value, (int, float)) and field in reasonable_ranges:
            min_val, max_val = reasonable_ranges[field]
            if min_val <= value <= max_val:
                validated[field] = value
            else:
                warnings.append(f"{field}: value {value} seems outside reasonable range ({min_val}-{max_val})")
                # Still include it, but flag it
                validated[field] = value
        else:
            validated[field] = value

    return {"values": validated, "warnings": warnings}
