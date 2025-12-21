"""
Voice-to-Text Clinical Documentation System

Comprehensive voice input system for hands-free clinical documentation:
- Speech-to-text transcription
- Voice commands recognition
- NLP extraction of clinical data (duration, severity, location)
- Clinical terminology recognition and expansion
- Auto-generation of SOAP notes from dictation
- Medical vocabulary understanding
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import io
import base64


class VoiceCommandType(Enum):
    """Types of voice commands"""
    NAVIGATION = "navigation"
    CAPTURE = "capture"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    SYSTEM = "system"


class ClinicalDataType(Enum):
    """Types of clinical data extracted from speech"""
    DURATION = "duration"
    SEVERITY = "severity"
    LOCATION = "location"
    SYMPTOM = "symptom"
    CHANGE = "change"
    TREATMENT = "treatment"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    HISTORY = "history"


@dataclass
class VoiceCommand:
    """Recognized voice command"""
    command_type: VoiceCommandType
    action: str
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str


@dataclass
class ExtractedClinicalData:
    """Clinical data extracted from speech"""
    data_type: ClinicalDataType
    value: Any
    normalized_value: Any
    unit: Optional[str]
    confidence: float
    source_text: str
    position: Tuple[int, int]  # Start, end position in text


@dataclass
class SOAPNote:
    """SOAP note structure"""
    subjective: str
    objective: str
    assessment: str
    plan: str

    # Extracted structured data
    chief_complaint: Optional[str] = None
    history_present_illness: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    location: Optional[str] = None
    associated_symptoms: List[str] = field(default_factory=list)
    aggravating_factors: List[str] = field(default_factory=list)
    alleviating_factors: List[str] = field(default_factory=list)

    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence_score: float = 0.0


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription"""
    text: str
    confidence: float
    language: str
    duration_seconds: float
    words: List[Dict[str, Any]]  # Word-level timing and confidence
    is_final: bool


class ClinicalTerminologyRecognizer:
    """
    Recognizes and expands clinical/medical terminology.
    """

    def __init__(self):
        # Medical abbreviations and their expansions
        self.abbreviations = {
            # Common dermatology abbreviations
            "bcc": "basal cell carcinoma",
            "scc": "squamous cell carcinoma",
            "ak": "actinic keratosis",
            "aks": "actinic keratoses",
            "nm": "nodular melanoma",
            "ssm": "superficial spreading melanoma",
            "lmm": "lentigo maligna melanoma",
            "alm": "acral lentiginous melanoma",
            "dn": "dysplastic nevus",
            "dns": "dysplastic nevi",
            "sk": "seborrheic keratosis",
            "sks": "seborrheic keratoses",
            "df": "dermatofibroma",
            "psk": "porokeratosis",
            "kp": "keratosis pilaris",

            # General medical abbreviations
            "hx": "history",
            "px": "physical examination",
            "dx": "diagnosis",
            "ddx": "differential diagnosis",
            "tx": "treatment",
            "rx": "prescription",
            "sx": "symptoms",
            "pt": "patient",
            "yo": "year old",
            "y/o": "year old",
            "m": "male",
            "f": "female",
            "h/o": "history of",
            "c/o": "complains of",
            "r/o": "rule out",
            "s/p": "status post",
            "w/": "with",
            "w/o": "without",
            "b/l": "bilateral",
            "prn": "as needed",
            "qd": "daily",
            "bid": "twice daily",
            "tid": "three times daily",
            "qid": "four times daily",
            "qhs": "at bedtime",
            "po": "by mouth",
            "top": "topical",
            "ung": "ointment",
            "cr": "cream",
            "lot": "lotion",
            "sol": "solution",

            # Body location abbreviations
            "rle": "right lower extremity",
            "lle": "left lower extremity",
            "rue": "right upper extremity",
            "lue": "left upper extremity",
            "ant": "anterior",
            "post": "posterior",
            "lat": "lateral",
            "med": "medial",
            "prox": "proximal",
            "dist": "distal",

            # Examination findings
            "wdwn": "well developed well nourished",
            "nad": "no acute distress",
            "wnl": "within normal limits",
            "nka": "no known allergies",
            "nkda": "no known drug allergies",
            "aox3": "alert and oriented times three",

            # Skin examination
            "hyper": "hyperpigmented",
            "hypo": "hypopigmented",
            "eryth": "erythematous",
            "macular": "macular",
            "papular": "papular",
            "nodular": "nodular",
            "vesicular": "vesicular",
            "pustular": "pustular",
            "crusted": "crusted",
            "scaly": "scaly",
            "ulcerated": "ulcerated",
        }

        # Medical terms that might be misheard and their corrections
        self.phonetic_corrections = {
            "melanoma": ["melanoma", "mela noma", "mella noma"],
            "carcinoma": ["carcinoma", "carci noma", "carcin oma"],
            "keratosis": ["keratosis", "kera tosis", "keratin osis"],
            "nevus": ["nevus", "nee vus", "nay vus"],
            "nevi": ["nevi", "nee vie", "nay vie"],
            "erythema": ["erythema", "erith ema", "eri thema"],
            "pruritus": ["pruritus", "prur itis", "proo ritus"],
            "papule": ["papule", "pap yule", "pap ule"],
            "macule": ["macule", "mack yule", "mack ule"],
            "vesicle": ["vesicle", "vess ickle", "vesi cle"],
            "pustule": ["pustule", "pus tool", "pust ule"],
            "plaque": ["plaque", "plack", "plak"],
            "nodule": ["nodule", "nod yule", "nodd ule"],
            "dermis": ["dermis", "derm is"],
            "epidermis": ["epidermis", "epi dermis"],
            "subcutaneous": ["subcutaneous", "sub cute aneous"],
            "biopsy": ["biopsy", "buy opsy", "bio psy"],
            "excision": ["excision", "ex cision", "exci sion"],
            "cryotherapy": ["cryotherapy", "cryo therapy"],
            "phototherapy": ["phototherapy", "photo therapy"],
            "dermoscopy": ["dermoscopy", "dermo scopy", "derma scopy"],
            "eczema": ["eczema", "ex ema", "ec zema"],
            "psoriasis": ["psoriasis", "sore eye asis", "so rye asis"],
            "rosacea": ["rosacea", "rose aysha", "rose asia"],
            "urticaria": ["urticaria", "urt icaria", "urti caria"],
            "angioedema": ["angioedema", "angio edema"],
            "dermatitis": ["dermatitis", "derma titis", "dermat itis"],
        }

        # Dermatological conditions vocabulary
        self.conditions = {
            "melanocytic": ["melanoma", "nevus", "nevi", "dysplastic nevus", "atypical mole",
                          "lentigo", "congenital nevus", "spitz nevus", "blue nevus"],
            "epithelial": ["basal cell carcinoma", "squamous cell carcinoma", "actinic keratosis",
                         "seborrheic keratosis", "keratoacanthoma", "bowens disease"],
            "inflammatory": ["eczema", "dermatitis", "psoriasis", "rosacea", "acne",
                           "lichen planus", "urticaria", "contact dermatitis"],
            "infectious": ["herpes", "shingles", "impetigo", "cellulitis", "tinea",
                         "ringworm", "molluscum", "warts", "verruca"],
            "vascular": ["hemangioma", "cherry angioma", "spider angioma",
                        "pyogenic granuloma", "venous lake"],
        }

        # Body locations for dermatology
        self.body_locations = {
            "head_face": ["face", "forehead", "temple", "cheek", "nose", "ear", "earlobe",
                        "scalp", "chin", "jaw", "lip", "eyelid", "eyebrow"],
            "neck": ["neck", "anterior neck", "posterior neck", "lateral neck"],
            "trunk": ["chest", "breast", "abdomen", "back", "upper back", "lower back",
                     "flank", "side", "umbilicus", "navel"],
            "upper_extremity": ["arm", "upper arm", "forearm", "elbow", "wrist",
                               "hand", "palm", "finger", "thumb", "nail", "shoulder", "axilla"],
            "lower_extremity": ["leg", "thigh", "knee", "shin", "calf", "ankle",
                               "foot", "sole", "toe", "toenail", "heel", "groin"],
            "genital": ["genital", "perineal", "perianal", "buttock", "gluteal"],
        }

    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations in text"""
        words = text.split()
        expanded = []

        for word in words:
            word_lower = word.lower().strip('.,;:!?')
            if word_lower in self.abbreviations:
                expanded.append(self.abbreviations[word_lower])
            else:
                expanded.append(word)

        return ' '.join(expanded)

    def correct_medical_terms(self, text: str) -> str:
        """Correct commonly misheard medical terms"""
        corrected = text.lower()

        for correct_term, variations in self.phonetic_corrections.items():
            for variation in variations[1:]:  # Skip the first (correct) one
                corrected = corrected.replace(variation.lower(), correct_term)

        return corrected

    def identify_medical_terms(self, text: str) -> List[Dict[str, Any]]:
        """Identify and tag medical terms in text"""
        identified = []
        text_lower = text.lower()

        # Check conditions
        for category, terms in self.conditions.items():
            for term in terms:
                if term in text_lower:
                    start = text_lower.find(term)
                    identified.append({
                        "term": term,
                        "category": category,
                        "type": "condition",
                        "position": (start, start + len(term))
                    })

        # Check body locations
        for region, locations in self.body_locations.items():
            for location in locations:
                if location in text_lower:
                    start = text_lower.find(location)
                    identified.append({
                        "term": location,
                        "category": region,
                        "type": "body_location",
                        "position": (start, start + len(location))
                    })

        return identified

    def normalize_text(self, text: str) -> str:
        """Normalize text with medical terminology corrections"""
        # First correct phonetic errors
        text = self.correct_medical_terms(text)
        # Then expand abbreviations
        text = self.expand_abbreviations(text)
        return text


class ClinicalNLPExtractor:
    """
    Extracts clinical data from natural language text.
    Pulls out duration, severity, location, symptoms, etc.
    """

    def __init__(self):
        self.terminology = ClinicalTerminologyRecognizer()

        # Duration patterns
        self.duration_patterns = [
            # Days
            (r'(\d+)\s*days?(?:\s+ago)?', 'days'),
            (r'for\s+(\d+)\s*days?', 'days'),
            (r'past\s+(\d+)\s*days?', 'days'),
            (r'since\s+(\d+)\s*days?', 'days'),
            # Weeks
            (r'(\d+)\s*weeks?(?:\s+ago)?', 'weeks'),
            (r'for\s+(\d+)\s*weeks?', 'weeks'),
            (r'past\s+(\d+)\s*weeks?', 'weeks'),
            (r'couple\s+(?:of\s+)?weeks?', 'weeks', 2),
            (r'few\s+weeks?', 'weeks', 3),
            # Months
            (r'(\d+)\s*months?(?:\s+ago)?', 'months'),
            (r'for\s+(\d+)\s*months?', 'months'),
            (r'past\s+(\d+)\s*months?', 'months'),
            (r'couple\s+(?:of\s+)?months?', 'months', 2),
            (r'few\s+months?', 'months', 3),
            (r'several\s+months?', 'months', 4),
            # Years
            (r'(\d+)\s*years?(?:\s+ago)?', 'years'),
            (r'for\s+(\d+)\s*years?', 'years'),
            (r'past\s+(\d+)\s*years?', 'years'),
            # Qualitative
            (r'recently', 'days', 7),
            (r'just\s+(?:started|noticed|appeared)', 'days', 3),
            (r'long\s+time', 'months', 6),
            (r'always\s+had', 'years', 5),
            (r'since\s+(?:birth|childhood)', 'years', 20),
        ]

        # Severity patterns
        self.severity_patterns = {
            'mild': [
                r'mild(?:ly)?', r'slight(?:ly)?', r'minor', r'little\s+bit',
                r'not\s+(?:too\s+)?bad', r'barely', r'faint'
            ],
            'moderate': [
                r'moderate(?:ly)?', r'somewhat', r'fairly', r'pretty',
                r'noticeable', r'bothersome', r'uncomfortable'
            ],
            'severe': [
                r'severe(?:ly)?', r'intense(?:ly)?', r'extreme(?:ly)?',
                r'very\s+(?:bad|painful|itchy)', r'unbearable', r'terrible',
                r'excruciating', r'worst', r'significant(?:ly)?'
            ]
        }

        # Pain scale patterns
        self.pain_scale_patterns = [
            (r'(\d+)\s*(?:out\s+of|\/)\s*10', lambda m: int(m.group(1))),
            (r'pain\s+(?:level\s+)?(?:is\s+)?(\d+)', lambda m: int(m.group(1))),
        ]

        # Symptom patterns
        self.symptom_keywords = {
            'itching': ['itch', 'itchy', 'itching', 'pruritus', 'pruritic'],
            'pain': ['pain', 'painful', 'hurt', 'hurts', 'hurting', 'sore', 'tender', 'ache', 'aching'],
            'burning': ['burn', 'burning', 'burns'],
            'bleeding': ['bleed', 'bleeding', 'bleeds', 'bled', 'blood', 'bloody'],
            'discharge': ['discharge', 'draining', 'oozing', 'weeping', 'pus'],
            'swelling': ['swollen', 'swelling', 'swelled', 'puffy', 'edema'],
            'redness': ['red', 'redness', 'erythema', 'erythematous', 'inflamed'],
            'scaling': ['scale', 'scaly', 'scaling', 'flaky', 'flaking', 'peeling'],
            'crusting': ['crust', 'crusted', 'crusting', 'scab', 'scabbing'],
            'ulceration': ['ulcer', 'ulcerated', 'ulceration', 'open sore'],
            'numbness': ['numb', 'numbness', 'tingling', 'pins and needles'],
        }

        # Change patterns
        self.change_patterns = {
            'growing': ['growing', 'getting bigger', 'enlarging', 'increasing in size', 'expanded'],
            'spreading': ['spreading', 'spread', 'expanding', 'getting more'],
            'changing_color': ['changing color', 'darker', 'lighter', 'color change', 'discolored'],
            'changing_shape': ['changing shape', 'irregular', 'uneven', 'asymmetric'],
            'new': ['new', 'just appeared', 'recently noticed', 'developed'],
            'stable': ['same', 'unchanged', 'stable', 'not changed', 'no change'],
            'improving': ['improving', 'getting better', 'healing', 'resolving', 'fading'],
            'worsening': ['worse', 'worsening', 'getting worse', 'deteriorating'],
        }

    def extract_all(self, text: str) -> List[ExtractedClinicalData]:
        """Extract all clinical data from text"""
        extracted = []

        # Normalize text first
        normalized = self.terminology.normalize_text(text)

        # Extract duration
        duration = self.extract_duration(normalized)
        if duration:
            extracted.append(duration)

        # Extract severity
        severity = self.extract_severity(normalized)
        if severity:
            extracted.append(severity)

        # Extract locations
        locations = self.extract_locations(normalized)
        extracted.extend(locations)

        # Extract symptoms
        symptoms = self.extract_symptoms(normalized)
        extracted.extend(symptoms)

        # Extract changes
        changes = self.extract_changes(normalized)
        extracted.extend(changes)

        return extracted

    def extract_duration(self, text: str) -> Optional[ExtractedClinicalData]:
        """Extract duration information"""
        text_lower = text.lower()

        for pattern_tuple in self.duration_patterns:
            if len(pattern_tuple) == 2:
                pattern, unit = pattern_tuple
                default_value = None
            else:
                pattern, unit, default_value = pattern_tuple

            match = re.search(pattern, text_lower)
            if match:
                if default_value is not None:
                    value = default_value
                else:
                    value = int(match.group(1))

                return ExtractedClinicalData(
                    data_type=ClinicalDataType.DURATION,
                    value=f"{value} {unit}",
                    normalized_value=self._normalize_duration(value, unit),
                    unit=unit,
                    confidence=0.85,
                    source_text=match.group(0),
                    position=(match.start(), match.end())
                )

        return None

    def _normalize_duration(self, value: int, unit: str) -> int:
        """Normalize duration to days"""
        multipliers = {
            'days': 1,
            'weeks': 7,
            'months': 30,
            'years': 365
        }
        return value * multipliers.get(unit, 1)

    def extract_severity(self, text: str) -> Optional[ExtractedClinicalData]:
        """Extract severity information"""
        text_lower = text.lower()

        # Check pain scale first (most specific)
        for pattern, extractor in self.pain_scale_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = extractor(match)
                if value <= 3:
                    severity = 'mild'
                elif value <= 6:
                    severity = 'moderate'
                else:
                    severity = 'severe'

                return ExtractedClinicalData(
                    data_type=ClinicalDataType.SEVERITY,
                    value=f"{value}/10",
                    normalized_value=severity,
                    unit="pain_scale",
                    confidence=0.95,
                    source_text=match.group(0),
                    position=(match.start(), match.end())
                )

        # Check severity keywords
        for severity, patterns in self.severity_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return ExtractedClinicalData(
                        data_type=ClinicalDataType.SEVERITY,
                        value=severity,
                        normalized_value=severity,
                        unit=None,
                        confidence=0.8,
                        source_text=match.group(0),
                        position=(match.start(), match.end())
                    )

        return None

    def extract_locations(self, text: str) -> List[ExtractedClinicalData]:
        """Extract body locations"""
        locations = []
        text_lower = text.lower()

        for region, location_terms in self.terminology.body_locations.items():
            for location in location_terms:
                if location in text_lower:
                    start = text_lower.find(location)
                    locations.append(ExtractedClinicalData(
                        data_type=ClinicalDataType.LOCATION,
                        value=location,
                        normalized_value=f"{region}:{location}",
                        unit=None,
                        confidence=0.9,
                        source_text=location,
                        position=(start, start + len(location))
                    ))

        return locations

    def extract_symptoms(self, text: str) -> List[ExtractedClinicalData]:
        """Extract symptoms"""
        symptoms = []
        text_lower = text.lower()

        for symptom, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    start = text_lower.find(keyword)
                    symptoms.append(ExtractedClinicalData(
                        data_type=ClinicalDataType.SYMPTOM,
                        value=keyword,
                        normalized_value=symptom,
                        unit=None,
                        confidence=0.85,
                        source_text=keyword,
                        position=(start, start + len(keyword))
                    ))
                    break  # Only add each symptom once

        return symptoms

    def extract_changes(self, text: str) -> List[ExtractedClinicalData]:
        """Extract change descriptions"""
        changes = []
        text_lower = text.lower()

        for change_type, patterns in self.change_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    start = text_lower.find(pattern)
                    changes.append(ExtractedClinicalData(
                        data_type=ClinicalDataType.CHANGE,
                        value=pattern,
                        normalized_value=change_type,
                        unit=None,
                        confidence=0.8,
                        source_text=pattern,
                        position=(start, start + len(pattern))
                    ))
                    break

        return changes


class SOAPNoteGenerator:
    """
    Generates SOAP notes from dictation.
    """

    def __init__(self):
        self.nlp_extractor = ClinicalNLPExtractor()
        self.terminology = ClinicalTerminologyRecognizer()

        # Section indicators in dictation
        self.section_markers = {
            'subjective': [
                'subjective', 'history', 'chief complaint', 'patient reports',
                'patient states', 'the patient', 'complains of', 'presents with'
            ],
            'objective': [
                'objective', 'physical exam', 'examination', 'on exam',
                'inspection reveals', 'skin exam', 'findings'
            ],
            'assessment': [
                'assessment', 'diagnosis', 'impression', 'differential',
                'ddx', 'dx', 'consistent with', 'suggestive of'
            ],
            'plan': [
                'plan', 'treatment', 'recommend', 'prescribe', 'follow up',
                'return', 'refer', 'biopsy', 'order'
            ]
        }

    def generate_from_dictation(self, dictation: str) -> SOAPNote:
        """Generate SOAP note from free-form dictation"""
        # Normalize the dictation
        normalized = self.terminology.normalize_text(dictation)

        # Extract clinical data
        extracted_data = self.nlp_extractor.extract_all(normalized)

        # Parse sections
        sections = self._parse_sections(normalized)

        # Generate each section
        subjective = self._generate_subjective(
            sections.get('subjective', normalized),
            extracted_data
        )

        objective = self._generate_objective(
            sections.get('objective', ''),
            extracted_data
        )

        assessment = self._generate_assessment(
            sections.get('assessment', ''),
            extracted_data
        )

        plan = self._generate_plan(
            sections.get('plan', ''),
            extracted_data
        )

        # Extract structured data
        duration = self._get_extracted_value(extracted_data, ClinicalDataType.DURATION)
        severity = self._get_extracted_value(extracted_data, ClinicalDataType.SEVERITY)
        locations = self._get_all_extracted_values(extracted_data, ClinicalDataType.LOCATION)
        symptoms = self._get_all_extracted_values(extracted_data, ClinicalDataType.SYMPTOM)

        # Calculate confidence
        confidence = self._calculate_confidence(extracted_data, sections)

        return SOAPNote(
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
            chief_complaint=self._extract_chief_complaint(normalized),
            history_present_illness=subjective,
            duration=duration,
            severity=severity,
            location=locations[0] if locations else None,
            associated_symptoms=symptoms,
            confidence_score=confidence
        )

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Parse SOAP sections from dictation"""
        sections = {}
        text_lower = text.lower()

        # Find section boundaries
        boundaries = []
        for section, markers in self.section_markers.items():
            for marker in markers:
                idx = text_lower.find(marker)
                if idx != -1:
                    boundaries.append((idx, section))
                    break

        # Sort by position
        boundaries.sort(key=lambda x: x[0])

        # Extract sections
        for i, (start, section) in enumerate(boundaries):
            if i + 1 < len(boundaries):
                end = boundaries[i + 1][0]
            else:
                end = len(text)
            sections[section] = text[start:end].strip()

        return sections

    def _generate_subjective(self, text: str, extracted: List[ExtractedClinicalData]) -> str:
        """Generate subjective section"""
        if not text:
            return "Patient presents with skin concern."

        # Clean up the text
        text = self._clean_section_text(text, ['subjective', 'history'])

        # Add structured elements if not present
        duration = self._get_extracted_value(extracted, ClinicalDataType.DURATION)
        if duration and duration not in text.lower():
            text += f" Duration: {duration}."

        symptoms = self._get_all_extracted_values(extracted, ClinicalDataType.SYMPTOM)
        symptom_str = ', '.join(symptoms) if symptoms else None
        if symptom_str and not any(s in text.lower() for s in symptoms):
            text += f" Associated symptoms: {symptom_str}."

        return text.strip()

    def _generate_objective(self, text: str, extracted: List[ExtractedClinicalData]) -> str:
        """Generate objective section"""
        if not text:
            locations = self._get_all_extracted_values(extracted, ClinicalDataType.LOCATION)
            if locations:
                return f"Skin examination reveals lesion(s) on {', '.join(locations)}."
            return "Skin examination performed."

        # Clean up the text
        text = self._clean_section_text(text, ['objective', 'physical exam', 'examination'])

        return text.strip()

    def _generate_assessment(self, text: str, extracted: List[ExtractedClinicalData]) -> str:
        """Generate assessment section"""
        if not text:
            return "Skin lesion - further evaluation recommended."

        # Clean up the text
        text = self._clean_section_text(text, ['assessment', 'diagnosis', 'impression'])

        return text.strip()

    def _generate_plan(self, text: str, extracted: List[ExtractedClinicalData]) -> str:
        """Generate plan section"""
        if not text:
            return "1. Clinical monitoring\n2. Patient education provided\n3. Follow up as needed"

        # Clean up the text
        text = self._clean_section_text(text, ['plan', 'treatment', 'recommend'])

        # Format as numbered list if not already
        if not re.search(r'^\d+\.', text):
            sentences = text.split('.')
            numbered = []
            for i, sentence in enumerate(sentences, 1):
                sentence = sentence.strip()
                if sentence:
                    numbered.append(f"{i}. {sentence}")
            if numbered:
                text = '\n'.join(numbered)

        return text.strip()

    def _clean_section_text(self, text: str, markers: List[str]) -> str:
        """Remove section markers from text"""
        for marker in markers:
            # Remove marker at start of text
            pattern = rf'^{re.escape(marker)}[:\s]*'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    def _extract_chief_complaint(self, text: str) -> Optional[str]:
        """Extract chief complaint"""
        patterns = [
            r'(?:chief complaint|c/o|complains of|presents with)[:\s]+([^.]+)',
            r'(?:patient (?:has|reports|states))[:\s]+([^.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Default to first sentence
        first_sentence = text.split('.')[0]
        if len(first_sentence) < 100:
            return first_sentence.strip()

        return None

    def _get_extracted_value(self, extracted: List[ExtractedClinicalData],
                            data_type: ClinicalDataType) -> Optional[str]:
        """Get single extracted value of given type"""
        for item in extracted:
            if item.data_type == data_type:
                return item.value
        return None

    def _get_all_extracted_values(self, extracted: List[ExtractedClinicalData],
                                  data_type: ClinicalDataType) -> List[str]:
        """Get all extracted values of given type"""
        return [item.normalized_value for item in extracted if item.data_type == data_type]

    def _calculate_confidence(self, extracted: List[ExtractedClinicalData],
                             sections: Dict[str, str]) -> float:
        """Calculate overall confidence score"""
        scores = []

        # Section coverage
        section_count = len([s for s in sections.values() if s])
        scores.append(min(section_count / 4.0, 1.0))

        # Extracted data confidence
        if extracted:
            avg_confidence = sum(e.confidence for e in extracted) / len(extracted)
            scores.append(avg_confidence)
        else:
            scores.append(0.5)

        # Data completeness
        has_duration = any(e.data_type == ClinicalDataType.DURATION for e in extracted)
        has_location = any(e.data_type == ClinicalDataType.LOCATION for e in extracted)
        has_symptoms = any(e.data_type == ClinicalDataType.SYMPTOM for e in extracted)
        completeness = sum([has_duration, has_location, has_symptoms]) / 3.0
        scores.append(completeness)

        return sum(scores) / len(scores)


class VoiceCommandProcessor:
    """
    Processes voice commands for hands-free operation.
    """

    def __init__(self):
        # Command patterns
        self.commands = {
            # Navigation commands
            VoiceCommandType.NAVIGATION: {
                'go_home': ['go home', 'home screen', 'main menu', 'back to home'],
                'go_back': ['go back', 'back', 'previous', 'return'],
                'go_history': ['show history', 'view history', 'analysis history', 'past analyses'],
                'go_settings': ['settings', 'preferences', 'options'],
                'next_patient': ['next patient', 'new patient', 'next case'],
            },
            # Capture commands
            VoiceCommandType.CAPTURE: {
                'take_photo': ['take photo', 'take picture', 'capture', 'snap', 'photograph'],
                'retake': ['retake', 'take again', 'try again', 'new photo'],
                'use_gallery': ['gallery', 'photo library', 'choose photo', 'select image'],
                'switch_camera': ['switch camera', 'flip camera', 'front camera', 'back camera'],
            },
            # Analysis commands
            VoiceCommandType.ANALYSIS: {
                'start_analysis': ['start analysis', 'analyze', 'begin analysis', 'run analysis'],
                'analyze_lesion': ['analyze lesion', 'check lesion', 'evaluate lesion'],
                'dermoscopy_analysis': ['dermoscopy', 'dermoscopic analysis'],
                'compare_images': ['compare', 'comparison', 'side by side'],
            },
            # Documentation commands
            VoiceCommandType.DOCUMENTATION: {
                'start_dictation': ['start dictation', 'begin dictation', 'start recording', 'dictate'],
                'stop_dictation': ['stop dictation', 'end dictation', 'stop recording', 'done dictating'],
                'generate_soap': ['generate soap', 'create soap note', 'soap note'],
                'add_notes': ['add notes', 'clinical notes', 'add documentation'],
                'save_notes': ['save notes', 'save documentation', 'save'],
            },
            # System commands
            VoiceCommandType.SYSTEM: {
                'help': ['help', 'what can you do', 'commands', 'voice commands'],
                'cancel': ['cancel', 'never mind', 'stop', 'abort'],
                'repeat': ['repeat', 'say again', 'what did you say'],
                'read_results': ['read results', 'speak results', 'tell me results'],
            }
        }

        # Build reverse lookup
        self._build_command_lookup()

    def _build_command_lookup(self):
        """Build reverse lookup for fast command matching"""
        self.command_lookup = {}
        for cmd_type, commands in self.commands.items():
            for action, phrases in commands.items():
                for phrase in phrases:
                    self.command_lookup[phrase.lower()] = (cmd_type, action)

    def process(self, text: str) -> Optional[VoiceCommand]:
        """Process text to identify voice command"""
        text_lower = text.lower().strip()

        # Direct match
        if text_lower in self.command_lookup:
            cmd_type, action = self.command_lookup[text_lower]
            return VoiceCommand(
                command_type=cmd_type,
                action=action,
                parameters={},
                confidence=1.0,
                raw_text=text
            )

        # Fuzzy match
        best_match = None
        best_score = 0.6  # Minimum threshold

        for phrase, (cmd_type, action) in self.command_lookup.items():
            score = self._fuzzy_match(text_lower, phrase)
            if score > best_score:
                best_score = score
                best_match = (cmd_type, action, phrase)

        if best_match:
            cmd_type, action, matched_phrase = best_match
            return VoiceCommand(
                command_type=cmd_type,
                action=action,
                parameters={'matched_phrase': matched_phrase},
                confidence=best_score,
                raw_text=text
            )

        return None

    def _fuzzy_match(self, text: str, pattern: str) -> float:
        """Simple fuzzy matching score"""
        # Check if pattern is contained in text
        if pattern in text:
            return 0.9

        # Check if all words of pattern are in text
        pattern_words = set(pattern.split())
        text_words = set(text.split())

        if pattern_words.issubset(text_words):
            return 0.8

        # Calculate word overlap
        overlap = len(pattern_words & text_words)
        if overlap > 0:
            return 0.5 + (overlap / len(pattern_words)) * 0.3

        return 0.0

    def get_available_commands(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get list of available commands for help display"""
        result = {}
        for cmd_type, commands in self.commands.items():
            result[cmd_type.value] = [
                {
                    'action': action,
                    'phrases': phrases,
                    'example': phrases[0]
                }
                for action, phrases in commands.items()
            ]
        return result


class VoiceClinicalDocumentationService:
    """
    Main service for voice-to-text clinical documentation.
    """

    def __init__(self):
        self.terminology = ClinicalTerminologyRecognizer()
        self.nlp_extractor = ClinicalNLPExtractor()
        self.soap_generator = SOAPNoteGenerator()
        self.command_processor = VoiceCommandProcessor()

        # Dictation session state (in production, use proper session management)
        self._dictation_sessions: Dict[str, Dict[str, Any]] = {}

    def transcribe_audio(self, audio_data: bytes,
                        language: str = "en-US") -> TranscriptionResult:
        """
        Transcribe audio to text.

        In production, this would integrate with:
        - Google Cloud Speech-to-Text
        - AWS Transcribe Medical
        - Azure Speech Services
        - OpenAI Whisper

        For now, returns a placeholder that expects text input.
        """
        # This is a placeholder - actual implementation would use a speech service
        # The frontend should handle actual speech recognition using expo-speech
        # or react-native-voice, then send the text here for processing

        return TranscriptionResult(
            text="",
            confidence=0.0,
            language=language,
            duration_seconds=0.0,
            words=[],
            is_final=False
        )

    def process_transcription(self, text: str,
                             mode: str = "dictation") -> Dict[str, Any]:
        """
        Process transcribed text.

        Args:
            text: Transcribed text from speech
            mode: Processing mode - 'command', 'dictation', or 'auto'

        Returns:
            Processed result with extracted data or command
        """
        result = {
            "original_text": text,
            "processed_text": self.terminology.normalize_text(text),
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        }

        if mode == "command" or mode == "auto":
            # Try to parse as command
            command = self.command_processor.process(text)
            if command:
                result["command"] = {
                    "type": command.command_type.value,
                    "action": command.action,
                    "parameters": command.parameters,
                    "confidence": command.confidence
                }
                if mode == "auto":
                    result["mode"] = "command"
                    return result

        if mode == "dictation" or mode == "auto":
            # Process as clinical dictation
            extracted = self.nlp_extractor.extract_all(text)
            medical_terms = self.terminology.identify_medical_terms(text)

            result["extracted_data"] = [
                {
                    "type": e.data_type.value,
                    "value": e.value,
                    "normalized": e.normalized_value,
                    "unit": e.unit,
                    "confidence": e.confidence,
                    "source": e.source_text
                }
                for e in extracted
            ]

            result["medical_terms"] = medical_terms
            result["mode"] = "dictation"

        return result

    def generate_soap_note(self, dictation: str) -> Dict[str, Any]:
        """Generate SOAP note from dictation"""
        soap = self.soap_generator.generate_from_dictation(dictation)

        return {
            "subjective": soap.subjective,
            "objective": soap.objective,
            "assessment": soap.assessment,
            "plan": soap.plan,
            "structured_data": {
                "chief_complaint": soap.chief_complaint,
                "duration": soap.duration,
                "severity": soap.severity,
                "location": soap.location,
                "associated_symptoms": soap.associated_symptoms
            },
            "confidence_score": soap.confidence_score,
            "generated_at": soap.generated_at
        }

    def start_dictation_session(self, user_id: str,
                                analysis_id: Optional[int] = None) -> str:
        """Start a new dictation session"""
        import uuid
        session_id = str(uuid.uuid4())

        self._dictation_sessions[session_id] = {
            "user_id": user_id,
            "analysis_id": analysis_id,
            "started_at": datetime.now().isoformat(),
            "segments": [],
            "current_text": ""
        }

        return session_id

    def add_to_dictation(self, session_id: str, text: str) -> Dict[str, Any]:
        """Add transcribed segment to dictation session"""
        if session_id not in self._dictation_sessions:
            raise ValueError("Invalid session ID")

        session = self._dictation_sessions[session_id]

        # Process the segment
        processed = self.process_transcription(text, mode="dictation")

        # Add to session
        session["segments"].append({
            "text": text,
            "processed": processed,
            "timestamp": datetime.now().isoformat()
        })

        # Update current full text
        session["current_text"] += " " + text
        session["current_text"] = session["current_text"].strip()

        return {
            "session_id": session_id,
            "segment_count": len(session["segments"]),
            "current_length": len(session["current_text"]),
            "latest_extraction": processed.get("extracted_data", [])
        }

    def end_dictation_session(self, session_id: str,
                              generate_soap: bool = True) -> Dict[str, Any]:
        """End dictation session and optionally generate SOAP note"""
        if session_id not in self._dictation_sessions:
            raise ValueError("Invalid session ID")

        session = self._dictation_sessions[session_id]

        result = {
            "session_id": session_id,
            "full_text": session["current_text"],
            "segment_count": len(session["segments"]),
            "started_at": session["started_at"],
            "ended_at": datetime.now().isoformat()
        }

        if generate_soap:
            soap = self.generate_soap_note(session["current_text"])
            result["soap_note"] = soap

        # Clean up session
        del self._dictation_sessions[session_id]

        return result

    def get_voice_commands(self) -> Dict[str, Any]:
        """Get available voice commands"""
        return self.command_processor.get_available_commands()


# Global service instance
_service = None

def get_voice_documentation_service() -> VoiceClinicalDocumentationService:
    """Get or create global service instance"""
    global _service
    if _service is None:
        _service = VoiceClinicalDocumentationService()
    return _service
