"""
DICOM Metadata Compliance Module

Generates DICOM-compliant metadata for medical photography.
Ensures photos meet clinical documentation standards.
"""

from typing import Dict, Optional
from datetime import datetime
import json


class DICOMMetadataGenerator:
    """
    Generate DICOM-compliant metadata for clinical photographs.

    DICOM (Digital Imaging and Communications in Medicine) is the
    international standard for medical images and related information.
    """

    def __init__(self):
        # DICOM VR (Value Representation) types
        self.VR_TYPES = {
            'PatientID': 'LO',  # Long String
            'StudyDate': 'DA',  # Date
            'StudyTime': 'TM',  # Time
            'Modality': 'CS',   # Code String
            'Manufacturer': 'LO',
            'BodyPartExamined': 'CS',
            'ViewPosition': 'CS',
            'PixelSpacing': 'DS',  # Decimal String
            'PhotometricInterpretation': 'CS',
            'SeriesDescription': 'LO',
            'PatientPosition': 'CS',
        }

    def generate_metadata(
        self,
        patient_id: str,
        body_site: str,
        laterality: Optional[str] = None,
        view_description: str = 'overview',
        pixel_spacing: Optional[float] = None,
        device_info: Optional[Dict] = None,
        lighting_info: Optional[Dict] = None,
        custom_tags: Optional[Dict] = None
    ) -> Dict:
        """
        Generate complete DICOM-compliant metadata.

        Args:
            patient_id: Anonymized patient identifier
            body_site: Anatomical location (e.g., 'left_arm', 'back')
            laterality: Left/Right/Bilateral
            view_description: Type of view (overview/close-up/macro)
            pixel_spacing: Millimeters per pixel
            device_info: Camera/device information
            lighting_info: Lighting conditions
            custom_tags: Additional custom metadata

        Returns:
            Dictionary with DICOM-compliant metadata
        """
        now = datetime.now()

        metadata = {
            # Patient Information (anonymized)
            'PatientID': self._anonymize_patient_id(patient_id),
            'PatientName': 'ANONYMOUS',
            'PatientBirthDate': '',  # Omitted for privacy

            # Study Information
            'StudyDate': now.strftime('%Y%m%d'),
            'StudyTime': now.strftime('%H%M%S'),
            'StudyID': self._generate_study_id(patient_id, now),
            'StudyDescription': f'Dermatology Clinical Photography - {body_site}',

            # Series Information
            'SeriesDate': now.strftime('%Y%m%d'),
            'SeriesTime': now.strftime('%H%M%S'),
            'SeriesNumber': '1',
            'SeriesDescription': f'{view_description} view of {body_site}',

            # Modality (XC = External-camera Photography)
            'Modality': 'XC',
            'ModalityDescription': 'Clinical Photography',

            # Equipment
            'Manufacturer': device_info.get('manufacturer', 'Unknown') if device_info else 'Mobile Device',
            'ManufacturerModelName': device_info.get('model', 'Unknown') if device_info else 'Smartphone',
            'DeviceSerialNumber': device_info.get('serial', 'ANON') if device_info else 'ANON',
            'SoftwareVersions': device_info.get('software', '1.0') if device_info else '1.0',

            # Anatomical Information
            'BodyPartExamined': self._normalize_body_part(body_site),
            'Laterality': self._normalize_laterality(laterality),
            'ViewPosition': self._normalize_view_position(view_description),

            # Image Acquisition
            'AcquisitionDateTime': now.isoformat(),
            'ContentDate': now.strftime('%Y%m%d'),
            'ContentTime': now.strftime('%H%M%S'),

            # Pixel Spacing (for scale calibration)
            'PixelSpacing': [pixel_spacing, pixel_spacing] if pixel_spacing else None,
            'ImagerPixelSpacing': [pixel_spacing, pixel_spacing] if pixel_spacing else None,

            # Image Characteristics
            'PhotometricInterpretation': 'RGB',
            'SamplesPerPixel': 3,
            'BitsAllocated': 8,
            'BitsStored': 8,
            'HighBit': 7,

            # Clinical Context
            'ClinicalTrialSponsorName': 'Dermatology Practice',
            'ClinicalTrialProtocolID': 'DERM-PHOTO-001',
            'ClinicalTrialSubjectID': self._anonymize_patient_id(patient_id),

            # Lighting Information
            'IlluminationTypeCodeSequence': self._encode_lighting_info(lighting_info) if lighting_info else None,

            # Institutional Information
            'InstitutionName': 'Teledermatology Service',
            'InstitutionalDepartmentName': 'Dermatology',

            # Quality Indicators
            'LossyImageCompression': '00',  # Not compressed
            'ImageType': ['ORIGINAL', 'PRIMARY', 'CLINICAL'],

            # Burned-in Annotation
            'BurnedInAnnotation': 'NO',  # No text burned into image

            # Patient Orientation
            'PatientOrientation': self._get_patient_orientation(body_site, view_description),

            # Additional Custom Tags
            'PrivateCreator': 'Teledermatology AI System',
            'CustomMetadata': custom_tags if custom_tags else {},
        }

        # Add timestamp
        metadata['MetadataGeneratedAt'] = now.isoformat()

        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return metadata

    def _anonymize_patient_id(self, patient_id: str) -> str:
        """Anonymize patient ID for DICOM compliance"""
        # Hash or truncate patient ID
        import hashlib
        hashed = hashlib.sha256(patient_id.encode()).hexdigest()
        return f"ANON-{hashed[:8].upper()}"

    def _generate_study_id(self, patient_id: str, timestamp: datetime) -> str:
        """Generate unique study ID"""
        date_str = timestamp.strftime('%Y%m%d')
        return f"STD-{self._anonymize_patient_id(patient_id)[-8:]}-{date_str}"

    def _normalize_body_part(self, body_site: str) -> str:
        """
        Normalize body site to DICOM-compliant body part code.

        Uses SNOMED CT or DICOM-defined codes.
        """
        body_part_map = {
            'head': 'HEAD',
            'face': 'FACE',
            'neck': 'NECK',
            'chest': 'CHEST',
            'back': 'BACK',
            'abdomen': 'ABDOMEN',
            'arm': 'ARM',
            'left_arm': 'ARM',
            'right_arm': 'ARM',
            'hand': 'HAND',
            'left_hand': 'HAND',
            'right_hand': 'HAND',
            'leg': 'LEG',
            'left_leg': 'LEG',
            'right_leg': 'LEG',
            'foot': 'FOOT',
            'left_foot': 'FOOT',
            'right_foot': 'FOOT',
        }

        normalized = body_site.lower().replace('_', ' ').strip()
        for key, value in body_part_map.items():
            if key in normalized:
                return value

        return 'SKIN'  # Default

    def _normalize_laterality(self, laterality: Optional[str]) -> Optional[str]:
        """Normalize laterality to DICOM standard"""
        if not laterality:
            return None

        laterality_map = {
            'left': 'L',
            'right': 'R',
            'bilateral': 'B',
            'unpaired': 'U',
        }

        return laterality_map.get(laterality.lower(), None)

    def _normalize_view_position(self, view_description: str) -> str:
        """Normalize view position to DICOM standard"""
        view_map = {
            'overview': 'AP',  # Anterior-Posterior
            'close-up': 'AP',
            'macro': 'AP',
            'front': 'AP',
            'back': 'PA',  # Posterior-Anterior
            'side': 'LAT',  # Lateral
            'left': 'LLT',  # Left Lateral
            'right': 'RLT',  # Right Lateral
        }

        for key, value in view_map.items():
            if key in view_description.lower():
                return value

        return 'AP'  # Default

    def _encode_lighting_info(self, lighting_info: Dict) -> Dict:
        """Encode lighting information in DICOM format"""
        return {
            'CodeValue': 'LIGHTING',
            'CodingSchemeDesignator': 'LOCAL',
            'CodeMeaning': 'Lighting Conditions',
            'LightingType': lighting_info.get('type', 'ambient'),
            'LightingQualityScore': lighting_info.get('quality_score', 0),
            'HasGlare': lighting_info.get('has_glare', False),
            'HasShadows': lighting_info.get('has_shadows', False),
        }

    def _get_patient_orientation(self, body_site: str, view_description: str) -> Optional[str]:
        """Get patient orientation string"""
        # DICOM patient orientation (row direction, column direction)
        # Common values: ['L','F'], ['R','F'], ['F','L'], etc.
        # L=Left, R=Right, A=Anterior, P=Posterior, H=Head, F=Foot

        if 'front' in view_description.lower() or 'anterior' in view_description.lower():
            return ['R', 'H']  # Right, Head
        elif 'back' in view_description.lower() or 'posterior' in view_description.lower():
            return ['L', 'H']  # Left, Head
        elif 'left' in body_site.lower():
            return ['A', 'H']  # Anterior, Head
        elif 'right' in body_site.lower():
            return ['P', 'H']  # Posterior, Head

        return None

    def generate_dicom_tags(self, metadata: Dict) -> Dict:
        """
        Generate DICOM tags with proper group/element numbers.

        Returns dictionary with DICOM tag numbers as keys.
        """
        # Standard DICOM tags (Group, Element)
        dicom_tags = {
            # Patient Module
            (0x0010, 0x0010): metadata.get('PatientName'),  # Patient's Name
            (0x0010, 0x0020): metadata.get('PatientID'),    # Patient ID
            (0x0010, 0x0030): metadata.get('PatientBirthDate'),  # Patient's Birth Date

            # Study Module
            (0x0020, 0x000D): metadata.get('StudyID'),      # Study Instance UID
            (0x0008, 0x0020): metadata.get('StudyDate'),    # Study Date
            (0x0008, 0x0030): metadata.get('StudyTime'),    # Study Time
            (0x0008, 0x1030): metadata.get('StudyDescription'),  # Study Description

            # Series Module
            (0x0020, 0x0011): metadata.get('SeriesNumber'), # Series Number
            (0x0008, 0x103E): metadata.get('SeriesDescription'),  # Series Description

            # Equipment Module
            (0x0008, 0x0070): metadata.get('Manufacturer'), # Manufacturer
            (0x0008, 0x1090): metadata.get('ManufacturerModelName'),  # Model Name

            # Image Module
            (0x0008, 0x0060): metadata.get('Modality'),     # Modality
            (0x0018, 0x0015): metadata.get('BodyPartExamined'),  # Body Part Examined
            (0x0020, 0x0060): metadata.get('Laterality'),   # Laterality
            (0x0028, 0x0030): metadata.get('PixelSpacing'), # Pixel Spacing
        }

        # Remove None values
        return {k: v for k, v in dicom_tags.items() if v is not None}

    def export_to_json(self, metadata: Dict, filepath: str):
        """Export metadata to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def validate_compliance(self, metadata: Dict) -> Dict:
        """
        Validate DICOM compliance of metadata.

        Returns validation report with issues and recommendations.
        """
        issues = []
        warnings = []
        required_fields = [
            'PatientID',
            'StudyDate',
            'StudyTime',
            'Modality',
            'BodyPartExamined',
        ]

        # Check required fields
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                issues.append(f"Missing required field: {field}")

        # Check recommended fields
        recommended_fields = [
            'PixelSpacing',
            'SeriesDescription',
            'ViewPosition',
        ]

        for field in recommended_fields:
            if field not in metadata or not metadata[field]:
                warnings.append(f"Missing recommended field: {field}")

        # Check pixel spacing for scale calibration
        if not metadata.get('PixelSpacing'):
            warnings.append("No pixel spacing (scale calibration) - size measurements will be inaccurate")

        is_compliant = len(issues) == 0

        return {
            'is_compliant': is_compliant,
            'issues': issues,
            'warnings': warnings,
            'compliance_score': max(0, 100 - len(issues) * 20 - len(warnings) * 5),
        }


# Global instance
_generator = None

def get_dicom_metadata_generator() -> DICOMMetadataGenerator:
    """Get or create global generator instance"""
    global _generator
    if _generator is None:
        _generator = DICOMMetadataGenerator()
    return _generator
