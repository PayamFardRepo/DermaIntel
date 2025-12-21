"""
De-identification Service for Publication-Ready Reports

This module provides utilities for anonymizing patient data in compliance
with basic de-identification requirements for medical case reports.

Features:
- Generate anonymous case IDs
- Convert ages to ranges
- Anonymize dates to relative timelines
- Remove direct identifiers (names, contacts, addresses)
"""

import hashlib
from datetime import datetime, date
from typing import Optional, Dict, Any, Union


class DeidentificationService:
    """
    Service for de-identifying patient data for publication-ready reports.

    Implements basic anonymization:
    - Removes direct identifiers (name, email, phone, address)
    - Converts exact ages to ranges
    - Converts dates to relative timelines or month/year only
    - Generates anonymous case IDs
    """

    # Age ranges for anonymization
    AGE_RANGES = [
        (0, 9, "0-9 years"),
        (10, 19, "10-19 years"),
        (20, 29, "20-29 years"),
        (30, 39, "30-39 years"),
        (40, 49, "40-49 years"),
        (50, 59, "50-59 years"),
        (60, 69, "60-69 years"),
        (70, 79, "70-79 years"),
        (80, 89, "80-89 years"),
        (90, 200, "90+ years"),
    ]

    @staticmethod
    def generate_case_id(user_id: int, analysis_id: int, prefix: str = "CASE") -> str:
        """
        Generate an anonymous case ID.

        Args:
            user_id: The user's database ID
            analysis_id: The analysis record ID
            prefix: Prefix for the case ID (default: "CASE")

        Returns:
            Anonymous case ID like "CASE-2024-0001"
        """
        year = datetime.utcnow().year
        # Create a hash-based suffix to ensure uniqueness without revealing IDs
        hash_input = f"{user_id}-{analysis_id}-{year}"
        hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:4].upper()

        # Format: PREFIX-YEAR-HASH
        return f"{prefix}-{year}-{hash_suffix}"

    @staticmethod
    def get_age_range(age: Optional[int]) -> str:
        """
        Convert exact age to an age range.

        Args:
            age: Exact age in years

        Returns:
            Age range string like "40-49 years"
        """
        if age is None:
            return "Age not specified"

        for min_age, max_age, label in DeidentificationService.AGE_RANGES:
            if min_age <= age <= max_age:
                return label

        return "Age not specified"

    @staticmethod
    def calculate_age(date_of_birth: Optional[Union[date, datetime]]) -> Optional[int]:
        """
        Calculate age from date of birth.

        Args:
            date_of_birth: Date of birth

        Returns:
            Age in years, or None if DOB not provided
        """
        if date_of_birth is None:
            return None

        today = date.today()
        if isinstance(date_of_birth, datetime):
            date_of_birth = date_of_birth.date()

        age = today.year - date_of_birth.year
        # Adjust if birthday hasn't occurred this year
        if (today.month, today.day) < (date_of_birth.month, date_of_birth.day):
            age -= 1

        return age

    @staticmethod
    def anonymize_date(
        input_date: Optional[Union[date, datetime]],
        reference_date: Optional[Union[date, datetime]] = None,
        format_type: str = "month_year"
    ) -> str:
        """
        Anonymize a date by converting to relative timeline or month/year.

        Args:
            input_date: The date to anonymize
            reference_date: Reference date for relative calculations (e.g., analysis date)
            format_type: "month_year" for "December 2024", "relative" for "Day 0", "Week 2"

        Returns:
            Anonymized date string
        """
        if input_date is None:
            return "Not specified"

        if isinstance(input_date, datetime):
            input_date = input_date.date()

        if format_type == "month_year":
            return input_date.strftime("%B %Y")

        elif format_type == "relative":
            if reference_date is None:
                reference_date = date.today()
            if isinstance(reference_date, datetime):
                reference_date = reference_date.date()

            delta = (input_date - reference_date).days

            if delta == 0:
                return "Day 0"
            elif abs(delta) < 7:
                return f"Day {delta:+d}"
            elif abs(delta) < 30:
                weeks = delta // 7
                return f"Week {weeks:+d}"
            elif abs(delta) < 365:
                months = delta // 30
                return f"Month {months:+d}"
            else:
                years = delta // 365
                return f"Year {years:+d}"

        return input_date.strftime("%B %Y")

    @staticmethod
    def anonymize_patient_data(
        user_data: Optional[Dict[str, Any]] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        analysis_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Anonymize patient data for publication.

        Args:
            user_data: User model data (id, email, full_name, etc.)
            profile_data: UserProfile model data (DOB, address, etc.)
            analysis_data: AnalysisHistory data

        Returns:
            Dictionary with anonymized patient demographics
        """
        user_data = user_data or {}
        profile_data = profile_data or {}
        analysis_data = analysis_data or {}

        # Calculate age from DOB if available
        age = None
        if profile_data.get("date_of_birth"):
            age = DeidentificationService.calculate_age(profile_data["date_of_birth"])
        elif profile_data.get("age"):
            age = profile_data["age"]
        elif user_data.get("age"):
            age = user_data["age"]

        # Build anonymized demographics
        anonymized = {
            "age_range": DeidentificationService.get_age_range(age),
            "gender": profile_data.get("gender") or user_data.get("gender") or "Not specified",
            "skin_type": profile_data.get("skin_type") or "Not specified",

            # Medical history (non-identifying)
            "has_family_history_skin_cancer": profile_data.get("family_history_skin_cancer", False),
            "has_previous_skin_cancers": profile_data.get("previous_skin_cancers", False),
            "is_immunosuppressed": profile_data.get("immunosuppression", False),

            # Remove all direct identifiers
            "name": "[REDACTED]",
            "email": "[REDACTED]",
            "phone": "[REDACTED]",
            "address": "[REDACTED]",

            # Referring physician anonymized
            "referring_physician": "Referring Physician" if analysis_data.get("dermatologist_name") else None,
        }

        return anonymized

    @staticmethod
    def anonymize_clinical_data(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize clinical presentation data.

        Args:
            analysis_data: AnalysisHistory data

        Returns:
            Anonymized clinical presentation
        """
        return {
            "body_location": analysis_data.get("body_location") or "Not specified",
            "body_sublocation": analysis_data.get("body_sublocation"),
            "body_side": analysis_data.get("body_side"),
            "symptom_duration": analysis_data.get("symptom_duration"),
            "symptom_changes": analysis_data.get("symptom_changes"),
            "symptom_itching": analysis_data.get("symptom_itching", False),
            "symptom_pain": analysis_data.get("symptom_pain", False),
            "symptom_bleeding": analysis_data.get("symptom_bleeding", False),
            # Remove any notes that might contain identifying information
            "clinical_notes": "[Clinical notes redacted for privacy]" if analysis_data.get("symptom_notes") else None,
        }

    @staticmethod
    def sanitize_text(text: Optional[str], max_length: int = 500) -> Optional[str]:
        """
        Sanitize free-text fields that might contain identifying information.

        Args:
            text: Text to sanitize
            max_length: Maximum length to return

        Returns:
            Sanitized text or None
        """
        if not text:
            return None

        # Common patterns that might contain PHI
        phi_patterns = [
            # Names (Mr., Mrs., Dr., etc.)
            r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+',
            # Phone numbers
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            # Email addresses
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            # Social Security Numbers
            r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
            # Medical Record Numbers (common patterns)
            r'\bMRN[:\s]*\d+\b',
            # Dates in various formats
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        ]

        import re
        sanitized = text
        for pattern in phi_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)

        # Truncate if needed
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."

        return sanitized


def anonymize_for_publication(
    user_id: int,
    analysis_id: int,
    user_data: Optional[Dict[str, Any]] = None,
    profile_data: Optional[Dict[str, Any]] = None,
    analysis_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to fully anonymize data for publication.

    Args:
        user_id: User database ID
        analysis_id: Analysis record ID
        user_data: User model data
        profile_data: UserProfile model data
        analysis_data: AnalysisHistory data

    Returns:
        Complete anonymized data package for report generation
    """
    service = DeidentificationService()

    return {
        "case_id": service.generate_case_id(user_id, analysis_id),
        "demographics": service.anonymize_patient_data(user_data, profile_data, analysis_data),
        "clinical_presentation": service.anonymize_clinical_data(analysis_data or {}),
        "analysis_date": service.anonymize_date(
            analysis_data.get("created_at") if analysis_data else None,
            format_type="month_year"
        ),
    }
