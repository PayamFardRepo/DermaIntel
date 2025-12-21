"""
Publication-Ready Case Report Generator

Generates professional PDF case reports suitable for medical journals,
case studies, and clinical presentations.

Uses ReportLab for PDF generation with:
- De-identified patient demographics
- Clinical images with annotations
- AI analysis results
- Biopsy correlation (if available)
- Medical disclaimer
"""

import io
import os
import base64
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import HorizontalBarChart

from PIL import Image as PILImage

from deidentification_service import DeidentificationService, anonymize_for_publication


class PublicationReportGenerator:
    """
    Generates publication-ready PDF case reports.

    Features:
    - Professional medical report layout
    - De-identified patient information
    - High-resolution clinical images
    - AI analysis with confidence scores
    - ABCDE assessment visualization
    - Biopsy correlation
    - Medical disclaimer
    """

    # Color scheme
    COLORS = {
        "header": colors.HexColor("#1e3a5f"),  # Dark blue
        "subheader": colors.HexColor("#2563eb"),  # Blue
        "high_risk": colors.HexColor("#dc2626"),  # Red
        "medium_risk": colors.HexColor("#f59e0b"),  # Amber
        "low_risk": colors.HexColor("#10b981"),  # Green
        "text": colors.HexColor("#1f2937"),  # Dark gray
        "light_text": colors.HexColor("#6b7280"),  # Gray
        "border": colors.HexColor("#e5e7eb"),  # Light gray
        "background": colors.HexColor("#f9fafb"),  # Very light gray
    }

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.styles = self._create_styles()
        self.deidentifier = DeidentificationService()

    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for the report."""
        base_styles = getSampleStyleSheet()

        styles = {
            "title": ParagraphStyle(
                "Title",
                parent=base_styles["Heading1"],
                fontSize=18,
                textColor=self.COLORS["header"],
                spaceAfter=12,
                alignment=TA_CENTER,
            ),
            "section_header": ParagraphStyle(
                "SectionHeader",
                parent=base_styles["Heading2"],
                fontSize=14,
                textColor=self.COLORS["header"],
                spaceBefore=16,
                spaceAfter=8,
                borderPadding=4,
            ),
            "subsection": ParagraphStyle(
                "Subsection",
                parent=base_styles["Heading3"],
                fontSize=11,
                textColor=self.COLORS["subheader"],
                spaceBefore=8,
                spaceAfter=4,
            ),
            "body": ParagraphStyle(
                "Body",
                parent=base_styles["Normal"],
                fontSize=10,
                textColor=self.COLORS["text"],
                spaceAfter=6,
                alignment=TA_JUSTIFY,
            ),
            "label": ParagraphStyle(
                "Label",
                parent=base_styles["Normal"],
                fontSize=9,
                textColor=self.COLORS["light_text"],
                spaceAfter=2,
            ),
            "value": ParagraphStyle(
                "Value",
                parent=base_styles["Normal"],
                fontSize=10,
                textColor=self.COLORS["text"],
                fontName="Helvetica-Bold",
                spaceAfter=4,
            ),
            "risk_high": ParagraphStyle(
                "RiskHigh",
                parent=base_styles["Normal"],
                fontSize=12,
                textColor=self.COLORS["high_risk"],
                fontName="Helvetica-Bold",
            ),
            "risk_medium": ParagraphStyle(
                "RiskMedium",
                parent=base_styles["Normal"],
                fontSize=12,
                textColor=self.COLORS["medium_risk"],
                fontName="Helvetica-Bold",
            ),
            "risk_low": ParagraphStyle(
                "RiskLow",
                parent=base_styles["Normal"],
                fontSize=12,
                textColor=self.COLORS["low_risk"],
                fontName="Helvetica-Bold",
            ),
            "disclaimer": ParagraphStyle(
                "Disclaimer",
                parent=base_styles["Normal"],
                fontSize=8,
                textColor=self.COLORS["light_text"],
                alignment=TA_CENTER,
                spaceBefore=20,
            ),
            "footer": ParagraphStyle(
                "Footer",
                parent=base_styles["Normal"],
                fontSize=8,
                textColor=self.COLORS["light_text"],
                alignment=TA_CENTER,
            ),
        }

        return styles

    def generate_report(
        self,
        analysis_data: Dict[str, Any],
        user_data: Optional[Dict[str, Any]] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Generate a publication-ready PDF case report.

        Args:
            analysis_data: AnalysisHistory record as dictionary
            user_data: User record as dictionary
            profile_data: UserProfile record as dictionary
            options: Report options (include_images, include_dermoscopy, etc.)

        Returns:
            PDF file as bytes
        """
        options = options or {}
        include_images = options.get("include_images", True)
        include_dermoscopy = options.get("include_dermoscopy", True)
        include_heatmap = options.get("include_heatmap", True)
        include_biopsy = options.get("include_biopsy", True)

        # Anonymize data
        user_id = analysis_data.get("user_id", 0)
        analysis_id = analysis_data.get("id", 0)
        anonymized = anonymize_for_publication(
            user_id, analysis_id, user_data, profile_data, analysis_data
        )

        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        # Build report elements
        elements = []

        # Header
        elements.extend(self._build_header(anonymized["case_id"], anonymized["analysis_date"]))

        # Abstract
        elements.extend(self._build_abstract_section(analysis_data, anonymized))

        # Introduction
        elements.extend(self._build_introduction_section(analysis_data))

        # Patient Demographics
        elements.extend(self._build_demographics_section(anonymized["demographics"]))

        # Clinical Presentation
        elements.extend(self._build_clinical_section(anonymized["clinical_presentation"]))

        # Clinical Images
        if include_images:
            elements.extend(self._build_images_section(
                analysis_data,
                include_dermoscopy=include_dermoscopy,
                include_heatmap=include_heatmap
            ))

        # Methods Section
        elements.extend(self._build_methods_section(analysis_data))

        # AI Analysis Results
        elements.extend(self._build_analysis_section(analysis_data))

        # ABCDE Assessment (if available)
        if analysis_data.get("red_flag_data"):
            elements.extend(self._build_abcde_section(analysis_data["red_flag_data"]))

        # Dermoscopy Features (if available)
        if include_dermoscopy and analysis_data.get("dermoscopy_data"):
            elements.extend(self._build_dermoscopy_section(analysis_data["dermoscopy_data"]))

        # Biopsy Correlation (if available and requested)
        if include_biopsy and analysis_data.get("biopsy_performed"):
            elements.extend(self._build_biopsy_section(analysis_data))

        # Treatment Recommendations
        elements.extend(self._build_treatment_section(analysis_data))

        # Discussion
        elements.extend(self._build_discussion_section(analysis_data))

        # Conclusion
        elements.extend(self._build_conclusion_section(analysis_data))

        # References
        elements.extend(self._build_references_section())

        # Footer with disclaimer
        elements.extend(self._build_footer(analysis_data))

        # Generate PDF
        doc.build(elements)
        buffer.seek(0)

        return buffer.getvalue()

    def _build_header(self, case_id: str, analysis_date: str) -> List:
        """Build the report header section."""
        elements = []

        # Title
        elements.append(Paragraph("CLINICAL CASE REPORT", self.styles["title"]))
        elements.append(Spacer(1, 4))

        # Case ID and Date in a table
        header_data = [
            [
                Paragraph(f"<b>Case ID:</b> {case_id}", self.styles["body"]),
                Paragraph(f"<b>Report Generated:</b> {datetime.utcnow().strftime('%B %Y')}", self.styles["body"]),
            ],
            [
                Paragraph(f"<b>Analysis Date:</b> {analysis_date}", self.styles["body"]),
                Paragraph("<b>Institution:</b> [Institution Name]", self.styles["body"]),
            ],
        ]

        header_table = Table(header_data, colWidths=[3.5 * inch, 3.5 * inch])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.COLORS["background"]),
            ("BOX", (0, 0), (-1, -1), 1, self.COLORS["border"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, self.COLORS["border"]),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))

        elements.append(header_table)
        elements.append(Spacer(1, 16))

        return elements

    def _build_demographics_section(self, demographics: Dict[str, Any]) -> List:
        """Build the patient demographics section."""
        elements = []

        elements.append(Paragraph("PATIENT DEMOGRAPHICS", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))

        # Demographics table
        demo_data = [
            ["Age Range", demographics.get("age_range", "Not specified")],
            ["Gender", demographics.get("gender", "Not specified")],
            ["Skin Type", demographics.get("skin_type", "Not specified")],
        ]

        # Add medical history flags
        if demographics.get("has_family_history_skin_cancer"):
            demo_data.append(["Family History", "Positive for skin cancer"])
        if demographics.get("has_previous_skin_cancers"):
            demo_data.append(["Personal History", "Previous skin cancers"])
        if demographics.get("is_immunosuppressed"):
            demo_data.append(["Immune Status", "Immunosuppressed"])

        demo_table = Table(demo_data, colWidths=[2 * inch, 5 * inch])
        demo_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), self.COLORS["light_text"]),
            ("TEXTCOLOR", (1, 0), (1, -1), self.COLORS["text"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))

        elements.append(Spacer(1, 8))
        elements.append(demo_table)
        elements.append(Spacer(1, 12))

        return elements

    def _build_clinical_section(self, clinical: Dict[str, Any]) -> List:
        """Build the clinical presentation section."""
        elements = []

        elements.append(Paragraph("CLINICAL PRESENTATION", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))

        # Location
        location_parts = [clinical.get("body_location", "Not specified")]
        if clinical.get("body_sublocation"):
            location_parts.append(clinical["body_sublocation"])
        if clinical.get("body_side"):
            location_parts.append(f"({clinical['body_side']} side)")
        location_str = " - ".join(filter(None, location_parts))

        clinical_data = [
            ["Location", location_str],
            ["Duration", clinical.get("symptom_duration") or "Not specified"],
        ]

        if clinical.get("symptom_changes"):
            clinical_data.append(["Changes Noted", clinical["symptom_changes"]])

        # Symptoms
        symptoms = []
        if clinical.get("symptom_itching"):
            symptoms.append("Itching")
        if clinical.get("symptom_pain"):
            symptoms.append("Pain")
        if clinical.get("symptom_bleeding"):
            symptoms.append("Bleeding")

        if symptoms:
            clinical_data.append(["Associated Symptoms", ", ".join(symptoms)])

        clinical_table = Table(clinical_data, colWidths=[2 * inch, 5 * inch])
        clinical_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), self.COLORS["light_text"]),
            ("TEXTCOLOR", (1, 0), (1, -1), self.COLORS["text"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        elements.append(Spacer(1, 8))
        elements.append(clinical_table)
        elements.append(Spacer(1, 12))

        return elements

    def _build_images_section(
        self,
        analysis_data: Dict[str, Any],
        include_dermoscopy: bool = True,
        include_heatmap: bool = True
    ) -> List:
        """Build the clinical images section."""
        elements = []

        elements.append(Paragraph("CLINICAL IMAGES", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        images_row = []

        # Main clinical image
        image_url = analysis_data.get("image_url")
        if image_url:
            try:
                # Handle relative paths
                if image_url.startswith("/uploads/"):
                    image_path = Path("uploads") / image_url.replace("/uploads/", "")
                else:
                    image_path = Path(image_url)

                if image_path.exists():
                    img = RLImage(str(image_path), width=2.5 * inch, height=2.5 * inch)
                    images_row.append([
                        img,
                        Paragraph("Clinical Image", self.styles["label"])
                    ])
            except Exception as e:
                images_row.append([
                    Paragraph("[Image not available]", self.styles["body"]),
                    Paragraph("Clinical Image", self.styles["label"])
                ])

        # Heatmap (if available and requested)
        if include_heatmap and analysis_data.get("explainability_heatmap"):
            try:
                heatmap_data = analysis_data["explainability_heatmap"]
                if heatmap_data.startswith("data:image"):
                    heatmap_data = heatmap_data.split(",")[1]

                heatmap_bytes = base64.b64decode(heatmap_data)
                heatmap_buffer = io.BytesIO(heatmap_bytes)
                img = RLImage(heatmap_buffer, width=2.5 * inch, height=2.5 * inch)
                images_row.append([
                    img,
                    Paragraph("AI Attention Map", self.styles["label"])
                ])
            except Exception:
                pass

        if images_row:
            # Create image table
            image_table = Table(
                [images_row],
                colWidths=[3 * inch] * len(images_row)
            )
            image_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]))
            elements.append(image_table)
        else:
            elements.append(Paragraph("No images available", self.styles["body"]))

        elements.append(Spacer(1, 12))
        return elements

    def _build_analysis_section(self, analysis_data: Dict[str, Any]) -> List:
        """Build the AI analysis results section."""
        elements = []

        elements.append(Paragraph("AI ANALYSIS RESULTS", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        # Primary diagnosis
        predicted_class = analysis_data.get("predicted_class", "Unknown")
        confidence = analysis_data.get("lesion_confidence") or analysis_data.get("binary_confidence") or 0
        risk_level = analysis_data.get("risk_level", "unknown")

        # Risk level styling
        risk_style = self.styles["risk_low"]
        if risk_level in ["high", "very_high"]:
            risk_style = self.styles["risk_high"]
        elif risk_level == "medium":
            risk_style = self.styles["risk_medium"]

        # Main diagnosis box
        diagnosis_content = [
            [
                Paragraph("Primary Diagnosis", self.styles["label"]),
                Paragraph("Confidence", self.styles["label"]),
                Paragraph("Risk Level", self.styles["label"]),
            ],
            [
                Paragraph(f"<b>{predicted_class}</b>", self.styles["value"]),
                Paragraph(f"<b>{confidence * 100:.1f}%</b>", self.styles["value"]),
                Paragraph(f"<b>{risk_level.upper().replace('_', ' ')}</b>", risk_style),
            ],
        ]

        diagnosis_table = Table(diagnosis_content, colWidths=[3 * inch, 2 * inch, 2 * inch])
        diagnosis_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.COLORS["background"]),
            ("BOX", (0, 0), (-1, -1), 1, self.COLORS["border"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, self.COLORS["border"]),
            ("PADDING", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))

        elements.append(diagnosis_table)
        elements.append(Spacer(1, 12))

        # Differential diagnoses
        differentials = analysis_data.get("differential_diagnoses") or []
        probabilities = analysis_data.get("lesion_probabilities") or {}

        if probabilities:
            elements.append(Paragraph("Differential Diagnoses", self.styles["subsection"]))

            # Sort by probability
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

            diff_data = [["Condition", "Probability"]]
            for condition, prob in sorted_probs:
                if prob > 0.01:  # Only show > 1%
                    diff_data.append([condition, f"{prob * 100:.1f}%"])

            if len(diff_data) > 1:
                diff_table = Table(diff_data, colWidths=[5 * inch, 2 * inch])
                diff_table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BACKGROUND", (0, 0), (-1, 0), self.COLORS["background"]),
                    ("GRID", (0, 0), (-1, -1), 0.5, self.COLORS["border"]),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]))
                elements.append(diff_table)
                elements.append(Spacer(1, 12))

        return elements

    def _build_abcde_section(self, red_flag_data: Dict[str, Any]) -> List:
        """Build the ABCDE assessment section."""
        elements = []

        elements.append(Paragraph("ABCDE ASSESSMENT", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        abcde_data = [["Feature", "Score", "Interpretation"]]

        features = [
            ("A - Asymmetry", red_flag_data.get("asymmetry_score"), "asymmetry"),
            ("B - Border", red_flag_data.get("border_score"), "border"),
            ("C - Color", red_flag_data.get("color_score"), "color"),
            ("D - Diameter", red_flag_data.get("diameter_mm"), "diameter"),
            ("E - Evolution", red_flag_data.get("evolution_score"), "evolution"),
        ]

        for name, score, key in features:
            if score is not None:
                if key == "diameter":
                    score_str = f"{score:.1f} mm"
                    interp = "Concerning" if score > 6 else "Within normal range"
                else:
                    score_str = f"{score:.2f}"
                    interp = "Concerning" if score > 0.5 else "Within normal range"
                abcde_data.append([name, score_str, interp])

        if len(abcde_data) > 1:
            abcde_table = Table(abcde_data, colWidths=[2.5 * inch, 2 * inch, 2.5 * inch])
            abcde_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), self.COLORS["background"]),
                ("GRID", (0, 0), (-1, -1), 0.5, self.COLORS["border"]),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            elements.append(abcde_table)
        else:
            elements.append(Paragraph("ABCDE data not available", self.styles["body"]))

        elements.append(Spacer(1, 12))
        return elements

    def _build_dermoscopy_section(self, dermoscopy_data: Dict[str, Any]) -> List:
        """Build the dermoscopy features section."""
        elements = []

        elements.append(Paragraph("DERMOSCOPY ANALYSIS", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        # 7-Point Checklist
        seven_point = dermoscopy_data.get("seven_point_checklist", {})
        if seven_point:
            score = seven_point.get("score", 0)
            elements.append(Paragraph(
                f"7-Point Checklist Score: <b>{score}</b> (threshold: 3)",
                self.styles["body"]
            ))
            elements.append(Spacer(1, 8))

        # Features detected
        features = dermoscopy_data.get("features", {})
        if features:
            feature_data = [["Feature", "Present", "Details"]]

            feature_names = [
                ("pigment_network", "Pigment Network"),
                ("globules", "Globules/Dots"),
                ("streaks", "Streaks"),
                ("blue_white_veil", "Blue-White Veil"),
                ("vascular_patterns", "Vascular Patterns"),
                ("regression_structures", "Regression Structures"),
            ]

            for key, name in feature_names:
                feat = features.get(key, {})
                if isinstance(feat, dict):
                    present = "Yes" if feat.get("present") else "No"
                    details = feat.get("type") or feat.get("count") or "-"
                    feature_data.append([name, present, str(details)])

            if len(feature_data) > 1:
                feat_table = Table(feature_data, colWidths=[2.5 * inch, 1.5 * inch, 3 * inch])
                feat_table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BACKGROUND", (0, 0), (-1, 0), self.COLORS["background"]),
                    ("GRID", (0, 0), (-1, -1), 0.5, self.COLORS["border"]),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]))
                elements.append(feat_table)

        elements.append(Spacer(1, 12))
        return elements

    def _build_biopsy_section(self, analysis_data: Dict[str, Any]) -> List:
        """Build the biopsy/pathology correlation section."""
        elements = []

        elements.append(Paragraph("PATHOLOGY CORRELATION", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        biopsy_data = [
            ["Biopsy Performed", "Yes"],
            ["Pathology Result", analysis_data.get("biopsy_result", "Pending")],
        ]

        if analysis_data.get("biopsy_date"):
            biopsy_date = self.deidentifier.anonymize_date(
                analysis_data["biopsy_date"],
                format_type="month_year"
            )
            biopsy_data.append(["Procedure Date", biopsy_date])

        if analysis_data.get("pathologist_name"):
            biopsy_data.append(["Pathologist", "Board-certified dermatopathologist"])

        # AI concordance
        if analysis_data.get("prediction_correct") is not None:
            concordance = "Concordant" if analysis_data["prediction_correct"] else "Discordant"
            biopsy_data.append(["AI-Pathology Agreement", concordance])

        if analysis_data.get("accuracy_category"):
            biopsy_data.append(["Concordance Type", analysis_data["accuracy_category"].replace("_", " ").title()])

        biopsy_table = Table(biopsy_data, colWidths=[2.5 * inch, 4.5 * inch])
        biopsy_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TEXTCOLOR", (0, 0), (0, -1), self.COLORS["light_text"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))

        elements.append(biopsy_table)
        elements.append(Spacer(1, 12))

        return elements

    def _build_discussion_section(self, analysis_data: Dict[str, Any]) -> List:
        """Build the discussion section."""
        elements = []

        elements.append(Paragraph("DISCUSSION", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        # Clinical significance
        risk_level = analysis_data.get("risk_level", "unknown")
        predicted_class = analysis_data.get("predicted_class", "Unknown")

        if risk_level in ["high", "very_high"]:
            significance = (
                f"This case demonstrates a {predicted_class} with high-risk features warranting "
                "immediate dermatological evaluation and consideration for biopsy. The AI analysis "
                "identified concerning characteristics consistent with potentially malignant pathology."
            )
        elif risk_level == "medium":
            significance = (
                f"This case presents with {predicted_class} showing moderate concern features. "
                "Close clinical monitoring is recommended with consideration for follow-up imaging "
                "or biopsy based on clinical judgment."
            )
        else:
            significance = (
                f"This case demonstrates {predicted_class} with low-risk features. "
                "Routine monitoring according to standard guidelines is appropriate."
            )

        elements.append(Paragraph(significance, self.styles["body"]))
        elements.append(Spacer(1, 8))

        # AI limitations
        elements.append(Paragraph("AI Model Limitations", self.styles["subsection"]))
        limitations = (
            "This analysis was performed using an AI-assisted diagnostic system. While the system "
            "demonstrates high accuracy in clinical validation studies, it is intended to support "
            "rather than replace clinical judgment. The AI model may have reduced accuracy for: "
            "(1) rare skin conditions not well-represented in training data, "
            "(2) lesions with atypical presentations, "
            "(3) images with suboptimal quality or lighting. "
            "All diagnoses should be confirmed by a qualified dermatologist."
        )
        elements.append(Paragraph(limitations, self.styles["body"]))

        elements.append(Spacer(1, 12))
        return elements

    def _build_abstract_section(self, analysis_data: Dict[str, Any], anonymized: Dict[str, Any]) -> List:
        """Build the abstract/summary section."""
        elements = []

        elements.append(Paragraph("ABSTRACT", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        predicted_class = analysis_data.get("predicted_class", "skin lesion")
        confidence = analysis_data.get("lesion_confidence") or analysis_data.get("binary_confidence") or 0
        risk_level = analysis_data.get("risk_level", "unknown")
        body_location = anonymized.get("clinical_presentation", {}).get("body_location", "unspecified location")
        age_range = anonymized.get("demographics", {}).get("age_range", "adult")

        abstract_text = f"""
        <b>Background:</b> Artificial intelligence (AI) systems are increasingly being utilized in dermatological
        practice for skin lesion classification and risk assessment. This case report presents an AI-assisted
        analysis of a cutaneous lesion using a deep learning-based classification system.
        <br/><br/>
        <b>Case Presentation:</b> We report the case of a {age_range} patient presenting with a skin lesion
        located on the {body_location}. The lesion was evaluated using an AI-powered dermatological analysis
        system employing convolutional neural networks trained on dermoscopic and clinical images.
        <br/><br/>
        <b>Results:</b> The AI system classified the lesion as <b>{predicted_class}</b> with a confidence
        score of {confidence * 100:.1f}%. The overall risk assessment was categorized as <b>{risk_level.upper()}</b>.
        The system provided differential diagnoses and generated attention maps highlighting regions of
        diagnostic significance.
        <br/><br/>
        <b>Conclusion:</b> This case demonstrates the application of AI-assisted dermatological analysis
        in clinical practice. While AI systems show promise as diagnostic aids, findings should be
        correlated with clinical judgment and histopathological confirmation when indicated.
        """

        elements.append(Paragraph(abstract_text, self.styles["body"]))
        elements.append(Spacer(1, 8))

        # Keywords
        keywords = ["artificial intelligence", "dermatology", "skin lesion classification",
                    "deep learning", "computer-aided diagnosis", predicted_class.lower()]
        elements.append(Paragraph(
            f"<b>Keywords:</b> {', '.join(keywords)}",
            self.styles["body"]
        ))

        elements.append(Spacer(1, 12))
        return elements

    def _build_introduction_section(self, analysis_data: Dict[str, Any]) -> List:
        """Build the introduction section."""
        elements = []

        elements.append(Paragraph("INTRODUCTION", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        predicted_class = analysis_data.get("predicted_class", "skin lesion")

        intro_text = """
        Skin cancer represents one of the most common malignancies worldwide, with melanoma alone accounting
        for the majority of skin cancer-related deaths despite comprising a small percentage of cases.
        Early detection and accurate classification of skin lesions are critical for improving patient
        outcomes, as the 5-year survival rate for early-stage melanoma exceeds 98% but drops significantly
        with disease progression.
        <br/><br/>
        Recent advances in artificial intelligence, particularly deep learning and convolutional neural
        networks (CNNs), have demonstrated remarkable capability in analyzing medical images. Studies have
        shown that well-trained AI systems can achieve diagnostic accuracy comparable to experienced
        dermatologists in classifying dermoscopic images of skin lesions. These systems analyze multiple
        features including color distribution, border irregularity, texture patterns, and structural
        elements that correlate with diagnostic criteria such as the ABCDE rule and 7-point checklist.
        <br/><br/>
        This case report presents the application of an AI-assisted diagnostic system for the evaluation
        of a skin lesion. The system employs an ensemble of deep learning models trained on large datasets
        of dermoscopic and clinical images, including data from the International Skin Imaging Collaboration
        (ISIC) archive. We present the clinical findings, AI analysis results, and discuss the implications
        for clinical practice.
        """

        elements.append(Paragraph(intro_text, self.styles["body"]))
        elements.append(Spacer(1, 12))
        return elements

    def _build_methods_section(self, analysis_data: Dict[str, Any]) -> List:
        """Build the methodology section."""
        elements = []

        elements.append(Paragraph("METHODS", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        # Image Acquisition
        elements.append(Paragraph("Image Acquisition", self.styles["subsection"]))
        acquisition_text = """
        Clinical images were captured using a standardized imaging protocol. For dermoscopic analysis,
        images were obtained using a polarized dermatoscope with 10x magnification. Images were acquired
        in JPEG format with minimum resolution of 1024x1024 pixels to ensure adequate detail for
        computational analysis.
        """
        elements.append(Paragraph(acquisition_text, self.styles["body"]))
        elements.append(Spacer(1, 6))

        # AI Analysis Pipeline
        elements.append(Paragraph("AI Analysis Pipeline", self.styles["subsection"]))
        model_version = analysis_data.get("model_version", "v2.0")
        analysis_type = analysis_data.get("analysis_type", "full")

        pipeline_text = f"""
        The AI analysis system (Model Version: {model_version}) employs a multi-stage classification
        pipeline:
        <br/><br/>
        <b>1. Preprocessing:</b> Images undergo automated quality assessment, followed by normalization,
        color constancy adjustment, and artifact removal. Images are resized to the model's input
        dimensions while preserving aspect ratio.
        <br/><br/>
        <b>2. Feature Extraction:</b> Deep convolutional neural networks extract hierarchical features
        from the input images. The system utilizes transfer learning from ImageNet-pretrained models
        (EfficientNet, ResNet architectures) fine-tuned on dermatological datasets.
        <br/><br/>
        <b>3. Classification:</b> An ensemble of classifiers produces probability distributions across
        diagnostic categories. The system outputs include primary diagnosis, confidence scores, and
        differential diagnoses ranked by probability.
        <br/><br/>
        <b>4. Explainability:</b> Gradient-weighted Class Activation Mapping (Grad-CAM) generates
        attention heatmaps highlighting image regions most influential to the classification decision,
        providing interpretable visual explanations.
        <br/><br/>
        <b>5. Risk Assessment:</b> A rule-based system integrates classification results with clinical
        parameters to generate an overall risk stratification (Low, Medium, High, Very High).
        """
        elements.append(Paragraph(pipeline_text, self.styles["body"]))
        elements.append(Spacer(1, 6))

        # Training Data
        elements.append(Paragraph("Training Data", self.styles["subsection"]))
        training_text = """
        The classification models were trained on a curated dataset comprising images from multiple
        sources including the ISIC Archive (>100,000 dermoscopic images), HAM10000 dataset, and
        institutional collections. The training set encompasses multiple diagnostic categories including
        melanoma, basal cell carcinoma, squamous cell carcinoma, actinic keratosis, benign keratoses,
        dermatofibroma, vascular lesions, and melanocytic nevi. Data augmentation techniques including
        rotation, flipping, color jittering, and elastic deformations were applied to improve model
        generalization.
        """
        elements.append(Paragraph(training_text, self.styles["body"]))

        elements.append(Spacer(1, 12))
        return elements

    def _build_treatment_section(self, analysis_data: Dict[str, Any]) -> List:
        """Build the treatment recommendations section."""
        elements = []

        elements.append(Paragraph("MANAGEMENT RECOMMENDATIONS", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        risk_level = analysis_data.get("risk_level", "unknown")
        predicted_class = analysis_data.get("predicted_class", "skin lesion")
        treatment_recs = analysis_data.get("treatment_recommendations") or []
        risk_recommendation = analysis_data.get("risk_recommendation", "")

        # Risk-based recommendations
        if risk_level in ["high", "very_high"]:
            urgency = "URGENT"
            urgency_color = self.COLORS["high_risk"]
            rec_text = f"""
            Based on the AI analysis indicating a high-risk classification ({predicted_class}),
            the following management approach is recommended:
            <br/><br/>
            <b>Immediate Actions:</b>
            <br/>• Urgent referral to dermatology for clinical evaluation
            <br/>• Complete skin examination to assess for additional lesions
            <br/>• Consideration for excisional biopsy with appropriate margins
            <br/>• Documentation with serial photography for comparison
            <br/><br/>
            <b>If Malignancy Confirmed:</b>
            <br/>• Surgical excision with histologically clear margins
            <br/>• Staging workup as indicated by histopathology
            <br/>• Multidisciplinary tumor board review for advanced cases
            <br/>• Patient education regarding sun protection and self-examination
            """
        elif risk_level == "medium":
            urgency = "MODERATE PRIORITY"
            urgency_color = self.COLORS["medium_risk"]
            rec_text = f"""
            The AI analysis suggests moderate concern for this {predicted_class}.
            Recommended management approach:
            <br/><br/>
            <b>Recommended Actions:</b>
            <br/>• Dermatological evaluation within 2-4 weeks
            <br/>• Dermoscopic examination and documentation
            <br/>• Consider biopsy if clinical concern warrants
            <br/>• Short-interval follow-up (3 months) if observation chosen
            <br/><br/>
            <b>Monitoring Protocol:</b>
            <br/>• Serial photography for objective comparison
            <br/>• Patient education on warning signs requiring earlier evaluation
            <br/>• Assessment of risk factors (UV exposure, family history)
            """
        else:
            urgency = "ROUTINE"
            urgency_color = self.COLORS["low_risk"]
            rec_text = f"""
            The AI analysis indicates low risk for this {predicted_class}.
            Recommended management approach:
            <br/><br/>
            <b>Recommended Actions:</b>
            <br/>• Routine dermatological follow-up as per standard guidelines
            <br/>• Annual skin examination for patients with risk factors
            <br/>• Patient reassurance with education on self-monitoring
            <br/><br/>
            <b>General Recommendations:</b>
            <br/>• Sun protection measures (SPF 30+, protective clothing)
            <br/>• Monthly self-examination of skin
            <br/>• Prompt evaluation if changes noted (size, shape, color, symptoms)
            """

        elements.append(Paragraph(f"<b>Priority Level: </b>", self.styles["body"]))
        elements.append(Paragraph(f"<b>{urgency}</b>", ParagraphStyle(
            "Urgency", parent=self.styles["body"], textColor=urgency_color, fontSize=12
        )))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(rec_text, self.styles["body"]))

        # Include AI-generated recommendations if available
        if treatment_recs:
            elements.append(Spacer(1, 8))
            elements.append(Paragraph("AI-Generated Specific Recommendations:", self.styles["subsection"]))
            for rec in treatment_recs[:5]:  # Limit to top 5
                elements.append(Paragraph(f"• {rec}", self.styles["body"]))

        if risk_recommendation:
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(f"<b>Additional Note:</b> {risk_recommendation}", self.styles["body"]))

        elements.append(Spacer(1, 12))
        return elements

    def _build_conclusion_section(self, analysis_data: Dict[str, Any]) -> List:
        """Build the conclusion section."""
        elements = []

        elements.append(Paragraph("CONCLUSION", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        predicted_class = analysis_data.get("predicted_class", "skin lesion")
        confidence = analysis_data.get("lesion_confidence") or analysis_data.get("binary_confidence") or 0
        risk_level = analysis_data.get("risk_level", "unknown")
        has_biopsy = analysis_data.get("biopsy_performed", False)
        prediction_correct = analysis_data.get("prediction_correct")

        conclusion_text = f"""
        This case report demonstrates the application of artificial intelligence in dermatological
        lesion analysis. The AI system classified the presented lesion as {predicted_class} with
        {confidence * 100:.1f}% confidence and assigned a {risk_level} risk classification.
        """

        if has_biopsy and prediction_correct is not None:
            concordance = "concordant" if prediction_correct else "discordant"
            conclusion_text += f"""
            <br/><br/>
            Histopathological examination was performed, with results {concordance} with the AI
            prediction. This case contributes to the growing body of evidence regarding AI
            performance in dermatological diagnosis.
            """

        conclusion_text += """
        <br/><br/>
        <b>Key Findings:</b>
        <br/>• AI-assisted analysis provided rapid, objective lesion assessment
        <br/>• Attention mapping offered interpretable visualization of diagnostic features
        <br/>• Risk stratification aided clinical decision-making
        <br/><br/>
        <b>Clinical Implications:</b>
        <br/>AI-assisted diagnostic systems represent a promising adjunct to clinical dermatology,
        potentially improving diagnostic accuracy, reducing time to diagnosis, and supporting
        clinical decision-making particularly in settings with limited access to specialist care.
        However, these systems should complement rather than replace clinical judgment, and
        definitive diagnosis should be confirmed through appropriate clinical correlation and
        histopathological examination when indicated.
        <br/><br/>
        <b>Future Directions:</b>
        <br/>Continued validation studies, integration with electronic health records, and
        development of regulatory frameworks will be essential for broader clinical adoption
        of AI-assisted dermatological diagnosis.
        """

        elements.append(Paragraph(conclusion_text, self.styles["body"]))
        elements.append(Spacer(1, 12))
        return elements

    def _build_references_section(self) -> List:
        """Build the references section."""
        elements = []

        elements.append(Paragraph("REFERENCES", self.styles["section_header"]))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))
        elements.append(Spacer(1, 8))

        references = [
            "1. Esteva A, Kuprel B, Novoa RA, et al. Dermatologist-level classification of skin cancer with deep neural networks. Nature. 2017;542(7639):115-118.",
            "2. Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data. 2018;5:180161.",
            "3. Codella NCF, Gutman D, Celebi ME, et al. Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI). IEEE. 2018.",
            "4. Haenssle HA, Fink C, Schneiderbauer R, et al. Man against machine: diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists. Ann Oncol. 2018;29(8):1836-1842.",
            "5. Brinker TJ, Hekler A, Enk AH, et al. Deep learning outperformed 136 of 157 dermatologists in a head-to-head dermoscopic melanoma image classification task. Eur J Cancer. 2019;113:47-54.",
            "6. Marchetti MA, Codella NCF, Dusza SW, et al. Results of the 2016 International Skin Imaging Collaboration International Symposium on Biomedical Imaging challenge: Comparison of the accuracy of computer algorithms to dermatologists. J Am Acad Dermatol. 2018;78(2):270-277.",
            "7. Celebi ME, Codella N, Halpern A. Dermoscopy Image Analysis: Overview and Future Directions. IEEE J Biomed Health Inform. 2019;23(2):474-478.",
            "8. American Academy of Dermatology. Guidelines of care for the management of primary cutaneous melanoma. J Am Acad Dermatol. 2019.",
            "9. Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. IEEE ICCV. 2017.",
            "10. World Health Organization. Skin cancers. WHO Fact Sheet. 2023.",
        ]

        for ref in references:
            elements.append(Paragraph(ref, ParagraphStyle(
                "Reference",
                parent=self.styles["body"],
                fontSize=9,
                leftIndent=20,
                firstLineIndent=-20,
                spaceAfter=4,
            )))

        elements.append(Spacer(1, 12))
        return elements

    def _build_footer(self, analysis_data: Dict[str, Any]) -> List:
        """Build the report footer with disclaimer."""
        elements = []

        elements.append(Spacer(1, 20))
        elements.append(HRFlowable(width="100%", thickness=1, color=self.COLORS["border"]))

        # Model version
        model_version = analysis_data.get("model_version", "Unknown")
        elements.append(Paragraph(
            f"AI Model Version: {model_version} | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            self.styles["footer"]
        ))

        # Medical disclaimer
        disclaimer = (
            "<b>DISCLAIMER:</b> This report is generated by an AI-assisted diagnostic system and is intended "
            "for educational and research purposes only. It does not constitute medical advice, diagnosis, "
            "or treatment recommendations. All clinical decisions should be made by qualified healthcare "
            "professionals based on complete patient evaluation. Patient data has been de-identified "
            "in accordance with privacy guidelines."
        )
        elements.append(Paragraph(disclaimer, self.styles["disclaimer"]))

        return elements

    def save_report(
        self,
        pdf_bytes: bytes,
        case_id: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Save a generated PDF report to disk.

        Args:
            pdf_bytes: PDF content as bytes
            case_id: Case ID for filename
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"case_report_{case_id}_{timestamp}.pdf"

        filepath = self.output_dir / filename
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)

        return str(filepath)
