"""
Patient Education PDF Generator

Auto-generates personalized PDF handouts based on diagnosis with:
- Condition-specific care instructions
- Treatment timeline with progress photos
- Warning signs
- Multi-language support
- Print-friendly formats
"""

import io
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from patient_education_content import get_patient_education_library


class PatientEducationPDFGenerator:
    """
    Generate personalized patient education PDF handouts.

    Features:
    - Auto-generated based on diagnosis
    - Multi-language support
    - Treatment timeline visualization
    - Warning signs highlighted
    - Print-friendly format
    """

    def __init__(self):
        self.library = get_patient_education_library()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )

        # Subheading style
        self.subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#3b82f6'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )

        # Body style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8
        )

        # Warning style
        self.warning_style = ParagraphStyle(
            'Warning',
            parent=self.styles['BodyText'],
            fontSize=11,
            textColor=colors.HexColor('#dc2626'),
            fontName='Helvetica-Bold',
            leading=14,
            spaceAfter=6
        )

        # Highlight style
        self.highlight_style = ParagraphStyle(
            'Highlight',
            parent=self.styles['BodyText'],
            fontSize=11,
            textColor=colors.HexColor('#059669'),
            fontName='Helvetica-Bold',
            leading=14,
            spaceAfter=6
        )

    def generate_handout(
        self,
        condition_id: str,
        patient_name: str,
        language: str = "en",
        include_timeline: bool = True,
        include_images: bool = True,
        personalization: Optional[Dict] = None
    ) -> bytes:
        """
        Generate personalized PDF handout.

        Args:
            condition_id: Condition identifier
            patient_name: Patient's name
            language: Language code (en, es, fr, etc.)
            include_timeline: Include treatment timeline
            include_images: Include reference images
            personalization: Custom personalization data

        Returns:
            PDF bytes
        """
        # Get content from library
        content = self.library.get_content(condition_id, language)
        if not content:
            raise ValueError(f"Condition '{condition_id}' not found in library")

        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch,
            title=f"{content['condition_name']} - Patient Education",
            author="Dermatology AI Assistant"
        )

        # Build document story
        story = []

        # Header with logo and date
        story.extend(self._build_header(patient_name, language))

        # Title
        story.append(Paragraph(content['condition_name'], self.title_style))
        story.append(Spacer(1, 0.3*inch))

        # Description
        story.append(Paragraph(self._translate("About Your Condition", language), self.heading_style))
        story.append(Paragraph(content['description'], self.body_style))
        story.append(Spacer(1, 0.2*inch))

        # Symptoms
        story.extend(self._build_bullet_section(
            self._translate("Common Symptoms", language),
            content['symptoms'],
            self.heading_style,
            self.body_style
        ))

        # Causes
        story.extend(self._build_bullet_section(
            self._translate("Causes & Risk Factors", language),
            content['causes'],
            self.heading_style,
            self.body_style
        ))

        # Care Instructions
        story.append(Paragraph(self._translate("How to Care for Your Skin", language), self.heading_style))
        story.append(Spacer(1, 0.1*inch))
        for instruction in content['care_instructions']:
            story.append(Paragraph(f"• {instruction}", self.body_style))
        story.append(Spacer(1, 0.2*inch))

        # DO and DON'T Lists (side by side)
        story.extend(self._build_do_dont_table(
            content['do_list'],
            content['dont_list'],
            language
        ))

        # Page break
        story.append(PageBreak())

        # Treatment Options
        story.append(Paragraph(self._translate("Treatment Options", language), self.heading_style))
        story.extend(self._build_treatment_options_table(content['treatment_options']))

        # Expected Timeline
        if include_timeline:
            story.extend(self._build_timeline_section(
                content['expected_timeline'],
                content['severity'],
                language
            ))

        # Warning Signs (Highlighted Box)
        story.extend(self._build_warning_box(content['warning_signs'], language))

        # When to Return
        story.extend(self._build_bullet_section(
            self._translate("When to Contact Your Doctor", language),
            content['when_to_return'],
            self.heading_style,
            self.warning_style
        ))

        # Prevention Tips
        story.extend(self._build_bullet_section(
            self._translate("Prevention Tips", language),
            content['prevention_tips'],
            self.heading_style,
            self.highlight_style
        ))

        # Additional Info Box
        story.extend(self._build_info_box(content, language))

        # Footer
        story.extend(self._build_footer(language))

        # Build PDF
        doc.build(story)

        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    def _build_header(self, patient_name: str, language: str) -> List:
        """Build document header"""
        elements = []

        # Patient info and date
        header_data = [
            [
                Paragraph(f"<b>{self._translate('Patient', language)}:</b> {patient_name}", self.body_style),
                Paragraph(f"<b>{self._translate('Date', language)}:</b> {datetime.now().strftime('%B %d, %Y')}", self.body_style),
            ]
        ]

        header_table = Table(header_data, colWidths=[3.5*inch, 3*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        elements.append(header_table)
        elements.append(Spacer(1, 0.2*inch))

        return elements

    def _build_bullet_section(self, title: str, items: List[str], title_style, item_style) -> List:
        """Build section with bullet points"""
        elements = []
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 0.1*inch))

        for item in items:
            elements.append(Paragraph(f"• {item}", item_style))

        elements.append(Spacer(1, 0.2*inch))
        return elements

    def _build_do_dont_table(self, do_list: List[str], dont_list: List[str], language: str) -> List:
        """Build side-by-side DO/DON'T table"""
        elements = []

        # Prepare data
        max_items = max(len(do_list), len(dont_list))
        dont_text = self._translate("DON'T", language)
        table_data = [
            [
                Paragraph(f"<b>{self._translate('DO', language)}</b>", self.highlight_style),
                Paragraph(f"<b>{dont_text}</b>", self.warning_style),
            ]
        ]

        for i in range(max_items):
            do_item = f"✓ {do_list[i]}" if i < len(do_list) else ""
            dont_item = f"✗ {dont_list[i]}" if i < len(dont_list) else ""

            table_data.append([
                Paragraph(do_item, self.body_style),
                Paragraph(dont_item, self.body_style),
            ])

        # Create table
        table = Table(table_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))

        return elements

    def _build_treatment_options_table(self, treatments: List[Dict]) -> List:
        """Build treatment options table"""
        elements = []

        table_data = [
            [
                Paragraph("<b>Treatment</b>", self.subheading_style),
                Paragraph("<b>Description</b>", self.subheading_style),
            ]
        ]

        for treatment in treatments:
            table_data.append([
                Paragraph(f"<b>{treatment['name']}</b>", self.body_style),
                Paragraph(treatment['description'], self.body_style),
            ])

        table = Table(table_data, colWidths=[2*inch, 4.5*inch])
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eff6ff')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _build_timeline_section(self, timeline: str, severity: str, language: str) -> List:
        """Build treatment timeline section"""
        elements = []

        elements.append(Paragraph(self._translate("Expected Timeline", language), self.heading_style))
        elements.append(Paragraph(timeline, self.body_style))
        elements.append(Spacer(1, 0.15*inch))

        # Timeline visualization
        if severity == "mild":
            weeks = [
                ("Week 1", self._translate("Starting treatment", language)),
                ("Week 2", self._translate("Initial improvement", language)),
                ("Week 4", self._translate("Significant improvement", language)),
            ]
        elif severity == "moderate":
            weeks = [
                ("Week 1-2", self._translate("Starting treatment", language)),
                ("Week 3-4", self._translate("Early improvement", language)),
                ("Week 6-8", self._translate("Noticeable improvement", language)),
                ("Week 12+", self._translate("Continued management", language)),
            ]
        else:  # severe
            weeks = [
                ("Week 1-4", self._translate("Diagnosis & treatment start", language)),
                ("Month 2-3", self._translate("Treatment response evaluation", language)),
                ("Month 6+", self._translate("Ongoing management", language)),
                ("Long-term", self._translate("Regular monitoring", language)),
            ]

        timeline_data = [[Paragraph(f"<b>{week}</b>", self.body_style), Paragraph(desc, self.body_style)] for week, desc in weeks]

        timeline_table = Table(timeline_data, colWidths=[1.5*inch, 5*inch])
        timeline_table.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, -2), 1, colors.HexColor('#e5e7eb')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(timeline_table)
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _build_warning_box(self, warnings: List[str], language: str) -> List:
        """Build highlighted warning box"""
        elements = []

        # Create warning box content
        warning_content = [
            Paragraph(f"<b>⚠️ {self._translate('IMPORTANT WARNING SIGNS', language)}</b>", self.warning_style),
            Spacer(1, 0.1*inch),
        ]

        for warning in warnings:
            warning_content.append(Paragraph(f"• {warning}", self.body_style))
            warning_content.append(Spacer(1, 0.05*inch))

        warning_content.append(Paragraph(
            f"<b>{self._translate('Seek immediate medical attention if you experience any of these symptoms.', language)}</b>",
            self.warning_style
        ))

        # Create table to act as box
        warning_table = Table([[KeepTogether(warning_content)]], colWidths=[6.5*inch])
        warning_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#dc2626')),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef2f2')),
            ('PADDING', (0, 0), (-1, -1), 15),
        ]))

        elements.append(warning_table)
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _build_info_box(self, content: Dict, language: str) -> List:
        """Build additional info box"""
        elements = []

        info_data = [
            [self._translate("Severity", language), content['severity'].title()],
            [self._translate("Contagious", language), self._translate("Yes" if content['is_contagious'] else "No", language)],
            [self._translate("Prescription Required", language), self._translate("Yes" if content['requires_prescription'] else "No", language)],
        ]

        info_table = Table(info_data, colWidths=[2*inch, 4.5*inch])
        info_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f9fafb')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ]))

        elements.append(Spacer(1, 0.2*inch))
        elements.append(info_table)
        elements.append(Spacer(1, 0.3*inch))

        return elements

    def _build_footer(self, language: str) -> List:
        """Build document footer"""
        elements = []

        footer_text = self._translate(
            "This educational material is for informational purposes only and does not replace professional medical advice. "
            "Always consult your healthcare provider for diagnosis and treatment.",
            language
        )

        footer_style = ParagraphStyle(
            'Footer',
            parent=self.body_style,
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        )

        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("─" * 80, footer_style))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(footer_text, footer_style))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(
            f"© {datetime.now().year} Dermatology AI Assistant | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            footer_style
        ))

        return elements

    def _translate(self, text: str, language: str) -> str:
        """Translate common UI strings"""
        translations = {
            "en": {
                "Patient": "Patient",
                "Date": "Date",
                "About Your Condition": "About Your Condition",
                "Common Symptoms": "Common Symptoms",
                "Causes & Risk Factors": "Causes & Risk Factors",
                "How to Care for Your Skin": "How to Care for Your Skin",
                "DO": "DO",
                "DON'T": "DON'T",
                "Treatment Options": "Treatment Options",
                "Expected Timeline": "Expected Timeline",
                "Starting treatment": "Starting treatment",
                "Initial improvement": "Initial improvement",
                "Significant improvement": "Significant improvement",
                "Early improvement": "Early improvement",
                "Noticeable improvement": "Noticeable improvement",
                "Continued management": "Continued management",
                "Diagnosis & treatment start": "Diagnosis & treatment start",
                "Treatment response evaluation": "Treatment response evaluation",
                "Ongoing management": "Ongoing management",
                "Regular monitoring": "Regular monitoring",
                "IMPORTANT WARNING SIGNS": "IMPORTANT WARNING SIGNS",
                "Seek immediate medical attention if you experience any of these symptoms.": "Seek immediate medical attention if you experience any of these symptoms.",
                "When to Contact Your Doctor": "When to Contact Your Doctor",
                "Prevention Tips": "Prevention Tips",
                "Severity": "Severity",
                "Contagious": "Contagious",
                "Prescription Required": "Prescription Required",
                "Yes": "Yes",
                "No": "No",
                "This educational material is for informational purposes only and does not replace professional medical advice. Always consult your healthcare provider for diagnosis and treatment.": "This educational material is for informational purposes only and does not replace professional medical advice. Always consult your healthcare provider for diagnosis and treatment.",
            },
            "es": {
                "Patient": "Paciente",
                "Date": "Fecha",
                "About Your Condition": "Acerca de Su Condición",
                "Common Symptoms": "Síntomas Comunes",
                "Causes & Risk Factors": "Causas y Factores de Riesgo",
                "How to Care for Your Skin": "Cómo Cuidar Su Piel",
                "DO": "HACER",
                "DON'T": "NO HACER",
                "Treatment Options": "Opciones de Tratamiento",
                "Expected Timeline": "Línea de Tiempo Esperada",
                "Starting treatment": "Iniciando tratamiento",
                "Initial improvement": "Mejoría inicial",
                "Significant improvement": "Mejoría significativa",
                "IMPORTANT WARNING SIGNS": "SEÑALES DE ADVERTENCIA IMPORTANTES",
                "Seek immediate medical attention if you experience any of these symptoms.": "Busque atención médica inmediata si experimenta alguno de estos síntomas.",
                "When to Contact Your Doctor": "Cuándo Contactar a Su Médico",
                "Prevention Tips": "Consejos de Prevención",
                "Severity": "Gravedad",
                "Contagious": "Contagioso",
                "Prescription Required": "Se Requiere Receta",
                "Yes": "Sí",
                "No": "No",
                "This educational material is for informational purposes only and does not replace professional medical advice. Always consult your healthcare provider for diagnosis and treatment.": "Este material educativo es solo para fines informativos y no reemplaza el consejo médico profesional. Siempre consulte a su proveedor de atención médica para diagnóstico y tratamiento.",
            },
            # Add more languages as needed
        }

        return translations.get(language, translations["en"]).get(text, text)


# Global instance
_generator = None

def get_patient_education_generator() -> PatientEducationPDFGenerator:
    """Get or create global generator instance"""
    global _generator
    if _generator is None:
        _generator = PatientEducationPDFGenerator()
    return _generator
