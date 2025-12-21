"""
Clinical Biopsy Report Generator

Generates professional PDF reports for histopathology/biopsy results.
Reports include:
- Patient information
- Specimen details
- Histopathology findings
- AI analysis correlation
- Diagnostic recommendations
- Quality metrics

Compatible with clinical documentation standards.
"""

import io
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class BiopsyReportGenerator:
    """Generate clinical biopsy reports in PDF and HTML formats."""

    def __init__(self):
        self.report_template = None

    def generate_report(
        self,
        analysis_data: Dict,
        patient_info: Optional[Dict] = None,
        specimen_info: Optional[Dict] = None,
        include_ai_correlation: bool = True,
        format: str = 'pdf'
    ) -> Union[bytes, str]:
        """
        Generate a clinical biopsy report.

        Args:
            analysis_data: Histopathology analysis results
            patient_info: Patient demographics (optional)
            specimen_info: Specimen collection details (optional)
            include_ai_correlation: Include AI vs biopsy correlation section
            format: Output format ('pdf', 'html', 'text')

        Returns:
            Report content (bytes for PDF, string for HTML/text)
        """
        if format == 'pdf':
            return self._generate_pdf_report(
                analysis_data, patient_info, specimen_info, include_ai_correlation
            )
        elif format == 'html':
            return self._generate_html_report(
                analysis_data, patient_info, specimen_info, include_ai_correlation
            )
        else:
            return self._generate_text_report(
                analysis_data, patient_info, specimen_info, include_ai_correlation
            )

    def _generate_pdf_report(
        self,
        analysis_data: Dict,
        patient_info: Optional[Dict],
        specimen_info: Optional[Dict],
        include_ai_correlation: bool
    ) -> bytes:
        """Generate PDF report using reportlab."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, Image
            )
        except ImportError:
            logger.warning("reportlab not available, falling back to HTML")
            html = self._generate_html_report(
                analysis_data, patient_info, specimen_info, include_ai_correlation
            )
            return html.encode('utf-8')

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        styles = getSampleStyleSheet()
        elements = []

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.HexColor('#1a365d')
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor('#2c5282')
        )

        # Title
        elements.append(Paragraph("HISTOPATHOLOGY REPORT", title_style))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2c5282')))
        elements.append(Spacer(1, 12))

        # Report metadata
        report_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        elements.append(Paragraph(f"<b>Report Date:</b> {report_date}", styles['Normal']))

        if analysis_data.get('timestamp'):
            analysis_date = analysis_data['timestamp'][:10] if isinstance(analysis_data['timestamp'], str) else analysis_data['timestamp']
            elements.append(Paragraph(f"<b>Analysis Date:</b> {analysis_date}", styles['Normal']))

        elements.append(Spacer(1, 12))

        # Patient Information (if provided)
        if patient_info:
            elements.append(Paragraph("PATIENT INFORMATION", heading_style))
            patient_data = [
                ['Name:', patient_info.get('name', 'N/A')],
                ['ID:', patient_info.get('id', 'N/A')],
                ['DOB:', patient_info.get('dob', 'N/A')],
                ['Gender:', patient_info.get('gender', 'N/A')]
            ]
            t = Table(patient_data, colWidths=[1.5*inch, 4*inch])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 12))

        # Specimen Information
        if specimen_info:
            elements.append(Paragraph("SPECIMEN INFORMATION", heading_style))
            spec_data = [
                ['Site:', specimen_info.get('site', 'Not specified')],
                ['Type:', specimen_info.get('type', 'Skin biopsy')],
                ['Collection Date:', specimen_info.get('collection_date', 'N/A')],
                ['Accession #:', specimen_info.get('accession_number', 'N/A')]
            ]
            t = Table(spec_data, colWidths=[1.5*inch, 4*inch])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 12))

        # Primary Diagnosis
        elements.append(Paragraph("DIAGNOSIS", heading_style))
        primary_diagnosis = analysis_data.get('primary_diagnosis', 'Not determined')
        diagnostic_category = analysis_data.get('diagnostic_category', '')

        diag_text = f"<b>{self._format_diagnosis(primary_diagnosis)}</b>"
        if diagnostic_category:
            diag_text += f" ({diagnostic_category})"

        elements.append(Paragraph(diag_text, styles['Normal']))
        elements.append(Spacer(1, 6))

        # Confidence
        confidence = analysis_data.get('primary_probability', 0)
        elements.append(Paragraph(
            f"<b>Confidence:</b> {confidence:.1%}",
            styles['Normal']
        ))

        # Risk Level
        malignancy = analysis_data.get('malignancy_assessment', {})
        risk_level = malignancy.get('risk_level', 'unknown')
        risk_color = {'high': '#c53030', 'moderate': '#d69e2e', 'low': '#38a169'}.get(risk_level, '#718096')

        elements.append(Paragraph(
            f"<b>Risk Level:</b> <font color='{risk_color}'>{risk_level.upper()}</font>",
            styles['Normal']
        ))
        elements.append(Spacer(1, 12))

        # Tissue Analysis
        elements.append(Paragraph("MICROSCOPIC FINDINGS", heading_style))
        tissue_types = analysis_data.get('tissue_types', [])

        if tissue_types:
            findings_data = [['Tissue Type', 'Confidence', 'Description']]
            for tissue in tissue_types[:5]:
                findings_data.append([
                    tissue.get('type', 'N/A'),
                    f"{tissue.get('confidence', 0):.1%}",
                    tissue.get('description', '')[:50] + '...' if len(tissue.get('description', '')) > 50 else tissue.get('description', '')
                ])

            t = Table(findings_data, colWidths=[2*inch, 1*inch, 3.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e2e8f0')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(t)
        elements.append(Spacer(1, 12))

        # Key Features
        key_features = malignancy.get('key_features', [])
        if key_features:
            elements.append(Paragraph("KEY FEATURES IDENTIFIED", heading_style))
            for feature in key_features:
                elements.append(Paragraph(f"• {feature}", styles['Normal']))
            elements.append(Spacer(1, 12))

        # Quality Metrics
        quality = analysis_data.get('quality_metrics', {})
        if quality:
            elements.append(Paragraph("IMAGE QUALITY ASSESSMENT", heading_style))
            quality_data = [
                ['Staining Quality:', quality.get('staining_quality', 'N/A')],
                ['Focus Quality:', quality.get('focus_quality', 'N/A')],
                ['Tissue Adequacy:', quality.get('tissue_adequacy', 'N/A')],
            ]
            if quality.get('quality_score'):
                quality_data.append(['Quality Score:', f"{quality['quality_score']:.2f}"])

            t = Table(quality_data, colWidths=[1.5*inch, 4*inch])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 12))

        # AI Correlation (if available)
        if include_ai_correlation and 'dermoscopy_correlation' in analysis_data:
            correlation = analysis_data['dermoscopy_correlation']
            elements.append(Paragraph("AI-HISTOPATHOLOGY CORRELATION", heading_style))

            is_concordant = correlation.get('is_concordant', False)
            concordance_text = "CONCORDANT" if is_concordant else "DISCORDANT"
            concordance_color = '#38a169' if is_concordant else '#c53030'

            elements.append(Paragraph(
                f"<b>Correlation Status:</b> <font color='{concordance_color}'>{concordance_text}</font>",
                styles['Normal']
            ))
            elements.append(Paragraph(
                f"<b>Dermoscopy AI Prediction:</b> {correlation.get('actual_dermoscopy_class', 'N/A')}",
                styles['Normal']
            ))
            elements.append(Paragraph(
                f"<b>Assessment:</b> {correlation.get('agreement_assessment', 'N/A')}",
                styles['Normal']
            ))
            elements.append(Spacer(1, 12))

        # Recommendations
        recommendations = analysis_data.get('recommendations', [])
        if recommendations:
            elements.append(Paragraph("RECOMMENDATIONS", heading_style))
            for rec in recommendations:
                elements.append(Paragraph(f"• {rec}", styles['Normal']))
            elements.append(Spacer(1, 12))

        # Disclaimer
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elements.append(Spacer(1, 6))
        disclaimer = """
        <font size=8 color='#718096'>
        <b>DISCLAIMER:</b> This report is generated by AI-assisted analysis and should be reviewed
        by a qualified pathologist. AI predictions are provided to assist clinical decision-making
        but do not replace professional medical judgment. Model: {model}
        </font>
        """.format(model=analysis_data.get('model_used', 'Hibou-L'))
        elements.append(Paragraph(disclaimer, styles['Normal']))

        # Build PDF
        doc.build(elements)
        return buffer.getvalue()

    def _generate_html_report(
        self,
        analysis_data: Dict,
        patient_info: Optional[Dict],
        specimen_info: Optional[Dict],
        include_ai_correlation: bool
    ) -> str:
        """Generate HTML report."""
        report_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        primary_diagnosis = analysis_data.get('primary_diagnosis', 'Not determined')
        diagnostic_category = analysis_data.get('diagnostic_category', '')
        confidence = analysis_data.get('primary_probability', 0)

        malignancy = analysis_data.get('malignancy_assessment', {})
        risk_level = malignancy.get('risk_level', 'unknown')
        risk_color = {'high': '#c53030', 'moderate': '#d69e2e', 'low': '#38a169'}.get(risk_level, '#718096')

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Histopathology Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
        h1 {{ color: #1a365d; border-bottom: 2px solid #2c5282; padding-bottom: 10px; }}
        h2 {{ color: #2c5282; margin-top: 20px; }}
        .metadata {{ color: #666; margin-bottom: 20px; }}
        .section {{ margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #e2e8f0; }}
        .risk-high {{ color: #c53030; font-weight: bold; }}
        .risk-moderate {{ color: #d69e2e; font-weight: bold; }}
        .risk-low {{ color: #38a169; font-weight: bold; }}
        .concordant {{ color: #38a169; }}
        .discordant {{ color: #c53030; }}
        .disclaimer {{ font-size: 0.8em; color: #718096; border-top: 1px solid #ccc; padding-top: 10px; margin-top: 30px; }}
        ul {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>HISTOPATHOLOGY REPORT</h1>

    <div class="metadata">
        <p><strong>Report Date:</strong> {report_date}</p>
        <p><strong>Analysis Date:</strong> {analysis_data.get('timestamp', 'N/A')[:10] if analysis_data.get('timestamp') else 'N/A'}</p>
    </div>
"""

        # Patient info
        if patient_info:
            html += f"""
    <div class="section">
        <h2>PATIENT INFORMATION</h2>
        <table>
            <tr><td><strong>Name:</strong></td><td>{patient_info.get('name', 'N/A')}</td></tr>
            <tr><td><strong>ID:</strong></td><td>{patient_info.get('id', 'N/A')}</td></tr>
            <tr><td><strong>DOB:</strong></td><td>{patient_info.get('dob', 'N/A')}</td></tr>
            <tr><td><strong>Gender:</strong></td><td>{patient_info.get('gender', 'N/A')}</td></tr>
        </table>
    </div>
"""

        # Diagnosis
        html += f"""
    <div class="section">
        <h2>DIAGNOSIS</h2>
        <p><strong>{self._format_diagnosis(primary_diagnosis)}</strong> ({diagnostic_category})</p>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <p><strong>Risk Level:</strong> <span class="risk-{risk_level}">{risk_level.upper()}</span></p>
    </div>
"""

        # Tissue types
        tissue_types = analysis_data.get('tissue_types', [])
        if tissue_types:
            html += """
    <div class="section">
        <h2>MICROSCOPIC FINDINGS</h2>
        <table>
            <tr><th>Tissue Type</th><th>Confidence</th><th>Description</th></tr>
"""
            for tissue in tissue_types[:5]:
                html += f"""
            <tr>
                <td>{tissue.get('type', 'N/A')}</td>
                <td>{tissue.get('confidence', 0):.1%}</td>
                <td>{tissue.get('description', '')}</td>
            </tr>
"""
            html += """
        </table>
    </div>
"""

        # Key features
        key_features = malignancy.get('key_features', [])
        if key_features:
            html += """
    <div class="section">
        <h2>KEY FEATURES</h2>
        <ul>
"""
            for feature in key_features:
                html += f"            <li>{feature}</li>\n"
            html += """
        </ul>
    </div>
"""

        # AI Correlation
        if include_ai_correlation and 'dermoscopy_correlation' in analysis_data:
            correlation = analysis_data['dermoscopy_correlation']
            is_concordant = correlation.get('is_concordant', False)
            concordance_class = 'concordant' if is_concordant else 'discordant'
            concordance_text = 'CONCORDANT' if is_concordant else 'DISCORDANT'

            html += f"""
    <div class="section">
        <h2>AI-HISTOPATHOLOGY CORRELATION</h2>
        <p><strong>Status:</strong> <span class="{concordance_class}">{concordance_text}</span></p>
        <p><strong>Dermoscopy AI Prediction:</strong> {correlation.get('actual_dermoscopy_class', 'N/A')}</p>
        <p><strong>Assessment:</strong> {correlation.get('agreement_assessment', 'N/A')}</p>
    </div>
"""

        # Recommendations
        recommendations = analysis_data.get('recommendations', [])
        if recommendations:
            html += """
    <div class="section">
        <h2>RECOMMENDATIONS</h2>
        <ul>
"""
            for rec in recommendations:
                html += f"            <li>{rec}</li>\n"
            html += """
        </ul>
    </div>
"""

        # Disclaimer
        html += f"""
    <div class="disclaimer">
        <strong>DISCLAIMER:</strong> This report is generated by AI-assisted analysis and should be
        reviewed by a qualified pathologist. AI predictions are provided to assist clinical
        decision-making but do not replace professional medical judgment.
        Model: {analysis_data.get('model_used', 'Hibou-L')}
    </div>
</body>
</html>
"""
        return html

    def _generate_text_report(
        self,
        analysis_data: Dict,
        patient_info: Optional[Dict],
        specimen_info: Optional[Dict],
        include_ai_correlation: bool
    ) -> str:
        """Generate plain text report."""
        lines = []
        lines.append("=" * 60)
        lines.append("HISTOPATHOLOGY REPORT")
        lines.append("=" * 60)
        lines.append(f"Report Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")

        # Diagnosis
        lines.append("-" * 40)
        lines.append("DIAGNOSIS")
        lines.append("-" * 40)
        lines.append(f"Primary: {self._format_diagnosis(analysis_data.get('primary_diagnosis', 'N/A'))}")
        lines.append(f"Category: {analysis_data.get('diagnostic_category', 'N/A')}")
        lines.append(f"Confidence: {analysis_data.get('primary_probability', 0):.1%}")

        malignancy = analysis_data.get('malignancy_assessment', {})
        lines.append(f"Risk Level: {malignancy.get('risk_level', 'unknown').upper()}")
        lines.append("")

        # Tissue types
        lines.append("-" * 40)
        lines.append("MICROSCOPIC FINDINGS")
        lines.append("-" * 40)
        for tissue in analysis_data.get('tissue_types', [])[:5]:
            lines.append(f"  - {tissue.get('type')}: {tissue.get('confidence', 0):.1%}")
        lines.append("")

        # Recommendations
        lines.append("-" * 40)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        for rec in analysis_data.get('recommendations', []):
            lines.append(f"  * {rec}")
        lines.append("")

        lines.append("=" * 60)
        lines.append("AI-assisted analysis - Review by qualified pathologist required")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _format_diagnosis(self, diagnosis: str) -> str:
        """Format diagnosis name for display."""
        if not diagnosis:
            return "Not determined"
        return diagnosis.replace('_', ' ').title()


def generate_biopsy_report(
    analysis_data: Dict,
    patient_info: Optional[Dict] = None,
    specimen_info: Optional[Dict] = None,
    format: str = 'pdf'
) -> Union[bytes, str]:
    """
    Convenience function to generate a biopsy report.

    Args:
        analysis_data: Histopathology analysis results
        patient_info: Optional patient info
        specimen_info: Optional specimen details
        format: Output format ('pdf', 'html', 'text')

    Returns:
        Report content
    """
    generator = BiopsyReportGenerator()
    return generator.generate_report(
        analysis_data=analysis_data,
        patient_info=patient_info,
        specimen_info=specimen_info,
        format=format
    )


if __name__ == '__main__':
    # Test report generation
    test_analysis = {
        'timestamp': '2024-01-15T10:30:00',
        'model_used': 'Hibou-L',
        'primary_diagnosis': 'melanocytic_nevus',
        'primary_probability': 0.87,
        'diagnostic_category': 'benign',
        'tissue_types': [
            {'type': 'melanocytic_nevus', 'confidence': 0.87, 'description': 'Benign mole'},
            {'type': 'dermis_normal', 'confidence': 0.08, 'description': 'Normal dermis'},
        ],
        'malignancy_assessment': {
            'risk_level': 'low',
            'malignant_probability': 0.05,
            'key_features': ['Nested melanocytes', 'Regular architecture']
        },
        'quality_metrics': {
            'staining_quality': 'Good',
            'focus_quality': 'Good',
            'tissue_adequacy': 'Adequate'
        },
        'recommendations': [
            'Benign findings - routine follow-up',
            'Continue regular skin examinations'
        ]
    }

    # Generate reports
    generator = BiopsyReportGenerator()

    # Text report
    text_report = generator.generate_report(test_analysis, format='text')
    print(text_report)

    # HTML report
    html_report = generator.generate_report(test_analysis, format='html')
    print(f"\nHTML report length: {len(html_report)} characters")

    # PDF report (if reportlab available)
    try:
        pdf_report = generator.generate_report(test_analysis, format='pdf')
        print(f"PDF report size: {len(pdf_report)} bytes")
    except Exception as e:
        print(f"PDF generation: {e}")
