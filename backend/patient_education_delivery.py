"""
Patient Education Delivery System

Email and SMS delivery for educational PDF handouts.
Supports Twilio (SMS) and SMTP (Email).
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Optional, Dict
from datetime import datetime
import base64


class PatientEducationDelivery:
    """
    Deliver patient education materials via email or SMS.

    Supports:
    - Email with PDF attachment (SMTP)
    - SMS with PDF link (Twilio)
    - Delivery tracking
    """

    def __init__(self):
        # Email configuration (from environment variables)
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_user)
        self.from_name = os.getenv('FROM_NAME', 'Dermatology AI Assistant')

        # Twilio configuration
        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID', '')
        self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN', '')
        self.twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER', '')

        # Base URL for PDF links
        self.base_url = os.getenv('BASE_URL', 'http://localhost:8000')

    def send_email(
        self,
        to_email: str,
        patient_name: str,
        condition_name: str,
        pdf_bytes: bytes,
        pdf_filename: str,
        language: str = "en"
    ) -> Dict:
        """
        Send PDF handout via email.

        Args:
            to_email: Recipient email address
            patient_name: Patient's name
            condition_name: Condition name
            pdf_bytes: PDF file bytes
            pdf_filename: PDF filename
            language: Language code

        Returns:
            Status dictionary with success/error info
        """
        try:
            # Check configuration
            if not self.smtp_user or not self.smtp_password:
                return {
                    "success": False,
                    "error": "Email not configured. Set SMTP_USER and SMTP_PASSWORD environment variables.",
                    "delivery_method": "email"
                }

            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            msg['Subject'] = self._get_email_subject(condition_name, language)

            # Email body
            body = self._get_email_body(patient_name, condition_name, language)
            msg.attach(MIMEText(body, 'html'))

            # Attach PDF
            pdf_attachment = MIMEApplication(pdf_bytes, _subtype='pdf')
            pdf_attachment.add_header('Content-Disposition', 'attachment', filename=pdf_filename)
            msg.attach(pdf_attachment)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            return {
                "success": True,
                "delivery_method": "email",
                "recipient": to_email,
                "timestamp": datetime.now().isoformat(),
                "message": f"Educational material sent to {to_email}"
            }

        except Exception as e:
            return {
                "success": False,
                "delivery_method": "email",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def send_sms(
        self,
        to_phone: str,
        patient_name: str,
        condition_name: str,
        pdf_link: str,
        language: str = "en"
    ) -> Dict:
        """
        Send PDF link via SMS.

        Args:
            to_phone: Recipient phone number (E.164 format: +1234567890)
            patient_name: Patient's name
            condition_name: Condition name
            pdf_link: URL to PDF
            language: Language code

        Returns:
            Status dictionary with success/error info
        """
        try:
            # Check configuration
            if not self.twilio_account_sid or not self.twilio_auth_token:
                return {
                    "success": False,
                    "error": "SMS not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER environment variables.",
                    "delivery_method": "sms"
                }

            # Import Twilio (only if configured)
            try:
                from twilio.rest import Client
            except ImportError:
                return {
                    "success": False,
                    "error": "Twilio package not installed. Run: pip install twilio",
                    "delivery_method": "sms"
                }

            # Create Twilio client
            client = Client(self.twilio_account_sid, self.twilio_auth_token)

            # Compose message
            message_body = self._get_sms_body(patient_name, condition_name, pdf_link, language)

            # Send SMS
            message = client.messages.create(
                body=message_body,
                from_=self.twilio_phone_number,
                to=to_phone
            )

            return {
                "success": True,
                "delivery_method": "sms",
                "recipient": to_phone,
                "message_sid": message.sid,
                "timestamp": datetime.now().isoformat(),
                "message": f"Educational material link sent to {to_phone}"
            }

        except Exception as e:
            return {
                "success": False,
                "delivery_method": "sms",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _get_email_subject(self, condition_name: str, language: str) -> str:
        """Get email subject in specified language"""
        subjects = {
            "en": f"Your {condition_name} Educational Material",
            "es": f"Su Material Educativo sobre {condition_name}",
            "fr": f"Votre Matériel Éducatif sur {condition_name}",
            "de": f"Ihr Lehrmaterial über {condition_name}",
        }
        return subjects.get(language, subjects["en"])

    def _get_email_body(self, patient_name: str, condition_name: str, language: str) -> str:
        """Get HTML email body"""
        templates = {
            "en": f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2563eb;">Patient Education Material</h2>

                    <p>Dear {patient_name},</p>

                    <p>Please find attached your personalized educational handout about <strong>{condition_name}</strong>.</p>

                    <p>This document includes:</p>
                    <ul>
                        <li>Description of your condition</li>
                        <li>Symptoms and causes</li>
                        <li>Care instructions</li>
                        <li>Treatment options</li>
                        <li>Warning signs to watch for</li>
                        <li>Prevention tips</li>
                    </ul>

                    <p><strong>Important:</strong> Please read this material carefully and follow the instructions provided. If you have any questions, don't hesitate to contact your healthcare provider.</p>

                    <p style="border-left: 4px solid #dc2626; padding-left: 15px; margin: 20px 0;">
                        <strong>When to Seek Immediate Care:</strong><br>
                        If you experience any warning signs mentioned in the document, contact your doctor immediately or seek emergency care if necessary.
                    </p>

                    <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 30px 0;">

                    <p style="font-size: 12px; color: #6b7280;">
                        This educational material is for informational purposes only and does not replace professional medical advice. Always consult your healthcare provider for diagnosis and treatment.
                    </p>

                    <p style="font-size: 12px; color: #6b7280;">
                        © {datetime.now().year} Dermatology AI Assistant
                    </p>
                </div>
            </body>
            </html>
            """,
            "es": f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2563eb;">Material Educativo para Pacientes</h2>

                    <p>Estimado/a {patient_name},</p>

                    <p>Adjunto encontrará su folleto educativo personalizado sobre <strong>{condition_name}</strong>.</p>

                    <p>Este documento incluye:</p>
                    <ul>
                        <li>Descripción de su condición</li>
                        <li>Síntomas y causas</li>
                        <li>Instrucciones de cuidado</li>
                        <li>Opciones de tratamiento</li>
                        <li>Señales de advertencia a vigilar</li>
                        <li>Consejos de prevención</li>
                    </ul>

                    <p><strong>Importante:</strong> Lea este material cuidadosamente y siga las instrucciones proporcionadas. Si tiene preguntas, no dude en contactar a su proveedor de atención médica.</p>

                    <p style="border-left: 4px solid #dc2626; padding-left: 15px; margin: 20px 0;">
                        <strong>Cuándo Buscar Atención Inmediata:</strong><br>
                        Si experimenta alguna señal de advertencia mencionada en el documento, contacte a su médico inmediatamente o busque atención de emergencia si es necesario.
                    </p>

                    <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 30px 0;">

                    <p style="font-size: 12px; color: #6b7280;">
                        Este material educativo es solo para fines informativos y no reemplaza el consejo médico profesional. Siempre consulte a su proveedor de atención médica para diagnóstico y tratamiento.
                    </p>

                    <p style="font-size: 12px; color: #6b7280;">
                        © {datetime.now().year} Asistente de IA para Dermatología
                    </p>
                </div>
            </body>
            </html>
            """,
        }
        return templates.get(language, templates["en"])

    def _get_sms_body(self, patient_name: str, condition_name: str, pdf_link: str, language: str) -> str:
        """Get SMS message body"""
        templates = {
            "en": f"Hi {patient_name}, your educational material about {condition_name} is ready. View/download: {pdf_link}",
            "es": f"Hola {patient_name}, su material educativo sobre {condition_name} está listo. Ver/descargar: {pdf_link}",
            "fr": f"Bonjour {patient_name}, votre matériel éducatif sur {condition_name} est prêt. Voir/télécharger: {pdf_link}",
        }
        return templates.get(language, templates["en"])


# Global instance
_delivery = None

def get_patient_education_delivery() -> PatientEducationDelivery:
    """Get or create global delivery instance"""
    global _delivery
    if _delivery is None:
        _delivery = PatientEducationDelivery()
    return _delivery


# Configuration checker
def check_delivery_config() -> Dict:
    """
    Check if email/SMS delivery is properly configured.

    Returns configuration status.
    """
    delivery = get_patient_education_delivery()

    email_configured = bool(delivery.smtp_user and delivery.smtp_password)
    sms_configured = bool(
        delivery.twilio_account_sid and
        delivery.twilio_auth_token and
        delivery.twilio_phone_number
    )

    twilio_installed = False
    try:
        import twilio
        twilio_installed = True
    except ImportError:
        pass

    return {
        "email": {
            "configured": email_configured,
            "smtp_host": delivery.smtp_host if email_configured else None,
            "smtp_user": delivery.smtp_user if email_configured else None,
        },
        "sms": {
            "configured": sms_configured,
            "twilio_installed": twilio_installed,
            "phone_number": delivery.twilio_phone_number if sms_configured else None,
        },
        "ready": email_configured or sms_configured
    }
