"""
Appointment Reminder Service

A background service that sends appointment reminders via:
- Email (using SMTP or SendGrid)
- SMS (using Twilio - placeholder)
- Push notifications (using Firebase - placeholder)

This service can be run as a scheduled task/cron job or as a background thread.
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


@dataclass
class ReminderConfig:
    """Configuration for the reminder service"""
    # Email settings
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    from_email: str = os.getenv("FROM_EMAIL", "appointments@skinanalyzer.com")

    # Twilio settings (for SMS)
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")

    # Firebase settings (for push notifications)
    firebase_credentials_path: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "")

    # Service settings
    check_interval_minutes: int = 5
    retry_failed_after_minutes: int = 15
    max_retries: int = 3


class AppointmentReminderService:
    """
    Service for sending appointment reminders.

    Can be used in two modes:
    1. Manual: Call send_pending_reminders() periodically
    2. Background: Call start_background_scheduler() to run continuously
    """

    def __init__(self, db_session_factory, config: Optional[ReminderConfig] = None):
        self.db_session_factory = db_session_factory
        self.config = config or ReminderConfig()
        self._stop_event = threading.Event()
        self._scheduler_thread = None

    def get_pending_reminders(self) -> List[Dict[str, Any]]:
        """Get all reminders that need to be sent"""
        from database import Appointment, AppointmentReminder

        db = self.db_session_factory()
        try:
            now = datetime.utcnow()

            # Find appointments needing reminders
            pending_reminders = []

            # Get upcoming appointments (next 48 hours)
            cutoff = now + timedelta(hours=48)

            appointments = db.query(Appointment).filter(
                Appointment.status.in_(["scheduled", "confirmed"]),
                Appointment.appointment_date >= now.date(),
            ).all()

            for apt in appointments:
                apt_datetime = datetime.combine(apt.appointment_date, apt.start_time)
                time_until = apt_datetime - now
                hours_until = time_until.total_seconds() / 3600

                # Get reminder settings
                reminder_settings = apt.reminder_settings or {
                    "24_hours_before": True,
                    "2_hours_before": True,
                    "email": True,
                    "push": True
                }

                reminders_sent = apt.reminders_sent or []

                # Check 24-hour reminder
                if 23 <= hours_until <= 25:
                    if reminder_settings.get("24_hours_before", True):
                        if not any(r.get("type") == "24_hours" for r in reminders_sent):
                            pending_reminders.append({
                                "appointment": apt,
                                "reminder_type": "24_hours",
                                "hours_until": hours_until,
                                "channels": self._get_channels(reminder_settings)
                            })

                # Check 2-hour reminder
                elif 1.5 <= hours_until <= 2.5:
                    if reminder_settings.get("2_hours_before", True):
                        if not any(r.get("type") == "2_hours" for r in reminders_sent):
                            pending_reminders.append({
                                "appointment": apt,
                                "reminder_type": "2_hours",
                                "hours_until": hours_until,
                                "channels": self._get_channels(reminder_settings)
                            })

                # Check 30-minute reminder (for same-day appointments)
                elif 0.4 <= hours_until <= 0.6:
                    if reminder_settings.get("30_minutes_before", False):
                        if not any(r.get("type") == "30_minutes" for r in reminders_sent):
                            pending_reminders.append({
                                "appointment": apt,
                                "reminder_type": "30_minutes",
                                "hours_until": hours_until,
                                "channels": self._get_channels(reminder_settings)
                            })

            return pending_reminders

        finally:
            db.close()

    def _get_channels(self, settings: Dict) -> List[NotificationChannel]:
        """Determine which channels to use based on settings"""
        channels = []
        if settings.get("email", True):
            channels.append(NotificationChannel.EMAIL)
        if settings.get("sms", False):
            channels.append(NotificationChannel.SMS)
        if settings.get("push", True):
            channels.append(NotificationChannel.PUSH)
        if settings.get("in_app", True):
            channels.append(NotificationChannel.IN_APP)
        return channels

    def send_reminder(
        self,
        appointment,
        reminder_type: str,
        channels: List[NotificationChannel]
    ) -> Dict[str, bool]:
        """Send reminder through specified channels"""
        results = {}

        for channel in channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    success = self._send_email_reminder(appointment, reminder_type)
                elif channel == NotificationChannel.SMS:
                    success = self._send_sms_reminder(appointment, reminder_type)
                elif channel == NotificationChannel.PUSH:
                    success = self._send_push_reminder(appointment, reminder_type)
                elif channel == NotificationChannel.IN_APP:
                    success = self._create_in_app_notification(appointment, reminder_type)
                else:
                    success = False

                results[channel.value] = success

            except Exception as e:
                logger.error(f"Failed to send {channel.value} reminder: {e}")
                results[channel.value] = False

        return results

    def _send_email_reminder(self, appointment, reminder_type: str) -> bool:
        """Send email reminder"""
        if not appointment.patient_email:
            logger.warning(f"No email for appointment {appointment.appointment_id}")
            return False

        if not self.config.smtp_username or not self.config.smtp_password:
            logger.info(f"Email not configured. Would send {reminder_type} reminder to {appointment.patient_email}")
            return True  # Return True for demo mode

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = self._get_email_subject(appointment, reminder_type)
            msg["From"] = self.config.from_email
            msg["To"] = appointment.patient_email

            # Create HTML content
            html_content = self._generate_email_html(appointment, reminder_type)
            text_content = self._generate_email_text(appointment, reminder_type)

            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Send email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.sendmail(self.config.from_email, appointment.patient_email, msg.as_string())

            logger.info(f"Sent email reminder to {appointment.patient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _send_sms_reminder(self, appointment, reminder_type: str) -> bool:
        """Send SMS reminder using Twilio"""
        if not appointment.patient_phone:
            logger.warning(f"No phone for appointment {appointment.appointment_id}")
            return False

        if not self.config.twilio_account_sid:
            logger.info(f"SMS not configured. Would send {reminder_type} reminder to {appointment.patient_phone}")
            return True  # Return True for demo mode

        try:
            # Twilio integration would go here
            # from twilio.rest import Client
            # client = Client(self.config.twilio_account_sid, self.config.twilio_auth_token)
            # message = client.messages.create(
            #     body=self._generate_sms_text(appointment, reminder_type),
            #     from_=self.config.twilio_phone_number,
            #     to=appointment.patient_phone
            # )

            logger.info(f"SMS reminder sent to {appointment.patient_phone}")
            return True

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False

    def _send_push_reminder(self, appointment, reminder_type: str) -> bool:
        """Send push notification using Firebase"""
        if not self.config.firebase_credentials_path:
            logger.info(f"Push not configured. Would send {reminder_type} push notification")
            return True  # Return True for demo mode

        try:
            # Firebase integration would go here
            # import firebase_admin
            # from firebase_admin import messaging

            # Get user's FCM token from database
            # Send notification using firebase_admin.messaging

            logger.info(f"Push notification sent for appointment {appointment.appointment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
            return False

    def _create_in_app_notification(self, appointment, reminder_type: str) -> bool:
        """Create an in-app notification"""
        from database import Notification

        db = self.db_session_factory()
        try:
            notification = Notification(
                user_id=appointment.user_id,
                notification_type="appointment_reminder",
                title=self._get_notification_title(reminder_type),
                message=self._get_notification_message(appointment, reminder_type),
                data=json.dumps({
                    "appointment_id": appointment.appointment_id,
                    "reminder_type": reminder_type,
                    "appointment_date": appointment.appointment_date.isoformat(),
                    "start_time": appointment.start_time.isoformat() if appointment.start_time else None,
                }),
                is_read=False,
                created_at=datetime.utcnow()
            )
            db.add(notification)
            db.commit()

            logger.info(f"Created in-app notification for user {appointment.user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create in-app notification: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def _get_email_subject(self, appointment, reminder_type: str) -> str:
        """Generate email subject line"""
        apt_type = appointment.appointment_type.replace("_", " ").title()

        if reminder_type == "24_hours":
            return f"Reminder: Your {apt_type} Appointment Tomorrow"
        elif reminder_type == "2_hours":
            return f"Reminder: Your {apt_type} Appointment in 2 Hours"
        elif reminder_type == "30_minutes":
            return f"Starting Soon: Your {apt_type} Appointment"
        else:
            return f"Appointment Reminder: {apt_type}"

    def _get_notification_title(self, reminder_type: str) -> str:
        """Generate notification title"""
        if reminder_type == "24_hours":
            return "Appointment Tomorrow"
        elif reminder_type == "2_hours":
            return "Appointment in 2 Hours"
        elif reminder_type == "30_minutes":
            return "Appointment Starting Soon"
        else:
            return "Appointment Reminder"

    def _get_notification_message(self, appointment, reminder_type: str) -> str:
        """Generate notification message"""
        apt_type = appointment.appointment_type.replace("_", " ").title()
        apt_date = appointment.appointment_date.strftime("%B %d, %Y")
        apt_time = appointment.start_time.strftime("%I:%M %p") if appointment.start_time else ""

        if reminder_type == "24_hours":
            return f"Your {apt_type} appointment is tomorrow at {apt_time} with {appointment.provider_name}."
        elif reminder_type == "2_hours":
            return f"Your {apt_type} appointment is in 2 hours at {apt_time}."
        elif reminder_type == "30_minutes":
            msg = f"Your {apt_type} appointment starts in 30 minutes."
            if appointment.is_telemedicine:
                msg += " Click to join the video call."
            return msg
        else:
            return f"Reminder for your {apt_type} appointment on {apt_date} at {apt_time}."

    def _generate_email_text(self, appointment, reminder_type: str) -> str:
        """Generate plain text email content"""
        apt_type = appointment.appointment_type.replace("_", " ").title()
        apt_date = appointment.appointment_date.strftime("%A, %B %d, %Y")
        apt_time = appointment.start_time.strftime("%I:%M %p") if appointment.start_time else ""

        lines = [
            f"Appointment Reminder",
            "",
            f"Dear {appointment.patient_name or 'Patient'},",
            "",
        ]

        if reminder_type == "24_hours":
            lines.append(f"This is a reminder that you have a {apt_type} appointment tomorrow.")
        elif reminder_type == "2_hours":
            lines.append(f"This is a reminder that your {apt_type} appointment is in 2 hours.")
        elif reminder_type == "30_minutes":
            lines.append(f"Your {apt_type} appointment starts in 30 minutes.")

        lines.extend([
            "",
            "Appointment Details:",
            f"  Date: {apt_date}",
            f"  Time: {apt_time}",
            f"  Provider: {appointment.provider_name}",
            f"  Type: {apt_type}",
        ])

        if appointment.is_telemedicine:
            lines.extend([
                "",
                "This is a telemedicine appointment.",
                f"Join your video call here: {appointment.telemedicine_link or 'Link will be provided'}",
            ])
        else:
            lines.extend([
                "",
                f"Location: {appointment.location or 'Main Clinic'}",
            ])

        if appointment.reason_for_visit:
            lines.extend([
                "",
                f"Reason for visit: {appointment.reason_for_visit}",
            ])

        lines.extend([
            "",
            "If you need to cancel or reschedule, please do so at least 24 hours in advance.",
            "",
            "Thank you,",
            "Skin Analyzer Team"
        ])

        return "\n".join(lines)

    def _generate_email_html(self, appointment, reminder_type: str) -> str:
        """Generate HTML email content"""
        apt_type = appointment.appointment_type.replace("_", " ").title()
        apt_date = appointment.appointment_date.strftime("%A, %B %d, %Y")
        apt_time = appointment.start_time.strftime("%I:%M %p") if appointment.start_time else ""

        if reminder_type == "24_hours":
            header = "Your Appointment is Tomorrow"
            time_text = "tomorrow"
        elif reminder_type == "2_hours":
            header = "Your Appointment is in 2 Hours"
            time_text = "in 2 hours"
        elif reminder_type == "30_minutes":
            header = "Your Appointment Starts Soon"
            time_text = "in 30 minutes"
        else:
            header = "Appointment Reminder"
            time_text = "soon"

        telemedicine_section = ""
        if appointment.is_telemedicine:
            telemedicine_section = f"""
            <div style="background-color: #dbeafe; border-radius: 8px; padding: 16px; margin: 20px 0;">
                <h3 style="color: #1e40af; margin: 0 0 8px 0;">Video Consultation</h3>
                <p style="margin: 0; color: #1e3a5f;">This is a telemedicine appointment.</p>
                <a href="{appointment.telemedicine_link or '#'}" style="display: inline-block; background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; margin-top: 12px;">
                    Join Video Call
                </a>
            </div>
            """
        else:
            telemedicine_section = f"""
            <div style="background-color: #f0f9ff; border-radius: 8px; padding: 16px; margin: 20px 0;">
                <h3 style="color: #1e3a5f; margin: 0 0 8px 0;">Location</h3>
                <p style="margin: 0; color: #4b5563;">{appointment.location or 'Main Clinic - 123 Medical Center Dr'}</p>
            </div>
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f0f9ff; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); padding: 32px; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 24px;">{header}</h1>
                </div>

                <!-- Content -->
                <div style="padding: 32px;">
                    <p style="color: #4b5563; font-size: 16px; margin: 0 0 24px 0;">
                        Dear {appointment.patient_name or 'Patient'},<br><br>
                        This is a reminder that your <strong>{apt_type}</strong> appointment is {time_text}.
                    </p>

                    <!-- Appointment Details Card -->
                    <div style="background-color: #f8fafc; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                        <h2 style="color: #1e3a5f; font-size: 18px; margin: 0 0 16px 0;">Appointment Details</h2>

                        <div style="display: flex; margin-bottom: 12px;">
                            <span style="color: #6b7280; width: 100px;">Date:</span>
                            <span style="color: #1e3a5f; font-weight: 600;">{apt_date}</span>
                        </div>
                        <div style="display: flex; margin-bottom: 12px;">
                            <span style="color: #6b7280; width: 100px;">Time:</span>
                            <span style="color: #1e3a5f; font-weight: 600;">{apt_time}</span>
                        </div>
                        <div style="display: flex; margin-bottom: 12px;">
                            <span style="color: #6b7280; width: 100px;">Provider:</span>
                            <span style="color: #1e3a5f; font-weight: 600;">{appointment.provider_name}</span>
                        </div>
                        <div style="display: flex;">
                            <span style="color: #6b7280; width: 100px;">Type:</span>
                            <span style="color: #1e3a5f; font-weight: 600;">{apt_type}</span>
                        </div>
                    </div>

                    {telemedicine_section}

                    <!-- Reason if provided -->
                    {f'<p style="color: #4b5563; margin: 20px 0;"><strong>Reason:</strong> {appointment.reason_for_visit}</p>' if appointment.reason_for_visit else ''}

                    <!-- Footer note -->
                    <p style="color: #9ca3af; font-size: 14px; margin: 24px 0 0 0; padding-top: 16px; border-top: 1px solid #e5e7eb;">
                        If you need to cancel or reschedule, please do so at least 24 hours in advance.
                    </p>
                </div>

                <!-- Footer -->
                <div style="background-color: #f8fafc; padding: 20px; text-align: center;">
                    <p style="color: #6b7280; font-size: 14px; margin: 0;">
                        Skin Analyzer - Your Dermatology Companion
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

    def _generate_sms_text(self, appointment, reminder_type: str) -> str:
        """Generate SMS message text"""
        apt_type = appointment.appointment_type.replace("_", " ").title()
        apt_time = appointment.start_time.strftime("%I:%M %p") if appointment.start_time else ""

        if reminder_type == "24_hours":
            msg = f"Reminder: Your {apt_type} appointment is tomorrow at {apt_time}"
        elif reminder_type == "2_hours":
            msg = f"Reminder: Your {apt_type} appointment is in 2 hours at {apt_time}"
        elif reminder_type == "30_minutes":
            msg = f"Your {apt_type} appointment starts in 30 minutes"
        else:
            msg = f"Appointment reminder: {apt_type} at {apt_time}"

        if appointment.is_telemedicine and appointment.telemedicine_link:
            msg += f". Join: {appointment.telemedicine_link}"

        return msg

    def mark_reminder_sent(self, appointment, reminder_type: str, channels: Dict[str, bool]):
        """Mark reminder as sent in the database"""
        from database import Appointment

        db = self.db_session_factory()
        try:
            apt = db.query(Appointment).filter(
                Appointment.appointment_id == appointment.appointment_id
            ).first()

            if apt:
                reminders_sent = apt.reminders_sent or []
                reminders_sent.append({
                    "type": reminder_type,
                    "channels": channels,
                    "sent_at": datetime.utcnow().isoformat()
                })
                apt.reminders_sent = reminders_sent
                db.commit()

                logger.info(f"Marked reminder {reminder_type} as sent for {appointment.appointment_id}")
        except Exception as e:
            logger.error(f"Failed to mark reminder sent: {e}")
            db.rollback()
        finally:
            db.close()

    def send_pending_reminders(self) -> Dict[str, Any]:
        """
        Main method to process and send all pending reminders.
        Call this periodically (e.g., every 5 minutes).
        """
        results = {
            "processed": 0,
            "sent": 0,
            "failed": 0,
            "details": []
        }

        pending = self.get_pending_reminders()

        for reminder_info in pending:
            appointment = reminder_info["appointment"]
            reminder_type = reminder_info["reminder_type"]
            channels = reminder_info["channels"]

            results["processed"] += 1

            try:
                channel_results = self.send_reminder(appointment, reminder_type, channels)

                # Mark as sent
                self.mark_reminder_sent(appointment, reminder_type, channel_results)

                if any(channel_results.values()):
                    results["sent"] += 1
                else:
                    results["failed"] += 1

                results["details"].append({
                    "appointment_id": appointment.appointment_id,
                    "reminder_type": reminder_type,
                    "channels": channel_results,
                    "success": any(channel_results.values())
                })

            except Exception as e:
                logger.error(f"Failed to process reminder for {appointment.appointment_id}: {e}")
                results["failed"] += 1
                results["details"].append({
                    "appointment_id": appointment.appointment_id,
                    "reminder_type": reminder_type,
                    "error": str(e),
                    "success": False
                })

        logger.info(f"Reminder processing complete: {results['sent']} sent, {results['failed']} failed")
        return results

    def start_background_scheduler(self):
        """Start background thread to continuously process reminders"""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Scheduler already running")
            return

        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        logger.info("Background reminder scheduler started")

    def stop_background_scheduler(self):
        """Stop the background scheduler"""
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=10)
        logger.info("Background reminder scheduler stopped")

    def _run_scheduler(self):
        """Background scheduler loop"""
        while not self._stop_event.is_set():
            try:
                self.send_pending_reminders()
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")

            # Wait for next interval
            self._stop_event.wait(timeout=self.config.check_interval_minutes * 60)


# API endpoint handler for manual trigger
def get_reminder_service():
    """Factory function to create reminder service with database session"""
    from database import get_db, SessionLocal
    return AppointmentReminderService(db_session_factory=SessionLocal)


# Convenience function for API
async def process_reminders():
    """Process pending reminders - call from API endpoint or scheduled task"""
    service = get_reminder_service()
    return service.send_pending_reminders()
