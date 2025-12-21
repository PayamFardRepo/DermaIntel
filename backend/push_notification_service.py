"""
Push Notification Service

Supports multiple platforms:
- Expo Push Notifications (for React Native/Expo apps)
- Firebase Cloud Messaging (FCM)
- Apple Push Notification Service (APNS) - via FCM

This service handles sending push notifications to mobile devices.
"""

import os
import json
import logging
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PushPlatform(Enum):
    EXPO = "expo"
    FCM = "fcm"
    APNS = "apns"


@dataclass
class PushNotificationConfig:
    """Configuration for push notification services"""
    # Expo settings
    expo_access_token: str = os.getenv("EXPO_ACCESS_TOKEN", "")

    # Firebase settings
    fcm_server_key: str = os.getenv("FCM_SERVER_KEY", "")
    fcm_project_id: str = os.getenv("FCM_PROJECT_ID", "")

    # Batch settings
    batch_size: int = 100  # Max notifications per batch


class PushNotificationService:
    """
    Service for sending push notifications to mobile devices.

    Supports Expo and Firebase Cloud Messaging.
    """

    EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
    FCM_URL = "https://fcm.googleapis.com/fcm/send"

    def __init__(self, config: Optional[PushNotificationConfig] = None):
        self.config = config or PushNotificationConfig()

    def send_notification(
        self,
        token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        platform: PushPlatform = PushPlatform.EXPO,
        badge: Optional[int] = None,
        sound: str = "default",
        priority: str = "high"
    ) -> Dict[str, Any]:
        """
        Send a push notification to a single device.

        Args:
            token: Device push token
            title: Notification title
            body: Notification body/message
            data: Additional data payload
            platform: Push platform (expo, fcm, apns)
            badge: Badge count for iOS
            sound: Notification sound
            priority: Notification priority (high, normal)

        Returns:
            Result dictionary with success status
        """
        if platform == PushPlatform.EXPO:
            return self._send_expo_notification(
                tokens=[token],
                title=title,
                body=body,
                data=data,
                badge=badge,
                sound=sound,
                priority=priority
            )
        elif platform == PushPlatform.FCM:
            return self._send_fcm_notification(
                token=token,
                title=title,
                body=body,
                data=data,
                priority=priority
            )
        else:
            return {"success": False, "error": f"Unsupported platform: {platform}"}

    def send_notifications_batch(
        self,
        notifications: List[Dict[str, Any]],
        platform: PushPlatform = PushPlatform.EXPO
    ) -> Dict[str, Any]:
        """
        Send multiple push notifications in a batch.

        Args:
            notifications: List of notification dicts with token, title, body, data
            platform: Push platform

        Returns:
            Results with success/failure counts
        """
        if platform == PushPlatform.EXPO:
            return self._send_expo_batch(notifications)
        else:
            # For FCM, send individually
            results = {"success": 0, "failed": 0, "errors": []}
            for notif in notifications:
                result = self.send_notification(
                    token=notif["token"],
                    title=notif["title"],
                    body=notif["body"],
                    data=notif.get("data"),
                    platform=platform
                )
                if result.get("success"):
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(result.get("error"))
            return results

    def _send_expo_notification(
        self,
        tokens: List[str],
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        badge: Optional[int] = None,
        sound: str = "default",
        priority: str = "high"
    ) -> Dict[str, Any]:
        """Send notification via Expo Push API"""

        # Build messages
        messages = []
        for token in tokens:
            # Validate Expo push token format
            if not token.startswith("ExponentPushToken[") and not token.startswith("ExpoPushToken["):
                logger.warning(f"Invalid Expo push token format: {token[:20]}...")
                continue

            message = {
                "to": token,
                "title": title,
                "body": body,
                "sound": sound,
                "priority": priority,
            }

            if data:
                message["data"] = data

            if badge is not None:
                message["badge"] = badge

            messages.append(message)

        if not messages:
            return {"success": False, "error": "No valid tokens"}

        # Prepare headers
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Content-Type": "application/json",
        }

        if self.config.expo_access_token:
            headers["Authorization"] = f"Bearer {self.config.expo_access_token}"

        try:
            # Send request
            response = requests.post(
                self.EXPO_PUSH_URL,
                headers=headers,
                json=messages,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                data = result.get("data", [])

                # Check for individual errors
                errors = []
                for i, item in enumerate(data):
                    if item.get("status") == "error":
                        errors.append({
                            "token": tokens[i] if i < len(tokens) else "unknown",
                            "message": item.get("message", "Unknown error"),
                            "details": item.get("details")
                        })

                if errors:
                    logger.warning(f"Some notifications failed: {errors}")

                return {
                    "success": len(data) - len(errors) > 0,
                    "sent": len(data) - len(errors),
                    "failed": len(errors),
                    "errors": errors
                }
            else:
                logger.error(f"Expo API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return {"success": False, "error": str(e)}

    def _send_expo_batch(self, notifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send batch of notifications via Expo Push API"""

        # Build messages
        messages = []
        for notif in notifications:
            token = notif.get("token", "")

            # Validate Expo push token format
            if not token.startswith("ExponentPushToken[") and not token.startswith("ExpoPushToken["):
                continue

            message = {
                "to": token,
                "title": notif.get("title", ""),
                "body": notif.get("body", ""),
                "sound": notif.get("sound", "default"),
                "priority": notif.get("priority", "high"),
            }

            if notif.get("data"):
                message["data"] = notif["data"]

            if notif.get("badge") is not None:
                message["badge"] = notif["badge"]

            messages.append(message)

        if not messages:
            return {"success": 0, "failed": len(notifications), "error": "No valid tokens"}

        # Prepare headers
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Content-Type": "application/json",
        }

        if self.config.expo_access_token:
            headers["Authorization"] = f"Bearer {self.config.expo_access_token}"

        # Split into batches
        results = {"success": 0, "failed": 0, "errors": []}

        for i in range(0, len(messages), self.config.batch_size):
            batch = messages[i:i + self.config.batch_size]

            try:
                response = requests.post(
                    self.EXPO_PUSH_URL,
                    headers=headers,
                    json=batch,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json().get("data", [])
                    for item in data:
                        if item.get("status") == "ok":
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                            results["errors"].append(item.get("message"))
                else:
                    results["failed"] += len(batch)
                    results["errors"].append(f"Batch failed: {response.status_code}")

            except requests.exceptions.RequestException as e:
                results["failed"] += len(batch)
                results["errors"].append(str(e))

        return results

    def _send_fcm_notification(
        self,
        token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        priority: str = "high"
    ) -> Dict[str, Any]:
        """Send notification via Firebase Cloud Messaging"""

        if not self.config.fcm_server_key:
            logger.info(f"FCM not configured. Would send to token: {token[:20]}...")
            return {"success": True, "demo_mode": True}

        headers = {
            "Authorization": f"key={self.config.fcm_server_key}",
            "Content-Type": "application/json",
        }

        message = {
            "to": token,
            "notification": {
                "title": title,
                "body": body,
            },
            "priority": priority,
        }

        if data:
            message["data"] = data

        try:
            response = requests.post(
                self.FCM_URL,
                headers=headers,
                json=message,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success", 0) > 0:
                    return {"success": True}
                else:
                    return {
                        "success": False,
                        "error": result.get("results", [{}])[0].get("error", "Unknown error")
                    }
            else:
                return {"success": False, "error": f"FCM error: {response.status_code}"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def send_appointment_reminder(
        self,
        token: str,
        appointment_id: str,
        appointment_type: str,
        appointment_datetime: str,
        reminder_type: str,  # "24_hours", "2_hours", "30_minutes"
        is_telemedicine: bool = False,
        telemedicine_link: Optional[str] = None,
        platform: PushPlatform = PushPlatform.EXPO
    ) -> Dict[str, Any]:
        """
        Send an appointment reminder push notification.

        This is a convenience method specifically for appointment reminders.
        """
        # Determine title and body based on reminder type
        type_name = appointment_type.replace("_", " ").title()

        if reminder_type == "24_hours":
            title = "Appointment Tomorrow"
            body = f"Your {type_name} appointment is tomorrow."
        elif reminder_type == "2_hours":
            title = "Appointment in 2 Hours"
            body = f"Your {type_name} appointment is in 2 hours."
        elif reminder_type == "30_minutes":
            title = "Appointment Starting Soon"
            body = f"Your {type_name} appointment starts in 30 minutes."
            if is_telemedicine:
                body += " Tap to join the video call."
        else:
            title = "Appointment Reminder"
            body = f"Reminder for your {type_name} appointment."

        # Build data payload
        data = {
            "type": "appointment_reminder",
            "appointment_id": appointment_id,
            "reminder_type": reminder_type,
            "is_telemedicine": str(is_telemedicine).lower(),
        }

        if telemedicine_link:
            data["telemedicine_link"] = telemedicine_link

        return self.send_notification(
            token=token,
            title=title,
            body=body,
            data=data,
            platform=platform,
            sound="default",
            priority="high"
        )


# Convenience function
def get_push_service() -> PushNotificationService:
    """Factory function to get push notification service"""
    return PushNotificationService()


# Test function
def test_push_notification():
    """Test sending a push notification (demo mode)"""
    service = get_push_service()

    # Test with a sample Expo token (won't actually send without valid token)
    result = service.send_notification(
        token="ExponentPushToken[TEST_TOKEN_HERE]",
        title="Test Notification",
        body="This is a test notification from Skin Analyzer",
        data={"screen": "appointments", "action": "view"},
        platform=PushPlatform.EXPO
    )

    print(f"Test result: {result}")
    return result


if __name__ == "__main__":
    test_push_notification()
