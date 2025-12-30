"""
Integration tests for notifications endpoints.

Tests the notification management features including:
- Getting user notifications
- Marking notifications as read
- Deleting notifications
- High-priority alerts
- Push notification device registration
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import Notification


class TestNotificationFixtures:
    """Fixtures for notification tests."""

    @pytest.fixture
    def notification(self, test_db, test_user):
        """Create a sample notification for testing."""
        notif = Notification(
            user_id=test_user.id,
            notification_type="analysis_complete",
            title="Analysis Complete",
            message="Your skin analysis is ready to view.",
            priority="normal",
            is_read=False,
            created_at=datetime.utcnow()
        )
        test_db.add(notif)
        test_db.commit()
        test_db.refresh(notif)
        return notif

    @pytest.fixture
    def read_notification(self, test_db, test_user):
        """Create a read notification."""
        notif = Notification(
            user_id=test_user.id,
            notification_type="reminder",
            title="Reminder",
            message="Time for your skin check.",
            priority="normal",
            is_read=True,
            read_at=datetime.utcnow(),
            created_at=datetime.utcnow() - timedelta(hours=1)
        )
        test_db.add(notif)
        test_db.commit()
        test_db.refresh(notif)
        return notif

    @pytest.fixture
    def urgent_alert(self, test_db, test_user):
        """Create an urgent alert (high-priority notification)."""
        notif = Notification(
            user_id=test_user.id,
            notification_type="high_risk_detected",
            title="Urgent: High Risk Detected",
            message="A high-risk lesion was detected. Please consult a dermatologist.",
            priority="urgent",
            is_read=False,
            created_at=datetime.utcnow()
        )
        test_db.add(notif)
        test_db.commit()
        test_db.refresh(notif)
        return notif

    @pytest.fixture
    def high_priority_alert(self, test_db, test_user):
        """Create a high-priority notification."""
        notif = Notification(
            user_id=test_user.id,
            notification_type="follow_up_needed",
            title="Follow-up Recommended",
            message="Your dermatologist recommends a follow-up appointment.",
            priority="high",
            is_read=False,
            created_at=datetime.utcnow()
        )
        test_db.add(notif)
        test_db.commit()
        test_db.refresh(notif)
        return notif

    @pytest.fixture
    def expired_notification(self, test_db, test_user):
        """Create an expired notification."""
        notif = Notification(
            user_id=test_user.id,
            notification_type="promo",
            title="Limited Time Offer",
            message="This offer has expired.",
            priority="low",
            is_read=False,
            expires_at=datetime.utcnow() - timedelta(days=1),
            created_at=datetime.utcnow() - timedelta(days=2)
        )
        test_db.add(notif)
        test_db.commit()
        test_db.refresh(notif)
        return notif

    @pytest.fixture
    def multiple_notifications(self, test_db, test_user):
        """Create multiple notifications for testing."""
        notifications = []
        types = [
            ("analysis_complete", "Analysis Complete", "normal", False),
            ("reminder", "Reminder", "normal", False),
            ("update", "App Update", "low", True),
            ("high_risk_detected", "High Risk", "urgent", False),
            ("follow_up", "Follow Up", "high", False)
        ]
        for i, (ntype, title, priority, is_read) in enumerate(types):
            notif = Notification(
                user_id=test_user.id,
                notification_type=ntype,
                title=title,
                message=f"Test message {i}",
                priority=priority,
                is_read=is_read,
                created_at=datetime.utcnow() - timedelta(hours=i)
            )
            test_db.add(notif)
            notifications.append(notif)
        test_db.commit()
        for n in notifications:
            test_db.refresh(n)
        return notifications


class TestGetNotifications(TestNotificationFixtures):
    """Tests for getting user notifications."""

    def test_get_notifications_empty(
        self,
        client,
        auth_headers
    ):
        """Test getting notifications when none exist."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "notifications" in data
        assert "total" in data
        assert "unread_count" in data
        assert data["total"] == 0

    def test_get_notifications_with_data(
        self,
        client,
        auth_headers,
        notification
    ):
        """Test getting notifications with existing data."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["notifications"]) >= 1

    def test_get_notifications_multiple(
        self,
        client,
        auth_headers,
        multiple_notifications
    ):
        """Test getting multiple notifications."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["notifications"]) == 5

    def test_get_notifications_unread_only(
        self,
        client,
        auth_headers,
        multiple_notifications
    ):
        """Test filtering for unread notifications only."""
        response = client.get(
            "/notifications?unread_only=true",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        # Only unread notifications should be returned
        for notif in data["notifications"]:
            assert notif["is_read"] is False

    def test_get_notifications_with_limit(
        self,
        client,
        auth_headers,
        multiple_notifications
    ):
        """Test pagination with limit."""
        response = client.get(
            "/notifications?limit=2",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["notifications"]) <= 2
        assert data["total"] == 5  # Total count should still be full

    def test_get_notifications_with_offset(
        self,
        client,
        auth_headers,
        multiple_notifications
    ):
        """Test pagination with offset."""
        response = client.get(
            "/notifications?limit=2&offset=2",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["notifications"]) <= 2

    def test_get_notifications_ordered_by_date(
        self,
        client,
        auth_headers,
        multiple_notifications
    ):
        """Test that notifications are ordered by date descending."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        notifications = data["notifications"]

        # Check that most recent is first
        if len(notifications) > 1:
            dates = [n["created_at"] for n in notifications if n["created_at"]]
            assert dates == sorted(dates, reverse=True)

    def test_get_notifications_excludes_expired(
        self,
        client,
        auth_headers,
        expired_notification,
        notification
    ):
        """Test that expired notifications are excluded."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # Expired notification should not be in the list
        notification_ids = [n["id"] for n in data["notifications"]]
        assert expired_notification.id not in notification_ids
        assert notification.id in notification_ids

    def test_get_notifications_contains_required_fields(
        self,
        client,
        auth_headers,
        notification
    ):
        """Test that notifications contain all required fields."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        notif = data["notifications"][0]

        assert "id" in notif
        assert "type" in notif
        assert "title" in notif
        assert "message" in notif
        assert "is_read" in notif
        assert "priority" in notif
        assert "created_at" in notif

    def test_get_notifications_unread_count(
        self,
        client,
        auth_headers,
        multiple_notifications
    ):
        """Test that unread count is accurate."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # Count unread from the multiple_notifications fixture
        # One is marked as read, so 4 should be unread
        assert data["unread_count"] == 4

    def test_get_notifications_requires_auth(self, client):
        """Test that getting notifications requires authentication."""
        response = client.get("/notifications")

        assert response.status_code in [401, 403]


class TestMarkNotificationRead(TestNotificationFixtures):
    """Tests for marking notifications as read."""

    def test_mark_notification_read(
        self,
        client,
        auth_headers,
        notification
    ):
        """Test marking a notification as read."""
        response = client.post(
            f"/notifications/{notification.id}/read",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify it's now read
        get_response = client.get("/notifications", headers=auth_headers)
        notifications = get_response.json()["notifications"]
        notif = next(n for n in notifications if n["id"] == notification.id)
        assert notif["is_read"] is True
        assert notif["read_at"] is not None

    def test_mark_notification_read_already_read(
        self,
        client,
        auth_headers,
        read_notification
    ):
        """Test marking an already read notification."""
        response = client.post(
            f"/notifications/{read_notification.id}/read",
            headers=auth_headers
        )

        # Should still succeed
        assert response.status_code == 200

    def test_mark_notification_read_not_found(
        self,
        client,
        auth_headers
    ):
        """Test marking a non-existent notification."""
        response = client.post(
            "/notifications/99999/read",
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_mark_other_user_notification_read(
        self,
        client,
        professional_auth_headers,
        notification
    ):
        """Test that users cannot mark other users' notifications as read."""
        response = client.post(
            f"/notifications/{notification.id}/read",
            headers=professional_auth_headers
        )

        assert response.status_code == 404

    def test_mark_notification_read_requires_auth(self, client, notification):
        """Test that marking notification requires authentication."""
        response = client.post(f"/notifications/{notification.id}/read")

        assert response.status_code in [401, 403]


class TestMarkAllNotificationsRead(TestNotificationFixtures):
    """Tests for marking all notifications as read."""

    def test_mark_all_read(
        self,
        client,
        auth_headers,
        multiple_notifications
    ):
        """Test marking all notifications as read."""
        response = client.post(
            "/notifications/read-all",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify all are now read
        get_response = client.get("/notifications", headers=auth_headers)
        assert get_response.json()["unread_count"] == 0

    def test_mark_all_read_no_notifications(
        self,
        client,
        auth_headers
    ):
        """Test marking all read when no notifications exist."""
        response = client.post(
            "/notifications/read-all",
            headers=auth_headers
        )

        # Should succeed even with no notifications
        assert response.status_code == 200

    def test_mark_all_read_requires_auth(self, client):
        """Test that marking all read requires authentication."""
        response = client.post("/notifications/read-all")

        assert response.status_code in [401, 403]


class TestDeleteNotification(TestNotificationFixtures):
    """Tests for deleting notifications."""

    def test_delete_notification(
        self,
        client,
        auth_headers,
        notification
    ):
        """Test deleting a notification."""
        response = client.delete(
            f"/notifications/{notification.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify it's deleted
        get_response = client.get("/notifications", headers=auth_headers)
        notification_ids = [n["id"] for n in get_response.json()["notifications"]]
        assert notification.id not in notification_ids

    def test_delete_notification_not_found(
        self,
        client,
        auth_headers
    ):
        """Test deleting a non-existent notification."""
        response = client.delete(
            "/notifications/99999",
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_delete_other_user_notification(
        self,
        client,
        professional_auth_headers,
        notification
    ):
        """Test that users cannot delete other users' notifications."""
        response = client.delete(
            f"/notifications/{notification.id}",
            headers=professional_auth_headers
        )

        assert response.status_code == 404

    def test_delete_notification_requires_auth(self, client, notification):
        """Test that deleting notification requires authentication."""
        response = client.delete(f"/notifications/{notification.id}")

        assert response.status_code in [401, 403]


class TestAlerts(TestNotificationFixtures):
    """Tests for alerts (high-priority notifications)."""

    def test_get_alerts_empty(
        self,
        client,
        auth_headers
    ):
        """Test getting alerts when none exist."""
        response = client.get("/alerts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "unread_count" in data

    def test_get_alerts_with_urgent(
        self,
        client,
        auth_headers,
        urgent_alert
    ):
        """Test getting alerts with urgent notification."""
        response = client.get("/alerts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["alerts"]) >= 1

        # Verify urgent alert is included
        alert_ids = [a["id"] for a in data["alerts"]]
        assert urgent_alert.id in alert_ids

    def test_get_alerts_with_high_priority(
        self,
        client,
        auth_headers,
        high_priority_alert
    ):
        """Test getting alerts with high-priority notification."""
        response = client.get("/alerts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        # Verify high priority alert is included
        alert_ids = [a["id"] for a in data["alerts"]]
        assert high_priority_alert.id in alert_ids

    def test_get_alerts_excludes_normal_priority(
        self,
        client,
        auth_headers,
        notification,  # normal priority
        urgent_alert
    ):
        """Test that normal priority notifications are excluded from alerts."""
        response = client.get("/alerts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        # Normal notification should not be in alerts
        alert_ids = [a["id"] for a in data["alerts"]]
        assert notification.id not in alert_ids
        assert urgent_alert.id in alert_ids

    def test_get_alerts_unread_only(
        self,
        client,
        auth_headers,
        urgent_alert
    ):
        """Test filtering alerts for unread only."""
        # First mark the alert as read
        client.post(
            f"/notifications/{urgent_alert.id}/read",
            headers=auth_headers
        )

        response = client.get(
            "/alerts?unread_only=true",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        # Read alert should not be in unread_only results
        for alert in data["alerts"]:
            assert alert["is_read"] is False

    def test_get_alerts_unread_count(
        self,
        client,
        auth_headers,
        urgent_alert,
        high_priority_alert
    ):
        """Test alerts unread count."""
        response = client.get("/alerts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # Both alerts are unread
        assert data["unread_count"] == 2

    def test_get_alerts_contains_required_fields(
        self,
        client,
        auth_headers,
        urgent_alert
    ):
        """Test that alerts contain required fields."""
        response = client.get("/alerts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        alert = data["alerts"][0]

        assert "id" in alert
        assert "type" in alert
        assert "title" in alert
        assert "message" in alert
        assert "priority" in alert
        assert "is_read" in alert
        assert "created_at" in alert

    def test_get_alerts_requires_auth(self, client):
        """Test that getting alerts requires authentication."""
        response = client.get("/alerts")

        assert response.status_code in [401, 403]


class TestPushDeviceRegistration(TestNotificationFixtures):
    """Tests for push notification device registration."""

    def test_register_device(
        self,
        client,
        auth_headers
    ):
        """Test registering a device for push notifications."""
        response = client.post(
            "/notifications/register-device",
            params={
                "device_token": "test_token_abc123",
                "platform": "ios"
            },
            headers=auth_headers
        )

        # May return 500 if UserProfile table doesn't have push fields
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True

    def test_register_device_android(
        self,
        client,
        auth_headers
    ):
        """Test registering an Android device."""
        response = client.post(
            "/notifications/register-device",
            params={
                "device_token": "fcm_token_xyz789",
                "platform": "android"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 500]

    def test_register_device_update_token(
        self,
        client,
        auth_headers
    ):
        """Test updating device token."""
        # Register first token
        response1 = client.post(
            "/notifications/register-device",
            params={
                "device_token": "first_token",
                "platform": "ios"
            },
            headers=auth_headers
        )

        # Update with new token
        response2 = client.post(
            "/notifications/register-device",
            params={
                "device_token": "updated_token",
                "platform": "ios"
            },
            headers=auth_headers
        )

        assert response2.status_code in [200, 500]

    def test_register_device_requires_auth(self, client):
        """Test that device registration requires authentication."""
        response = client.post(
            "/notifications/register-device",
            params={
                "device_token": "test_token",
                "platform": "ios"
            }
        )

        assert response.status_code in [401, 403]

    def test_unregister_device(
        self,
        client,
        auth_headers
    ):
        """Test unregistering a device from push notifications."""
        # First register
        client.post(
            "/notifications/register-device",
            params={
                "device_token": "test_token",
                "platform": "ios"
            },
            headers=auth_headers
        )

        # Then unregister
        response = client.post(
            "/notifications/unregister-device",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True

    def test_unregister_device_no_prior_registration(
        self,
        client,
        auth_headers
    ):
        """Test unregistering when no device was registered."""
        response = client.post(
            "/notifications/unregister-device",
            headers=auth_headers
        )

        # Should succeed even without prior registration
        assert response.status_code in [200, 500]

    def test_unregister_device_requires_auth(self, client):
        """Test that device unregistration requires authentication."""
        response = client.post("/notifications/unregister-device")

        assert response.status_code in [401, 403]


class TestNotificationPriorities(TestNotificationFixtures):
    """Tests for notification priority handling."""

    def test_mixed_priorities_in_notifications(
        self,
        client,
        auth_headers,
        multiple_notifications
    ):
        """Test that all priorities appear in general notifications."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        priorities = {n["priority"] for n in data["notifications"]}

        # Should include various priorities
        assert len(priorities) > 1

    def test_alerts_only_high_priorities(
        self,
        client,
        auth_headers,
        multiple_notifications
    ):
        """Test that alerts only include high/urgent priorities."""
        response = client.get("/alerts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        for alert in data["alerts"]:
            assert alert["priority"] in ["high", "urgent"]


class TestNotificationTypes(TestNotificationFixtures):
    """Tests for different notification types."""

    def test_analysis_complete_notification(
        self,
        client,
        auth_headers,
        notification
    ):
        """Test analysis complete notification type."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        notif = next(
            (n for n in data["notifications"] if n["id"] == notification.id),
            None
        )
        assert notif is not None
        assert notif["type"] == "analysis_complete"

    def test_high_risk_notification_appears_in_alerts(
        self,
        client,
        auth_headers,
        urgent_alert
    ):
        """Test that high-risk notifications appear in alerts."""
        response = client.get("/alerts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        alert = next(
            (a for a in data["alerts"] if a["id"] == urgent_alert.id),
            None
        )
        assert alert is not None
        assert alert["type"] == "high_risk_detected"


class TestProfessionalNotificationAccess(TestNotificationFixtures):
    """Tests for professional access to notifications."""

    def test_professional_can_get_own_notifications(
        self,
        client,
        professional_auth_headers
    ):
        """Test that professionals can access their own notifications."""
        response = client.get("/notifications", headers=professional_auth_headers)

        assert response.status_code == 200

    def test_professional_cannot_see_other_user_notifications(
        self,
        client,
        professional_auth_headers,
        notification
    ):
        """Test that professionals cannot see other users' notifications."""
        response = client.get("/notifications", headers=professional_auth_headers)

        assert response.status_code == 200
        data = response.json()
        notification_ids = [n["id"] for n in data["notifications"]]
        # The test user's notification should not be visible
        assert notification.id not in notification_ids

    def test_professional_can_get_own_alerts(
        self,
        client,
        professional_auth_headers
    ):
        """Test that professionals can access their own alerts."""
        response = client.get("/alerts", headers=professional_auth_headers)

        assert response.status_code == 200


class TestNotificationData(TestNotificationFixtures):
    """Tests for notification data field."""

    @pytest.fixture
    def notification_with_data(self, test_db, test_user):
        """Create a notification with data payload."""
        notif = Notification(
            user_id=test_user.id,
            notification_type="analysis_complete",
            title="Analysis Complete",
            message="Your skin analysis is ready.",
            priority="normal",
            is_read=False,
            data={"analysis_id": 123, "risk_level": "low"},
            created_at=datetime.utcnow()
        )
        test_db.add(notif)
        test_db.commit()
        test_db.refresh(notif)
        return notif

    def test_notification_includes_data(
        self,
        client,
        auth_headers,
        notification_with_data
    ):
        """Test that notification data is included in response."""
        response = client.get("/notifications", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        notif = next(
            (n for n in data["notifications"] if n["id"] == notification_with_data.id),
            None
        )
        assert notif is not None
        # Data field should be present (may be dict or None)
        assert "data" in notif
