"""
Integration tests for authentication endpoints.

Tests the complete auth flow including:
- User registration
- User login
- Token-based authentication
- Profile management
"""

import pytest
from datetime import timedelta
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestUserRegistration:
    """Tests for user registration endpoint."""

    def test_register_success(self, client):
        """Test successful user registration."""
        response = client.post("/register", json={
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "SecurePassword123!",
            "full_name": "New User"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert data["is_active"] is True
        assert "password" not in data  # Password should not be in response
        assert "hashed_password" not in data

    def test_register_with_account_type(self, client):
        """Test registration with professional account type."""
        response = client.post("/register", json={
            "username": "druser",
            "email": "dr@example.com",
            "password": "DoctorPass123!",
            "full_name": "Dr. User",
            "account_type": "professional",
            "display_mode": "professional"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["account_type"] == "professional"
        assert data["display_mode"] == "professional"

    def test_register_duplicate_username(self, client, test_user):
        """Test that duplicate username is rejected."""
        response = client.post("/register", json={
            "username": test_user.username,  # Same username
            "email": "different@example.com",
            "password": "Password123!"
        })

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_register_duplicate_email(self, client, test_user):
        """Test that duplicate email is rejected."""
        response = client.post("/register", json={
            "username": "differentuser",
            "email": test_user.email,  # Same email
            "password": "Password123!"
        })

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_register_missing_required_fields(self, client):
        """Test registration with missing required fields."""
        # Missing username
        response = client.post("/register", json={
            "email": "test@example.com",
            "password": "Password123!"
        })
        assert response.status_code == 422  # Validation error

        # Missing email
        response = client.post("/register", json={
            "username": "testuser",
            "password": "Password123!"
        })
        assert response.status_code == 422

        # Missing password
        response = client.post("/register", json={
            "username": "testuser",
            "email": "test@example.com"
        })
        assert response.status_code == 422

    def test_register_invalid_email_format(self, client):
        """Test registration with invalid email format."""
        response = client.post("/register", json={
            "username": "testuser",
            "email": "not-an-email",
            "password": "Password123!"
        })
        assert response.status_code == 422


class TestUserLogin:
    """Tests for user login endpoint."""

    def test_login_success(self, client, test_user, test_user_data):
        """Test successful login returns token."""
        response = client.post("/login", json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert len(data["access_token"]) > 0

    def test_login_invalid_password(self, client, test_user, test_user_data):
        """Test login with wrong password."""
        response = client.post("/login", json={
            "username": test_user_data["username"],
            "password": "WrongPassword123!"
        })

        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user."""
        response = client.post("/login", json={
            "username": "nonexistent",
            "password": "Password123!"
        })

        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    def test_login_empty_credentials(self, client):
        """Test login with empty credentials."""
        response = client.post("/login", json={
            "username": "",
            "password": ""
        })

        # Should fail validation or auth
        assert response.status_code in [401, 422]

    def test_login_case_sensitive_username(self, client, test_user, test_user_data):
        """Test that username is case-sensitive."""
        response = client.post("/login", json={
            "username": test_user_data["username"].upper(),
            "password": test_user_data["password"]
        })

        # Should fail because username doesn't match
        assert response.status_code == 401


class TestGetCurrentUser:
    """Tests for /me endpoint."""

    def test_get_current_user_success(self, client, auth_headers, test_user):
        """Test getting current user with valid token."""
        response = client.get("/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user.username
        assert data["email"] == test_user.email
        assert data["id"] == test_user.id

    def test_get_current_user_no_token(self, client):
        """Test getting current user without token."""
        response = client.get("/me")

        assert response.status_code in [401, 403]

    def test_get_current_user_invalid_token(self, client):
        """Test getting current user with invalid token."""
        response = client.get("/me", headers={
            "Authorization": "Bearer invalid_token_here"
        })

        assert response.status_code == 401

    def test_get_current_user_expired_token(self, client, expired_token):
        """Test getting current user with expired token."""
        response = client.get("/me", headers={
            "Authorization": f"Bearer {expired_token}"
        })

        assert response.status_code == 401

    def test_get_current_user_malformed_auth_header(self, client):
        """Test with malformed authorization header."""
        # Missing "Bearer" prefix
        response = client.get("/me", headers={
            "Authorization": "some_token"
        })
        assert response.status_code in [401, 403, 422]

        # Empty bearer
        response = client.get("/me", headers={
            "Authorization": "Bearer "
        })
        assert response.status_code in [401, 403, 422]


class TestUpdateUserSettings:
    """Tests for /me/settings endpoint."""

    def test_update_display_mode(self, client, auth_headers):
        """Test updating display mode."""
        response = client.put(
            "/me/settings?display_mode=professional",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["display_mode"] == "professional"

    def test_update_account_type(self, client, auth_headers):
        """Test updating account type."""
        response = client.put(
            "/me/settings?account_type=professional",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["account_type"] == "professional"

    def test_update_invalid_display_mode(self, client, auth_headers):
        """Test updating with invalid display mode."""
        response = client.put(
            "/me/settings?display_mode=invalid",
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_update_invalid_account_type(self, client, auth_headers):
        """Test updating with invalid account type."""
        response = client.put(
            "/me/settings?account_type=invalid",
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_update_settings_without_auth(self, client):
        """Test updating settings without authentication."""
        response = client.put("/me/settings?display_mode=professional")

        assert response.status_code in [401, 403]


class TestExtendedUserInfo:
    """Tests for /me/extended endpoint."""

    def test_get_extended_info_success(self, client, auth_headers, test_user):
        """Test getting extended user info."""
        response = client.get("/me/extended", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user.username
        assert "total_analyses" in data
        assert "last_analysis_date" in data

    def test_get_extended_info_with_analyses(self, client, auth_headers, multiple_analyses):
        """Test extended info includes analysis count."""
        response = client.get("/me/extended", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["total_analyses"] == len(multiple_analyses)

    def test_get_extended_info_no_auth(self, client):
        """Test extended info requires authentication."""
        response = client.get("/me/extended")

        assert response.status_code in [401, 403]


class TestProfessionalVerification:
    """Tests for professional verification endpoints."""

    def test_submit_verification_request(self, client, auth_headers):
        """Test submitting professional verification."""
        response = client.post(
            "/me/professional-verification",
            data={
                "license_number": "MD123456",
                "license_state": "CA",
                "npi_number": "1234567890"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["license_number"] == "MD123456"
        assert data["license_state"] == "CA"

    def test_submit_verification_without_npi(self, client, auth_headers):
        """Test verification request without optional NPI."""
        response = client.post(
            "/me/professional-verification",
            data={
                "license_number": "MD123456",
                "license_state": "NY"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["license_number"] == "MD123456"

    def test_get_professional_status(self, client, professional_auth_headers):
        """Test getting professional verification status."""
        response = client.get(
            "/me/professional-status",
            headers=professional_auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "is_verified" in data or "is_verified_professional" in data

    def test_verification_requires_auth(self, client):
        """Test verification endpoints require authentication."""
        response = client.post(
            "/me/professional-verification",
            data={
                "license_number": "MD123456",
                "license_state": "CA"
            }
        )

        assert response.status_code in [401, 403]


class TestInactiveUser:
    """Tests for inactive user handling."""

    def test_inactive_user_cannot_access_protected_endpoints(self, client, inactive_user, test_db):
        """Test that inactive users cannot access protected endpoints."""
        from auth import create_access_token
        from datetime import timedelta

        # Create token for inactive user
        token = create_access_token(
            data={"sub": inactive_user.username},
            expires_delta=timedelta(minutes=30)
        )

        response = client.get("/me", headers={
            "Authorization": f"Bearer {token}"
        })

        # Should be rejected because user is inactive
        assert response.status_code == 400


class TestTokenFlow:
    """Tests for the complete token authentication flow."""

    def test_full_auth_flow(self, client):
        """Test complete registration -> login -> access flow."""
        # 1. Register
        register_response = client.post("/register", json={
            "username": "flowuser",
            "email": "flow@example.com",
            "password": "FlowPass123!",
            "full_name": "Flow User"
        })
        assert register_response.status_code == 200

        # 2. Login
        login_response = client.post("/login", json={
            "username": "flowuser",
            "password": "FlowPass123!"
        })
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]

        # 3. Access protected endpoint
        me_response = client.get("/me", headers={
            "Authorization": f"Bearer {token}"
        })
        assert me_response.status_code == 200
        assert me_response.json()["username"] == "flowuser"

    def test_token_used_across_multiple_requests(self, client, auth_headers):
        """Test that the same token works for multiple requests."""
        # First request
        response1 = client.get("/me", headers=auth_headers)
        assert response1.status_code == 200

        # Second request with same token
        response2 = client.get("/me", headers=auth_headers)
        assert response2.status_code == 200

        # Third request
        response3 = client.get("/me/extended", headers=auth_headers)
        assert response3.status_code == 200
