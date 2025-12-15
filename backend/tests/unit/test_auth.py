"""
Unit tests for authentication functions.

Tests the core authentication logic including:
- Password hashing and verification
- JWT token creation and validation
- User retrieval functions
"""

import pytest
from datetime import datetime, timedelta
from jose import jwt, JWTError
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_user,
    authenticate_user,
    SECRET_KEY,
    ALGORITHM,
    TokenData
)


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_password_hashing_creates_hash(self):
        """Test that password hashing creates a non-empty hash."""
        password = "TestPassword123!"
        hashed = get_password_hash(password)

        assert hashed is not None
        assert len(hashed) > 0
        assert hashed != password  # Hash should not equal plain password

    def test_password_hashing_creates_unique_hashes(self):
        """Test that the same password creates different hashes (due to salting)."""
        password = "TestPassword123!"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # Bcrypt uses random salt, so hashes should differ
        assert hash1 != hash2

    def test_password_hashing_handles_special_characters(self):
        """Test password hashing with special characters."""
        password = "P@$$w0rd!#$%^&*()_+{}[]|\\:\";<>?,./"
        hashed = get_password_hash(password)

        assert hashed is not None
        assert verify_password(password, hashed)

    def test_password_hashing_handles_unicode(self):
        """Test password hashing with unicode characters."""
        password = "Passw0rd\u00e9\u00f1\u00fc\u00df"
        hashed = get_password_hash(password)

        assert hashed is not None
        assert verify_password(password, hashed)

    def test_password_hashing_handles_long_passwords(self):
        """Test password hashing with very long passwords (bcrypt truncates at 72 bytes)."""
        # Bcrypt has a 72-byte limit
        long_password = "A" * 100
        hashed = get_password_hash(long_password)

        assert hashed is not None
        # Should work with the truncated version
        assert verify_password(long_password, hashed)


class TestPasswordVerification:
    """Tests for password verification functions."""

    def test_verify_correct_password(self):
        """Test that correct password verifies successfully."""
        password = "TestPassword123!"
        hashed = get_password_hash(password)

        assert verify_password(password, hashed) is True

    def test_verify_incorrect_password(self):
        """Test that incorrect password fails verification."""
        password = "TestPassword123!"
        wrong_password = "WrongPassword456!"
        hashed = get_password_hash(password)

        assert verify_password(wrong_password, hashed) is False

    def test_verify_empty_password(self):
        """Test verification with empty password."""
        password = "TestPassword123!"
        hashed = get_password_hash(password)

        assert verify_password("", hashed) is False

    def test_verify_case_sensitive(self):
        """Test that password verification is case-sensitive."""
        password = "TestPassword123!"
        hashed = get_password_hash(password)

        assert verify_password("testpassword123!", hashed) is False
        assert verify_password("TESTPASSWORD123!", hashed) is False


class TestJWTTokenCreation:
    """Tests for JWT token creation."""

    def test_create_token_with_default_expiration(self):
        """Test token creation with default expiration."""
        data = {"sub": "testuser"}
        token = create_access_token(data)

        assert token is not None
        assert len(token) > 0

        # Decode and verify
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert "exp" in payload

    def test_create_token_with_custom_expiration(self):
        """Test token creation with custom expiration delta."""
        data = {"sub": "testuser"}
        expires_delta = timedelta(hours=2)
        token = create_access_token(data, expires_delta=expires_delta)

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Verify expiration is approximately 2 hours from now
        exp_time = datetime.utcfromtimestamp(payload["exp"])
        expected_exp = datetime.utcnow() + expires_delta
        # Allow 10 seconds tolerance
        assert abs((exp_time - expected_exp).total_seconds()) < 10

    def test_create_token_with_additional_claims(self):
        """Test token creation with additional custom claims."""
        data = {
            "sub": "testuser",
            "role": "admin",
            "user_id": 123
        }
        token = create_access_token(data)

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"
        assert payload["role"] == "admin"
        assert payload["user_id"] == 123

    def test_token_is_valid_jwt_format(self):
        """Test that created token has valid JWT format (3 parts separated by dots)."""
        data = {"sub": "testuser"}
        token = create_access_token(data)

        parts = token.split(".")
        assert len(parts) == 3  # Header, payload, signature


class TestJWTTokenValidation:
    """Tests for JWT token validation."""

    def test_valid_token_decodes_successfully(self):
        """Test that valid token decodes without errors."""
        data = {"sub": "testuser"}
        token = create_access_token(data, expires_delta=timedelta(minutes=30))

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "testuser"

    def test_expired_token_raises_error(self):
        """Test that expired token raises JWTError."""
        data = {"sub": "testuser"}
        # Create token that expired 10 minutes ago
        token = create_access_token(data, expires_delta=timedelta(minutes=-10))

        with pytest.raises(JWTError):
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    def test_invalid_signature_raises_error(self):
        """Test that token with invalid signature raises error."""
        data = {"sub": "testuser"}
        token = create_access_token(data)

        # Try to decode with wrong secret
        with pytest.raises(JWTError):
            jwt.decode(token, "wrong-secret-key", algorithms=[ALGORITHM])

    def test_invalid_algorithm_raises_error(self):
        """Test that decoding with wrong algorithm raises error."""
        data = {"sub": "testuser"}
        token = create_access_token(data)

        with pytest.raises(JWTError):
            jwt.decode(token, SECRET_KEY, algorithms=["HS384"])

    def test_tampered_token_raises_error(self):
        """Test that tampered token raises error."""
        data = {"sub": "testuser"}
        token = create_access_token(data)

        # Tamper with the token by changing a character
        tampered = token[:-1] + ("X" if token[-1] != "X" else "Y")

        with pytest.raises(JWTError):
            jwt.decode(tampered, SECRET_KEY, algorithms=[ALGORITHM])


class TestTokenData:
    """Tests for TokenData class."""

    def test_token_data_initialization(self):
        """Test TokenData initialization."""
        token_data = TokenData(username="testuser")
        assert token_data.username == "testuser"

    def test_token_data_default_none(self):
        """Test TokenData defaults to None."""
        token_data = TokenData()
        assert token_data.username is None


class TestGetUser:
    """Tests for get_user function."""

    def test_get_existing_user(self, test_db, test_user):
        """Test getting an existing user by username."""
        user = get_user(test_db, test_user.username)

        assert user is not None
        assert user.username == test_user.username
        assert user.email == test_user.email

    def test_get_nonexistent_user(self, test_db):
        """Test getting a non-existent user returns None."""
        user = get_user(test_db, "nonexistent_user")
        assert user is None

    def test_get_user_case_sensitive(self, test_db, test_user):
        """Test that username lookup is case-sensitive."""
        # Should not find user with different case
        user = get_user(test_db, test_user.username.upper())
        assert user is None


class TestAuthenticateUser:
    """Tests for authenticate_user function."""

    def test_authenticate_valid_credentials(self, test_db, test_user, test_user_data):
        """Test authentication with valid credentials."""
        user = authenticate_user(
            test_db,
            test_user_data["username"],
            test_user_data["password"]
        )

        assert user is not False
        assert user.username == test_user_data["username"]

    def test_authenticate_wrong_password(self, test_db, test_user, test_user_data):
        """Test authentication with wrong password."""
        result = authenticate_user(
            test_db,
            test_user_data["username"],
            "WrongPassword123!"
        )

        assert result is False

    def test_authenticate_nonexistent_user(self, test_db):
        """Test authentication with non-existent user."""
        result = authenticate_user(
            test_db,
            "nonexistent_user",
            "SomePassword123!"
        )

        assert result is False

    def test_authenticate_empty_password(self, test_db, test_user, test_user_data):
        """Test authentication with empty password."""
        result = authenticate_user(
            test_db,
            test_user_data["username"],
            ""
        )

        assert result is False

    def test_authenticate_empty_username(self, test_db):
        """Test authentication with empty username."""
        result = authenticate_user(
            test_db,
            "",
            "SomePassword123!"
        )

        assert result is False
