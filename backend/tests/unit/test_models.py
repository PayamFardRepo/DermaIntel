"""
Unit tests for database models.

Tests the database model definitions including:
- Model creation and field defaults
- Unique constraints
- Foreign key relationships
- Cascade deletions
- JSON field storage
"""

import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import (
    User,
    AnalysisHistory,
    UserProfile,
    LesionGroup
)
from auth import get_password_hash


class TestUserModel:
    """Tests for User model."""

    def test_user_creation_with_required_fields(self, test_db):
        """Test creating a user with required fields."""
        user = User(
            username="newuser",
            email="newuser@example.com",
            hashed_password=get_password_hash("Password123!")
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)

        assert user.id is not None
        assert user.username == "newuser"
        assert user.email == "newuser@example.com"
        assert user.is_active is True  # Default
        assert user.role == "patient"  # Default

    def test_user_creation_with_all_fields(self, test_db):
        """Test creating a user with all fields populated."""
        user = User(
            username="fulluser",
            email="fulluser@example.com",
            hashed_password=get_password_hash("Password123!"),
            full_name="Full User",
            age=35,
            gender="female",
            is_active=True,
            role="dermatologist",
            display_mode="professional",
            is_verified_professional=True,
            professional_license_number="MD999999",
            professional_license_state="NY",
            npi_number="9876543210",
            verification_date=datetime.utcnow(),
            device_token="test_token_123",
            phone_number="+1234567890"
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)

        assert user.id is not None
        assert user.full_name == "Full User"
        assert user.age == 35
        assert user.gender == "female"
        assert user.role == "dermatologist"
        assert user.is_verified_professional is True

    def test_user_unique_username_constraint(self, test_db, test_user):
        """Test that duplicate usernames are rejected."""
        duplicate_user = User(
            username=test_user.username,  # Same username
            email="different@example.com",
            hashed_password=get_password_hash("Password123!")
        )
        test_db.add(duplicate_user)

        with pytest.raises(IntegrityError):
            test_db.commit()

    def test_user_unique_email_constraint(self, test_db, test_user):
        """Test that duplicate emails are rejected."""
        duplicate_user = User(
            username="differentuser",
            email=test_user.email,  # Same email
            hashed_password=get_password_hash("Password123!")
        )
        test_db.add(duplicate_user)

        with pytest.raises(IntegrityError):
            test_db.commit()

    def test_user_default_values(self, test_db):
        """Test that default values are applied correctly."""
        user = User(
            username="defaultuser",
            email="default@example.com",
            hashed_password=get_password_hash("Password123!")
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)

        assert user.is_active is True
        assert user.role == "patient"
        assert user.display_mode == "simple"
        assert user.second_opinion_credits == 0
        assert user.is_verified_professional is False

    def test_user_created_at_auto_set(self, test_db):
        """Test that created_at is automatically set."""
        before_creation = datetime.utcnow()

        user = User(
            username="timestampuser",
            email="timestamp@example.com",
            hashed_password=get_password_hash("Password123!")
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)

        after_creation = datetime.utcnow()

        assert user.created_at is not None
        assert before_creation <= user.created_at <= after_creation


class TestAnalysisHistoryModel:
    """Tests for AnalysisHistory model."""

    def test_analysis_creation(self, test_db, test_user):
        """Test creating an analysis record."""
        analysis = AnalysisHistory(
            user_id=test_user.id,
            image_filename="test.jpg",
            analysis_type="full",
            is_lesion=True,
            binary_confidence=0.95,
            predicted_class="melanocytic_nevus",
            lesion_confidence=0.87,
            risk_level="low"
        )
        test_db.add(analysis)
        test_db.commit()
        test_db.refresh(analysis)

        assert analysis.id is not None
        assert analysis.user_id == test_user.id
        assert analysis.predicted_class == "melanocytic_nevus"

    def test_analysis_user_relationship(self, test_db, test_user):
        """Test that analysis is linked to user via relationship."""
        analysis = AnalysisHistory(
            user_id=test_user.id,
            image_filename="relationship_test.jpg",
            analysis_type="full"
        )
        test_db.add(analysis)
        test_db.commit()
        test_db.refresh(analysis)

        # Access via relationship
        assert analysis.user.id == test_user.id
        assert analysis.user.username == test_user.username

    def test_analysis_json_field_storage(self, test_db, test_user):
        """Test that JSON fields store and retrieve correctly."""
        probabilities = {
            "nv": 0.87,
            "mel": 0.05,
            "bkl": 0.03,
            "bcc": 0.02
        }
        differential = [
            {"condition": "nevus", "probability": 0.87},
            {"condition": "melanoma", "probability": 0.05}
        ]

        analysis = AnalysisHistory(
            user_id=test_user.id,
            image_filename="json_test.jpg",
            analysis_type="full",
            lesion_probabilities=probabilities,
            differential_diagnoses=differential
        )
        test_db.add(analysis)
        test_db.commit()
        test_db.refresh(analysis)

        # Verify JSON data is preserved
        assert analysis.lesion_probabilities == probabilities
        assert analysis.differential_diagnoses == differential
        assert analysis.lesion_probabilities["nv"] == 0.87

    def test_analysis_burn_fields(self, test_db, test_user):
        """Test burn classification fields."""
        analysis = AnalysisHistory(
            user_id=test_user.id,
            image_filename="burn_test.jpg",
            analysis_type="burn",
            burn_severity="second_degree",
            burn_confidence=0.82,
            burn_severity_level=2,
            is_burn_detected=True,
            burn_urgency="Medium urgency",
            burn_treatment_advice="Seek medical attention",
            burn_medical_attention_required=True
        )
        test_db.add(analysis)
        test_db.commit()
        test_db.refresh(analysis)

        assert analysis.burn_severity == "second_degree"
        assert analysis.burn_severity_level == 2
        assert analysis.burn_medical_attention_required is True

    def test_analysis_dermoscopy_fields(self, test_db, test_user):
        """Test dermoscopy analysis fields."""
        dermoscopy_data = {
            "seven_point_score": 3,
            "abcd_score": 4.2,
            "features": {
                "pigment_network": {"present": True, "atypical": False}
            }
        }

        analysis = AnalysisHistory(
            user_id=test_user.id,
            image_filename="dermoscopy_test.jpg",
            analysis_type="dermoscopy",
            dermoscopy_data=dermoscopy_data
        )
        test_db.add(analysis)
        test_db.commit()
        test_db.refresh(analysis)

        assert analysis.dermoscopy_data["seven_point_score"] == 3
        assert analysis.dermoscopy_data["features"]["pigment_network"]["present"] is True

    def test_analysis_body_location_fields(self, test_db, test_user):
        """Test body location tracking fields."""
        analysis = AnalysisHistory(
            user_id=test_user.id,
            image_filename="location_test.jpg",
            analysis_type="full",
            body_location="arm_left",
            body_sublocation="forearm",
            body_side="left",
            body_map_coordinates={"x": 120, "y": 450}
        )
        test_db.add(analysis)
        test_db.commit()
        test_db.refresh(analysis)

        assert analysis.body_location == "arm_left"
        assert analysis.body_sublocation == "forearm"
        assert analysis.body_map_coordinates["x"] == 120

    def test_analysis_biopsy_fields(self, test_db, test_user):
        """Test biopsy correlation fields."""
        analysis = AnalysisHistory(
            user_id=test_user.id,
            image_filename="biopsy_test.jpg",
            analysis_type="full",
            predicted_class="melanoma",
            biopsy_performed=True,
            biopsy_result="melanoma_in_situ",
            biopsy_date=datetime.utcnow(),
            biopsy_facility="Test Lab",
            pathologist_name="Dr. Smith",
            prediction_correct=True,
            accuracy_category="category_match"
        )
        test_db.add(analysis)
        test_db.commit()
        test_db.refresh(analysis)

        assert analysis.biopsy_performed is True
        assert analysis.biopsy_result == "melanoma_in_situ"
        assert analysis.prediction_correct is True


class TestUserProfileModel:
    """Tests for UserProfile model."""

    def test_profile_creation(self, test_db, test_user):
        """Test creating a user profile."""
        profile = UserProfile(
            user_id=test_user.id,
            gender="male",
            skin_type="III",  # Fitzpatrick type
            city="Los Angeles",
            state="CA",
            country="USA",
            zip_code="90001"
        )
        test_db.add(profile)
        test_db.commit()
        test_db.refresh(profile)

        assert profile.id is not None
        assert profile.user_id == test_user.id
        assert profile.skin_type == "III"

    def test_profile_user_relationship(self, test_db, test_user):
        """Test profile-user relationship."""
        profile = UserProfile(
            user_id=test_user.id,
            city="New York",
            state="NY"
        )
        test_db.add(profile)
        test_db.commit()
        test_db.refresh(profile)

        # Access user via relationship
        assert profile.user.username == test_user.username

    def test_profile_unique_per_user(self, test_db, test_user):
        """Test that only one profile per user is allowed."""
        profile1 = UserProfile(user_id=test_user.id, city="LA")
        test_db.add(profile1)
        test_db.commit()

        profile2 = UserProfile(user_id=test_user.id, city="NYC")
        test_db.add(profile2)

        with pytest.raises(IntegrityError):
            test_db.commit()

    def test_profile_default_values(self, test_db, test_user):
        """Test profile default values."""
        profile = UserProfile(user_id=test_user.id)
        test_db.add(profile)
        test_db.commit()
        test_db.refresh(profile)

        assert profile.country == "USA"
        assert profile.preferred_distance_miles == 50
        assert profile.total_analyses == 0


class TestLesionGroupModel:
    """Tests for LesionGroup model."""

    def test_lesion_group_creation(self, test_db, test_user):
        """Test creating a lesion group."""
        lesion = LesionGroup(
            user_id=test_user.id,
            lesion_name="Mole on left shoulder",
            lesion_description="Small brown spot noticed 2 years ago",
            body_location="back",
            body_sublocation="upper_back",
            body_side="left",
            monitoring_frequency="monthly"
        )
        test_db.add(lesion)
        test_db.commit()
        test_db.refresh(lesion)

        assert lesion.id is not None
        assert lesion.lesion_name == "Mole on left shoulder"
        assert lesion.monitoring_frequency == "monthly"

    def test_lesion_group_user_relationship(self, test_db, test_user):
        """Test lesion group-user relationship."""
        lesion = LesionGroup(
            user_id=test_user.id,
            lesion_name="Test lesion"
        )
        test_db.add(lesion)
        test_db.commit()
        test_db.refresh(lesion)

        assert lesion.user.username == test_user.username

    def test_lesion_group_default_values(self, test_db, test_user):
        """Test lesion group default values."""
        lesion = LesionGroup(
            user_id=test_user.id,
            lesion_name="Default test"
        )
        test_db.add(lesion)
        test_db.commit()
        test_db.refresh(lesion)

        assert lesion.total_analyses == 0
        assert lesion.change_detected is False
        assert lesion.requires_attention is False
        assert lesion.is_active is True
        assert lesion.archived is False

    def test_lesion_group_analysis_relationship(self, test_db, test_user):
        """Test lesion group-analysis relationship."""
        lesion = LesionGroup(
            user_id=test_user.id,
            lesion_name="Tracked lesion"
        )
        test_db.add(lesion)
        test_db.commit()
        test_db.refresh(lesion)

        # Create analysis linked to lesion group
        analysis = AnalysisHistory(
            user_id=test_user.id,
            lesion_group_id=lesion.id,
            image_filename="tracked_lesion.jpg",
            analysis_type="full"
        )
        test_db.add(analysis)
        test_db.commit()
        test_db.refresh(analysis)
        test_db.refresh(lesion)

        # Verify relationship
        assert analysis.lesion_group_id == lesion.id
        assert len(lesion.analyses) == 1
        assert lesion.analyses[0].id == analysis.id


class TestCascadeDelete:
    """Tests for cascade delete behavior."""

    def test_user_deletion_cascades_to_analyses(self, test_db):
        """Test that deleting a user cascades to their analyses."""
        # Create user with analysis
        user = User(
            username="cascade_test",
            email="cascade@test.com",
            hashed_password=get_password_hash("Password123!")
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)

        analysis = AnalysisHistory(
            user_id=user.id,
            image_filename="cascade_test.jpg",
            analysis_type="full"
        )
        test_db.add(analysis)
        test_db.commit()

        analysis_id = analysis.id

        # Delete user
        test_db.delete(user)
        test_db.commit()

        # Verify analysis was also deleted
        remaining = test_db.query(AnalysisHistory).filter(
            AnalysisHistory.id == analysis_id
        ).first()
        assert remaining is None


class TestModelQueries:
    """Tests for common model queries."""

    def test_query_user_by_username(self, test_db, test_user):
        """Test querying user by username."""
        found = test_db.query(User).filter(
            User.username == test_user.username
        ).first()

        assert found is not None
        assert found.id == test_user.id

    def test_query_user_by_email(self, test_db, test_user):
        """Test querying user by email."""
        found = test_db.query(User).filter(
            User.email == test_user.email
        ).first()

        assert found is not None
        assert found.id == test_user.id

    def test_query_analyses_by_user(self, test_db, test_user, multiple_analyses):
        """Test querying all analyses for a user."""
        analyses = test_db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == test_user.id
        ).all()

        assert len(analyses) == len(multiple_analyses)

    def test_query_high_risk_analyses(self, test_db, test_user, multiple_analyses):
        """Test querying high-risk analyses."""
        high_risk = test_db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == test_user.id,
            AnalysisHistory.risk_level == "high"
        ).all()

        expected_count = sum(
            1 for a in multiple_analyses if a.risk_level == "high"
        )
        assert len(high_risk) == expected_count

    def test_query_active_users(self, test_db, test_user, inactive_user):
        """Test querying active users only."""
        active_users = test_db.query(User).filter(
            User.is_active == True
        ).all()

        usernames = [u.username for u in active_users]
        assert test_user.username in usernames
        assert inactive_user.username not in usernames
