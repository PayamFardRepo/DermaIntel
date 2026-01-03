from sqlalchemy import create_engine, Column, Integer, String, DateTime, Date, Time, Boolean, Text, JSON, Float, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database configuration with environment variable support
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./skin_classifier.db"  # Default to SQLite for development
)

# Handle PostgreSQL URL format differences
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with appropriate connection args
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    age = Column(Integer)  # User age for demographic analysis
    gender = Column(String)  # User gender for demographic patterns
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Second opinion credits
    second_opinion_credits = Column(Integer, default=0)  # Credits for second opinions

    # User role (legacy - kept for backwards compatibility)
    role = Column(String, default="patient")  # "patient", "dermatologist", "admin"

    # NEW: Account type for hybrid model
    # "personal" = Regular user (patient or doctor using for themselves)
    # "personal+clinic_admin" = User who also administers a clinic
    # "personal+clinic_provider" = User who is a provider at a clinic
    # "personal+clinic_staff" = User who is staff at a clinic
    account_type = Column(String, default="personal")

    # NEW: Display mode preference
    # "simple" = Patient-friendly view with plain English
    # "professional" = Dermatologist view with ABCDE scores, calibrated metrics, etc.
    display_mode = Column(String, default="simple")

    # NEW: Professional verification (for clinic features)
    is_verified_professional = Column(Boolean, default=False)
    professional_license_number = Column(String)
    professional_license_state = Column(String)
    npi_number = Column(String)  # National Provider Identifier
    verification_date = Column(DateTime)
    verification_documents = Column(JSON)  # Uploaded verification docs

    # Device token for push notifications
    device_token = Column(String)  # Firebase/Expo push token
    phone_number = Column(String)  # For SMS notifications

    # Relationship to analysis history
    analyses = relationship("AnalysisHistory", back_populates="user", cascade="all, delete-orphan")


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=True, index=True)  # Link to lesion group for tracking

    # Analysis metadata
    image_filename = Column(String)
    image_url = Column(String)  # For storing image path/URL
    analysis_type = Column(String, nullable=False)  # "binary", "full", "detailed"

    # Binary analysis results
    is_lesion = Column(Boolean)
    binary_confidence = Column(Float)
    binary_probabilities = Column(JSON)  # Store non_lesion and lesion probabilities

    # Full analysis results (when lesion is detected)
    predicted_class = Column(String)  # The main diagnosis
    lesion_confidence = Column(Float)
    lesion_probabilities = Column(JSON)  # Store all class probabilities

    # Inflammatory condition analysis results
    inflammatory_condition = Column(String)  # Detected inflammatory condition
    inflammatory_confidence = Column(Float)
    inflammatory_probabilities = Column(JSON)  # Store all inflammatory condition probabilities

    # Infectious disease analysis results
    infectious_disease = Column(String)  # Detected infectious disease (bacterial, fungal, viral, parasitic)
    infectious_confidence = Column(Float)
    infectious_probabilities = Column(JSON)  # Store all infectious disease class probabilities
    infection_type = Column(String)  # "bacterial", "fungal", "viral", "parasitic", "none"
    infectious_severity = Column(String)  # "mild", "moderate", "severe"
    contagious = Column(Boolean)  # Is the infection contagious
    transmission_risk = Column(String)  # "low", "medium", "high"

    # Burn severity classification results
    burn_severity = Column(String)  # Detected burn severity class
    burn_confidence = Column(Float)  # Confidence score for burn classification
    burn_probabilities = Column(JSON)  # Store all burn severity class probabilities
    burn_severity_level = Column(Integer)  # 0-3 (Normal, 1st, 2nd, 3rd degree)
    burn_urgency = Column(String)  # Urgency message for burn
    burn_treatment_advice = Column(String)  # Treatment recommendations for burn
    burn_medical_attention_required = Column(Boolean)  # Emergency flag for severe burns
    is_burn_detected = Column(Boolean)  # Any burn vs healthy skin
    burn_overall_probability = Column(Float)  # Probability of any burn
    burn_severe_probability = Column(Float)  # Probability of 2nd/3rd degree burn

    # Differential diagnoses
    differential_diagnoses = Column(JSON)  # Store differential diagnosis lists for lesion, inflammatory, and infectious

    # Treatment recommendations
    treatment_recommendations = Column(JSON)  # Store evidence-based treatment suggestions

    # Clinical Decision Support System
    clinical_decision_support = Column(JSON)  # Store complete clinical protocols, medications, drug interactions, etc.

    # Literature References
    literature_references = Column(JSON)  # Store literature references for lesion and inflammatory conditions

    # Insurance Pre-Authorization
    insurance_preauthorization = Column(JSON)  # Store comprehensive insurance pre-authorization documentation
    preauth_status = Column(String)  # DRAFT, SUBMITTED, UNDER_REVIEW, APPROVED, DENIED, ADDITIONAL_INFO_REQUIRED
    preauth_submitted_date = Column(DateTime)  # Date submitted to insurance
    preauth_decision_date = Column(DateTime)  # Date insurance made decision
    preauth_approval_notes = Column(Text)  # Notes about approval/denial

    # Risk assessment
    risk_level = Column(String)  # "low", "medium", "high"
    risk_recommendation = Column(Text)

    # Image quality metrics
    image_quality_score = Column(Float)
    image_quality_passed = Column(Boolean)
    quality_issues = Column(JSON)  # Store detected quality issues

    # Processing metadata
    processing_time_seconds = Column(Float)
    model_version = Column(String)

    # Red flag indicators (ABCDE criteria)
    red_flag_data = Column(JSON)  # Store ABCDE analysis results

    # Dermoscopic Feature Analysis
    dermoscopy_data = Column(JSON)  # Store comprehensive dermoscopic analysis (7-point checklist, ABCD score, features)

    # Explainability
    explainability_heatmap = Column(Text)  # Store base64-encoded heatmap image

    # Calibration & Measurement
    calibration_found = Column(Boolean)
    calibration_type = Column(String)  # "coin", "ruler", "manual", etc.
    pixels_per_mm = Column(Float)  # Conversion ratio
    calibration_confidence = Column(Float)
    calibration_data = Column(JSON)  # Store detected calibration objects
    measurements = Column(JSON)  # Store user-drawn measurements with real-world dimensions

    # Biopsy Correlation - Track actual pathology results vs AI predictions
    biopsy_performed = Column(Boolean, default=False)  # Whether biopsy was performed
    biopsy_result = Column(String)  # Actual pathology diagnosis
    biopsy_date = Column(DateTime)  # Date biopsy was performed
    biopsy_notes = Column(Text)  # Additional pathology notes
    biopsy_facility = Column(String)  # Lab/facility that performed biopsy
    pathologist_name = Column(String)  # Pathologist who reviewed the sample
    prediction_correct = Column(Boolean)  # Whether AI prediction matched biopsy
    accuracy_category = Column(String)  # "exact_match", "category_match", "mismatch", "pending"

    # Histopathology Analysis - AI analysis of biopsy slide images
    histopathology_performed = Column(Boolean, default=False)  # Whether histopathology analysis was done
    histopathology_result = Column(String)  # Primary histopathology diagnosis from AI
    histopathology_malignant = Column(Boolean)  # Whether AI detected malignancy
    histopathology_confidence = Column(Float)  # AI confidence score (0-1)
    histopathology_date = Column(DateTime)  # When histopathology analysis was performed
    histopathology_tissue_type = Column(String)  # Primary tissue classification
    histopathology_risk_level = Column(String)  # "low", "moderate", "high"
    histopathology_features = Column(JSON)  # Key pathological features detected
    histopathology_recommendations = Column(Text)  # Clinical recommendations from AI
    histopathology_image_quality = Column(JSON)  # Image quality metrics (focus, staining, etc.)
    histopathology_predictions = Column(JSON)  # Full predictions with confidence intervals
    ai_concordance = Column(Boolean)  # Whether AI dermoscopy prediction matches histopathology
    ai_concordance_type = Column(String)  # "exact_match", "category_match", "discordant"
    ai_concordance_notes = Column(Text)  # Notes on concordance/discordance

    # Body Map - Track lesion locations on body
    body_location = Column(String)  # Primary body location (e.g., "face", "chest", "arm_left", "leg_right")
    body_sublocation = Column(String)  # More specific location (e.g., "cheek", "upper_arm", "lower_leg")
    body_side = Column(String)  # "left", "right", "center", "bilateral"
    body_map_coordinates = Column(JSON)  # Store x, y coordinates on body diagram for visual display

    # Symptom Tracker - Record symptoms and changes over time
    symptom_duration = Column(String)  # How long lesion has been present (e.g., "2 weeks", "3 months")
    symptom_duration_value = Column(Integer)  # Numeric value for duration
    symptom_duration_unit = Column(String)  # "days", "weeks", "months", "years"
    symptom_changes = Column(Text)  # Description of changes over time
    symptom_itching = Column(Boolean, default=False)  # Is there itching?
    symptom_itching_severity = Column(Integer)  # 1-10 scale
    symptom_pain = Column(Boolean, default=False)  # Is there pain?
    symptom_pain_severity = Column(Integer)  # 1-10 scale
    symptom_bleeding = Column(Boolean, default=False)  # Is there bleeding?
    symptom_bleeding_frequency = Column(String)  # "rare", "occasional", "frequent"
    symptom_other = Column(JSON)  # Other symptoms as key-value pairs
    symptom_notes = Column(Text)  # Additional symptom notes

    # Medication List - Document drugs that can cause skin reactions
    medications = Column(JSON)  # Array of medication objects with name, dosage, start_date, purpose, skin_reaction
    medications_updated_at = Column(DateTime)  # When medications were last updated

    # Medical History - Track risk factors for skin conditions
    family_history_skin_cancer = Column(Boolean, default=False)  # Family history of skin cancer
    family_history_details = Column(Text)  # Details about family history
    previous_skin_cancers = Column(Boolean, default=False)  # Patient has had skin cancer before
    previous_skin_cancers_details = Column(Text)  # Details about previous skin cancers
    immunosuppression = Column(Boolean, default=False)  # Is patient immunosuppressed
    immunosuppression_details = Column(Text)  # Details about immunosuppression (transplant, HIV, medications, etc.)
    sun_exposure_level = Column(String)  # "minimal", "moderate", "high", "very_high"
    sun_exposure_details = Column(Text)  # Details about sun exposure (outdoor work, geographic location, etc.)
    history_of_sunburns = Column(Boolean, default=False)  # History of severe sunburns
    sunburn_details = Column(Text)  # Details about sunburns
    tanning_bed_use = Column(Boolean, default=False)  # Has used tanning beds
    tanning_bed_frequency = Column(String)  # Frequency of tanning bed use
    other_risk_factors = Column(Text)  # Other relevant risk factors
    medical_history_updated_at = Column(DateTime)  # When medical history was last updated

    # Teledermatology Integration - Share results with dermatologists
    shared_with_dermatologist = Column(Boolean, default=False)  # Has been shared with a dermatologist
    dermatologist_email = Column(String)  # Email of the dermatologist
    dermatologist_name = Column(String)  # Name of the dermatologist
    share_date = Column(DateTime)  # When the analysis was shared
    share_message = Column(Text)  # Patient's message to dermatologist
    share_token = Column(String)  # Unique token for dermatologist access
    dermatologist_reviewed = Column(Boolean, default=False)  # Has dermatologist reviewed
    dermatologist_notes = Column(Text)  # Dermatologist's review notes
    dermatologist_recommendation = Column(Text)  # Dermatologist's recommendations
    dermatologist_review_date = Column(DateTime)  # When dermatologist reviewed

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    biopsy_updated_at = Column(DateTime)  # When biopsy results were added
    symptom_updated_at = Column(DateTime)  # When symptoms were last updated

    # Multimodal Analysis Tracking
    multimodal_enabled = Column(Boolean, default=False)  # Whether multimodal analysis was used
    labs_integrated = Column(Boolean, default=False)  # Whether lab results were integrated
    lab_result_id = Column(Integer, ForeignKey("lab_results.id"), nullable=True)  # FK to lab_results
    history_integrated = Column(Boolean, default=False)  # Whether patient history was integrated
    lesion_tracking_used = Column(Boolean, default=False)  # Whether lesion tracking was used
    confidence_adjustments = Column(JSON)  # Breakdown of how confidence was modified
    data_sources_used = Column(JSON)  # ["image", "labs", "history", "lesion_tracking"]
    raw_image_confidence = Column(Float)  # Original model confidence before adjustments
    clinical_adjustment_delta = Column(Float)  # Change from clinical context
    lab_adjustment_delta = Column(Float)  # Change from lab integration
    multimodal_risk_factors = Column(JSON)  # List of risk factors identified
    multimodal_recommendations = Column(JSON)  # List of recommendations

    # Relationships
    user = relationship("User", back_populates="analyses")
    lesion_group = relationship("LesionGroup", back_populates="analyses")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Extended profile information
    date_of_birth = Column(DateTime)
    gender = Column(String)
    phone_number = Column(String)
    address = Column(Text)
    emergency_contact = Column(String)
    medical_history = Column(Text)
    skin_type = Column(String)  # Fitzpatrick skin type
    family_history = Column(Text)
    ethnicity = Column(String)  # For demographic and risk assessment

    # Phenotype characteristics for risk assessment
    natural_hair_color = Column(String)  # red, blonde, light_brown, dark_brown, black
    natural_eye_color = Column(String)  # blue, green, hazel, brown
    freckles = Column(String)  # none, few, some, many

    # Family history of skin conditions
    family_history_skin_cancer = Column(Boolean, default=False)
    family_history_melanoma = Column(Boolean, default=False)

    # Location information (for dermatologist matching)
    city = Column(String, index=True)
    state = Column(String, index=True)
    country = Column(String, default="USA")
    zip_code = Column(String)
    latitude = Column(Float)  # For distance calculations
    longitude = Column(Float)  # For distance calculations
    preferred_distance_miles = Column(Integer, default=50)  # Max distance for in-person consultations

    # Preferences
    notification_preferences = Column(JSON)
    privacy_settings = Column(JSON)

    # Statistics
    total_analyses = Column(Integer, default=0)
    last_analysis_date = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="profile")


class LesionGroup(Base):
    """
    Lesion Group - Track the same lesion over time for comparison and change detection
    Groups multiple analysis records of the same physical lesion to enable temporal tracking
    """
    __tablename__ = "lesion_groups"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Lesion identification
    lesion_name = Column(String, nullable=False)  # User-provided name (e.g., "Mole on left shoulder")
    lesion_description = Column(Text)  # Additional description

    # Body location (inherited from first analysis)
    body_location = Column(String)  # Primary body location
    body_sublocation = Column(String)  # Specific location
    body_side = Column(String)  # "left", "right", "center"
    body_map_coordinates = Column(JSON)  # x, y coordinates on body map

    # Lesion metadata
    first_noticed_date = Column(DateTime)  # When user first noticed the lesion
    monitoring_frequency = Column(String)  # "weekly", "monthly", "quarterly", "biannual", "annual"
    next_check_date = Column(DateTime)  # When next check is scheduled

    # Risk and status
    current_risk_level = Column(String)  # "low", "medium", "high" (from latest analysis)
    requires_attention = Column(Boolean, default=False)  # Flag for concerning changes
    attention_reason = Column(Text)  # Why attention is needed

    # Change detection summary
    total_analyses = Column(Integer, default=0)  # Number of times this lesion has been analyzed
    change_detected = Column(Boolean, default=False)  # Whether significant changes have been detected
    change_summary = Column(JSON)  # Summary of detected changes over time
    growth_rate = Column(Float)  # Estimated growth rate (mm per month)

    # Monitoring settings
    is_active = Column(Boolean, default=True)  # Whether actively monitoring
    archived = Column(Boolean, default=False)  # Archived (e.g., removed/treated)
    archive_reason = Column(String)  # "removed_surgically", "resolved", "false_alarm", etc.
    archive_date = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_analyzed_at = Column(DateTime)  # Most recent analysis timestamp
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="lesion_groups")
    analyses = relationship("AnalysisHistory", back_populates="lesion_group", order_by="AnalysisHistory.created_at")


class LesionComparison(Base):
    """
    Lesion Comparison - Store AI-powered change detection results between two analyses
    Tracks size changes, color changes, shape changes, and ABCDE criteria evolution
    """
    __tablename__ = "lesion_comparisons"

    id = Column(Integer, primary_key=True, index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=False, index=True)

    # The two analyses being compared
    baseline_analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=False)  # Earlier analysis
    current_analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=False)  # Later analysis

    # Time between analyses
    time_difference_days = Column(Float)  # Days between the two analyses

    # Overall change assessment
    change_detected = Column(Boolean, default=False)  # Whether significant change was detected
    change_severity = Column(String)  # "none", "minimal", "moderate", "significant", "concerning"
    change_score = Column(Float)  # 0-1 score of overall change magnitude

    # Size changes
    size_changed = Column(Boolean, default=False)
    size_change_percent = Column(Float)  # Percentage change in size
    size_change_mm = Column(Float)  # Absolute change in mm (if measurements available)
    size_trend = Column(String)  # "stable", "growing", "shrinking"

    # Color changes
    color_changed = Column(Boolean, default=False)
    color_change_score = Column(Float)  # 0-1 score of color variation
    color_description = Column(Text)  # Description of color changes
    new_colors_appeared = Column(Boolean, default=False)  # New colors detected

    # Shape/Border changes (Asymmetry & Border from ABCDE)
    shape_changed = Column(Boolean, default=False)
    asymmetry_increased = Column(Boolean, default=False)
    border_irregularity_increased = Column(Boolean, default=False)
    shape_change_score = Column(Float)  # 0-1 score of shape change

    # Texture changes
    texture_changed = Column(Boolean, default=False)
    texture_description = Column(Text)  # Description of texture changes

    # Symptom changes
    new_symptoms = Column(Boolean, default=False)  # New symptoms appeared
    symptom_worsening = Column(Boolean, default=False)  # Existing symptoms worsened
    symptom_changes_list = Column(JSON)  # List of symptom changes

    # Diagnosis consistency
    diagnosis_changed = Column(Boolean, default=False)  # Whether AI diagnosis changed
    baseline_diagnosis = Column(String)  # Original diagnosis
    current_diagnosis = Column(String)  # Current diagnosis
    diagnosis_consistency_score = Column(Float)  # 0-1 score (1 = consistent, 0 = completely different)

    # Risk level changes
    risk_escalated = Column(Boolean, default=False)  # Risk level increased
    baseline_risk = Column(String)  # Original risk level
    current_risk = Column(String)  # Current risk level

    # ABCDE criteria evolution
    abcde_worsening = Column(Boolean, default=False)  # Whether ABCDE criteria worsened
    abcde_comparison = Column(JSON)  # Detailed ABCDE comparison data

    # AI-powered analysis
    feature_vector_distance = Column(Float)  # Cosine distance between CNN feature vectors
    visual_similarity_score = Column(Float)  # 0-1 score (1 = identical, 0 = completely different)
    change_heatmap = Column(Text)  # Base64-encoded difference heatmap

    # Clinical recommendations
    action_required = Column(Boolean, default=False)  # Whether immediate action needed
    recommendation = Column(Text)  # Clinical recommendation based on changes
    urgency_level = Column(String)  # "routine", "soon", "urgent", "emergency"

    # Alert flags
    alert_triggered = Column(Boolean, default=False)  # Whether automated alert was triggered
    alert_reasons = Column(JSON)  # List of reasons for alert
    alert_sent_at = Column(DateTime)  # When alert was sent to user

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    lesion_group = relationship("LesionGroup", backref="comparisons")
    baseline_analysis = relationship("AnalysisHistory", foreign_keys=[baseline_analysis_id])
    current_analysis = relationship("AnalysisHistory", foreign_keys=[current_analysis_id])


class AuditLog(Base):
    """
    Audit Trail for Quality Assurance and Legal Documentation
    Logs all AI predictions and system events for compliance and traceability
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)

    # User and session information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    username = Column(String, index=True)  # Denormalized for quick access
    session_id = Column(String, index=True)  # Track user session

    # Event classification
    event_type = Column(String, nullable=False, index=True)  # "prediction", "login", "export", "share", etc.
    event_category = Column(String, index=True)  # "ai_inference", "user_action", "system_event", "data_access"
    severity = Column(String, default="info")  # "info", "warning", "error", "critical"

    # AI Prediction specific fields
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=True, index=True)
    model_name = Column(String)  # "binary_classifier", "lesion_classifier", "inflammatory_classifier"
    model_version = Column(String)  # Model version/checkpoint identifier
    prediction_result = Column(String)  # Primary predicted class
    confidence_score = Column(Float)  # Confidence of prediction (0-1)
    prediction_probabilities = Column(JSON)  # Full probability distribution

    # Input data tracking
    input_data_hash = Column(String)  # SHA-256 hash of input image for integrity verification
    input_metadata = Column(JSON)  # Image size, format, quality metrics

    # Processing metadata
    processing_time_ms = Column(Float)  # Time taken for inference in milliseconds
    gpu_used = Column(Boolean, default=False)  # Whether GPU was used
    system_resources = Column(JSON)  # CPU/GPU/memory usage during inference

    # Quality assurance flags
    quality_passed = Column(Boolean)  # Whether image quality checks passed
    uncertainty_metrics = Column(JSON)  # Monte Carlo Dropout uncertainty metrics
    reliability_score = Column(Float)  # Overall reliability score (0-1)
    flags = Column(JSON)  # Any warnings or flags raised (e.g., "low_confidence", "bias_detected")

    # Legal and compliance
    consent_obtained = Column(Boolean, default=True)  # User consent for data processing
    data_retention_days = Column(Integer, default=2555)  # 7 years for medical records
    anonymized = Column(Boolean, default=False)  # Whether PII has been removed

    # IP and location for security
    ip_address = Column(String)  # User's IP address
    user_agent = Column(String)  # Browser/device information
    geo_location = Column(String)  # Country/region (for regulatory compliance)

    # Action details
    action = Column(String)  # Specific action taken (e.g., "classify_image", "export_report")
    endpoint = Column(String)  # API endpoint called
    http_method = Column(String)  # GET, POST, etc.
    request_params = Column(JSON)  # Request parameters (sanitized)
    response_status = Column(Integer)  # HTTP response code

    # Error tracking
    error_occurred = Column(Boolean, default=False)
    error_message = Column(Text)  # Error details if any
    error_stack_trace = Column(Text)  # Stack trace for debugging

    # Audit metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_by = Column(String)  # System user or admin who triggered the event
    audit_reviewed = Column(Boolean, default=False)  # For periodic audit reviews
    audit_notes = Column(Text)  # Notes from audit reviewers

    # Relationships
    user = relationship("User", backref="audit_logs")
    analysis = relationship("AnalysisHistory", backref="audit_logs")


class FamilyMember(Base):
    """
    Family Member - Track family medical history for genetic risk assessment
    """
    __tablename__ = "family_members"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Relationship to user
    relationship_type = Column(String, nullable=False)  # "parent", "sibling", "grandparent", "aunt_uncle", "cousin", "child"
    relationship_side = Column(String)  # "maternal", "paternal", "own" (for siblings/children)

    # Basic information
    name = Column(String)  # Optional: name or identifier (e.g., "Mom", "Paternal Grandfather")
    gender = Column(String)  # "male", "female", "other"
    year_of_birth = Column(Integer)  # For calculating age-related risk
    is_alive = Column(Boolean, default=True)
    age_at_death = Column(Integer)  # If deceased

    # Skin cancer history
    has_skin_cancer = Column(Boolean, default=False)
    skin_cancer_types = Column(JSON)  # Array of {"type": "melanoma", "age_diagnosed": 45, "location": "back"}
    skin_cancer_count = Column(Integer, default=0)  # Number of skin cancers
    earliest_diagnosis_age = Column(Integer)  # Age at first skin cancer diagnosis

    # Melanoma-specific (higher hereditary risk)
    has_melanoma = Column(Boolean, default=False)
    melanoma_count = Column(Integer, default=0)
    melanoma_age_at_diagnosis = Column(Integer)  # Age when first melanoma was diagnosed
    melanoma_outcome = Column(String)  # "survived", "deceased", "unknown"
    melanoma_subtypes = Column(JSON)  # Array of melanoma subtypes if known
    melanoma_familial_syndrome = Column(Boolean, default=False)  # Part of familial melanoma syndrome

    # Other relevant cancers (may indicate genetic syndromes)
    has_other_cancers = Column(Boolean, default=False)
    other_cancers = Column(JSON)  # Array of {"type": "breast", "age_diagnosed": 50}

    # Skin characteristics (hereditary traits)
    skin_type = Column(String)  # Fitzpatrick skin type I-VI
    hair_color = Column(String)  # "blonde", "red", "brown", "black" - red hair increases melanoma risk
    eye_color = Column(String)  # "blue", "green", "hazel", "brown" - light eyes increase risk
    has_many_moles = Column(Boolean, default=False)  # >50 moles increases risk
    has_atypical_moles = Column(Boolean, default=False)  # Dysplastic nevus syndrome

    # Genetic testing (if available)
    genetic_testing_done = Column(Boolean, default=False)
    genetic_mutations = Column(JSON)  # Array of known mutations (CDKN2A, CDK4, BAP1, etc.)
    genetic_test_date = Column(DateTime)
    genetic_test_notes = Column(Text)

    # Additional notes
    notes = Column(Text)  # Other relevant medical information

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="family_members")


class GeneticRiskProfile(Base):
    """
    Genetic Risk Profile - Calculated risk scores based on family history and personal factors
    """
    __tablename__ = "genetic_risk_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Overall risk scores (0-100 scale)
    overall_genetic_risk_score = Column(Float, default=0.0)  # Comprehensive risk score
    melanoma_risk_score = Column(Float, default=0.0)  # Specific melanoma risk
    basal_cell_carcinoma_risk_score = Column(Float, default=0.0)  # BCC risk
    squamous_cell_carcinoma_risk_score = Column(Float, default=0.0)  # SCC risk

    # Risk level categories
    overall_risk_level = Column(String, default="low")  # "low", "moderate", "high", "very_high"
    melanoma_risk_level = Column(String, default="low")

    # Family history factors (contributing to risk)
    family_history_score = Column(Float, default=0.0)  # 0-100
    first_degree_relatives_affected = Column(Integer, default=0)  # Parents, siblings, children
    second_degree_relatives_affected = Column(Integer, default=0)  # Grandparents, aunts/uncles
    total_relatives_with_skin_cancer = Column(Integer, default=0)
    has_multiple_family_melanomas = Column(Boolean, default=False)  # >2 family members with melanoma
    has_early_onset_melanoma = Column(Boolean, default=False)  # Family member diagnosed <40 years
    familial_melanoma_syndrome_suspected = Column(Boolean, default=False)

    # Personal risk factors (contributing to risk)
    personal_risk_score = Column(Float, default=0.0)  # Based on user's own characteristics
    high_risk_phenotype = Column(Boolean, default=False)  # Fair skin, red hair, light eyes, many moles
    atypical_mole_syndrome = Column(Boolean, default=False)
    previous_skin_cancers_count = Column(Integer, default=0)

    # Genetic factors (if testing done)
    known_genetic_mutations = Column(JSON)  # Array of mutations found in user or family
    has_high_risk_mutation = Column(Boolean, default=False)  # CDKN2A, CDK4, BAP1
    genetic_counseling_recommended = Column(Boolean, default=False)

    # Risk-based recommendations
    recommended_screening_frequency = Column(String)  # "monthly", "quarterly", "biannual", "annual"
    recommended_professional_frequency = Column(String)  # How often to see dermatologist
    high_priority_monitoring = Column(Boolean, default=False)  # Flag for very high risk patients
    risk_reduction_recommendations = Column(JSON)  # Array of personalized recommendations

    # Risk calculation metadata
    last_calculated = Column(DateTime, default=datetime.utcnow)
    calculation_version = Column(String, default="1.0")  # Version of risk algorithm
    factors_considered = Column(JSON)  # List of factors included in calculation
    confidence_level = Column(Float)  # Confidence in risk assessment (0-1)

    # Hereditary pattern insights
    inheritance_pattern = Column(String)  # "sporadic", "possible_hereditary", "likely_hereditary", "confirmed_hereditary"
    affected_lineages = Column(JSON)  # {"maternal": true, "paternal": false} - which family lines affected
    generation_pattern = Column(JSON)  # Pattern across generations for visualization

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="genetic_risk_profile")


class GeneticTest(Base):
    """
    Genetic Test - Store uploaded genetic test results
    """
    __tablename__ = "genetic_tests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Test information
    test_name = Column(String, nullable=False)
    test_date = Column(String)  # YYYY-MM-DD format
    laboratory = Column(String)  # Testing laboratory name

    # Mutations detected
    mutations_detected = Column(JSON)  # Array of mutation identifiers (e.g., ["CDKN2A", "MC1R"])

    # File storage
    file_url = Column(String)  # Path to uploaded test report file
    file_name = Column(String)  # Original file name
    file_type = Column(String)  # MIME type

    # Additional information
    notes = Column(Text)  # Additional notes or observations

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="genetic_tests")


class UserAlert(Base):
    """
    User Alert - Notification system for high-risk users and important events
    """
    __tablename__ = "user_alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Alert classification
    alert_type = Column(String, nullable=False)  # "risk_escalation", "screening_reminder", "lesion_change", "genetic_result"
    severity = Column(String, default="info")  # "info", "warning", "urgent", "critical"
    priority = Column(Integer, default=0)  # 0-10 scale for sorting

    # Alert content
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    action_required = Column(String)  # Specific action user should take
    action_url = Column(String)  # Deep link to relevant screen

    # Alert metadata
    is_read = Column(Boolean, default=False)
    is_dismissed = Column(Boolean, default=False)
    read_at = Column(DateTime)
    dismissed_at = Column(DateTime)

    # Related entities
    related_entity_type = Column(String)  # "risk_profile", "lesion_group", "genetic_test", "analysis"
    related_entity_id = Column(Integer)

    # Reminder settings
    is_recurring = Column(Boolean, default=False)
    recurrence_frequency = Column(String)  # "daily", "weekly", "monthly", "quarterly"
    next_reminder_date = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="alerts")


class SystemAlert(Base):
    """
    System Alert - Alerts for model failures, degraded performance, and system issues.

    These are admin-level alerts not tied to specific users.
    Used for monitoring ML model health and system performance.
    """
    __tablename__ = "system_alerts"

    id = Column(Integer, primary_key=True, index=True)

    # Alert identification
    alert_id = Column(String, unique=True, index=True, nullable=False)  # External ID like "alert-000001"
    alert_type = Column(String, nullable=False, index=True)  # model_failure, high_error_rate, slow_inference, etc.
    severity = Column(String, nullable=False, index=True)  # info, warning, error, critical

    # Source information
    model_name = Column(String, index=True)  # Which model triggered the alert
    source_component = Column(String)  # analysis_router, tasks, etc.

    # Alert content
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)

    # Metrics at time of alert
    metrics = Column(JSON)  # Detailed metrics snapshot

    # Status tracking
    status = Column(String, default="active", index=True)  # active, acknowledged, resolved, expired
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String)  # Username of admin who acknowledged
    acknowledged_at = Column(DateTime)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolved_by = Column(String)
    resolution_notes = Column(Text)

    # Notification tracking
    notification_sent = Column(Boolean, default=False)
    notification_sent_at = Column(DateTime)
    notification_channels = Column(JSON)  # ["email", "slack", "push"]

    # Deduplication
    fingerprint = Column(String, index=True)  # Hash of alert type + model for deduplication
    occurrence_count = Column(Integer, default=1)  # Times this alert has occurred
    first_occurrence = Column(DateTime, default=datetime.utcnow)
    last_occurrence = Column(DateTime, default=datetime.utcnow)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime)  # Auto-expire old alerts


class GrowthForecast(Base):
    """
    Growth Forecast - ML-based predictions for lesion growth and changes
    """
    __tablename__ = "growth_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Forecast metadata
    forecast_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    forecast_horizon_days = Column(Integer, default=90)  # Forecasting 90 days ahead
    model_version = Column(String, default="1.0")
    confidence_score = Column(Float)  # 0-1 confidence in forecast

    # Historical data used
    historical_analyses_count = Column(Integer)  # Number of analyses used for forecast
    time_span_days = Column(Float)  # Time span of historical data

    # Size predictions (if measurements available)
    current_size_mm = Column(Float)  # Current size in mm
    predicted_size_30d = Column(Float)  # Predicted size in 30 days
    predicted_size_60d = Column(Float)  # Predicted size in 60 days
    predicted_size_90d = Column(Float)  # Predicted size in 90 days
    growth_rate_mm_per_month = Column(Float)  # Average growth rate
    growth_acceleration = Column(Float)  # Whether growth is accelerating

    # Change predictions
    predicted_color_change = Column(Boolean, default=False)
    predicted_shape_change = Column(Boolean, default=False)
    predicted_border_change = Column(Boolean, default=False)
    change_probability = Column(Float)  # 0-1 probability of significant change

    # Risk predictions
    current_risk_level = Column(String)  # Current risk level
    predicted_risk_level_30d = Column(String)  # Predicted risk in 30 days
    predicted_risk_level_60d = Column(String)
    predicted_risk_level_90d = Column(String)
    risk_escalation_probability = Column(Float)  # Probability risk will increase

    # Feature importance
    primary_risk_factors = Column(JSON)  # List of primary factors driving the forecast
    growth_trend = Column(String)  # "stable", "slow_growth", "moderate_growth", "rapid_growth", "shrinking"

    # Recommendations based on forecast
    recommended_action = Column(String)  # "continue_monitoring", "increase_frequency", "urgent_consultation"
    next_check_date = Column(DateTime)  # Recommended next check date
    monitoring_frequency = Column(String)  # "weekly", "biweekly", "monthly", "quarterly"

    # Forecast data for visualization
    forecast_data = Column(JSON)  # Time series data for charts

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    lesion_group = relationship("LesionGroup", backref="growth_forecasts")
    user = relationship("User", backref="growth_forecasts")


class ScreeningSchedule(Base):
    """
    Screening Schedule - Personalized screening recommendations based on risk profile and history
    """
    __tablename__ = "screening_schedules"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Schedule metadata
    schedule_type = Column(String, nullable=False)  # "self_exam", "dermatologist_visit", "lesion_check", "genetic_counseling"
    priority = Column(Integer, default=0)  # 0-10 priority scale
    is_recurring = Column(Boolean, default=True)
    recurrence_frequency = Column(String)  # "weekly", "monthly", "quarterly", "biannual", "annual"

    # Schedule details
    title = Column(String, nullable=False)
    description = Column(Text)
    recommended_date = Column(DateTime, nullable=False, index=True)
    due_date = Column(DateTime)  # Hard deadline if applicable
    completed_date = Column(DateTime)
    is_completed = Column(Boolean, default=False)
    is_overdue = Column(Boolean, default=False)

    # Related entities
    related_entity_type = Column(String)  # "lesion_group", "risk_profile", "genetic_test", "forecast"
    related_entity_id = Column(Integer)

    # Personalization factors
    based_on_risk_level = Column(String)  # Risk level that triggered this schedule
    based_on_genetic_risk = Column(Boolean, default=False)
    based_on_lesion_changes = Column(Boolean, default=False)
    based_on_family_history = Column(Boolean, default=False)

    # Notification settings
    notify_before_days = Column(Integer, default=7)  # Notify 7 days before
    notification_sent = Column(Boolean, default=False)
    notification_sent_at = Column(DateTime)

    # Next occurrence (for recurring schedules)
    next_occurrence_date = Column(DateTime)

    # Completion tracking
    completion_notes = Column(Text)
    completion_result = Column(String)  # "normal", "concerning", "action_required"

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="screening_schedules")


class RiskTrend(Base):
    """
    Risk Trend - Track how user's risk level changes over time for analytics
    """
    __tablename__ = "risk_trends"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Snapshot date
    snapshot_date = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Overall risk metrics
    overall_risk_score = Column(Float)
    overall_risk_level = Column(String)
    melanoma_risk_score = Column(Float)

    # Contributing factors
    total_lesions_tracked = Column(Integer, default=0)
    high_risk_lesions_count = Column(Integer, default=0)
    new_lesions_this_period = Column(Integer, default=0)
    changed_lesions_this_period = Column(Integer, default=0)

    # Genetic factors
    genetic_risk_score = Column(Float)
    family_members_affected = Column(Integer)

    # Behavioral factors
    analyses_this_period = Column(Integer, default=0)  # User engagement
    compliance_score = Column(Float)  # How well user follows recommendations (0-1)

    # Trend analysis
    risk_trend = Column(String)  # "improving", "stable", "worsening"
    risk_change_rate = Column(Float)  # Rate of change in risk score
    predicted_future_risk = Column(Float)  # ML prediction of future risk

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship back to user
    user = relationship("User", backref="risk_trends")


class SkinCancerRiskAssessment(Base):
    """
    Skin Cancer Risk Assessment - Comprehensive risk calculation results
    Combines family history, sun exposure, skin type, and AI findings
    """
    __tablename__ = "skin_cancer_risk_assessments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    assessment_id = Column(String, unique=True, nullable=False, index=True)

    # Overall Risk Scores (0-100 scale)
    overall_risk_score = Column(Float, nullable=False)
    risk_category = Column(String, nullable=False)  # very_low, low, moderate, high, very_high

    # Cancer-Specific Relative Risks
    melanoma_relative_risk = Column(Float)
    melanoma_lifetime_risk_percent = Column(Float)
    bcc_relative_risk = Column(Float)
    scc_relative_risk = Column(Float)

    # Component Scores (0-100 scale)
    genetic_score = Column(Float)
    phenotype_score = Column(Float)
    sun_exposure_score = Column(Float)
    behavioral_score = Column(Float)
    medical_history_score = Column(Float)
    ai_findings_score = Column(Float)

    # Input Data (stored for audit and recalculation)
    input_data = Column(JSON)  # Full input data for the assessment

    # Risk Factors Identified
    risk_factors = Column(JSON)  # Array of identified risk factors with details

    # Recommendations Generated
    recommendations = Column(JSON)  # Array of personalized recommendations

    # Screening Recommendations
    recommended_self_exam_frequency = Column(String)  # weekly, biweekly, monthly
    recommended_professional_exam_frequency = Column(String)  # quarterly, biannual, annual
    urgent_dermatology_referral = Column(Boolean, default=False)

    # Confidence and Methodology
    confidence_score = Column(Float)  # 0-1 confidence in the assessment
    methodology_version = Column(String, default="1.0")
    models_used = Column(JSON)  # Which validated models were applied

    # AI Integration Data (if AI analysis was included)
    ai_analysis_ids = Column(JSON)  # IDs of AI analyses incorporated
    ai_high_risk_lesions_count = Column(Integer, default=0)
    ai_uncertainty_flag = Column(Boolean, default=False)

    # Comparison to Previous
    previous_assessment_id = Column(String)  # Link to previous assessment
    risk_change = Column(Float)  # Change from previous assessment
    risk_trend = Column(String)  # improving, stable, worsening

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="skin_cancer_risk_assessments")


class Appointment(Base):
    """
    Appointment - Schedule dermatology appointments with providers
    """
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    appointment_id = Column(String, unique=True, nullable=False, index=True)

    # Provider information
    provider_id = Column(String, index=True)
    provider_name = Column(String)

    # Appointment timing
    appointment_date = Column(Date, nullable=False, index=True)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    timezone = Column(String, default="America/New_York")

    # Appointment type and status
    appointment_type = Column(String, nullable=False)  # initial_consultation, follow_up, skin_check, etc.
    status = Column(String, default="scheduled")  # scheduled, confirmed, checked_in, completed, cancelled, no_show
    is_telemedicine = Column(Boolean, default=False)

    # Location
    location = Column(String)
    room = Column(String)
    telemedicine_link = Column(String)

    # Patient information
    patient_name = Column(String)
    patient_phone = Column(String)
    patient_email = Column(String)

    # Reason and notes
    reason_for_visit = Column(Text)
    patient_notes = Column(Text)
    provider_notes = Column(Text)

    # Related records
    related_analysis_ids = Column(JSON)  # List of analysis IDs
    related_lesion_ids = Column(JSON)  # List of lesion IDs

    # Reminders
    reminder_settings = Column(JSON)  # {24_hours_before: true, 2_hours_before: true, etc.}
    reminders_sent = Column(JSON)  # [{type: "24_hours", sent_at: "...", channel: "email"}]

    # Recurrence
    recurrence_pattern = Column(String)  # none, weekly, monthly, quarterly, etc.
    recurrence_end_date = Column(Date)
    parent_appointment_id = Column(String)

    # Insurance
    insurance_verified = Column(Boolean, default=False)
    copay_amount = Column(Float)

    # Status timestamps
    confirmed_at = Column(DateTime)
    checked_in_at = Column(DateTime)
    completed_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    cancellation_reason = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="appointments")


class AppointmentWaitlist(Base):
    """
    Appointment Waitlist - Patients waiting for earlier appointment slots
    """
    __tablename__ = "appointment_waitlist"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    waitlist_id = Column(String, unique=True, nullable=False, index=True)

    # Preferences
    provider_id = Column(String)
    appointment_type = Column(String, nullable=False)
    preferred_dates = Column(JSON)  # List of preferred dates
    preferred_times = Column(JSON)  # ["morning", "afternoon", "evening"]
    flexibility = Column(String)  # flexible, somewhat_flexible, specific

    # Details
    reason = Column(Text)
    priority = Column(Integer, default=5)  # 1-10, 1 being highest

    # Notifications
    notified_slots = Column(JSON)  # List of slot IDs patient was notified about
    status = Column(String, default="active")  # active, fulfilled, expired, cancelled

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="waitlist_entries")


class AppointmentReminder(Base):
    """
    Appointment Reminder - Track scheduled and sent reminders
    """
    __tablename__ = "appointment_reminders"

    id = Column(Integer, primary_key=True, index=True)
    appointment_id = Column(String, ForeignKey("appointments.appointment_id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Reminder details
    reminder_type = Column(String, nullable=False)  # 24_hours, 2_hours, 30_minutes
    channel = Column(String, nullable=False)  # email, sms, push, in_app
    scheduled_for = Column(DateTime, nullable=False)

    # Status
    status = Column(String, default="pending")  # pending, sent, failed, cancelled
    sent_at = Column(DateTime)
    error_message = Column(Text)

    # Content
    subject = Column(String)
    message = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship back to user
    user = relationship("User", backref="appointment_reminders")


class SunExposure(Base):
    """
    Sun Exposure - Track individual sun exposure events for preventive care
    """
    __tablename__ = "sun_exposures"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # When and where
    exposure_date = Column(DateTime, nullable=False, index=True)
    duration_minutes = Column(Integer, nullable=False)
    time_of_day = Column(String)  # "morning", "midday", "afternoon", "evening"
    location = Column(String)  # "beach", "mountains", "park", "outdoor_work", "sports_field", etc.
    activity = Column(String)  # "work", "exercise", "leisure", "sports", "travel", etc.

    # UV conditions
    uv_index = Column(Float)  # 0-11+ scale
    uv_index_source = Column(String)  # "manual", "weather_api", "device_sensor"
    weather_conditions = Column(String)  # "sunny", "partly_cloudy", "cloudy", "overcast"
    altitude_meters = Column(Float)  # Higher altitude = more UV exposure

    # Sun protection used
    sun_protection_used = Column(Boolean, default=False)
    sunscreen_applied = Column(Boolean, default=False)
    spf_level = Column(Integer)  # SPF 15, 30, 50, etc.
    sunscreen_reapplied = Column(Boolean, default=False)
    protective_clothing = Column(Boolean, default=False)
    hat_worn = Column(Boolean, default=False)
    sunglasses_worn = Column(Boolean, default=False)
    shade_sought = Column(Boolean, default=False)

    # Body areas exposed
    exposed_body_areas = Column(JSON)  # List of exposed areas: ["face", "arms", "legs", "back", etc.]

    # Skin reaction
    skin_reaction = Column(String)  # "none", "mild_redness", "moderate_burn", "severe_burn", "tanning"
    reaction_severity = Column(Integer)  # 0-10 scale
    pain_level = Column(Integer)  # 0-10 scale
    peeling_occurred = Column(Boolean, default=False)

    # Additional context
    intentional_tanning = Column(Boolean, default=False)
    indoor_tanning = Column(Boolean, default=False)  # Tanning beds
    notes = Column(Text)

    # Risk calculation
    calculated_uv_dose = Column(Float)  # UV index × duration × protection factor
    risk_score = Column(Float)  # Overall risk score for this exposure (0-100)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", backref="sun_exposures")


class SunExposureCorrelation(Base):
    """
    Sun Exposure Correlation - Link sun exposure patterns to lesion development
    """
    __tablename__ = "sun_exposure_correlations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=True, index=True)

    # Analysis period
    analysis_period_start = Column(DateTime, nullable=False)
    analysis_period_end = Column(DateTime, nullable=False)

    # Aggregated exposure data
    total_exposure_hours = Column(Float)  # Total sun exposure in hours
    total_exposure_events = Column(Integer)  # Number of recorded exposures
    average_uv_index = Column(Float)
    max_uv_index_encountered = Column(Float)

    # High-risk exposure counts
    high_uv_exposures_count = Column(Integer, default=0)  # UV index >= 8
    extreme_uv_exposures_count = Column(Integer, default=0)  # UV index >= 11
    sunburn_events_count = Column(Integer, default=0)
    unprotected_exposures_count = Column(Integer, default=0)
    midday_exposures_count = Column(Integer, default=0)  # 10am-4pm exposures

    # Protection habits
    average_spf_used = Column(Float)
    protection_compliance_rate = Column(Float)  # 0-1, how often protection was used

    # Calculated UV dose
    cumulative_uv_dose = Column(Float)  # Total UV radiation dose over period

    # Correlation with lesion
    lesion_body_area = Column(String)  # Body area where lesion developed
    lesion_first_detected = Column(DateTime)  # When lesion was first noticed
    exposure_in_lesion_area_hours = Column(Float)  # Exposure specifically to affected area

    # Correlation analysis
    correlation_score = Column(Float)  # 0-1, statistical correlation strength
    correlation_confidence = Column(Float)  # 0-1, confidence in correlation
    correlation_type = Column(String)  # "strong_positive", "moderate_positive", "weak", "none"

    # Risk factors identified
    risk_factors = Column(JSON)  # List of identified risk factors
    protective_factors = Column(JSON)  # List of protective behaviors

    # Recommendations
    prevention_recommendations = Column(JSON)  # Personalized prevention advice
    screening_urgency = Column(String)  # "low", "medium", "high", "urgent"

    # Analysis metadata
    correlation_notes = Column(Text)
    analysis_algorithm_version = Column(String)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="sun_exposure_correlations")
    lesion_group = relationship("LesionGroup", backref="sun_exposure_correlations")


class Treatment(Base):
    """
    Treatment - Track medications, procedures, and topical treatments
    """
    __tablename__ = "treatments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=True, index=True)

    # Treatment details
    treatment_name = Column(String, nullable=False)
    treatment_type = Column(String, nullable=False)  # "topical", "oral_medication", "injection", "procedure", "phototherapy", "cryotherapy"

    # Medication details
    active_ingredient = Column(String)
    brand_name = Column(String)
    dosage = Column(String)  # e.g., "0.05%", "20mg", "2ml"
    dosage_unit = Column(String)  # "mg", "ml", "g", "%"

    # Administration
    route = Column(String)  # "topical", "oral", "injection", "IV"
    frequency = Column(String)  # "once_daily", "twice_daily", "as_needed", "weekly"
    instructions = Column(Text)  # Special instructions

    # Schedule
    start_date = Column(DateTime, nullable=False)
    planned_end_date = Column(DateTime)
    actual_end_date = Column(DateTime)
    duration_weeks = Column(Integer)

    # Purpose
    indication = Column(String)  # What condition is being treated
    prescriber_name = Column(String)
    prescription_number = Column(String)

    # Target lesion
    target_body_area = Column(String)

    # Treatment goals
    treatment_goals = Column(JSON)  # List of goals: ["reduce_size", "reduce_inflammation", "prevent_growth"]

    # Status
    is_active = Column(Boolean, default=True)
    discontinued = Column(Boolean, default=False)
    discontinuation_reason = Column(String)  # "completed", "side_effects", "ineffective", "improved"
    discontinuation_date = Column(DateTime)

    # Cost tracking
    cost_per_unit = Column(Float)
    insurance_covered = Column(Boolean, default=False)

    # Notes
    notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="treatments")
    lesion_group = relationship("LesionGroup", backref="treatments")


class TreatmentLog(Base):
    """
    Treatment Log - Track each application/dose of treatment
    """
    __tablename__ = "treatment_logs"

    id = Column(Integer, primary_key=True, index=True)
    treatment_id = Column(Integer, ForeignKey("treatments.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # When taken/applied
    administered_date = Column(DateTime, nullable=False, index=True)

    # Dose details
    dose_amount = Column(Float)
    dose_unit = Column(String)

    # Application details (for topical)
    application_area = Column(String)  # Where it was applied
    application_method = Column(String)  # "thin_layer", "thick_layer", "spot_treatment"

    # Compliance
    taken_as_prescribed = Column(Boolean, default=True)
    missed_dose = Column(Boolean, default=False)
    late_dose = Column(Boolean, default=False)
    hours_late = Column(Float)

    # Immediate observations
    immediate_reaction = Column(String)  # "none", "burning", "stinging", "redness", "itching"
    reaction_severity = Column(Integer)  # 0-10 scale

    # Photos
    photo_before_url = Column(String)
    photo_after_url = Column(String)  # Photo taken hours/days after application

    # Notes
    notes = Column(Text)

    # Reminders
    reminder_sent = Column(Boolean, default=False)
    reminder_acknowledged = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    treatment = relationship("Treatment", backref="treatment_logs")
    user = relationship("User", backref="treatment_logs")


class TreatmentEffectiveness(Base):
    """
    Treatment Effectiveness - Before/after comparisons and effectiveness tracking
    """
    __tablename__ = "treatment_effectiveness"

    id = Column(Integer, primary_key=True, index=True)
    treatment_id = Column(Integer, ForeignKey("treatments.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=True, index=True)

    # Assessment period
    assessment_date = Column(DateTime, nullable=False, index=True)
    days_into_treatment = Column(Integer)  # How many days since treatment started

    # Before baseline (from analysis before treatment started)
    baseline_analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=True)
    baseline_size_mm = Column(Float)
    baseline_color_score = Column(Float)  # 0-10 darkness/redness
    baseline_inflammation_score = Column(Float)  # 0-10
    baseline_symptoms_score = Column(Float)  # 0-10 based on itching, pain, bleeding

    # Current assessment
    current_analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=True)
    current_size_mm = Column(Float)
    current_color_score = Column(Float)
    current_inflammation_score = Column(Float)
    current_symptoms_score = Column(Float)

    # Changes (calculated)
    size_change_mm = Column(Float)  # Negative = smaller, positive = larger
    size_change_percent = Column(Float)
    color_change = Column(Float)  # Negative = lighter, positive = darker
    inflammation_change = Column(Float)
    symptoms_change = Column(Float)

    # Overall effectiveness rating
    patient_effectiveness_rating = Column(Integer)  # 1-5 stars
    objective_effectiveness_score = Column(Float)  # 0-100 calculated score

    # Specific improvements
    improvements_noted = Column(JSON)  # ["size_reduced", "less_red", "less_itchy", "flatter"]
    concerns_noted = Column(JSON)  # ["spreading", "darker", "more_painful"]

    # Side effects
    side_effects = Column(JSON)  # List of side effects experienced
    side_effects_severity = Column(Integer)  # 0-10
    side_effects_tolerable = Column(Boolean, default=True)

    # Treatment adherence during period
    adherence_rate = Column(Float)  # 0-1, percentage of doses taken as prescribed
    total_doses_prescribed = Column(Integer)
    total_doses_taken = Column(Integer)
    missed_doses = Column(Integer)

    # Comparison images
    before_image_url = Column(String)
    after_image_url = Column(String)
    comparison_heatmap_url = Column(String)  # Visual difference map

    # Doctor assessment (if available)
    doctor_assessment = Column(Text)
    doctor_effectiveness_rating = Column(Integer)  # 1-5
    doctor_recommendation = Column(String)  # "continue", "increase_dose", "decrease_dose", "discontinue", "change_treatment"

    # Overall outcome
    treatment_outcome = Column(String)  # "improving", "stable", "worsening", "resolved", "no_change"
    continue_treatment = Column(Boolean, default=True)

    # Notes
    patient_notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    treatment = relationship("Treatment", backref="effectiveness_assessments")
    user = relationship("User", backref="treatment_effectiveness")
    lesion_group = relationship("LesionGroup", backref="treatment_effectiveness")


class DermatologistProfile(Base):
    """
    Dermatologist Profile - Directory of available dermatologists for consultations and referrals
    """
    __tablename__ = "dermatologist_profiles"

    id = Column(Integer, primary_key=True, index=True)

    # Basic information
    full_name = Column(String, nullable=False, index=True)
    credentials = Column(String)  # "MD", "DO", "FAAD", etc.
    email = Column(String, nullable=False, index=True)
    phone_number = Column(String)

    # Practice information
    practice_name = Column(String)
    practice_address = Column(String)
    city = Column(String, index=True)
    state = Column(String, index=True)
    country = Column(String, default="USA")
    zip_code = Column(String)
    latitude = Column(Float)  # For distance calculations
    longitude = Column(Float)  # For distance calculations

    # Workload tracking
    current_queue_size = Column(Integer, default=0)  # Current pending reviews
    max_queue_size = Column(Integer, default=20)  # Maximum queue capacity
    avg_response_hours = Column(Float, default=48.0)  # Average response time in hours

    # Specializations and expertise
    specializations = Column(JSON)  # ["Mohs Surgery", "Cosmetic", "Pediatric", "Skin Cancer"]
    languages_spoken = Column(JSON)  # ["English", "Spanish", "French"]
    board_certifications = Column(JSON)

    # Availability
    accepts_video_consultations = Column(Boolean, default=True)
    accepts_referrals = Column(Boolean, default=True)
    accepts_second_opinions = Column(Boolean, default=True)
    availability_status = Column(String, default="accepting")  # "accepting", "limited", "waitlist", "not_accepting"

    # Scheduling
    typical_wait_time_days = Column(Integer)  # Average wait for appointment
    consultation_duration_minutes = Column(Integer, default=30)
    available_days = Column(JSON)  # ["Monday", "Tuesday", "Wednesday"]
    available_hours = Column(String)  # "9:00 AM - 5:00 PM"
    timezone = Column(String, default="America/New_York")

    # Ratings and reviews
    average_rating = Column(Float)  # 0-5 stars
    total_reviews = Column(Integer, default=0)
    total_consultations = Column(Integer, default=0)

    # Integration
    video_platform = Column(String)  # "zoom", "teams", "google_meet", "telehealth_platform"
    video_platform_url = Column(String)  # Link to their video platform
    booking_url = Column(String)  # External booking system link

    # Profile
    bio = Column(Text)
    photo_url = Column(String)
    years_experience = Column(Integer)
    medical_school = Column(String)
    residency = Column(String)

    # Status
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VideoConsultation(Base):
    """
    Video Consultation - Scheduled video appointments with dermatologists
    """
    __tablename__ = "video_consultations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    dermatologist_id = Column(Integer, ForeignKey("dermatologist_profiles.id"), nullable=True, index=True)  # Nullable for pending assignment

    # Related analysis/lesion
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=True, index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=True, index=True)

    # Consultation details
    consultation_type = Column(String, nullable=False)  # "initial", "follow_up", "second_opinion", "urgent"
    consultation_reason = Column(Text)  # Patient's reason for consultation

    # Scheduling
    scheduled_datetime = Column(DateTime, nullable=False, index=True)
    duration_minutes = Column(Integer, default=30)
    timezone = Column(String)

    # Status tracking
    status = Column(String, nullable=False, default="scheduled", index=True)
    # "scheduled", "confirmed", "in_progress", "completed", "cancelled", "no_show", "rescheduled"

    # Video meeting
    video_platform = Column(String)  # "zoom", "teams", "google_meet"
    video_meeting_url = Column(String)
    video_meeting_id = Column(String)
    video_meeting_password = Column(String)

    # Reminders
    reminder_sent = Column(Boolean, default=False)
    reminder_sent_at = Column(DateTime)

    # Pre-consultation
    patient_notes = Column(Text)  # What patient wants to discuss
    patient_questions = Column(JSON)  # List of specific questions
    attachments = Column(JSON)  # Additional photos/documents shared

    # Consultation outcome
    consultation_completed_at = Column(DateTime)
    duration_actual_minutes = Column(Integer)
    dermatologist_notes = Column(Text)
    diagnosis = Column(String)
    treatment_plan = Column(Text)
    prescriptions = Column(JSON)  # Medications prescribed
    referrals_made = Column(JSON)  # Referrals to specialists
    follow_up_needed = Column(Boolean, default=False)
    follow_up_timeframe = Column(String)  # "2 weeks", "1 month", "3 months"

    # Billing
    consultation_fee = Column(Float)
    insurance_covered = Column(Boolean, default=False)
    payment_status = Column(String)  # "pending", "paid", "insurance_submitted", "refunded"

    # Patient feedback
    patient_rating = Column(Integer)  # 1-5 stars
    patient_feedback = Column(Text)
    would_recommend = Column(Boolean)

    # Administrative
    cancellation_reason = Column(String)
    cancelled_by = Column(String)  # "patient", "dermatologist", "system"
    cancelled_at = Column(DateTime)
    rescheduled_from_consultation_id = Column(Integer)  # Link to original if rescheduled

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="video_consultations")
    dermatologist = relationship("DermatologistProfile", backref="consultations")
    analysis = relationship("AnalysisHistory", backref="consultations")
    lesion_group = relationship("LesionGroup", backref="consultations")


class Referral(Base):
    """
    Referral - Formal referrals from primary care or other sources to dermatologists
    """
    __tablename__ = "referrals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Source of referral
    referring_provider_name = Column(String)
    referring_provider_specialty = Column(String)
    referring_provider_contact = Column(String)
    referral_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Target dermatologist
    dermatologist_id = Column(Integer, ForeignKey("dermatologist_profiles.id"), nullable=True, index=True)
    dermatologist_preference = Column(String)  # If patient has preference

    # Clinical information
    referral_reason = Column(String, nullable=False)  # "suspicious_lesion", "skin_cancer_screening", "treatment_resistant", "specialized_care"
    primary_concern = Column(Text, nullable=False)
    clinical_summary = Column(Text)  # Summary from referring provider
    urgency_level = Column(String, default="routine")  # "urgent", "semi_urgent", "routine"

    # Related data
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=True, index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=True, index=True)
    supporting_documents = Column(JSON)  # Medical records, test results, images

    # Status tracking
    status = Column(String, nullable=False, default="pending", index=True)
    # "pending", "accepted", "appointment_scheduled", "seen", "declined", "cancelled"

    status_notes = Column(Text)
    status_updated_at = Column(DateTime)

    # Appointment scheduling
    appointment_scheduled_date = Column(DateTime)
    appointment_completed_date = Column(DateTime)
    linked_consultation_id = Column(Integer, ForeignKey("video_consultations.id"), nullable=True)

    # Insurance and authorization
    insurance_authorization_required = Column(Boolean, default=False)
    insurance_authorization_number = Column(String)
    insurance_approved = Column(Boolean)

    # Outcome
    dermatologist_accepted = Column(Boolean)
    dermatologist_response = Column(Text)
    dermatologist_diagnosis = Column(String)
    treatment_provided = Column(Text)
    patient_seen = Column(Boolean, default=False)

    # Follow-up
    follow_up_with_referring_provider = Column(Boolean, default=False)
    referring_provider_notified = Column(Boolean, default=False)
    outcome_report = Column(Text)  # Report sent back to referring provider

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="referrals")
    dermatologist = relationship("DermatologistProfile", backref="referrals")
    analysis = relationship("AnalysisHistory", backref="referrals")
    lesion_group = relationship("LesionGroup", backref="referrals")


class SecondOpinion(Base):
    """
    Second Opinion - Requests for additional expert review of diagnoses or treatment plans
    """
    __tablename__ = "second_opinions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Original diagnosis information
    original_diagnosis = Column(String, nullable=False)
    original_provider_name = Column(String)
    original_diagnosis_date = Column(DateTime)
    original_treatment_plan = Column(Text)

    # Why seeking second opinion
    reason_for_second_opinion = Column(String, nullable=False)
    # "uncertainty", "high_risk_diagnosis", "treatment_concerns", "poor_response", "peace_of_mind"
    specific_questions = Column(JSON)  # List of questions for second opinion provider
    concerns = Column(Text)

    # Related data
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=True, index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=True, index=True)
    original_biopsy_results = Column(Text)
    original_pathology_report = Column(String)  # File path/URL

    # Additional documentation
    supporting_images = Column(JSON)  # Additional photos
    medical_records = Column(JSON)  # Relevant records
    test_results = Column(JSON)  # Lab results, imaging

    # Second opinion provider
    dermatologist_id = Column(Integer, ForeignKey("dermatologist_profiles.id"), nullable=True, index=True)
    dermatologist_specialty_requested = Column(String)  # Specific expertise needed

    # Request urgency
    urgency = Column(String, default="routine")  # "urgent", "semi_urgent", "routine"
    patient_anxiety_level = Column(Integer)  # 1-10 scale

    # Status
    status = Column(String, nullable=False, default="submitted", index=True)
    # "submitted", "under_review", "additional_info_needed", "completed", "cancelled"

    # Second opinion outcome
    second_opinion_date = Column(DateTime)
    second_opinion_diagnosis = Column(String)
    second_opinion_notes = Column(Text)
    second_opinion_treatment_plan = Column(Text)

    # Agreement with original
    agrees_with_original_diagnosis = Column(Boolean)
    diagnosis_confidence_level = Column(Integer)  # 1-10
    differences_from_original = Column(Text)

    # Recommendations
    recommended_action = Column(String)
    # "proceed_as_planned", "modify_treatment", "additional_testing", "specialist_referral", "urgent_intervention"
    recommended_next_steps = Column(JSON)

    # Additional testing recommended
    additional_tests_needed = Column(JSON)
    biopsy_recommended = Column(Boolean, default=False)
    imaging_recommended = Column(Boolean, default=False)

    # Consultation details
    consultation_method = Column(String)  # "video", "in_person", "chart_review"
    linked_consultation_id = Column(Integer, ForeignKey("video_consultations.id"), nullable=True)

    # Patient response
    patient_satisfied = Column(Boolean)
    patient_clarity_improved = Column(Boolean)
    patient_anxiety_reduced = Column(Boolean)
    patient_rating = Column(Integer)  # 1-5 stars
    patient_feedback = Column(Text)

    # Follow-up
    follow_up_needed = Column(Boolean, default=False)
    follow_up_scheduled = Column(Boolean, default=False)

    # Billing
    fee = Column(Float)
    insurance_covered = Column(Boolean, default=False)
    payment_status = Column(String)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="second_opinions")
    dermatologist = relationship("DermatologistProfile", backref="second_opinions")
    analysis = relationship("AnalysisHistory", backref="second_opinions")
    lesion_group = relationship("LesionGroup", backref="second_opinions")


class ConsultationNote(Base):
    """
    Consultation Notes - Detailed notes from any consultation (video, in-person, second opinion)
    """
    __tablename__ = "consultation_notes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    dermatologist_id = Column(Integer, ForeignKey("dermatologist_profiles.id"), nullable=False, index=True)

    # Link to consultation
    consultation_id = Column(Integer, ForeignKey("video_consultations.id"), nullable=True, index=True)
    second_opinion_id = Column(Integer, ForeignKey("second_opinions.id"), nullable=True, index=True)
    referral_id = Column(Integer, ForeignKey("referrals.id"), nullable=True, index=True)

    # Note details
    note_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    note_type = Column(String)  # "consultation", "follow_up", "procedure", "phone_call"

    # Clinical documentation
    chief_complaint = Column(Text)
    history_of_present_illness = Column(Text)
    review_of_systems = Column(Text)
    physical_examination = Column(Text)

    # Assessment
    diagnosis = Column(String)
    differential_diagnoses = Column(JSON)
    icd_codes = Column(JSON)  # ICD-10 diagnostic codes

    # Plan
    treatment_plan = Column(Text)
    prescriptions = Column(JSON)  # Medications with dosage, frequency
    procedures_performed = Column(JSON)
    procedures_recommended = Column(JSON)

    # Testing
    tests_ordered = Column(JSON)  # Lab tests, imaging, biopsy
    test_results = Column(JSON)

    # Recommendations
    lifestyle_modifications = Column(JSON)
    skincare_recommendations = Column(JSON)
    sun_protection_advice = Column(Text)

    # Follow-up
    follow_up_recommended = Column(Boolean, default=False)
    follow_up_timeframe = Column(String)
    follow_up_reason = Column(String)

    # Education
    patient_education_provided = Column(JSON)  # Topics discussed
    educational_materials_given = Column(JSON)  # Handouts, links

    # Full note
    full_soap_note = Column(Text)  # Complete SOAP note

    # Administrative
    note_status = Column(String, default="draft")  # "draft", "final", "amended"
    signed_by_provider = Column(Boolean, default=False)
    signed_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="consultation_notes")
    dermatologist = relationship("DermatologistProfile", backref="notes_written")
    consultation = relationship("VideoConsultation", backref="notes")


class Notification(Base):
    """
    Notification - Store in-app notifications for users
    """
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Notification details
    notification_type = Column(String, nullable=False)  # "appointment_reminder", "analysis_complete", "alert", etc.
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSON)  # Additional data (appointment_id, analysis_id, etc.)

    # Status
    is_read = Column(Boolean, default=False, index=True)
    read_at = Column(DateTime)

    # Priority and expiration
    priority = Column(String, default="normal")  # "low", "normal", "high", "urgent"
    expires_at = Column(DateTime)  # Optional expiration

    # Delivery tracking
    push_sent = Column(Boolean, default=False)
    push_sent_at = Column(DateTime)
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship back to user
    user = relationship("User", backref="notifications")


class ProviderAvailability(Base):
    """
    Provider Availability - Manage provider schedules and availability
    """
    __tablename__ = "provider_availability"

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(String, nullable=False, index=True)
    provider_name = Column(String, nullable=False)

    # Working hours by day (0=Monday, 6=Sunday)
    day_of_week = Column(Integer, nullable=False)  # 0-6
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)

    # Break times
    break_start = Column(Time)
    break_end = Column(Time)

    # Location
    location = Column(String)
    is_telemedicine_day = Column(Boolean, default=False)

    # Status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProviderBlockedTime(Base):
    """
    Provider Blocked Time - Track when providers are unavailable
    """
    __tablename__ = "provider_blocked_times"

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(String, nullable=False, index=True)

    # Block details
    start_datetime = Column(DateTime, nullable=False, index=True)
    end_datetime = Column(DateTime, nullable=False)
    reason = Column(String)  # "vacation", "conference", "personal", "emergency", etc.
    notes = Column(Text)

    # Recurring blocks
    is_recurring = Column(Boolean, default=False)
    recurrence_pattern = Column(String)  # "weekly", "monthly", etc.
    recurrence_end_date = Column(Date)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# =============================================================================
# CLINIC/PRACTICE MODELS - For the hybrid user role system
# =============================================================================

class Clinic(Base):
    """
    Clinic - A medical practice that can have multiple providers and patients.
    This is the core entity for the doctor-facing side of the hybrid model.
    """
    __tablename__ = "clinics"

    id = Column(Integer, primary_key=True, index=True)

    # Basic information
    name = Column(String, nullable=False, index=True)
    description = Column(Text)
    clinic_type = Column(String, default="dermatology")  # "dermatology", "primary_care", "multi_specialty"

    # Contact information
    email = Column(String, nullable=False)
    phone = Column(String)
    website = Column(String)

    # Address
    address_line1 = Column(String)
    address_line2 = Column(String)
    city = Column(String, index=True)
    state = Column(String, index=True)
    zip_code = Column(String)
    country = Column(String, default="USA")
    latitude = Column(Float)
    longitude = Column(Float)

    # Verification
    is_verified = Column(Boolean, default=False)
    verification_date = Column(DateTime)
    npi_organization = Column(String)  # Organization NPI if applicable
    tax_id = Column(String)  # For billing/insurance

    # Clinic settings
    accepts_new_patients = Column(Boolean, default=True)
    accepts_telemedicine = Column(Boolean, default=True)
    typical_wait_days = Column(Integer, default=7)

    # Branding
    logo_url = Column(String)
    primary_color = Column(String, default="#3B82F6")  # For clinic branding

    # Unique clinic code for patient linking
    clinic_code = Column(String, unique=True, index=True)  # e.g., "DERM-ABC123"

    # Status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    staff_members = relationship("ClinicStaff", back_populates="clinic", cascade="all, delete-orphan")
    patient_links = relationship("ClinicPatient", back_populates="clinic", cascade="all, delete-orphan")
    clinical_notes = relationship("ClinicNote", back_populates="clinic", cascade="all, delete-orphan")


class ClinicStaff(Base):
    """
    Clinic Staff - Links users to clinics with specific roles.
    A user can be staff at multiple clinics.
    """
    __tablename__ = "clinic_staff"

    id = Column(Integer, primary_key=True, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Role in this clinic
    # "admin" = Full access, can manage clinic settings and staff
    # "provider" = Can see patients, write notes, make diagnoses
    # "staff" = Limited access, can see patient list but not clinical details
    role = Column(String, nullable=False, default="provider")

    # Provider-specific fields
    title = Column(String)  # "MD", "DO", "PA", "NP", etc.
    specialty = Column(String)  # "General Dermatology", "Mohs Surgery", etc.
    license_number = Column(String)
    license_state = Column(String)
    is_license_verified = Column(Boolean, default=False)

    # Availability in this clinic
    is_accepting_patients = Column(Boolean, default=True)
    max_daily_patients = Column(Integer, default=20)

    # Status
    is_active = Column(Boolean, default=True)
    joined_at = Column(DateTime, default=datetime.utcnow)
    left_at = Column(DateTime)  # If they leave the clinic

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    clinic = relationship("Clinic", back_populates="staff_members")
    user = relationship("User", backref="clinic_memberships")

    # Unique constraint: one user can have one role per clinic
    __table_args__ = (
        # UniqueConstraint('clinic_id', 'user_id', name='unique_clinic_user'),
    )


class ClinicPatient(Base):
    """
    Clinic Patient - Links patients to clinics with consent-based data sharing.
    Patients own their data and choose what to share with each clinic.
    """
    __tablename__ = "clinic_patients"

    id = Column(Integer, primary_key=True, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=False, index=True)
    patient_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # How the patient was linked
    linked_via = Column(String, default="code")  # "code", "qr", "referral", "manual"
    linked_by_user_id = Column(Integer, ForeignKey("users.id"))  # Staff who added them, if manual

    # Consent levels - PATIENT CONTROLS THIS
    # "share_selected" = Patient manually selects which analyses to share
    # "share_new" = Auto-share new analyses, but patient can revoke individual ones
    # "share_all" = Share all past and future analyses
    # "view_only" = Clinic can view but patient must explicitly share each item
    consent_level = Column(String, default="share_selected")

    # What data is shared (granular control)
    share_analyses = Column(Boolean, default=True)
    share_photos = Column(Boolean, default=True)
    share_medical_history = Column(Boolean, default=False)
    share_family_history = Column(Boolean, default=False)
    share_medications = Column(Boolean, default=False)

    # Patient info visible to clinic
    patient_notes = Column(Text)  # Notes from patient to clinic
    preferred_provider_id = Column(Integer, ForeignKey("clinic_staff.id"))

    # Clinic-assigned fields
    patient_mrn = Column(String, index=True)  # Medical Record Number in clinic's system
    clinic_notes = Column(Text)  # Internal notes about patient (not visible to patient)

    # Status
    is_active = Column(Boolean, default=True)
    status = Column(String, default="active")  # "active", "inactive", "pending", "revoked"

    # Timestamps
    linked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    consent_updated_at = Column(DateTime, default=datetime.utcnow)
    last_visit_at = Column(DateTime)
    revoked_at = Column(DateTime)  # If patient revokes access

    # Relationships
    clinic = relationship("Clinic", back_populates="patient_links")
    patient = relationship("User", foreign_keys=[patient_user_id], backref="clinic_links")
    linked_by = relationship("User", foreign_keys=[linked_by_user_id])


class SharedAnalysis(Base):
    """
    Shared Analysis - Tracks which analyses a patient has shared with which clinics.
    Patients can share/unshare individual analyses.
    """
    __tablename__ = "shared_analyses"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=False, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=False, index=True)
    patient_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Sharing details
    shared_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    shared_by = Column(String, default="patient")  # "patient", "auto" (based on consent level)

    # What's shared (can be granular)
    share_image = Column(Boolean, default=True)
    share_ai_results = Column(Boolean, default=True)
    share_clinical_context = Column(Boolean, default=True)

    # Patient's message when sharing
    patient_message = Column(Text)

    # Status
    is_active = Column(Boolean, default=True)
    revoked_at = Column(DateTime)

    # Clinic's response
    reviewed_by_user_id = Column(Integer, ForeignKey("users.id"))
    reviewed_at = Column(DateTime)
    review_status = Column(String)  # "pending", "reviewed", "needs_followup", "urgent"

    # Relationships
    analysis = relationship("AnalysisHistory", backref="clinic_shares")
    clinic = relationship("Clinic", backref="shared_analyses")
    patient = relationship("User", foreign_keys=[patient_user_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by_user_id])


class ClinicNote(Base):
    """
    Clinic Note - Clinical notes written by clinic staff about patient analyses.
    These are owned by the clinic, not visible to patient unless explicitly shared.
    """
    __tablename__ = "clinic_notes"

    id = Column(Integer, primary_key=True, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=False, index=True)
    patient_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    provider_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Related entities (optional)
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"), index=True)
    shared_analysis_id = Column(Integer, ForeignKey("shared_analyses.id"), index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), index=True)

    # Note content
    note_type = Column(String, nullable=False)  # "assessment", "plan", "followup", "general"
    title = Column(String)
    content = Column(Text, nullable=False)

    # Clinical assessment
    clinical_impression = Column(String)  # Provider's diagnosis/impression
    icd_codes = Column(JSON)  # ICD-10 codes
    cpt_codes = Column(JSON)  # Procedure codes

    # Plan
    treatment_plan = Column(Text)
    follow_up_recommended = Column(Boolean, default=False)
    follow_up_timeframe = Column(String)  # "1 week", "1 month", "3 months"
    referral_needed = Column(Boolean, default=False)
    referral_to = Column(String)

    # Visibility
    visible_to_patient = Column(Boolean, default=False)  # Can patient see this note?
    shared_with_patient_at = Column(DateTime)

    # Status
    status = Column(String, default="draft")  # "draft", "final", "amended"
    signed_at = Column(DateTime)
    amended_at = Column(DateTime)
    amendment_reason = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    clinic = relationship("Clinic", back_populates="clinical_notes")
    patient = relationship("User", foreign_keys=[patient_user_id], backref="clinic_notes_about_me")
    provider = relationship("User", foreign_keys=[provider_user_id], backref="clinic_notes_written")
    analysis = relationship("AnalysisHistory", backref="clinic_notes")
    lesion_group = relationship("LesionGroup", backref="clinic_notes")


class ClinicInvitation(Base):
    """
    Clinic Invitation - Invitations for patients to link with a clinic.
    Can be sent via email or generated as a QR code.
    """
    __tablename__ = "clinic_invitations"

    id = Column(Integer, primary_key=True, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"), nullable=False, index=True)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Invitation details
    invitation_code = Column(String, unique=True, nullable=False, index=True)  # Unique code
    invitation_type = Column(String, default="general")  # "general", "specific_patient", "bulk"

    # For specific patient invitations
    patient_email = Column(String)
    patient_name = Column(String)

    # Suggested consent level (patient can change)
    suggested_consent_level = Column(String, default="share_selected")

    # Usage limits
    max_uses = Column(Integer, default=1)  # 1 for specific, higher for bulk
    current_uses = Column(Integer, default=0)

    # Validity
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

    # Tracking
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    clinic = relationship("Clinic", backref="invitations")
    created_by = relationship("User", backref="created_invitations")


class LabResults(Base):
    """
    Store user lab results (blood, urine, stool) for enhanced skin analysis.
    Lab values provide systemic context for skin conditions.
    """
    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Lab test metadata
    test_date = Column(Date, nullable=False)  # When the lab test was done
    lab_name = Column(String)  # Name of the laboratory
    ordering_physician = Column(String)  # Doctor who ordered the test
    test_type = Column(String, nullable=False)  # "blood", "urine", "stool", "comprehensive"

    # Upload information
    uploaded_file = Column(String)  # Path to uploaded PDF/image
    is_manually_entered = Column(Boolean, default=False)  # Manual entry vs OCR

    # ============================================
    # BLOOD PANEL VALUES
    # ============================================

    # Complete Blood Count (CBC)
    wbc = Column(Float)  # White Blood Cells (x10^9/L) - normal: 3.8-10.8
    rbc = Column(Float)  # Red Blood Cells (x10^12/L) - normal: 4.2-5.8 (M), 3.8-5.1 (F)
    hemoglobin = Column(Float)  # g/dL - normal: 13.2-17.1 (M), 11.7-15.5 (F)
    hematocrit = Column(Float)  # % - normal: 38.5-50.0 (M), 35-45 (F)
    platelets = Column(Float)  # x10^9/L - normal: 140-400
    mcv = Column(Float)  # Mean Corpuscular Volume fL - normal: 80-100
    mch = Column(Float)  # Mean Corpuscular Hemoglobin pg - normal: 27-33
    mchc = Column(Float)  # Mean Corpuscular Hemoglobin Concentration g/dL - normal: 32-36
    rdw = Column(Float)  # Red Cell Distribution Width % - normal: 11-15
    mpv = Column(Float)  # Mean Platelet Volume fL - normal: 7.5-12.5

    # White Blood Cell Differential (%)
    neutrophils = Column(Float)  # % - normal: 40-70%
    lymphocytes = Column(Float)  # % - normal: 20-40%
    monocytes = Column(Float)  # % - normal: 2-8%
    eosinophils = Column(Float)  # % - normal: 1-4% (elevated in allergies)
    basophils = Column(Float)  # % - normal: 0-1%

    # Absolute White Blood Cell Counts (cells/uL)
    neutrophils_abs = Column(Float)  # Absolute Neutrophils - normal: 1500-7800
    lymphocytes_abs = Column(Float)  # Absolute Lymphocytes - normal: 850-3900
    monocytes_abs = Column(Float)  # Absolute Monocytes - normal: 200-950
    eosinophils_abs = Column(Float)  # Absolute Eosinophils - normal: 15-500
    basophils_abs = Column(Float)  # Absolute Basophils - normal: 0-200

    # Metabolic Panel
    glucose_fasting = Column(Float)  # mg/dL - normal: 65-99
    glucose_random = Column(Float)  # mg/dL - normal: <140
    hba1c = Column(Float)  # % - normal: <5.7%, prediabetes: 5.7-6.4%, diabetes: ≥6.5%
    eag = Column(Float)  # Estimated Average Glucose mg/dL (calculated from HbA1c)
    bun = Column(Float)  # Blood Urea Nitrogen mg/dL - normal: 7-25
    creatinine = Column(Float)  # mg/dL - normal: 0.70-1.25 (M), 0.50-1.00 (F)
    bun_creatinine_ratio = Column(Float)  # normal: 6-22
    egfr = Column(Float)  # eGFR Non-African American mL/min/1.73m² - normal: >60
    egfr_african_american = Column(Float)  # eGFR African American mL/min/1.73m² - normal: >60
    sodium = Column(Float)  # mmol/L - normal: 135-146
    potassium = Column(Float)  # mmol/L - normal: 3.5-5.3
    chloride = Column(Float)  # mmol/L - normal: 98-110
    co2 = Column(Float)  # Carbon Dioxide mmol/L - normal: 20-32
    calcium = Column(Float)  # mg/dL - normal: 8.6-10.3
    magnesium = Column(Float)  # mg/dL - normal: 1.6-2.3
    phosphorus = Column(Float)  # mg/dL - normal: 2.5-4.5

    # Liver Function
    alt = Column(Float)  # Alanine Aminotransferase U/L - normal: 9-46
    ast = Column(Float)  # Aspartate Aminotransferase U/L - normal: 10-35
    alp = Column(Float)  # Alkaline Phosphatase U/L - normal: 40-115
    bilirubin_total = Column(Float)  # mg/dL - normal: 0.2-1.2
    bilirubin_direct = Column(Float)  # mg/dL - normal: 0-0.3
    albumin = Column(Float)  # g/dL - normal: 3.6-5.1
    total_protein = Column(Float)  # g/dL - normal: 6.1-8.1
    globulin = Column(Float)  # g/dL - normal: 1.9-3.7 (calculated)
    albumin_globulin_ratio = Column(Float)  # normal: 1.0-2.5

    # Lipid Panel
    cholesterol_total = Column(Float)  # mg/dL - desirable: <200
    ldl = Column(Float)  # mg/dL - optimal: <100
    hdl = Column(Float)  # mg/dL - desirable: >40 (M), >50 (F)
    triglycerides = Column(Float)  # mg/dL - normal: <150
    chol_hdl_ratio = Column(Float)  # Cholesterol/HDL ratio - normal: <5.0
    non_hdl_cholesterol = Column(Float)  # Non-HDL Cholesterol mg/dL - normal: <130

    # Thyroid Panel
    tsh = Column(Float)  # mIU/L - normal: 0.40-4.50
    t3_uptake = Column(Float)  # T3 Uptake % - normal: 22-35
    t4_total = Column(Float)  # T4 (Thyroxine) Total mcg/dL - normal: 4.9-10.5
    free_t4_index = Column(Float)  # Free T4 Index (T7) - normal: 1.4-3.8
    t3_free = Column(Float)  # Free T3 pg/mL - normal: 2.3-4.2
    t4_free = Column(Float)  # Free T4 ng/dL - normal: 0.8-1.8
    t3_total = Column(Float)  # Total T3 ng/dL - normal: 80-200

    # Iron Studies
    iron = Column(Float)  # µg/dL - normal: 60-170
    ferritin = Column(Float)  # ng/mL - normal: 12-150 (F), 12-300 (M)
    tibc = Column(Float)  # Total Iron Binding Capacity µg/dL - normal: 250-370
    transferrin_saturation = Column(Float)  # % - normal: 20-50%

    # Vitamins
    vitamin_d = Column(Float)  # ng/mL - normal: 30-100, insufficient: 20-29, deficient: <20
    vitamin_b12 = Column(Float)  # pg/mL - normal: 200-900
    folate = Column(Float)  # ng/mL - normal: 2.7-17.0

    # Inflammatory Markers
    crp = Column(Float)  # C-Reactive Protein mg/L - normal: <3.0
    esr = Column(Float)  # Erythrocyte Sedimentation Rate mm/hr - normal: 0-22 (M), 0-29 (F)
    homocysteine = Column(Float)  # µmol/L - normal: 5-15

    # Autoimmune Markers
    ana_positive = Column(Boolean)  # Antinuclear Antibody
    ana_titer = Column(String)  # e.g., "1:80", "1:160"
    ana_pattern = Column(String)  # "homogeneous", "speckled", "nucleolar", etc.
    rf = Column(Float)  # Rheumatoid Factor IU/mL - normal: <14
    anti_ccp = Column(Float)  # Anti-Cyclic Citrullinated Peptide U/mL

    # Allergy
    ige_total = Column(Float)  # Total IgE IU/mL - normal: <100

    # ============================================
    # URINALYSIS VALUES
    # ============================================
    # Physical Characteristics
    urine_color = Column(String)  # "yellow", "amber", "red", etc. - normal: yellow
    urine_appearance = Column(String)  # "clear", "cloudy", "turbid" - normal: clear
    urine_clarity = Column(String)  # alias for appearance
    urine_specific_gravity = Column(Float)  # normal: 1.001-1.035
    urine_ph = Column(Float)  # normal: 5.0-8.0

    # Chemical Analysis
    urine_protein = Column(String)  # "negative", "trace", "1+", "2+", "3+" - normal: negative
    urine_glucose = Column(String)  # "negative", "trace", "1+", etc. - normal: negative
    urine_ketones = Column(String)  # "negative", "trace", "small", "moderate", "large"
    urine_blood = Column(String)  # "negative", "trace", "1+", etc. (occult blood)
    urine_bilirubin = Column(String)  # "negative", "1+", "2+", "3+"
    urine_urobilinogen = Column(String)  # normal: 0.2-1.0 mg/dL or "normal"
    urine_nitrite = Column(String)  # "positive", "negative"
    urine_leukocyte_esterase = Column(String)  # "negative", "trace", "1+", "2+", "3+"
    urine_reducing_substances = Column(String)  # "negative", or positive

    # Microscopic Examination
    urine_wbc = Column(String)  # /HPF - normal: <5 or "none seen"
    urine_rbc = Column(String)  # /HPF - normal: <2 or "none seen"
    urine_squamous_epithelial = Column(String)  # /HPF - normal: <5 or "none seen"
    urine_transitional_epithelial = Column(String)  # /HPF - normal: <5
    urine_renal_epithelial = Column(String)  # /HPF - normal: <3
    urine_bacteria = Column(String)  # "none seen", "few", "moderate", "many"
    urine_yeast = Column(String)  # "none seen" or present

    # Crystals
    urine_calcium_oxalate = Column(String)  # "none", "few", "moderate", "many"
    urine_triple_phosphate = Column(String)  # "none", "few", "moderate", "many"
    urine_uric_acid_crystals = Column(String)  # "none", "few", "moderate", "many"
    urine_amorphous_sediment = Column(String)  # "none", "few", "moderate"

    # Casts
    urine_hyaline_cast = Column(String)  # /LPF - normal: none seen
    urine_granular_cast = Column(String)  # /LPF - normal: none seen
    urine_casts_other = Column(String)  # Other casts

    # ============================================
    # STOOL TEST VALUES
    # ============================================
    stool_color = Column(String)  # "brown", "black", "red", "clay", etc.
    stool_consistency = Column(String)  # "formed", "loose", "watery", "hard"
    stool_occult_blood = Column(String)  # "positive", "negative"
    stool_wbc = Column(String)  # "present", "absent"
    stool_parasites = Column(String)  # "none detected", or specific parasite
    stool_ova = Column(String)  # "none detected", or specific finding
    stool_h_pylori = Column(String)  # "positive", "negative"
    stool_calprotectin = Column(Float)  # µg/g - normal: <50, elevated: >50

    # ============================================
    # METADATA & ANALYSIS
    # ============================================
    notes = Column(Text)  # User or physician notes
    abnormal_flags = Column(JSON)  # List of abnormal values detected
    skin_relevance_analysis = Column(JSON)  # AI analysis of skin-relevant findings

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", backref="lab_results")


# =============================================================================
# GENETIC TESTING MODELS
# =============================================================================

class GeneticTestResult(Base):
    """
    Genetic Test Result - Stores genetic test results for dermatology-related variants
    """
    __tablename__ = "genetic_test_results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Test metadata
    test_id = Column(String, unique=True, index=True)  # External lab test ID
    test_type = Column(String, nullable=False)  # "panel", "wes", "wgs", "targeted"
    test_name = Column(String)  # e.g., "Melanoma Risk Panel", "Skin Cancer Panel"
    lab_name = Column(String)  # e.g., "Foundation Medicine", "Tempus", "Invitae"
    lab_report_id = Column(String)  # Lab's report identifier
    ordering_physician = Column(String)
    test_date = Column(DateTime, nullable=False)
    report_date = Column(DateTime)

    # Sample info
    sample_type = Column(String)  # "blood", "saliva", "tissue", "tumor"
    sample_id = Column(String)

    # Results summary
    total_variants_tested = Column(Integer)
    pathogenic_variants_found = Column(Integer, default=0)
    likely_pathogenic_found = Column(Integer, default=0)
    vus_found = Column(Integer, default=0)  # Variants of uncertain significance
    benign_found = Column(Integer, default=0)

    # Overall risk assessment
    overall_risk_level = Column(String)  # "low", "moderate", "high", "very_high"
    risk_score = Column(Float)  # Numeric risk score if available
    risk_percentile = Column(Float)  # Population percentile

    # Specific condition risks (JSON with condition -> risk mapping)
    melanoma_risk = Column(JSON)  # {"relative_risk": 2.5, "lifetime_risk": "10%", "genes": ["CDKN2A"]}
    bcc_risk = Column(JSON)  # Basal cell carcinoma
    scc_risk = Column(JSON)  # Squamous cell carcinoma
    other_skin_risks = Column(JSON)  # Other skin-related genetic risks

    # Raw variant data
    variants = Column(JSON)  # List of detected variants with details
    vcf_file_path = Column(String)  # Path to uploaded VCF file if applicable

    # Pharmacogenomics relevant to dermatology
    pharmacogenomics = Column(JSON)  # Drug metabolism variants

    # Clinical recommendations
    recommendations = Column(JSON)  # List of clinical recommendations
    screening_recommendations = Column(JSON)  # Recommended screening schedule
    genetic_counseling_recommended = Column(Boolean, default=False)

    # Integration with other modules
    linked_to_risk_calculator = Column(Boolean, default=False)
    risk_calculator_adjustment = Column(Float)  # Multiplier for risk calculator

    # Report
    report_summary = Column(Text)
    report_pdf_path = Column(String)

    # Status
    status = Column(String, default="pending")  # "pending", "processing", "completed", "failed"
    processing_notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="genetic_test_results")


class GeneticVariant(Base):
    """
    Genetic Variant - Individual variant detected in a genetic test
    Links to parent GeneticTestResult
    """
    __tablename__ = "genetic_variants"

    id = Column(Integer, primary_key=True, index=True)
    test_result_id = Column(Integer, ForeignKey("genetic_test_results.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Variant identification (VCF-style)
    chromosome = Column(String, nullable=False)  # "chr1", "chr9", etc.
    position = Column(Integer, nullable=False)  # Genomic position
    reference = Column(String, nullable=False)  # Reference allele
    alternate = Column(String, nullable=False)  # Alternate allele
    rsid = Column(String, index=True)  # dbSNP ID if available (e.g., "rs1801516")

    # Gene information
    gene_symbol = Column(String, nullable=False, index=True)  # e.g., "CDKN2A", "MC1R", "BRAF"
    gene_name = Column(String)  # Full gene name
    transcript_id = Column(String)  # e.g., "NM_000077.5"

    # Variant effect
    variant_type = Column(String)  # "SNV", "insertion", "deletion", "indel"
    consequence = Column(String)  # "missense", "nonsense", "frameshift", "splice_site", etc.
    hgvs_c = Column(String)  # cDNA notation, e.g., "c.442G>A"
    hgvs_p = Column(String)  # Protein notation, e.g., "p.Ala148Thr"
    exon = Column(String)  # Exon number if applicable
    codon_change = Column(String)

    # Classification (ACMG guidelines)
    classification = Column(String, nullable=False)  # "pathogenic", "likely_pathogenic", "vus", "likely_benign", "benign"
    classification_criteria = Column(JSON)  # ACMG criteria used
    classification_date = Column(DateTime)
    classification_source = Column(String)  # "ClinVar", "lab_internal", etc.

    # Clinical significance
    clinvar_id = Column(String)
    clinvar_significance = Column(String)
    clinvar_review_status = Column(String)

    # Allele frequency
    gnomad_af = Column(Float)  # gnomAD allele frequency
    population_af = Column(JSON)  # Population-specific frequencies

    # Zygosity
    zygosity = Column(String)  # "heterozygous", "homozygous", "hemizygous"
    allele_depth = Column(JSON)  # Read depths for each allele

    # Dermatology relevance
    skin_condition_associations = Column(JSON)  # Associated skin conditions
    melanoma_risk_modifier = Column(Float)  # Risk multiplier for melanoma
    uv_sensitivity_impact = Column(String)  # "increased", "normal", "decreased"
    pigmentation_impact = Column(String)  # Impact on skin/hair pigmentation

    # Pharmacogenomics
    drug_interactions = Column(JSON)  # Relevant drug interactions

    # Evidence
    literature_references = Column(JSON)  # PubMed IDs and citations
    functional_studies = Column(Text)

    # Quality metrics
    quality_score = Column(Float)  # Variant quality score
    read_depth = Column(Integer)  # Coverage at this position
    filter_status = Column(String)  # "PASS" or filter reasons

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    test_result = relationship("GeneticTestResult", backref="detected_variants")
    user = relationship("User", backref="genetic_variants")


class DermatologyGeneReference(Base):
    """
    Reference database of genes relevant to dermatology
    Pre-populated with known skin-related genes and their clinical significance
    """
    __tablename__ = "dermatology_gene_reference"

    id = Column(Integer, primary_key=True, index=True)

    # Gene identification
    gene_symbol = Column(String, unique=True, nullable=False, index=True)
    gene_name = Column(String, nullable=False)
    gene_id = Column(String)  # NCBI Gene ID
    ensembl_id = Column(String)  # Ensembl Gene ID

    # Chromosomal location
    chromosome = Column(String)
    start_position = Column(Integer)
    end_position = Column(Integer)

    # Clinical relevance
    category = Column(String, nullable=False)  # "melanoma", "keratinocyte_cancer", "pigmentation", "photosensitivity", "hereditary_skin_disorder", "pharmacogenomics"
    subcategory = Column(String)

    # Associated conditions
    associated_conditions = Column(JSON)  # List of associated skin conditions
    inheritance_pattern = Column(String)  # "AD", "AR", "XL", "complex"

    # Risk information
    penetrance = Column(String)  # "high", "moderate", "low", "variable"
    typical_risk_increase = Column(String)  # e.g., "2-5x increased risk"

    # Key variants
    key_pathogenic_variants = Column(JSON)  # List of well-known pathogenic variants
    hotspot_regions = Column(JSON)  # Mutation hotspots

    # Clinical guidelines
    acmg_actionability = Column(Boolean, default=False)  # Is gene on ACMG actionable list
    screening_recommendations = Column(JSON)
    management_guidelines = Column(JSON)

    # References
    omim_id = Column(String)
    gene_reviews_link = Column(String)
    literature_summary = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def create_tables():
    Base.metadata.create_all(bind=engine)

def migrate_inflammatory_fields():
    """Add inflammatory condition fields to existing analysis_history table"""
    import sqlite3
    from sqlalchemy import text

    try:
        # Check if columns already exist
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(analysis_history)"))
            columns = [row[1] for row in result.fetchall()]

            if 'inflammatory_condition' not in columns:
                print("Adding inflammatory condition fields to database...")

                # Add the new columns
                conn.execute(text("ALTER TABLE analysis_history ADD COLUMN inflammatory_condition VARCHAR"))
                conn.execute(text("ALTER TABLE analysis_history ADD COLUMN inflammatory_confidence FLOAT"))
                conn.execute(text("ALTER TABLE analysis_history ADD COLUMN inflammatory_probabilities JSON"))
                conn.commit()

                print("Successfully added inflammatory condition fields!")
            else:
                print("Inflammatory condition fields already exist.")

            # Add differential_diagnoses column if it doesn't exist
            if 'differential_diagnoses' not in columns:
                print("Adding differential_diagnoses field to database...")
                conn.execute(text("ALTER TABLE analysis_history ADD COLUMN differential_diagnoses JSON"))
                conn.commit()
                print("Successfully added differential_diagnoses field!")
            else:
                print("Differential diagnoses field already exists.")

            # Add red_flag_data column if it doesn't exist
            if 'red_flag_data' not in columns:
                print("Adding red_flag_data field to database...")
                conn.execute(text("ALTER TABLE analysis_history ADD COLUMN red_flag_data JSON"))
                conn.commit()
                print("Successfully added red_flag_data field!")
            else:
                print("Red flag data field already exists.")

            # Add explainability_heatmap column if it doesn't exist
            if 'explainability_heatmap' not in columns:
                print("Adding explainability_heatmap field to database...")
                conn.execute(text("ALTER TABLE analysis_history ADD COLUMN explainability_heatmap TEXT"))
                conn.commit()
                print("Successfully added explainability_heatmap field!")
            else:
                print("Explainability heatmap field already exists.")

            # Add calibration and measurement columns
            calibration_fields = {
                'calibration_found': 'BOOLEAN',
                'calibration_type': 'VARCHAR',
                'pixels_per_mm': 'FLOAT',
                'calibration_confidence': 'FLOAT',
                'calibration_data': 'JSON',
                'measurements': 'JSON'
            }

            for field_name, field_type in calibration_fields.items():
                if field_name not in columns:
                    print(f"Adding {field_name} field to database...")
                    conn.execute(text(f"ALTER TABLE analysis_history ADD COLUMN {field_name} {field_type}"))
                    conn.commit()
                    print(f"Successfully added {field_name} field!")
                else:
                    print(f"{field_name} field already exists.")

    except Exception as e:
        print(f"Error during migration: {e}")


def migrate_user_roles():
    """Add hybrid user role system fields to existing users table"""
    from sqlalchemy import text

    try:
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(users)"))
            columns = [row[1] for row in result.fetchall()]

            user_role_fields = {
                'account_type': "VARCHAR DEFAULT 'personal'",
                'display_mode': "VARCHAR DEFAULT 'simple'",
                'is_verified_professional': 'BOOLEAN DEFAULT 0',
                'professional_license_number': 'VARCHAR',
                'professional_license_state': 'VARCHAR',
                'npi_number': 'VARCHAR',
                'verification_date': 'DATETIME',
                'verification_documents': 'JSON'
            }

            for field_name, field_def in user_role_fields.items():
                if field_name not in columns:
                    print(f"Adding {field_name} field to users table...")
                    conn.execute(text(f"ALTER TABLE users ADD COLUMN {field_name} {field_def}"))
                    conn.commit()
                    print(f"Successfully added {field_name} field!")
                else:
                    print(f"{field_name} field already exists.")

            print("User roles migration completed.")

    except Exception as e:
        print(f"Error during user roles migration: {e}")


# =============================================================================
# CONSENSUS REVIEW SYSTEM
# =============================================================================

class ConsensusCase(Base):
    """
    Consensus Case - Multi-specialist review request for complex cases
    """
    __tablename__ = "consensus_cases"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, unique=True, nullable=False, index=True)  # e.g., "CON-ABC12345"

    # Requesting user
    requesting_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Case details
    case_summary = Column(Text, nullable=False)
    clinical_history = Column(Text)
    images = Column(JSON)  # List of image URLs/paths
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"))  # Link to AI analysis if available

    # Specialist assignment
    specialist_ids = Column(JSON, nullable=False)  # List of assigned dermatologist IDs
    required_opinions = Column(Integer, default=3)  # Number of opinions needed

    # Urgency and deadline
    urgency = Column(String, default="standard")  # "standard", "urgent", "emergency"
    deadline = Column(DateTime)

    # Status tracking
    status = Column(String, default="pending", index=True)  # "pending", "in_review", "consensus_reached", "disagreement", "escalated", "closed"

    # Consensus result (populated when complete)
    consensus_diagnosis = Column(String)
    consensus_confidence = Column(Float)
    agreement_ratio = Column(Float)  # e.g., 0.67 for 2/3 agreement
    recommended_actions = Column(JSON)
    consensus_notes = Column(Text)

    # Escalation
    escalated = Column(Boolean, default=False)
    escalation_reason = Column(Text)
    escalated_to_id = Column(Integer, ForeignKey("dermatologist_profiles.id"))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)

    # Relationships
    requesting_user = relationship("User", backref="consensus_cases_requested")
    analysis = relationship("AnalysisHistory", backref="consensus_cases")
    escalated_to = relationship("DermatologistProfile", backref="escalated_consensus_cases")


class ConsensusOpinion(Base):
    """
    Consensus Opinion - Individual specialist opinion for a consensus case
    """
    __tablename__ = "consensus_opinions"

    id = Column(Integer, primary_key=True, index=True)

    # Links
    case_id = Column(Integer, ForeignKey("consensus_cases.id"), nullable=False, index=True)
    specialist_id = Column(Integer, ForeignKey("dermatologist_profiles.id"), nullable=False, index=True)

    # Diagnosis
    diagnosis = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    differential_diagnoses = Column(JSON)  # List of other possible diagnoses

    # Recommendations
    recommended_actions = Column(JSON)  # List of recommended actions
    recommended_treatments = Column(JSON)
    follow_up_timeline = Column(String)  # e.g., "2 weeks", "1 month"

    # AI comparison
    agrees_with_ai = Column(Boolean)
    ai_agreement_notes = Column(Text)

    # Detailed notes
    clinical_notes = Column(Text)
    reasoning = Column(Text)  # Explanation of diagnosis reasoning

    # Status
    submitted = Column(Boolean, default=False)
    submitted_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    case = relationship("ConsensusCase", backref="opinions")
    specialist = relationship("DermatologistProfile", backref="consensus_opinions")


class BatchSkinCheck(Base):
    """
    Batch Skin Check - Track full-body skin check sessions
    Used for processing multiple images in a single session via job queue.
    """
    __tablename__ = "batch_skin_checks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Task tracking
    task_id = Column(String, index=True)  # Celery task ID
    status = Column(String, default="pending")  # pending, processing, completed, failed

    # Results summary
    total_images = Column(Integer, default=0)
    images_processed = Column(Integer, default=0)
    lesions_detected = Column(Integer, default=0)
    high_risk_count = Column(Integer, default=0)

    # Detailed results (JSON)
    results = Column(JSON)
    error_message = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Processing time
    processing_time_seconds = Column(Float)

    # Relationships
    user = relationship("User", backref="batch_skin_checks")


class ConsensusAssignment(Base):
    """
    Consensus Assignment - Track specialist assignments and notification status
    """
    __tablename__ = "consensus_assignments"

    id = Column(Integer, primary_key=True, index=True)

    # Links
    case_id = Column(Integer, ForeignKey("consensus_cases.id"), nullable=False, index=True)
    specialist_id = Column(Integer, ForeignKey("dermatologist_profiles.id"), nullable=False, index=True)

    # Assignment status
    status = Column(String, default="assigned")  # "assigned", "viewed", "in_progress", "completed", "declined"
    assigned_at = Column(DateTime, default=datetime.utcnow)
    viewed_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Notification tracking
    notification_sent = Column(Boolean, default=False)
    notification_sent_at = Column(DateTime)
    reminder_count = Column(Integer, default=0)
    last_reminder_at = Column(DateTime)

    # Decline tracking
    declined = Column(Boolean, default=False)
    decline_reason = Column(Text)

    # Relationships
    case = relationship("ConsensusCase", backref="assignments")
    specialist = relationship("DermatologistProfile", backref="consensus_assignments")


# =============================================================================
# CLINICAL TRIALS MODELS
# =============================================================================

class ClinicalTrial(Base):
    """
    Clinical Trial - Stores clinical trial information synced from ClinicalTrials.gov
    """
    __tablename__ = "clinical_trials"

    id = Column(Integer, primary_key=True, index=True)
    nct_id = Column(String, unique=True, index=True)  # ClinicalTrials.gov ID (e.g., NCT12345678)

    # Basic information
    title = Column(String, nullable=False)
    brief_summary = Column(Text)
    detailed_description = Column(Text)

    # Trial details
    phase = Column(String)  # Phase 1, Phase 2, Phase 3, Phase 4, N/A
    status = Column(String, index=True)  # Recruiting, Active, Completed, Suspended, Terminated
    study_type = Column(String)  # Interventional, Observational

    # Conditions & interventions (JSON arrays)
    conditions = Column(JSON)  # ["melanoma", "basal cell carcinoma"]
    interventions = Column(JSON)  # [{"type": "Drug", "name": "Pembrolizumab"}]

    # Eligibility criteria
    eligibility_criteria = Column(Text)  # Raw eligibility text from API
    min_age = Column(Integer)  # Minimum age in years
    max_age = Column(Integer)  # Maximum age in years (999 = no max)
    gender = Column(String)  # All, Male, Female

    # Locations (JSON array of location objects)
    locations = Column(JSON)  # [{facility, city, state, country, zip, lat, lng, status}]

    # Contact information
    contact_name = Column(String)
    contact_email = Column(String)
    contact_phone = Column(String)
    principal_investigator = Column(String)

    # Enrollment
    target_enrollment = Column(Integer)
    current_enrollment = Column(Integer)

    # Sponsor & collaborators
    sponsor = Column(String)
    collaborators = Column(JSON)  # ["Organization 1", "Organization 2"]

    # Key dates
    start_date = Column(DateTime)
    completion_date = Column(DateTime)
    primary_completion_date = Column(DateTime)
    last_update_posted = Column(DateTime)

    # URL
    url = Column(String)  # ClinicalTrials.gov URL

    # Genetic/Biomarker requirements (extracted from eligibility criteria)
    required_biomarkers = Column(JSON)  # ["BRAF V600E", "BRAF V600K", "NRAS mutation"]
    excluded_biomarkers = Column(JSON)  # ["BRAF wild-type"]
    genetic_requirements = Column(JSON)  # Structured: [{"gene": "BRAF", "variants": ["V600E", "V600K"], "required": true}]
    biomarker_keywords = Column(JSON)  # Raw keywords found: ["BRAF", "PD-L1", "TMB"]
    requires_genetic_testing = Column(Boolean, default=False)  # Trial requires genetic/biomarker testing
    targeted_therapy_trial = Column(Boolean, default=False)  # Is this a targeted therapy trial

    # Sync tracking
    synced_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrialMatch(Base):
    """
    Trial Match - Stores personalized trial matches for users based on their profile and diagnosis history
    """
    __tablename__ = "trial_matches"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    trial_id = Column(Integer, ForeignKey("clinical_trials.id"), nullable=False, index=True)

    # Match scoring (0-100)
    match_score = Column(Float)
    match_reasons = Column(JSON)  # ["diagnosis_exact_match", "location_nearby", "age_eligible"]
    unmet_criteria = Column(JSON)  # ["requires_biopsy_confirmation", "excludes_prior_immunotherapy"]

    # Diagnosis matching details
    matched_conditions = Column(JSON)  # Conditions that matched
    diagnosis_score = Column(Float)  # Score from diagnosis matching (0-60)

    # Location details
    distance_miles = Column(Float)
    nearest_location = Column(JSON)  # {facility, city, state, zip}

    # Demographics matching
    age_eligible = Column(Boolean)
    gender_eligible = Column(Boolean)

    # Genetic/Biomarker matching details
    genetic_score = Column(Float, default=0)  # Score from genetic matching (0-15)
    matched_biomarkers = Column(JSON)  # ["BRAF V600E", "NRAS Q61K"]
    missing_biomarkers = Column(JSON)  # Required but user doesn't have
    excluded_biomarkers_found = Column(JSON)  # User has biomarker that excludes them
    genetic_eligible = Column(Boolean, default=True)  # False if excluded by biomarker
    genetic_match_type = Column(String)  # "exact", "partial", "none", "excluded"

    # Status tracking
    status = Column(String, default="matched", index=True)  # matched, viewed, interested, contacted, enrolled, declined, expired

    # User interaction
    viewed_at = Column(DateTime)
    dismissed = Column(Boolean, default=False)
    dismissed_at = Column(DateTime)
    dismiss_reason = Column(String)

    # Timestamps
    matched_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="trial_matches")
    trial = relationship("ClinicalTrial", backref="matches")


class TrialInterest(Base):
    """
    Trial Interest - Tracks when users express interest in clinical trials
    """
    __tablename__ = "trial_interests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    trial_id = Column(Integer, ForeignKey("clinical_trials.id"), nullable=False, index=True)
    match_id = Column(Integer, ForeignKey("trial_matches.id"))

    # Interest details
    interest_level = Column(String)  # high, medium, exploring
    preferred_contact = Column(String)  # email, phone, either
    contact_email = Column(String)  # User's preferred contact email
    contact_phone = Column(String)  # User's preferred contact phone
    notes = Column(Text)  # User's notes or questions

    # Related analysis (if interest sparked by specific diagnosis)
    related_analysis_id = Column(Integer, ForeignKey("analysis_history.id"))

    # Contact tracking
    contacted_trial = Column(Boolean, default=False)
    contacted_at = Column(DateTime)
    contact_method = Column(String)  # email, phone, website
    contact_response = Column(String)  # responded, no_response, enrolled, declined

    # Enrollment outcome
    enrolled = Column(Boolean, default=False)
    enrolled_at = Column(DateTime)
    enrollment_status = Column(String)  # screening, enrolled, completed, withdrawn

    # Withdrawal tracking
    withdrawn = Column(Boolean, default=False)
    withdrawn_at = Column(DateTime)
    withdrawal_reason = Column(Text)

    # Timestamps
    expressed_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="trial_interests")
    trial = relationship("ClinicalTrial", backref="interests")
    match = relationship("TrialMatch", backref="interests")


# =============================================================================
# WEARABLE INTEGRATION MODELS
# =============================================================================

class WearableDevice(Base):
    """
    Connected wearable devices for UV exposure tracking.
    Supports Apple Watch, Fitbit, Garmin, and other devices.
    """
    __tablename__ = "wearable_devices"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Device identification
    device_type = Column(String, nullable=False)  # "apple_watch", "fitbit", "garmin", "samsung", "withings", "other"
    device_model = Column(String)  # "Apple Watch Series 9", "Fitbit Sense 2", "Garmin Venu 3"
    device_name = Column(String)  # User-given name for the device
    device_id = Column(String, unique=True, index=True)  # Unique device identifier from platform

    # Connection status
    is_connected = Column(Boolean, default=True)
    connection_status = Column(String, default="active")  # "active", "disconnected", "expired", "revoked"
    last_sync_at = Column(DateTime)
    sync_frequency_minutes = Column(Integer, default=60)  # How often to sync

    # OAuth tokens (encrypted in production)
    access_token = Column(Text)  # Encrypted OAuth access token
    refresh_token = Column(Text)  # Encrypted OAuth refresh token
    token_expires_at = Column(DateTime)
    scopes = Column(JSON)  # List of granted permission scopes

    # Capabilities
    has_uv_sensor = Column(Boolean, default=False)  # Device has built-in UV sensor
    has_location = Column(Boolean, default=True)  # Can provide location for UV index lookup
    has_activity_tracking = Column(Boolean, default=True)  # Can detect outdoor activity
    has_heart_rate = Column(Boolean, default=True)
    supported_data_types = Column(JSON)  # ["uv_exposure", "activity", "location", "heart_rate", "sleep"]

    # Sync statistics
    total_syncs = Column(Integer, default=0)
    successful_syncs = Column(Integer, default=0)
    failed_syncs = Column(Integer, default=0)
    total_uv_readings = Column(Integer, default=0)

    # Settings
    auto_sync_enabled = Column(Boolean, default=True)
    uv_alert_threshold = Column(Float, default=6.0)  # UV index threshold for alerts
    outdoor_detection_enabled = Column(Boolean, default=True)  # Auto-detect outdoor activity

    # Timestamps
    connected_at = Column(DateTime, default=datetime.utcnow)
    disconnected_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="wearable_devices")


class WearableUVReading(Base):
    """
    Individual UV exposure readings from wearable devices.
    Stores granular UV data for correlation with lesion changes.
    """
    __tablename__ = "wearable_uv_readings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    device_id = Column(Integer, ForeignKey("wearable_devices.id"), nullable=False, index=True)

    # Reading timestamp and duration
    reading_timestamp = Column(DateTime, nullable=False, index=True)
    duration_seconds = Column(Integer, default=60)  # Duration this reading covers
    reading_date = Column(Date, nullable=False, index=True)  # For daily aggregation

    # UV data
    uv_index = Column(Float)  # Current UV index (0-11+)
    uv_dose = Column(Float)  # UV dose in J/m² or MED (Minimal Erythemal Dose)
    uv_dose_unit = Column(String, default="index_minutes")  # "index_minutes", "joules_per_m2", "med"

    # UV components (if available from device)
    uva_level = Column(Float)  # UVA component
    uvb_level = Column(Float)  # UVB component

    # Source of UV data
    uv_source = Column(String, nullable=False)  # "device_sensor", "location_api", "weather_api", "manual"

    # Location context (for UV index lookup if no sensor)
    latitude = Column(Float)
    longitude = Column(Float)
    altitude_meters = Column(Float)
    location_name = Column(String)  # "Central Park, NYC" or reverse geocoded

    # Activity context
    activity_type = Column(String)  # "outdoor_walk", "run", "cycling", "swimming", "stationary", "indoor"
    activity_confidence = Column(Float)  # 0-1 confidence that user was outdoors
    is_outdoor = Column(Boolean)  # Definitive outdoor flag
    steps_during_reading = Column(Integer)

    # Environmental context
    weather_condition = Column(String)  # "sunny", "partly_cloudy", "cloudy", "rainy"
    cloud_cover_percent = Column(Float)  # 0-100
    temperature_celsius = Column(Float)
    humidity_percent = Column(Float)

    # Protection detection (from activity patterns)
    likely_protected = Column(Boolean)  # Inferred if user was likely using protection
    indoor_transition_detected = Column(Boolean)  # Moved indoors during reading

    # Body exposure estimation
    estimated_body_exposure = Column(JSON)  # {"face": 0.8, "arms": 0.6, "legs": 0.3} based on activity

    # Risk scoring
    reading_risk_score = Column(Float)  # 0-100 risk score for this reading
    cumulative_daily_dose = Column(Float)  # Running total for the day

    # Sync metadata
    synced_at = Column(DateTime, default=datetime.utcnow)
    raw_device_data = Column(JSON)  # Original data from device API

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="wearable_uv_readings")
    device = relationship("WearableDevice", backref="uv_readings")


class WearableDailyUVSummary(Base):
    """
    Daily aggregated UV exposure summary from wearables.
    Used for trend analysis and lesion correlation.
    """
    __tablename__ = "wearable_daily_uv_summaries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    summary_date = Column(Date, nullable=False, index=True)

    # Unique constraint: one summary per user per day
    __table_args__ = (
        # UniqueConstraint('user_id', 'summary_date', name='unique_user_daily_summary'),
    )

    # Aggregated UV exposure
    total_outdoor_minutes = Column(Integer, default=0)
    total_uv_dose = Column(Float, default=0)  # Cumulative UV dose
    average_uv_index = Column(Float)
    max_uv_index = Column(Float)
    min_uv_index = Column(Float)

    # Time-based breakdown
    morning_exposure_minutes = Column(Integer, default=0)  # Before 10am
    midday_exposure_minutes = Column(Integer, default=0)  # 10am-4pm (highest UV)
    afternoon_exposure_minutes = Column(Integer, default=0)  # After 4pm
    peak_uv_hour = Column(Integer)  # Hour with highest UV exposure (0-23)

    # High exposure events
    high_uv_minutes = Column(Integer, default=0)  # Minutes at UV >= 6
    very_high_uv_minutes = Column(Integer, default=0)  # Minutes at UV >= 8
    extreme_uv_minutes = Column(Integer, default=0)  # Minutes at UV >= 11

    # Activity breakdown
    active_outdoor_minutes = Column(Integer, default=0)  # Walking, running, etc.
    stationary_outdoor_minutes = Column(Integer, default=0)  # Beach, park, etc.

    # Location patterns
    primary_location = Column(String)  # Most common outdoor location
    locations_visited = Column(JSON)  # List of outdoor locations

    # Protection compliance (if tracked)
    protected_exposure_minutes = Column(Integer)  # When protection was likely used
    unprotected_exposure_minutes = Column(Integer)

    # Risk assessment
    daily_risk_score = Column(Float)  # 0-100 risk score for the day
    risk_category = Column(String)  # "low", "moderate", "high", "very_high"
    exceeded_recommended_dose = Column(Boolean, default=False)

    # Data quality
    reading_count = Column(Integer, default=0)  # Number of readings aggregated
    data_completeness = Column(Float)  # 0-1 how complete the day's data is
    devices_contributing = Column(JSON)  # List of device IDs that contributed

    # Alerts generated
    alerts_triggered = Column(JSON)  # List of alerts sent (high UV, etc.)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="wearable_daily_summaries")


class WearableLesionCorrelation(Base):
    """
    Correlation analysis between wearable UV data and lesion changes.
    Links UV exposure patterns to specific lesion developments.
    """
    __tablename__ = "wearable_lesion_correlations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    lesion_group_id = Column(Integer, ForeignKey("lesion_groups.id"), nullable=False, index=True)

    # Analysis window
    analysis_start_date = Column(Date, nullable=False)
    analysis_end_date = Column(Date, nullable=False)
    days_analyzed = Column(Integer)

    # Lesion change metrics
    lesion_body_location = Column(String)  # "face", "left_arm", "back", etc.
    lesion_change_type = Column(String)  # "new", "growth", "color_change", "border_change", "regression"
    lesion_change_severity = Column(Float)  # 0-10 severity of change
    lesion_first_detected = Column(DateTime)
    lesion_change_detected = Column(DateTime)

    # UV exposure to affected area
    uv_exposure_to_area_minutes = Column(Integer)  # Total exposure to lesion body area
    uv_dose_to_area = Column(Float)  # UV dose specifically to affected area
    high_uv_events_to_area = Column(Integer)  # High UV events exposing that area
    sunburn_events_to_area = Column(Integer)

    # Historical UV patterns (30/60/90 days before change)
    uv_30_day_avg = Column(Float)  # Average daily UV dose 30 days before
    uv_60_day_avg = Column(Float)
    uv_90_day_avg = Column(Float)
    uv_trend = Column(String)  # "increasing", "stable", "decreasing"

    # Cumulative exposure
    cumulative_uv_dose = Column(Float)  # Total UV dose during analysis period
    cumulative_outdoor_hours = Column(Float)
    sunburn_count = Column(Integer)  # Number of sunburn events

    # Correlation statistics
    correlation_coefficient = Column(Float)  # Pearson correlation -1 to 1
    correlation_p_value = Column(Float)  # Statistical significance
    correlation_strength = Column(String)  # "strong", "moderate", "weak", "none", "inverse"
    correlation_confidence = Column(Float)  # 0-1 confidence in correlation

    # Contributing factors
    contributing_factors = Column(JSON)  # ["high_midday_exposure", "no_protection", "altitude"]
    protective_factors = Column(JSON)  # ["consistent_sunscreen", "morning_only_exposure"]

    # Risk assessment
    uv_contribution_score = Column(Float)  # 0-100 how much UV likely contributed
    other_factors_score = Column(Float)  # 0-100 non-UV factors
    overall_risk_attribution = Column(String)  # "primarily_uv", "partially_uv", "unlikely_uv"

    # Recommendations
    personalized_recommendations = Column(JSON)  # List of specific recommendations
    recommended_max_daily_exposure = Column(Integer)  # Minutes
    recommended_spf = Column(Integer)
    high_risk_times = Column(JSON)  # ["10:00-14:00"] times to avoid

    # AI analysis
    ai_analysis_summary = Column(Text)  # LLM-generated summary
    ai_confidence = Column(Float)  # AI confidence in analysis

    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="wearable_lesion_correlations")
    lesion_group = relationship("LesionGroup", backref="wearable_correlations")


# =============================================================================
# INSURANCE APPEALS MODELS
# =============================================================================

class InsuranceAppeal(Base):
    """
    Insurance Appeal - Track insurance claim appeals and their outcomes.
    Supports multiple appeal levels and tracks the full appeal workflow.
    """
    __tablename__ = "insurance_appeals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=True, index=True)

    # Appeal identification
    appeal_id = Column(String, unique=True, index=True)  # External appeal ID (e.g., "APL-20241218-001")
    claim_number = Column(String, index=True)  # Original insurance claim number

    # Insurance details
    insurance_company = Column(String, nullable=False)
    policy_number = Column(String)
    group_number = Column(String)
    subscriber_id = Column(String)

    # Original claim details
    date_of_service = Column(DateTime)
    original_claim_amount = Column(Float)
    diagnosis = Column(String)
    icd10_code = Column(String)
    procedure = Column(String)
    cpt_code = Column(String)

    # Denial information
    denial_date = Column(DateTime)
    denial_reason = Column(String, nullable=False)  # "medical_necessity", "not_covered", etc.
    denial_reason_text = Column(Text)  # Full text of denial reason
    denial_code = Column(String)  # Insurance company denial code

    # Appeal details
    appeal_level = Column(String, default="first_level")  # "first_level", "second_level", "external_review", "state_insurance"
    appeal_status = Column(String, default="draft", index=True)  # "draft", "submitted", "under_review", "additional_info_requested", "approved", "denied", "escalated"

    # Appeal letter content
    letter_content = Column(Text)  # Generated appeal letter
    subject_line = Column(String)
    key_arguments = Column(JSON)  # List of key arguments made
    supporting_evidence = Column(JSON)  # List of supporting evidence mentioned
    supporting_documents = Column(JSON)  # List of documents attached

    # Success metrics
    success_likelihood = Column(Integer)  # 0-100 estimated success likelihood
    recommended_next_steps = Column(JSON)  # List of recommended actions

    # Provider information
    provider_name = Column(String)
    provider_npi = Column(String)
    provider_address = Column(Text)
    provider_phone = Column(String)
    provider_fax = Column(String)

    # Patient information (denormalized for appeal letter)
    patient_name = Column(String)
    patient_dob = Column(String)
    patient_address = Column(Text)
    patient_phone = Column(String)

    # Timeline tracking
    submitted_date = Column(DateTime)
    deadline = Column(DateTime)  # Appeal deadline
    response_due_date = Column(DateTime)  # When insurance must respond
    decision_date = Column(DateTime)  # When decision was received

    # Outcome
    outcome = Column(String)  # "approved", "denied", "partial_approval"
    outcome_amount = Column(Float)  # Amount approved if partial
    outcome_notes = Column(Text)  # Notes about the outcome

    # Escalation tracking
    escalated_from_id = Column(Integer, ForeignKey("insurance_appeals.id"), nullable=True)
    escalation_reason = Column(Text)

    # Communication log
    communications = Column(JSON)  # List of {date, type, summary, attachments}

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="insurance_appeals")
    analysis = relationship("AnalysisHistory", backref="insurance_appeals")
    escalated_from = relationship("InsuranceAppeal", remote_side=[id], backref="escalations")


def migrate_insurance_appeals_table():
    """Create insurance_appeals table if it doesn't exist."""
    try:
        InsuranceAppeal.__table__.create(engine, checkfirst=True)
        print("InsuranceAppeal table created/verified successfully!")
    except Exception as e:
        print(f"Error creating InsuranceAppeal table: {e}")


def migrate_wearable_tables():
    """Create wearable integration tables if they don't exist."""
    try:
        WearableDevice.__table__.create(engine, checkfirst=True)
        WearableUVReading.__table__.create(engine, checkfirst=True)
        WearableDailyUVSummary.__table__.create(engine, checkfirst=True)
        WearableLesionCorrelation.__table__.create(engine, checkfirst=True)
        print("Wearable integration tables created/verified successfully!")
    except Exception as e:
        print(f"Error creating wearable tables: {e}")


def migrate_clinical_trials_tables():
    """Create clinical trials related tables if they don't exist."""
    try:
        ClinicalTrial.__table__.create(engine, checkfirst=True)
        TrialMatch.__table__.create(engine, checkfirst=True)
        TrialInterest.__table__.create(engine, checkfirst=True)
        print("Clinical trials tables created/verified successfully!")
    except Exception as e:
        print(f"Error creating clinical trials tables: {e}")


def migrate_system_alerts_table():
    """Create system_alerts table if it doesn't exist."""
    try:
        SystemAlert.__table__.create(engine, checkfirst=True)
        print("SystemAlert table created/verified successfully!")
    except Exception as e:
        print(f"Error creating SystemAlert table: {e}")


def migrate_batch_skin_check_table():
    """Create batch_skin_checks table if it doesn't exist."""
    try:
        BatchSkinCheck.__table__.create(engine, checkfirst=True)
        print("BatchSkinCheck table created/verified successfully!")
    except Exception as e:
        print(f"Error creating BatchSkinCheck table: {e}")


def migrate_consensus_tables():
    """Create consensus-related tables if they don't exist."""
    try:
        # Create system alerts table
        migrate_system_alerts_table()

        # Create batch skin check table
        migrate_batch_skin_check_table()

        # Create the tables
        ConsensusCase.__table__.create(engine, checkfirst=True)
        ConsensusOpinion.__table__.create(engine, checkfirst=True)
        ConsensusAssignment.__table__.create(engine, checkfirst=True)
        print("Consensus tables created/verified successfully!")

        # Add user_id to dermatologist_profiles if it doesn't exist
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(dermatologist_profiles)"))
            columns = [row[1] for row in result.fetchall()]

            if 'user_id' not in columns:
                print("Adding user_id field to dermatologist_profiles...")
                conn.execute(text("ALTER TABLE dermatologist_profiles ADD COLUMN user_id INTEGER REFERENCES users(id)"))
                conn.commit()
                print("Successfully added user_id to dermatologist_profiles!")

    except Exception as e:
        print(f"Error creating consensus tables: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()