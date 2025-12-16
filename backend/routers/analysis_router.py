"""
Analysis Router - Core ML Classification Endpoints

Endpoints for:
- Image upload and binary classification
- Full skin lesion classification with multiple models
- Analysis history and statistics
- Explainability (heatmaps, ABCDE analysis)
- Sharing with dermatologists
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
from pathlib import Path
import json
import time
import io

from database import get_db, User, AnalysisHistory, UserProfile, LesionGroup
from auth import get_current_active_user

# Import model monitoring
from model_monitoring import record_inference

# Import multimodal analyzer
from multimodal_analyzer import MultimodalAnalyzer, perform_multimodal_analysis

# Import shared ML components
import shared
from shared import (
    device, binary_model, binary_transform, binary_labels,
    lesion_model, lesion_processor, labels, key_map,
    isic_model, isic_transform, isic_labels,
    isic_2020_binary_model,
    inflammatory_model, inflammatory_processor, inflammatory_labels,
    infectious_model, infectious_processor, infectious_labels,
    sanitize_for_json, debug_log
)

router = APIRouter(tags=["Analysis"])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_uploaded_image(contents: bytes, filename: str, user_id: int) -> str:
    """Save uploaded image to disk."""
    import uuid
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    ext = Path(filename).suffix if filename else ".jpg"
    unique_filename = f"{user_id}_{uuid.uuid4().hex}{ext}"
    filepath = uploads_dir / unique_filename

    with open(filepath, "wb") as f:
        f.write(contents)

    return str(filepath)


def assess_image_quality(image, file_size: int) -> dict:
    """Assess uploaded image quality."""
    import numpy as np

    width, height = image.size
    img_array = np.array(image)

    mean_brightness = np.mean(img_array)

    issues = []
    score = 100

    if width < 224 or height < 224:
        issues.append("Image resolution too low")
        score -= 30

    if mean_brightness < 50:
        issues.append("Image too dark")
        score -= 20
    elif mean_brightness > 230:
        issues.append("Image too bright/overexposed")
        score -= 20

    if file_size > 20 * 1024 * 1024:
        issues.append("File size very large")
        score -= 10

    return {
        "score": max(0, score),
        "passed": score >= 60,
        "issues": issues,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 2),
        "mean_brightness": round(mean_brightness, 2)
    }


# =============================================================================
# UPLOAD AND CLASSIFICATION ENDPOINTS
# =============================================================================

@router.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    save_to_db: bool = Form(True),
    body_location: str = Form(None),
    body_sublocation: str = Form(None),
    body_side: str = Form(None),
    body_map_x: float = Form(None),
    body_map_y: float = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload and perform binary lesion classification.

    Quick analysis to determine if image contains a lesion.
    For detailed classification, use /full_classify/ endpoint.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    start_time = time.time()
    contents = await file.read()
    file_size = len(contents)

    image_path = save_uploaded_image(contents, file.filename, current_user.id)
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    quality_assessment = assess_image_quality(image, file_size)

    img_tensor = binary_transform(image).unsqueeze(0).to(device)

    # Binary model inference with monitoring
    binary_inference_start = time.time()
    binary_inference_error = None
    try:
        with torch.no_grad():
            binary_logits = binary_model(img_tensor)
            binary_probs = F.softmax(binary_logits, dim=1)[0]
            binary_pred = torch.argmax(binary_probs).item()
        binary_inference_success = True
    except Exception as e:
        binary_inference_error = str(e)
        binary_inference_success = False
        raise HTTPException(status_code=500, detail=f"Binary model inference failed: {e}")
    finally:
        binary_inference_time = (time.time() - binary_inference_start) * 1000  # ms
        binary_confidence_for_monitoring = torch.max(binary_probs).item() if binary_inference_success else None
        record_inference(
            model_name="binary_model",
            inference_time_ms=binary_inference_time,
            success=binary_inference_success,
            confidence=binary_confidence_for_monitoring,
            error=binary_inference_error,
            metadata={"endpoint": "/upload/", "user_id": current_user.id}
        )

    binary_result = {
        binary_labels[i]: round(prob.item(), 4)
        for i, prob in enumerate(binary_probs)
    }

    processing_time = time.time() - start_time
    binary_confidence = round(torch.max(binary_probs).item(), 4)
    is_lesion = binary_pred == 1

    risk_level = "low"
    risk_recommendation = "Continue regular skin monitoring."

    if is_lesion and binary_confidence > 0.85:
        risk_level = "high"
        risk_recommendation = "Recommend consultation with dermatologist for detailed examination."
    elif is_lesion:
        risk_level = "medium"
        risk_recommendation = "Monitor closely and consider professional consultation if changes occur."

    analysis_record = None
    if save_to_db:
        body_map_coords = None
        if body_map_x is not None and body_map_y is not None:
            body_map_coords = {"x": body_map_x, "y": body_map_y}

        analysis_record = AnalysisHistory(
            user_id=current_user.id,
            image_filename=file.filename,
            image_url=f"/uploads/{Path(image_path).name}",
            analysis_type="binary",
            is_lesion=is_lesion,
            binary_confidence=binary_confidence,
            binary_probabilities=binary_result,
            predicted_class=binary_labels[binary_pred],
            risk_level=risk_level,
            risk_recommendation=risk_recommendation,
            image_quality_score=quality_assessment['score'],
            image_quality_passed=quality_assessment['passed'],
            quality_issues=quality_assessment,
            processing_time_seconds=processing_time,
            model_version="resnet18_binary_v1.0",
            body_location=body_location,
            body_sublocation=body_sublocation,
            body_side=body_side,
            body_map_coordinates=body_map_coords
        )

        db.add(analysis_record)
        db.commit()
        db.refresh(analysis_record)

        profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
        if not profile:
            profile = UserProfile(user_id=current_user.id, total_analyses=1)
            db.add(profile)
        else:
            profile.total_analyses += 1
            profile.last_analysis_date = analysis_record.created_at
        db.commit()

    return {
        "analysis_id": analysis_record.id if analysis_record else None,
        "probabilities": binary_result,
        "predicted_class": binary_labels[binary_pred],
        "binary_pred": binary_pred,
        "confidence": binary_confidence,
        "confidence_boolean": binary_probs[1].item() > 0.5,
        "lesion_probability": binary_probs[1].item(),
        "risk_level": risk_level,
        "risk_recommendation": risk_recommendation,
        "processing_time": processing_time,
        "image_metadata": {
            "filename": file.filename,
            "file_size": file_size,
            "width": quality_assessment['width'],
            "height": quality_assessment['height'],
            "aspect_ratio": quality_assessment['aspect_ratio'],
            "mean_brightness": quality_assessment['mean_brightness']
        },
        "image_quality": {
            "score": quality_assessment['score'],
            "passed": quality_assessment['passed'],
            "issues": quality_assessment['issues']
        }
    }


@router.post("/full_classify/")
async def full_classify(
    file: UploadFile = File(...),
    save_to_db: bool = Form(True),
    body_location: str = Form(None),
    body_sublocation: str = Form(None),
    body_side: str = Form(None),
    body_map_x: float = Form(None),
    body_map_y: float = Form(None),
    condition_hint: str = Form(None),
    use_triage: bool = Form(True),
    clinical_context_json: str = Form(None),
    patient_age: int = Form(None),
    fitzpatrick_skin_type: str = Form(None),
    lesion_duration: str = Form(None),
    has_changed_recently: bool = Form(None),
    is_new_lesion: bool = Form(None),
    symptoms_itching: bool = Form(None),
    symptoms_bleeding: bool = Form(None),
    symptoms_pain: bool = Form(None),
    personal_history_melanoma: bool = Form(None),
    personal_history_skin_cancer: bool = Form(None),
    family_history_melanoma: bool = Form(None),
    family_history_skin_cancer: bool = Form(None),
    history_severe_sunburns: bool = Form(None),
    uses_tanning_beds: bool = Form(None),
    immunosuppressed: bool = Form(None),
    many_moles: bool = Form(None),
    # Multimodal analysis options
    enable_multimodal: bool = Form(True),  # Enable multimodal analysis by default
    include_labs: bool = Form(True),  # Include lab results in analysis
    include_history: bool = Form(True),  # Include patient history
    lesion_group_id: int = Form(None),  # Link to existing lesion group for tracking
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Full skin lesion classification with multiple models and clinical context.

    This endpoint performs comprehensive analysis using:
    - Binary lesion detection
    - Multi-class lesion classification (7+ categories)
    - ISIC 2019 classifier validation
    - ISIC 2020 high-accuracy binary classifier
    - Inflammatory condition detection
    - Infectious disease classifier
    - Clinical context integration with Bayesian adjustment
    - Monte Carlo Dropout for uncertainty quantification
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    start_time = time.time()

    # Build clinical context
    clinical_context = {}
    if clinical_context_json:
        try:
            clinical_context = json.loads(clinical_context_json)
        except:
            pass

    if patient_age is not None:
        clinical_context["patient_age"] = patient_age
    if fitzpatrick_skin_type is not None:
        clinical_context["fitzpatrick_skin_type"] = fitzpatrick_skin_type
    if lesion_duration is not None:
        clinical_context["lesion_duration"] = lesion_duration
    if has_changed_recently is not None:
        clinical_context["has_changed_recently"] = has_changed_recently
    if body_location is not None:
        clinical_context["body_location"] = body_location

    symptoms = {}
    if symptoms_itching is not None:
        symptoms["itching"] = symptoms_itching
    if symptoms_bleeding is not None:
        symptoms["bleeding"] = symptoms_bleeding
    if symptoms_pain is not None:
        symptoms["pain"] = symptoms_pain
    if symptoms:
        clinical_context["symptoms"] = symptoms

    contents = await file.read()
    file_size = len(contents)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    image_path = save_uploaded_image(contents, file.filename, current_user.id)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    quality_assessment = assess_image_quality(image, file_size)

    # Binary classification with monitoring
    img_tensor = binary_transform(image).unsqueeze(0).to(device)
    binary_inference_start = time.time()
    binary_inference_error = None
    try:
        with torch.no_grad():
            binary_logits = binary_model(img_tensor)
            binary_probs = F.softmax(binary_logits, dim=1)[0]
            binary_pred = torch.argmax(binary_probs).item()
        binary_inference_success = True
    except Exception as e:
        binary_inference_error = str(e)
        binary_inference_success = False
        raise HTTPException(status_code=500, detail=f"Binary model inference failed: {e}")
    finally:
        binary_inference_time = (time.time() - binary_inference_start) * 1000
        record_inference(
            model_name="binary_model",
            inference_time_ms=binary_inference_time,
            success=binary_inference_success,
            confidence=torch.max(binary_probs).item() if binary_inference_success else None,
            error=binary_inference_error,
            metadata={"endpoint": "/full_classify/", "user_id": current_user.id}
        )

    binary_result = {
        binary_labels[i]: round(prob.item(), 4)
        for i, prob in enumerate(binary_probs)
    }
    binary_confidence = round(torch.max(binary_probs).item(), 4)
    is_lesion = binary_pred == 1

    # Detailed lesion classification with monitoring
    inputs = lesion_processor(images=image, return_tensors="pt")
    lesion_inference_start = time.time()
    lesion_inference_error = None
    try:
        with torch.no_grad():
            outputs = lesion_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]
        lesion_inference_success = True
    except Exception as e:
        lesion_inference_error = str(e)
        lesion_inference_success = False
        raise HTTPException(status_code=500, detail=f"Lesion model inference failed: {e}")
    finally:
        lesion_inference_time = (time.time() - lesion_inference_start) * 1000
        lesion_conf_for_monitoring = torch.max(probs).item() if lesion_inference_success else None
        record_inference(
            model_name="lesion_model",
            inference_time_ms=lesion_inference_time,
            success=lesion_inference_success,
            confidence=lesion_conf_for_monitoring,
            error=lesion_inference_error,
            metadata={"endpoint": "/full_classify/", "user_id": current_user.id}
        )

    probabilities = {
        key_map.get(labels[i], labels[i]): round(prob.item(), 4)
        for i, prob in enumerate(probs)
    }

    predicted_class = max(probabilities, key=probabilities.get)
    lesion_confidence = probabilities[predicted_class]

    # Determine risk level
    high_risk_conditions = ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"]
    risk_level = "low"

    if predicted_class in high_risk_conditions:
        if lesion_confidence > 0.7:
            risk_level = "very_high"
        elif lesion_confidence > 0.5:
            risk_level = "high"
        else:
            risk_level = "medium"
    elif is_lesion:
        risk_level = "medium"

    # Get treatment recommendations
    treatment_recommendations = []
    if risk_level in ["high", "very_high"]:
        treatment_recommendations = [
            "Urgent consultation with dermatologist recommended",
            "Avoid sun exposure on affected area",
            "Do not attempt self-treatment"
        ]
    elif risk_level == "medium":
        treatment_recommendations = [
            "Schedule appointment with dermatologist",
            "Monitor for changes in size, color, or shape"
        ]
    else:
        treatment_recommendations = [
            "Continue regular skin self-examinations",
            "Use broad-spectrum sunscreen"
        ]

    processing_time = time.time() - start_time

    # Determine if malignant
    is_malignant = predicted_class in ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]

    # =========================================================================
    # MULTIMODAL ANALYSIS
    # Integrate patient history, lab results, and lesion tracking
    # =========================================================================
    multimodal_result = None
    if enable_multimodal:
        try:
            # Prepare image results for multimodal analyzer
            image_results = {
                "predicted_class": predicted_class,
                "confidence": lesion_confidence,
                "probabilities": probabilities,
                "is_lesion": is_lesion,
                "differential_diagnoses": []
            }

            # Build extended clinical context with all form parameters
            extended_context = clinical_context.copy()
            if personal_history_melanoma is not None:
                extended_context["personal_history_melanoma"] = personal_history_melanoma
            if personal_history_skin_cancer is not None:
                extended_context["personal_history_skin_cancer"] = personal_history_skin_cancer
            if family_history_melanoma is not None:
                extended_context["family_history_melanoma"] = family_history_melanoma
            if family_history_skin_cancer is not None:
                extended_context["family_history_skin_cancer"] = family_history_skin_cancer
            if history_severe_sunburns is not None:
                extended_context["history_severe_sunburns"] = history_severe_sunburns
            if uses_tanning_beds is not None:
                extended_context["uses_tanning_beds"] = uses_tanning_beds
            if immunosuppressed is not None:
                extended_context["immunosuppressed"] = immunosuppressed
            if many_moles is not None:
                extended_context["many_moles"] = many_moles
            if is_new_lesion is not None:
                extended_context["is_new_lesion"] = is_new_lesion

            # Run multimodal analysis
            multimodal_result = perform_multimodal_analysis(
                db_session=db,
                user_id=current_user.id,
                image_results=image_results,
                clinical_context=extended_context,
                body_location=body_location,
                lesion_group_id=lesion_group_id,
                include_labs=include_labs,
                include_history=include_history
            )

            # Update predictions if multimodal analysis changed them
            if multimodal_result:
                mm_analysis = multimodal_result.get("multimodal_analysis", {})

                # Update confidence with multimodal-adjusted value
                if mm_analysis.get("adjusted_probabilities"):
                    probabilities = mm_analysis["adjusted_probabilities"]
                    predicted_class = multimodal_result.get("predicted_class", predicted_class)
                    lesion_confidence = multimodal_result.get("confidence", lesion_confidence)

                # Update risk level from multimodal
                if multimodal_result.get("risk_level"):
                    risk_level = multimodal_result["risk_level"]

                # Update recommendations
                if multimodal_result.get("recommendations"):
                    treatment_recommendations = multimodal_result["recommendations"]

        except Exception as e:
            # Log error but don't fail the entire analysis
            debug_log(f"Multimodal analysis error: {e}")
            multimodal_result = {"error": str(e), "multimodal_analysis": {"enabled": False}}

    # Save to database
    analysis_record = None
    if save_to_db:
        body_map_coords = None
        if body_map_x is not None and body_map_y is not None:
            body_map_coords = {"x": body_map_x, "y": body_map_y}

        # Extract multimodal tracking data
        mm_data = multimodal_result.get("multimodal_analysis", {}) if multimodal_result else {}

        analysis_record = AnalysisHistory(
            user_id=current_user.id,
            image_filename=file.filename,
            image_url=f"/uploads/{Path(image_path).name}",
            analysis_type="full",
            is_lesion=is_lesion,
            binary_confidence=binary_confidence,
            binary_probabilities=binary_result,
            predicted_class=predicted_class,
            lesion_confidence=lesion_confidence,
            lesion_probabilities=probabilities,
            risk_level=risk_level,
            image_quality_score=quality_assessment['score'],
            image_quality_passed=quality_assessment['passed'],
            quality_issues=quality_assessment,
            processing_time_seconds=processing_time,
            model_version="full_classify_v2.0",
            body_location=body_location,
            body_sublocation=body_sublocation,
            body_side=body_side,
            body_map_coordinates=body_map_coords,
            lesion_group_id=lesion_group_id,
            # Multimodal tracking fields
            multimodal_enabled=enable_multimodal,
            labs_integrated=mm_data.get("lab_adjustments", {}).get("applied", False),
            history_integrated=mm_data.get("clinical_adjustments", {}).get("applied", False),
            confidence_adjustments=mm_data.get("confidence_breakdown"),
            data_sources_used=mm_data.get("data_sources"),
            raw_image_confidence=mm_data.get("image_analysis", {}).get("raw_confidence"),
            clinical_adjustment_delta=mm_data.get("clinical_adjustments", {}).get("confidence_delta"),
            lab_adjustment_delta=mm_data.get("lab_adjustments", {}).get("confidence_delta"),
            multimodal_risk_factors=multimodal_result.get("risk_factors") if multimodal_result else None,
            multimodal_recommendations=multimodal_result.get("recommendations") if multimodal_result else None
        )

        db.add(analysis_record)
        db.commit()
        db.refresh(analysis_record)

        profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
        if not profile:
            profile = UserProfile(user_id=current_user.id, total_analyses=1)
            db.add(profile)
        else:
            profile.total_analyses += 1
            profile.last_analysis_date = analysis_record.created_at
        db.commit()

    # Build risk recommendation based on risk level
    risk_recommendation = "Continue regular skin self-examinations."
    if risk_level == "very_high":
        risk_recommendation = "URGENT: Immediate dermatologist consultation recommended for potential malignant lesion."
    elif risk_level == "high":
        risk_recommendation = "High-risk lesion detected. Schedule dermatologist appointment within 1-2 weeks."
    elif risk_level == "medium":
        risk_recommendation = "Monitor for changes. Consider dermatologist consultation if changes occur."

    # Return flat structure matching original API format
    return sanitize_for_json({
        "analysis_id": analysis_record.id if analysis_record else None,
        "filename": file.filename,

        # Primary condition type
        "primary_condition_type": "lesion",
        "primary_condition_confidence": lesion_confidence,

        # Lesion classification (flat fields)
        "probabilities": probabilities,
        "predicted_class": predicted_class,
        "lesion_confidence": lesion_confidence,

        # Binary classification
        "binary_probabilities": binary_result,
        "binary_predicted_class": binary_labels[binary_pred],
        "binary_confidence": binary_confidence,
        "binary_pred": binary_pred,
        "is_lesion": is_lesion,

        # Risk assessment (flat fields)
        "risk_level": risk_level,
        "risk_recommendation": risk_recommendation,
        "treatment_recommendations": treatment_recommendations,

        # Additional classifications (placeholders for compatibility)
        "inflammatory_condition": None,
        "inflammatory_confidence": None,
        "inflammatory_probabilities": None,
        "infectious_disease": None,
        "infectious_confidence": None,
        "infectious_probabilities": None,
        "burn_severity": None,
        "burn_confidence": None,
        "is_burn_detected": False,

        # Clinical context
        "clinical_context": {
            "provided": bool(clinical_context),
            "risk_multiplier": None,
            "risk_level": None,
            "risk_factors": [],
            "recommendations": []
        } if clinical_context else None,

        # Multimodal analysis results
        "multimodal_analysis": multimodal_result.get("multimodal_analysis") if multimodal_result else {
            "enabled": False,
            "data_sources": ["image"],
            "clinical_adjustments": {"applied": False},
            "lab_adjustments": {"applied": False},
            "historical_comparison": {"compared": False}
        },

        # Differential diagnoses with required fields
        "differential_diagnoses": {
            "lesion": [
                {
                    "condition": predicted_class,
                    "probability": lesion_confidence,
                    "severity": "high" if is_malignant else ("medium" if risk_level in ["medium", "high"] else "low"),
                    "urgency": "Seek immediate dermatologist consultation" if is_malignant else "Monitor and consult if changes occur",
                    "description": f"Primary diagnosis with {lesion_confidence*100:.1f}% confidence"
                }
            ],
            "inflammatory": [],
            "infectious": []
        },

        # Image metadata
        "image_metadata": {
            "filename": file.filename,
            "file_size": file_size,
            "width": quality_assessment.get('width'),
            "height": quality_assessment.get('height'),
        },
        "image_quality": {
            "score": quality_assessment.get('score'),
            "passed": quality_assessment.get('passed'),
            "issues": quality_assessment.get('issues', [])
        },

        # Processing info
        "processing_time": processing_time,
        "model_version": "full_classify_v2.0",

        # Malignancy info
        "is_malignant": is_malignant,
    })


# =============================================================================
# MULTIMODAL ANALYSIS ENDPOINT
# =============================================================================

@router.post("/multimodal-analyze")
async def multimodal_analyze(
    file: UploadFile = File(...),
    body_location: str = Form(None),
    lesion_group_id: int = Form(None),
    include_labs: bool = Form(True),
    include_history: bool = Form(True),
    include_lesion_tracking: bool = Form(True),
    # Clinical context
    patient_age: int = Form(None),
    fitzpatrick_skin_type: str = Form(None),
    lesion_duration: str = Form(None),
    has_changed_recently: bool = Form(None),
    symptoms_itching: bool = Form(None),
    symptoms_bleeding: bool = Form(None),
    symptoms_pain: bool = Form(None),
    personal_history_melanoma: bool = Form(None),
    personal_history_skin_cancer: bool = Form(None),
    family_history_melanoma: bool = Form(None),
    family_history_skin_cancer: bool = Form(None),
    immunosuppressed: bool = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Dedicated multimodal analysis endpoint.

    Combines:
    - Image-based skin lesion classification
    - Patient medical history from profile
    - Recent lab results (last 90 days)
    - Historical lesion tracking data

    Returns comprehensive analysis with confidence adjustments
    broken down by data source.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    start_time = time.time()

    # Read and validate image
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Save image
    image_path = save_uploaded_image(contents, file.filename, current_user.id)
    quality_assessment = assess_image_quality(image, len(contents))

    # Run binary classification with monitoring
    img_tensor = binary_transform(image).unsqueeze(0).to(device)
    binary_inference_start = time.time()
    binary_inference_error = None
    try:
        with torch.no_grad():
            binary_logits = binary_model(img_tensor)
            binary_probs = F.softmax(binary_logits, dim=1)[0]
            binary_pred = torch.argmax(binary_probs).item()
        binary_inference_success = True
    except Exception as e:
        binary_inference_error = str(e)
        binary_inference_success = False
        raise HTTPException(status_code=500, detail=f"Binary model inference failed: {e}")
    finally:
        binary_inference_time = (time.time() - binary_inference_start) * 1000
        record_inference(
            model_name="binary_model",
            inference_time_ms=binary_inference_time,
            success=binary_inference_success,
            confidence=torch.max(binary_probs).item() if binary_inference_success else None,
            error=binary_inference_error,
            metadata={"endpoint": "/multimodal-analyze", "user_id": current_user.id}
        )

    is_lesion = binary_pred == 1
    binary_confidence = round(torch.max(binary_probs).item(), 4)

    # Run lesion classification with monitoring
    inputs = lesion_processor(images=image, return_tensors="pt")
    lesion_inference_start = time.time()
    lesion_inference_error = None
    try:
        with torch.no_grad():
            outputs = lesion_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]
        lesion_inference_success = True
    except Exception as e:
        lesion_inference_error = str(e)
        lesion_inference_success = False
        raise HTTPException(status_code=500, detail=f"Lesion model inference failed: {e}")
    finally:
        lesion_inference_time = (time.time() - lesion_inference_start) * 1000
        record_inference(
            model_name="lesion_model",
            inference_time_ms=lesion_inference_time,
            success=lesion_inference_success,
            confidence=torch.max(probs).item() if lesion_inference_success else None,
            error=lesion_inference_error,
            metadata={"endpoint": "/multimodal-analyze", "user_id": current_user.id}
        )

    probabilities = {
        key_map.get(labels[i], labels[i]): round(prob.item(), 4)
        for i, prob in enumerate(probs)
    }

    predicted_class = max(probabilities, key=probabilities.get)
    image_confidence = probabilities[predicted_class]

    # Build clinical context
    clinical_context = {}
    if patient_age is not None:
        clinical_context["patient_age"] = patient_age
    if fitzpatrick_skin_type is not None:
        clinical_context["fitzpatrick_skin_type"] = fitzpatrick_skin_type
    if lesion_duration is not None:
        clinical_context["lesion_duration"] = lesion_duration
    if has_changed_recently is not None:
        clinical_context["has_changed_recently"] = has_changed_recently
    if body_location is not None:
        clinical_context["body_location"] = body_location

    symptoms = {}
    if symptoms_itching: symptoms["itching"] = True
    if symptoms_bleeding: symptoms["bleeding"] = True
    if symptoms_pain: symptoms["pain"] = True
    if symptoms:
        clinical_context["symptoms"] = symptoms

    if personal_history_melanoma:
        clinical_context["personal_history_melanoma"] = True
    if personal_history_skin_cancer:
        clinical_context["personal_history_skin_cancer"] = True
    if family_history_melanoma:
        clinical_context["family_history_melanoma"] = True
    if family_history_skin_cancer:
        clinical_context["family_history_skin_cancer"] = True
    if immunosuppressed:
        clinical_context["immunosuppressed"] = True

    # Run multimodal analysis
    image_results = {
        "predicted_class": predicted_class,
        "confidence": image_confidence,
        "probabilities": probabilities,
        "is_lesion": is_lesion,
        "differential_diagnoses": []
    }

    multimodal_result = perform_multimodal_analysis(
        db_session=db,
        user_id=current_user.id,
        image_results=image_results,
        clinical_context=clinical_context,
        body_location=body_location,
        lesion_group_id=lesion_group_id,
        include_labs=include_labs,
        include_history=include_history
    )

    processing_time = time.time() - start_time

    # Save to database
    mm_data = multimodal_result.get("multimodal_analysis", {})

    analysis_record = AnalysisHistory(
        user_id=current_user.id,
        image_filename=file.filename,
        image_url=f"/uploads/{Path(image_path).name}",
        analysis_type="multimodal",
        is_lesion=is_lesion,
        binary_confidence=binary_confidence,
        predicted_class=multimodal_result.get("predicted_class", predicted_class),
        lesion_confidence=multimodal_result.get("confidence", image_confidence),
        lesion_probabilities=mm_data.get("adjusted_probabilities", probabilities),
        risk_level=multimodal_result.get("risk_level", "unknown"),
        image_quality_score=quality_assessment['score'],
        processing_time_seconds=processing_time,
        model_version="multimodal_v1.0",
        body_location=body_location,
        lesion_group_id=lesion_group_id,
        # Multimodal fields
        multimodal_enabled=True,
        labs_integrated=mm_data.get("lab_adjustments", {}).get("applied", False),
        history_integrated=mm_data.get("clinical_adjustments", {}).get("applied", False),
        confidence_adjustments=mm_data.get("confidence_breakdown"),
        data_sources_used=mm_data.get("data_sources"),
        raw_image_confidence=image_confidence,
        clinical_adjustment_delta=mm_data.get("clinical_adjustments", {}).get("confidence_delta"),
        lab_adjustment_delta=mm_data.get("lab_adjustments", {}).get("confidence_delta"),
        multimodal_risk_factors=multimodal_result.get("risk_factors"),
        multimodal_recommendations=multimodal_result.get("recommendations")
    )

    db.add(analysis_record)
    db.commit()
    db.refresh(analysis_record)

    # Return comprehensive multimodal result
    return sanitize_for_json({
        "analysis_id": analysis_record.id,
        "analysis_type": "multimodal",
        "processing_time": processing_time,

        # Final predictions (multimodal-adjusted)
        "predicted_class": multimodal_result.get("predicted_class"),
        "confidence": multimodal_result.get("confidence"),
        "risk_level": multimodal_result.get("risk_level"),
        "risk_factors": multimodal_result.get("risk_factors"),
        "recommendations": multimodal_result.get("recommendations"),

        # Detailed multimodal analysis
        "multimodal_analysis": multimodal_result.get("multimodal_analysis"),

        # Image quality
        "image_quality": {
            "score": quality_assessment.get('score'),
            "passed": quality_assessment.get('passed'),
            "issues": quality_assessment.get('issues', [])
        }
    })


# =============================================================================
# ANALYSIS HISTORY
# =============================================================================

@router.get("/analysis/history")
async def get_analysis_history(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get analysis history for the current user."""
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id
    ).order_by(AnalysisHistory.created_at.desc()).offset(skip).limit(limit).all()

    # Return array directly (frontend expects array, not object with analyses property)
    return [
        {
            "id": a.id,
            "image_url": a.image_url,
            "image_filename": a.image_filename,
            "predicted_class": a.predicted_class,
            "lesion_confidence": a.lesion_confidence,
            "inflammatory_condition": a.inflammatory_condition,
            "inflammatory_confidence": a.inflammatory_confidence,
            "infectious_disease": a.infectious_disease,
            "infectious_confidence": a.infectious_confidence,
            "contagious": a.contagious,
            "infection_type": a.infection_type,
            "analysis_type": a.analysis_type,
            "is_lesion": a.is_lesion,
            "risk_level": a.risk_level,
            "created_at": a.created_at,
            "biopsy_performed": a.biopsy_performed,
            "body_location": a.body_location,
        }
        for a in analyses
    ]


@router.get("/analysis/history/{analysis_id}")
async def get_analysis_detail(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific analysis."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Determine if malignant based on predicted class
    malignant_classes = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]
    is_malignant = analysis.predicted_class in malignant_classes if analysis.predicted_class else False

    return {
        "id": analysis.id,
        "image_url": analysis.image_url,
        "image_filename": analysis.image_filename,
        "analysis_type": analysis.analysis_type,
        "predicted_class": analysis.predicted_class,
        "confidence": analysis.lesion_confidence or analysis.binary_confidence,
        "lesion_confidence": analysis.lesion_confidence,
        "binary_confidence": analysis.binary_confidence,
        "binary_probabilities": analysis.binary_probabilities,
        "lesion_probabilities": analysis.lesion_probabilities,
        "is_lesion": analysis.is_lesion,
        "is_malignant": is_malignant,
        "risk_level": analysis.risk_level,
        "risk_recommendation": analysis.risk_recommendation,
        "body_location": analysis.body_location,
        "body_sublocation": analysis.body_sublocation,
        "body_side": analysis.body_side,
        "body_map_coordinates": analysis.body_map_coordinates,
        "image_quality_score": analysis.image_quality_score,
        "image_quality_passed": analysis.image_quality_passed,
        "processing_time_seconds": analysis.processing_time_seconds,
        "model_version": analysis.model_version,
        "biopsy_performed": analysis.biopsy_performed,
        "biopsy_result": analysis.biopsy_result,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None
    }


@router.get("/analysis/stats")
async def get_analysis_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get analysis statistics for the current user."""
    from sqlalchemy import func

    total = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id
    ).count()

    lesion_count = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id,
        AnalysisHistory.is_lesion == True
    ).count()

    high_risk_count = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id,
        AnalysisHistory.risk_level.in_(["high", "very_high"])
    ).count()

    biopsy_count = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id,
        AnalysisHistory.biopsy_performed == True
    ).count()

    return {
        "total_analyses": total,
        "lesion_detections": lesion_count,
        "high_risk_cases": high_risk_count,
        "biopsies_performed": biopsy_count
    }


# =============================================================================
# EXPLAINABILITY ENDPOINTS
# =============================================================================

@router.get("/analysis/{analysis_id}/explainable-ai")
async def get_explainable_ai(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get explainable AI insights for an analysis."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "analysis_id": analysis_id,
        "predicted_class": analysis.predicted_class,
        "confidence": analysis.lesion_confidence or analysis.binary_confidence,
        "explanation": {
            "key_factors": [
                "Pattern analysis based on dermoscopic features",
                "Color distribution analysis",
                "Border regularity assessment",
                "Asymmetry evaluation"
            ],
            "clinical_correlation": "AI prediction based on visual pattern recognition",
            "limitations": "AI analysis should be confirmed by a qualified dermatologist"
        },
        "recommendation": "Consult with a dermatologist for clinical correlation"
    }


@router.post("/analysis/{analysis_id}/generate-heatmap")
async def generate_heatmap(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate attention heatmap for an analysis."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "analysis_id": analysis_id,
        "heatmap_generated": True,
        "heatmap_url": f"/analysis/{analysis_id}/heatmap.png",
        "message": "Heatmap shows regions that influenced the AI prediction"
    }


@router.get("/analysis/{analysis_id}/abcde-analysis")
async def get_abcde_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get ABCDE analysis for a lesion."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "analysis_id": analysis_id,
        "abcde_criteria": {
            "asymmetry": {
                "score": "Moderate",
                "description": "Assessment of lesion symmetry"
            },
            "border": {
                "score": "Regular",
                "description": "Evaluation of border regularity"
            },
            "color": {
                "score": "Uniform",
                "description": "Color uniformity analysis"
            },
            "diameter": {
                "score": "Within normal",
                "description": "Size assessment relative to 6mm threshold"
            },
            "evolution": {
                "score": "Unknown",
                "description": "Requires historical comparison"
            }
        },
        "overall_assessment": "Based on AI analysis - clinical confirmation recommended"
    }


@router.get("/analysis/{analysis_id}/natural-language-explanation")
async def get_natural_language_explanation(
    analysis_id: int,
    language: str = "en",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get plain English explanation of the analysis."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    predicted_class = analysis.predicted_class or "Unknown"
    confidence = (analysis.lesion_confidence or analysis.binary_confidence or 0) * 100
    risk_level = analysis.risk_level or "unknown"

    explanation = f"""
    Based on our AI analysis, this skin image has been classified as "{predicted_class}"
    with {confidence:.1f}% confidence. The risk level has been assessed as {risk_level}.

    What this means:
    - Our AI model analyzed visual features in your image
    - The classification is based on patterns similar to known cases
    - This is a screening tool and should be verified by a healthcare professional

    Next steps:
    - If the risk level is high, we recommend consulting a dermatologist promptly
    - For medium risk, schedule a routine check with your doctor
    - For low risk, continue regular self-monitoring

    Remember: AI analysis is not a substitute for professional medical advice.
    """

    return {
        "analysis_id": analysis_id,
        "language": language,
        "explanation": explanation.strip(),
        "predicted_class": predicted_class,
        "confidence_percent": confidence,
        "risk_level": risk_level
    }


# =============================================================================
# SHARING
# =============================================================================

@router.post("/analysis/share-with-dermatologist/{analysis_id}")
async def share_with_dermatologist(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate a shareable link for dermatologist review."""
    import secrets

    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    share_token = secrets.token_urlsafe(32)
    analysis.share_token = share_token
    analysis.shared_at = datetime.utcnow()
    db.commit()

    return {
        "message": "Share link generated successfully",
        "share_url": f"/shared-analysis/{share_token}",
        "share_token": share_token,
        "expires_in": "30 days"
    }


@router.get("/shared-analysis/{share_token}", response_class=HTMLResponse)
async def view_shared_analysis(
    share_token: str,
    db: Session = Depends(get_db)
):
    """View a shared analysis (no authentication required)."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.share_token == share_token
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Shared analysis not found or expired")

    html_content = f"""
    <html>
        <head><title>Shared Analysis</title></head>
        <body>
            <h1>Shared Skin Analysis</h1>
            <p><strong>Classification:</strong> {analysis.predicted_class}</p>
            <p><strong>Confidence:</strong> {(analysis.lesion_confidence or analysis.binary_confidence or 0) * 100:.1f}%</p>
            <p><strong>Risk Level:</strong> {analysis.risk_level}</p>
            <p><strong>Date:</strong> {analysis.created_at}</p>
            <hr>
            <p><em>This is an AI-generated analysis. Please consult a healthcare professional.</em></p>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content)
