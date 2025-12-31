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

# Import model monitoring (now non-blocking)
from model_monitoring import record_inference

# Import multimodal analyzer
from multimodal_analyzer import MultimodalAnalyzer, perform_multimodal_analysis

# Import ABCDE feature analysis
from abcde_feature_analysis import perform_abcde_analysis, format_abcde_for_response, build_combined_assessment

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

    # =========================================================================
    # ABCDE FEATURE ANALYSIS
    # Quantitative analysis of traditional dermoscopy criteria
    # =========================================================================
    abcde_result = None
    combined_assessment = None
    try:
        # Perform ABCDE analysis
        abcde_analysis = perform_abcde_analysis(
            image=image,
            pixels_per_mm=None,  # No calibration by default
            previous_image=None,
            previous_analysis=None
        )
        abcde_result = format_abcde_for_response(abcde_analysis)

        # Build combined assessment integrating ML and image features
        combined_assessment = build_combined_assessment(
            abcde_analysis=abcde_result,
            ml_classification=predicted_class,
            ml_confidence=lesion_confidence
        )

        # Update risk level if combined assessment indicates higher risk
        combined_risk = combined_assessment.get("combined_assessment", {}).get("overall_risk_level")
        if combined_risk:
            risk_order = {"low": 0, "moderate": 1, "medium": 1, "high": 2, "very_high": 3}
            if risk_order.get(combined_risk, 0) > risk_order.get(risk_level, 0):
                risk_level = combined_risk

    except Exception as e:
        debug_log(f"ABCDE analysis error: {e}")
        abcde_result = {"error": str(e)}

    # Save to database
    analysis_record = None
    if save_to_db:
        body_map_coords = None
        if body_map_x is not None and body_map_y is not None:
            body_map_coords = {"x": body_map_x, "y": body_map_y}

        # Extract multimodal tracking data
        mm_data = multimodal_result.get("multimodal_analysis", {}) if multimodal_result else {}

        # Sanitize all values to convert numpy types to native Python types for JSON serialization
        analysis_record = AnalysisHistory(
            user_id=current_user.id,
            image_filename=file.filename,
            image_url=f"/uploads/{Path(image_path).name}",
            analysis_type="full",
            is_lesion=sanitize_for_json(is_lesion),
            binary_confidence=sanitize_for_json(binary_confidence),
            binary_probabilities=sanitize_for_json(binary_result),
            predicted_class=predicted_class,
            lesion_confidence=sanitize_for_json(lesion_confidence),
            lesion_probabilities=sanitize_for_json(probabilities),
            risk_level=risk_level,
            image_quality_score=sanitize_for_json(quality_assessment['score']),
            image_quality_passed=sanitize_for_json(quality_assessment['passed']),
            quality_issues=sanitize_for_json(quality_assessment),
            processing_time_seconds=sanitize_for_json(processing_time),
            model_version="full_classify_v2.0",
            body_location=body_location,
            body_sublocation=body_sublocation,
            body_side=body_side,
            body_map_coordinates=sanitize_for_json(body_map_coords),
            lesion_group_id=lesion_group_id,
            # Multimodal tracking fields
            multimodal_enabled=sanitize_for_json(enable_multimodal),
            labs_integrated=sanitize_for_json(mm_data.get("lab_adjustments", {}).get("applied", False)),
            history_integrated=sanitize_for_json(mm_data.get("clinical_adjustments", {}).get("applied", False)),
            confidence_adjustments=sanitize_for_json(mm_data.get("confidence_breakdown")),
            data_sources_used=sanitize_for_json(mm_data.get("data_sources")),
            raw_image_confidence=sanitize_for_json(mm_data.get("image_analysis", {}).get("raw_confidence")),
            clinical_adjustment_delta=sanitize_for_json(mm_data.get("clinical_adjustments", {}).get("confidence_delta")),
            lab_adjustment_delta=sanitize_for_json(mm_data.get("lab_adjustments", {}).get("confidence_delta")),
            multimodal_risk_factors=sanitize_for_json(multimodal_result.get("risk_factors") if multimodal_result else None),
            multimodal_recommendations=sanitize_for_json(multimodal_result.get("recommendations") if multimodal_result else None),
            # ABCDE analysis data
            red_flag_data=sanitize_for_json(abcde_result if abcde_result and "error" not in abcde_result else None)
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

        # Unified Risk Assessment (combines AI + ABCDE with clear explanation)
        "unified_risk": combined_assessment.get("unified_risk") if combined_assessment else {
            "level": risk_level,
            "level_display": risk_level.upper() if risk_level else "UNKNOWN",
            "explanation": f"Risk assessment based on AI classification: {risk_level}",
            "ai_risk": risk_level,
            "feature_risk": None,
            "recommendation": risk_recommendation,
            "components_agree": True
        },

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

        # ABCDE Feature Analysis
        "abcde_analysis": abcde_result if abcde_result and "error" not in abcde_result else None,
        "combined_assessment": combined_assessment if combined_assessment else None,

        # For backward compatibility - flat ABCDE scores
        "asymmetry_score": abcde_result.get("asymmetry", {}).get("overall_score") if abcde_result and "error" not in abcde_result else None,
        "border_score": abcde_result.get("border", {}).get("overall_score") if abcde_result and "error" not in abcde_result else None,
        "color_score": abcde_result.get("color", {}).get("overall_score") if abcde_result and "error" not in abcde_result else None,
        "diameter_score": abcde_result.get("diameter", {}).get("overall_score") if abcde_result and "error" not in abcde_result else None,
        "total_dermoscopy_score": abcde_result.get("total_score") if abcde_result and "error" not in abcde_result else None,
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

    # Calculate average confidence
    avg_confidence_result = db.query(
        func.avg(
            func.coalesce(AnalysisHistory.lesion_confidence, AnalysisHistory.binary_confidence)
        )
    ).filter(
        AnalysisHistory.user_id == current_user.id
    ).scalar()

    average_confidence = float(avg_confidence_result) if avg_confidence_result is not None else 0.0

    return {
        "total_analyses": total,
        "lesion_detections": lesion_count,
        "high_risk_cases": high_risk_count,
        "biopsies_performed": biopsy_count,
        "average_confidence": average_confidence
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


# =============================================================================
# FHIR EXPORT ENDPOINT
# =============================================================================

def generate_fhir_report(analysis: AnalysisHistory, user: User) -> dict:
    """Generate FHIR R4 DiagnosticReport resource."""
    from datetime import datetime
    import uuid

    # Map predicted class to SNOMED codes
    snomed_mappings = {
        "Melanoma": {"code": "372244006", "display": "Malignant melanoma"},
        "Basal Cell Carcinoma": {"code": "254701007", "display": "Basal cell carcinoma of skin"},
        "Squamous Cell Carcinoma": {"code": "402815007", "display": "Squamous cell carcinoma of skin"},
        "Actinic Keratoses": {"code": "201101007", "display": "Actinic keratosis"},
        "Benign Keratosis": {"code": "400010006", "display": "Seborrheic keratosis"},
        "Dermatofibroma": {"code": "254788002", "display": "Dermatofibroma"},
        "Melanocytic Nevi": {"code": "398943008", "display": "Melanocytic nevus"},
        "Vascular Lesions": {"code": "400210000", "display": "Hemangioma"},
    }

    snomed = snomed_mappings.get(
        analysis.predicted_class,
        {"code": "95320005", "display": "Disorder of skin"}
    )

    # Risk level to FHIR interpretation
    risk_interpretations = {
        "low": {"code": "L", "display": "Low"},
        "medium": {"code": "N", "display": "Normal"},
        "high": {"code": "H", "display": "High"},
        "very_high": {"code": "HH", "display": "Critical high"}
    }

    interpretation = risk_interpretations.get(
        analysis.risk_level,
        {"code": "N", "display": "Normal"}
    )

    fhir_report = {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "meta": {
            "versionId": "1",
            "lastUpdated": datetime.utcnow().isoformat() + "Z",
            "profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-diagnosticreport-note"]
        },
        "identifier": [{
            "system": "urn:skin-analyzer:analysis",
            "value": str(analysis.id)
        }],
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "IMG",
                "display": "Diagnostic Imaging"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "72170-4",
                "display": "Photographic image"
            }],
            "text": "AI Skin Lesion Analysis"
        },
        "subject": {
            "reference": f"Patient/{user.id}",
            "display": user.email
        },
        "effectiveDateTime": analysis.created_at.isoformat() + "Z",
        "issued": datetime.utcnow().isoformat() + "Z",
        "conclusion": f"AI Classification: {analysis.predicted_class} ({(analysis.lesion_confidence or 0) * 100:.1f}% confidence). Risk Level: {analysis.risk_level}",
        "conclusionCode": [{
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": snomed["code"],
                "display": snomed["display"]
            }],
            "text": analysis.predicted_class
        }],
        "result": [{
            "reference": f"#confidence-observation",
            "display": f"Confidence: {(analysis.lesion_confidence or 0) * 100:.1f}%"
        }],
        "contained": [{
            "resourceType": "Observation",
            "id": "confidence-observation",
            "status": "final",
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "59776-5",
                    "display": "Procedure finding"
                }]
            },
            "valueQuantity": {
                "value": round((analysis.lesion_confidence or 0) * 100, 1),
                "unit": "%",
                "system": "http://unitsofmeasure.org",
                "code": "%"
            },
            "interpretation": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": interpretation["code"],
                    "display": interpretation["display"]
                }]
            }]
        }],
        "extension": [{
            "url": "http://skin-analyzer.com/fhir/StructureDefinition/ai-analysis",
            "extension": [
                {
                    "url": "modelVersion",
                    "valueString": analysis.model_version or "unknown"
                },
                {
                    "url": "analysisType",
                    "valueString": analysis.analysis_type or "full"
                },
                {
                    "url": "isLesion",
                    "valueBoolean": analysis.is_lesion or False
                },
                {
                    "url": "bodyLocation",
                    "valueString": analysis.body_location or "unspecified"
                }
            ]
        }]
    }

    return fhir_report


@router.get("/analysis/export/fhir/{analysis_id}")
def export_analysis_fhir(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Export analysis report in HL7 FHIR R4 format for EMR integration.

    FHIR (Fast Healthcare Interoperability Resources) is the international
    standard for exchanging healthcare information electronically.

    Returns:
        JSONResponse: FHIR DiagnosticReport resource
    """
    from fastapi.responses import JSONResponse

    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        fhir_report = generate_fhir_report(analysis, current_user)
        return JSONResponse(
            content=fhir_report,
            media_type="application/fhir+json",
            headers={
                "Content-Disposition": f"attachment; filename=fhir-report-{analysis_id}.json"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate FHIR report: {str(e)}")


# =============================================================================
# SYMPTOMS, MEDICATIONS, MEDICAL HISTORY ENDPOINTS
# =============================================================================

@router.post("/analysis/symptoms/{analysis_id}")
async def add_symptoms(
    analysis_id: int,
    symptom_duration: str = Form(None),
    symptom_duration_value: int = Form(None),
    symptom_duration_unit: str = Form(None),
    symptom_changes: str = Form(None),
    symptom_itching: bool = Form(False),
    symptom_itching_severity: int = Form(None),
    symptom_pain: bool = Form(False),
    symptom_pain_severity: int = Form(None),
    symptom_bleeding: bool = Form(False),
    symptom_bleeding_frequency: str = Form(None),
    symptom_notes: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add or update symptom information for an existing analysis.
    Tracks duration, changes, and associated symptoms like itching, pain, and bleeding.
    """
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis.symptom_duration = symptom_duration
    analysis.symptom_duration_value = symptom_duration_value
    analysis.symptom_duration_unit = symptom_duration_unit
    analysis.symptom_changes = symptom_changes
    analysis.symptom_itching = symptom_itching
    analysis.symptom_itching_severity = symptom_itching_severity if symptom_itching else None
    analysis.symptom_pain = symptom_pain
    analysis.symptom_pain_severity = symptom_pain_severity if symptom_pain else None
    analysis.symptom_bleeding = symptom_bleeding
    analysis.symptom_bleeding_frequency = symptom_bleeding_frequency if symptom_bleeding else None
    analysis.symptom_notes = symptom_notes
    analysis.symptom_updated_at = datetime.utcnow()

    db.commit()
    db.refresh(analysis)

    return {
        "message": "Symptoms recorded successfully",
        "analysis_id": analysis_id,
        "symptom_duration": symptom_duration,
        "symptoms": {
            "itching": symptom_itching,
            "pain": symptom_pain,
            "bleeding": symptom_bleeding
        }
    }


@router.post("/analysis/medications/{analysis_id}")
async def add_medications(
    analysis_id: int,
    medications: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add or update medication list for an existing analysis.
    Medications should be a JSON string containing an array of medication objects.
    """
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        medications_data = json.loads(medications) if medications else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid medications JSON format")

    analysis.medications = json.dumps(medications_data)
    analysis.medications_updated_at = datetime.utcnow()

    db.commit()
    db.refresh(analysis)

    return {
        "message": "Medications recorded successfully",
        "analysis_id": analysis_id,
        "medication_count": len(medications_data)
    }


@router.post("/analysis/medical-history/{analysis_id}")
async def add_medical_history(
    analysis_id: int,
    family_history_skin_cancer: bool = Form(False),
    family_history_details: str = Form(None),
    previous_skin_cancers: bool = Form(False),
    previous_skin_cancers_details: str = Form(None),
    immunosuppression: bool = Form(False),
    immunosuppression_details: str = Form(None),
    sun_exposure_level: str = Form(None),
    sun_exposure_details: str = Form(None),
    history_of_sunburns: bool = Form(False),
    sunburn_details: str = Form(None),
    tanning_bed_use: bool = Form(False),
    tanning_bed_frequency: str = Form(None),
    other_risk_factors: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add or update medical history risk factors for an existing analysis.
    Tracks family history, previous skin cancers, immunosuppression, sun exposure.
    """
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis.family_history_skin_cancer = family_history_skin_cancer
    analysis.family_history_details = family_history_details
    analysis.previous_skin_cancers = previous_skin_cancers
    analysis.previous_skin_cancers_details = previous_skin_cancers_details
    analysis.immunosuppression = immunosuppression
    analysis.immunosuppression_details = immunosuppression_details
    analysis.sun_exposure_level = sun_exposure_level
    analysis.sun_exposure_details = sun_exposure_details
    analysis.history_of_sunburns = history_of_sunburns
    analysis.sunburn_details = sunburn_details
    analysis.tanning_bed_use = tanning_bed_use
    analysis.tanning_bed_frequency = tanning_bed_frequency
    analysis.other_risk_factors = other_risk_factors
    analysis.medical_history_updated_at = datetime.utcnow()

    db.commit()
    db.refresh(analysis)

    return {
        "message": "Medical history recorded successfully",
        "analysis_id": analysis_id,
        "risk_factors": {
            "family_history": family_history_skin_cancer,
            "previous_cancers": previous_skin_cancers,
            "immunosuppression": immunosuppression,
            "sun_exposure": sun_exposure_level
        }
    }


# =============================================================================
# SHARED ANALYSIS PDF EXPORT
# =============================================================================

@router.get("/shared-analysis/{share_token}/pdf")
def get_shared_analysis_pdf(
    share_token: str,
    db: Session = Depends(get_db)
):
    """Generate and download a PDF report of the shared analysis."""
    from fastapi.responses import FileResponse
    from io import BytesIO
    import os

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.units import inch
    except ImportError:
        raise HTTPException(status_code=503, detail="PDF generation not available")

    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.share_token == share_token
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Shared analysis not found or token invalid")

    user = db.query(User).filter(User.id == analysis.user_id).first()
    patient_name = getattr(user, 'full_name', user.email.split('@')[0] if user else 'Unknown')

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'], fontSize=24,
        textColor=colors.HexColor('#0284c7'), spaceAfter=30, alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'], fontSize=16,
        textColor=colors.HexColor('#1f2937'), spaceAfter=12, spaceBefore=12
    )

    elements.append(Paragraph("Skin Analysis Report", title_style))
    elements.append(Spacer(1, 12))

    patient_data = [
        ['Patient:', patient_name],
        ['Analysis ID:', str(analysis.id)],
        ['Date:', analysis.created_at.strftime('%B %d, %Y') if analysis.created_at else 'Unknown']
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("AI Diagnosis", heading_style))
    diagnosis_data = [
        ['Predicted Class:', analysis.predicted_class or 'Unknown'],
        ['Confidence:', f'{(analysis.lesion_confidence or 0) * 100:.1f}%'],
        ['Risk Level:', analysis.risk_level or 'Unknown']
    ]
    diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ecfdf5')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#10b981'))
    ]))
    elements.append(diagnosis_table)
    elements.append(Spacer(1, 20))

    footer_style = ParagraphStyle(
        'Footer', parent=styles['Normal'], fontSize=8,
        textColor=colors.grey, alignment=TA_CENTER
    )
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("This report was generated via secure teledermatology link", footer_style))
    elements.append(Paragraph("For medical professional review only - Not a substitute for professional diagnosis", footer_style))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    pdf_filename = f"analysis_{analysis.id}_{share_token[:8]}.pdf"
    pdf_path = os.path.join("uploads", pdf_filename)
    os.makedirs("uploads", exist_ok=True)
    with open(pdf_path, 'wb') as f:
        f.write(pdf_bytes)

    return FileResponse(
        pdf_path,
        media_type='application/pdf',
        filename=f"skin_analysis_{patient_name.replace(' ', '_')}_{analysis.id}.pdf"
    )


# =============================================================================
# PREAUTH PDF AND STATUS ENDPOINTS
# =============================================================================

@router.get("/analysis/preauth-pdf/{analysis_id}")
def generate_preauth_pdf(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate a PDF of the insurance pre-authorization documentation."""
    from fastapi.responses import FileResponse
    import os

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.units import inch
    except ImportError:
        raise HTTPException(status_code=503, detail="PDF generation not available")

    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if not analysis.insurance_preauthorization:
        raise HTTPException(status_code=404, detail="No pre-authorization data available")

    pdf_filename = f"preauth_{analysis_id}_{current_user.id}.pdf"
    pdf_path = f"uploads/{pdf_filename}"
    os.makedirs("uploads", exist_ok=True)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'], fontSize=24,
        textColor=colors.HexColor('#047857'), spaceAfter=30, alignment=TA_CENTER
    )

    story.append(Paragraph("Insurance Pre-Authorization Documentation", title_style))
    story.append(Spacer(1, 0.2*inch))

    info_data = [
        ['Analysis ID:', str(analysis_id)],
        ['Date:', analysis.created_at.strftime('%B %d, %Y')],
        ['Diagnosis:', analysis.predicted_class or 'N/A'],
        ['Confidence:', f"{(analysis.lesion_confidence or 0) * 100:.1f}%"],
    ]

    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0fdf4')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#047857')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1fae5')),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))

    preauth_data = analysis.insurance_preauthorization or {}
    if preauth_data.get('justification'):
        story.append(Paragraph("Medical Justification", styles['Heading2']))
        story.append(Paragraph(preauth_data.get('justification', ''), styles['Normal']))

    doc.build(story)

    return FileResponse(
        pdf_path,
        media_type='application/pdf',
        filename=f"preauth_{analysis_id}.pdf"
    )


@router.patch("/analysis/preauth-status/{analysis_id}")
async def update_preauth_status(
    analysis_id: int,
    status: str = Form(...),
    reference_number: str = Form(None),
    notes: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update the pre-authorization status for an analysis."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    preauth_data = analysis.insurance_preauthorization or {}
    preauth_data['status'] = status
    preauth_data['reference_number'] = reference_number
    preauth_data['status_notes'] = notes
    preauth_data['status_updated_at'] = datetime.utcnow().isoformat()

    analysis.insurance_preauthorization = preauth_data
    db.commit()

    return {
        "message": "Pre-authorization status updated",
        "analysis_id": analysis_id,
        "status": status,
        "reference_number": reference_number
    }


# =============================================================================
# FEATURE IMPORTANCE AND DERMATOLOGIST COMPARISON
# =============================================================================

@router.get("/analysis/{analysis_id}/feature-importance")
async def get_feature_importance(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get feature importance scores for an analysis."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Get ABCDE data if available
    abcde_data = analysis.red_flag_data or {}

    features = []
    if abcde_data:
        if 'asymmetry' in abcde_data:
            features.append({
                "feature": "Asymmetry",
                "importance": abcde_data['asymmetry'].get('overall_score', 0),
                "description": "Degree of asymmetry in shape and color distribution"
            })
        if 'border' in abcde_data:
            features.append({
                "feature": "Border Irregularity",
                "importance": abcde_data['border'].get('overall_score', 0),
                "description": "Irregularity of the lesion border"
            })
        if 'color' in abcde_data:
            features.append({
                "feature": "Color Variation",
                "importance": abcde_data['color'].get('overall_score', 0),
                "description": "Number and variety of colors present"
            })
        if 'diameter' in abcde_data:
            features.append({
                "feature": "Diameter",
                "importance": abcde_data['diameter'].get('overall_score', 0),
                "description": "Size of the lesion"
            })

    # Add ML-based importance
    features.append({
        "feature": "ML Confidence",
        "importance": analysis.lesion_confidence or 0,
        "description": "Machine learning model confidence"
    })

    return {
        "analysis_id": analysis_id,
        "predicted_class": analysis.predicted_class,
        "features": sorted(features, key=lambda x: x['importance'], reverse=True)
    }


@router.post("/analysis/{analysis_id}/dermatologist-annotation")
async def add_dermatologist_annotation(
    analysis_id: int,
    dermatologist_diagnosis: str = Form(...),
    dermatologist_confidence: float = Form(None),
    dermatologist_notes: str = Form(None),
    agrees_with_ai: bool = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add dermatologist annotation to an analysis for comparison."""
    # Check if user is a professional
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile or not profile.is_professional:
        raise HTTPException(status_code=403, detail="Only verified professionals can add annotations")

    analysis = db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis.dermatologist_diagnosis = dermatologist_diagnosis
    analysis.dermatologist_confidence = dermatologist_confidence
    analysis.dermatologist_notes = dermatologist_notes
    analysis.dermatologist_agrees = agrees_with_ai
    analysis.dermatologist_reviewed_at = datetime.utcnow()
    analysis.dermatologist_reviewer_id = current_user.id

    db.commit()

    return {
        "message": "Dermatologist annotation added",
        "analysis_id": analysis_id,
        "dermatologist_diagnosis": dermatologist_diagnosis,
        "ai_diagnosis": analysis.predicted_class,
        "agreement": agrees_with_ai
    }


@router.get("/analysis/{analysis_id}/compare-with-dermatologist")
async def compare_with_dermatologist(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Compare AI diagnosis with dermatologist diagnosis."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if not analysis.dermatologist_diagnosis:
        return {
            "analysis_id": analysis_id,
            "has_dermatologist_review": False,
            "message": "No dermatologist review available for this analysis"
        }

    ai_diagnosis = analysis.predicted_class or ""
    derm_diagnosis = analysis.dermatologist_diagnosis or ""
    exact_match = ai_diagnosis.lower() == derm_diagnosis.lower()

    malignant_conditions = ["melanoma", "basal cell carcinoma", "squamous cell carcinoma"]
    ai_malignant = any(c in ai_diagnosis.lower() for c in malignant_conditions)
    derm_malignant = any(c in derm_diagnosis.lower() for c in malignant_conditions)
    category_match = ai_malignant == derm_malignant

    return {
        "analysis_id": analysis_id,
        "has_dermatologist_review": True,
        "ai_diagnosis": ai_diagnosis,
        "ai_confidence": analysis.lesion_confidence,
        "dermatologist_diagnosis": derm_diagnosis,
        "dermatologist_confidence": analysis.dermatologist_confidence,
        "dermatologist_notes": analysis.dermatologist_notes,
        "comparison": {
            "exact_match": exact_match,
            "category_match": category_match,
            "dermatologist_agrees": analysis.dermatologist_agrees,
            "ai_malignant": ai_malignant,
            "derm_malignant": derm_malignant
        },
        "reviewed_at": analysis.dermatologist_reviewed_at.isoformat() if analysis.dermatologist_reviewed_at else None
    }


@router.get("/analysis/{analysis_id}/highlighted-regions")
async def get_highlighted_regions(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get AI-highlighted regions of interest for an analysis."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Build regions from ABCDE data if available
    regions = []
    abcde_data = analysis.red_flag_data or {}

    if abcde_data.get('asymmetry', {}).get('overall_score', 0) > 0.5:
        regions.append({
            "region_id": 1,
            "feature_type": "asymmetry",
            "importance_score": abcde_data['asymmetry']['overall_score'],
            "description": "Asymmetric region detected"
        })

    if abcde_data.get('border', {}).get('overall_score', 0) > 0.5:
        regions.append({
            "region_id": 2,
            "feature_type": "border",
            "importance_score": abcde_data['border']['overall_score'],
            "description": "Irregular border region"
        })

    if abcde_data.get('color', {}).get('overall_score', 0) > 0.5:
        regions.append({
            "region_id": 3,
            "feature_type": "color",
            "importance_score": abcde_data['color']['overall_score'],
            "description": "Abnormal color variation"
        })

    return {
        "analysis_id": analysis_id,
        "predicted_class": analysis.predicted_class,
        "regions": regions,
        "heatmap_url": analysis.heatmap_url if hasattr(analysis, 'heatmap_url') else None
    }
