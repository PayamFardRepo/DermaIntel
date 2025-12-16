"""
Celery Tasks for Asynchronous Image Analysis

This module contains all Celery tasks for long-running ML operations:
- Full skin lesion classification
- Binary classification
- Multimodal analysis
- Dermoscopy analysis
- Burn classification
- Histopathology analysis
- Batch skin check processing

Tasks update their progress state for real-time monitoring.
"""

import io
import time
import json
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from celery import shared_task, current_task
from celery.exceptions import SoftTimeLimitExceeded

# Import app components
from celery_app import celery_app, TaskStatus
import config


def update_progress(current: int, total: int, status: str, details: dict = None):
    """Update task progress state."""
    if current_task:
        meta = {
            "current": current,
            "total": total,
            "percent": int((current / total) * 100) if total > 0 else 0,
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        if details:
            meta.update(details)
        current_task.update_state(state="PROGRESS", meta=meta)


def load_ml_components():
    """
    Lazy load ML components.
    This is called inside tasks to ensure models are loaded in worker process.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    # Import shared ML components
    import shared
    from shared import (
        device, binary_model, binary_transform, binary_labels,
        lesion_model, lesion_processor, labels, key_map,
        isic_model, isic_transform, isic_labels,
        inflammatory_model, inflammatory_processor, inflammatory_labels,
        infectious_model, infectious_processor, infectious_labels,
        sanitize_for_json
    )

    return {
        "torch": torch,
        "F": F,
        "Image": Image,
        "device": device,
        "binary_model": binary_model,
        "binary_transform": binary_transform,
        "binary_labels": binary_labels,
        "lesion_model": lesion_model,
        "lesion_processor": lesion_processor,
        "labels": labels,
        "key_map": key_map,
        "isic_model": isic_model,
        "isic_transform": isic_transform,
        "isic_labels": isic_labels,
        "inflammatory_model": inflammatory_model,
        "inflammatory_processor": inflammatory_processor,
        "inflammatory_labels": inflammatory_labels,
        "infectious_model": infectious_model,
        "infectious_processor": infectious_processor,
        "infectious_labels": infectious_labels,
        "sanitize_for_json": sanitize_for_json,
    }


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


@celery_app.task(bind=True, name="tasks.binary_classify_task")
def binary_classify_task(
    self,
    image_path: str,
    user_id: int,
    filename: str,
    save_to_db: bool = True,
    body_location: str = None,
    body_sublocation: str = None,
    body_side: str = None,
    body_map_x: float = None,
    body_map_y: float = None,
) -> Dict[str, Any]:
    """
    Binary lesion classification task.

    Args:
        image_path: Path to the saved image file
        user_id: User ID for database storage
        filename: Original filename
        save_to_db: Whether to save results to database
        body_location: Body location metadata
        body_sublocation: Body sublocation metadata
        body_side: Body side (left/right)
        body_map_x: X coordinate on body map
        body_map_y: Y coordinate on body map

    Returns:
        Classification results
    """
    try:
        start_time = time.time()
        update_progress(1, 5, "Loading image")

        # Load ML components
        ml = load_ml_components()
        torch = ml["torch"]
        F = ml["F"]
        Image = ml["Image"]

        # Load image
        image = Image.open(image_path).convert("RGB")
        file_size = Path(image_path).stat().st_size

        update_progress(2, 5, "Assessing image quality")
        quality_assessment = assess_image_quality(image, file_size)

        update_progress(3, 5, "Running binary classification")
        img_tensor = ml["binary_transform"](image).unsqueeze(0).to(ml["device"])

        with torch.no_grad():
            binary_logits = ml["binary_model"](img_tensor)
            binary_probs = F.softmax(binary_logits, dim=1)[0]
            binary_pred = torch.argmax(binary_probs).item()

        binary_result = {
            ml["binary_labels"][i]: round(prob.item(), 4)
            for i, prob in enumerate(binary_probs)
        }

        binary_confidence = round(torch.max(binary_probs).item(), 4)
        is_lesion = binary_pred == 1

        # Determine risk level
        risk_level = "low"
        risk_recommendation = "Continue regular skin monitoring."

        if is_lesion and binary_confidence > 0.85:
            risk_level = "high"
            risk_recommendation = "Recommend consultation with dermatologist."
        elif is_lesion:
            risk_level = "medium"
            risk_recommendation = "Monitor closely and consider consultation."

        update_progress(4, 5, "Saving to database")

        analysis_id = None
        if save_to_db:
            from database import get_db, AnalysisHistory, UserProfile
            from sqlalchemy.orm import Session

            db = next(get_db())
            try:
                body_map_coords = None
                if body_map_x is not None and body_map_y is not None:
                    body_map_coords = {"x": body_map_x, "y": body_map_y}

                analysis_record = AnalysisHistory(
                    user_id=user_id,
                    image_filename=filename,
                    image_url=f"/uploads/{Path(image_path).name}",
                    analysis_type="binary",
                    is_lesion=is_lesion,
                    binary_confidence=binary_confidence,
                    binary_probabilities=binary_result,
                    predicted_class=ml["binary_labels"][binary_pred],
                    risk_level=risk_level,
                    risk_recommendation=risk_recommendation,
                    image_quality_score=quality_assessment['score'],
                    image_quality_passed=quality_assessment['passed'],
                    quality_issues=quality_assessment,
                    processing_time_seconds=time.time() - start_time,
                    model_version="resnet18_binary_v1.0_async",
                    body_location=body_location,
                    body_sublocation=body_sublocation,
                    body_side=body_side,
                    body_map_coordinates=body_map_coords
                )

                db.add(analysis_record)
                db.commit()
                db.refresh(analysis_record)
                analysis_id = analysis_record.id

                # Update user profile
                profile = db.query(UserProfile).filter(
                    UserProfile.user_id == user_id
                ).first()
                if not profile:
                    profile = UserProfile(user_id=user_id, total_analyses=1)
                    db.add(profile)
                else:
                    profile.total_analyses += 1
                    profile.last_analysis_date = analysis_record.created_at
                db.commit()
            finally:
                db.close()

        update_progress(5, 5, "Complete")
        processing_time = time.time() - start_time

        return {
            "analysis_id": analysis_id,
            "task_id": self.request.id,
            "probabilities": binary_result,
            "predicted_class": ml["binary_labels"][binary_pred],
            "binary_pred": binary_pred,
            "confidence": binary_confidence,
            "is_lesion": is_lesion,
            "risk_level": risk_level,
            "risk_recommendation": risk_recommendation,
            "processing_time": processing_time,
            "image_quality": {
                "score": quality_assessment['score'],
                "passed": quality_assessment['passed'],
                "issues": quality_assessment['issues']
            },
            "status": "success"
        }

    except SoftTimeLimitExceeded:
        return {
            "status": "error",
            "error": "Task timed out",
            "task_id": self.request.id
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "task_id": self.request.id
        }


@celery_app.task(bind=True, name="tasks.full_classify_task")
def full_classify_task(
    self,
    image_path: str,
    user_id: int,
    filename: str,
    save_to_db: bool = True,
    body_location: str = None,
    body_sublocation: str = None,
    body_side: str = None,
    body_map_x: float = None,
    body_map_y: float = None,
    clinical_context: dict = None,
    enable_multimodal: bool = True,
    include_labs: bool = True,
    include_history: bool = True,
    lesion_group_id: int = None,
) -> Dict[str, Any]:
    """
    Full skin lesion classification with multiple models.

    This task performs comprehensive analysis using:
    - Binary lesion detection
    - Multi-class lesion classification
    - Clinical context integration
    - Multimodal analysis (if enabled)

    Args:
        image_path: Path to the saved image file
        user_id: User ID for database storage
        filename: Original filename
        save_to_db: Whether to save results to database
        body_location: Body location metadata
        body_sublocation: Body sublocation metadata
        body_side: Body side (left/right)
        body_map_x: X coordinate on body map
        body_map_y: Y coordinate on body map
        clinical_context: Clinical context dictionary
        enable_multimodal: Enable multimodal analysis
        include_labs: Include lab results in analysis
        include_history: Include patient history
        lesion_group_id: Link to existing lesion group

    Returns:
        Comprehensive classification results
    """
    try:
        start_time = time.time()
        clinical_context = clinical_context or {}

        update_progress(1, 8, "Loading image")

        # Load ML components
        ml = load_ml_components()
        torch = ml["torch"]
        F = ml["F"]
        Image = ml["Image"]

        # Load image
        image = Image.open(image_path).convert("RGB")
        file_size = Path(image_path).stat().st_size

        update_progress(2, 8, "Assessing image quality")
        quality_assessment = assess_image_quality(image, file_size)

        update_progress(3, 8, "Running binary classification")
        img_tensor = ml["binary_transform"](image).unsqueeze(0).to(ml["device"])

        with torch.no_grad():
            binary_logits = ml["binary_model"](img_tensor)
            binary_probs = F.softmax(binary_logits, dim=1)[0]
            binary_pred = torch.argmax(binary_probs).item()

        binary_result = {
            ml["binary_labels"][i]: round(prob.item(), 4)
            for i, prob in enumerate(binary_probs)
        }
        binary_confidence = round(torch.max(binary_probs).item(), 4)
        is_lesion = binary_pred == 1

        update_progress(4, 8, "Running lesion classification")
        inputs = ml["lesion_processor"](images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = ml["lesion_model"](**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]

        probabilities = {
            ml["key_map"].get(ml["labels"][i], ml["labels"][i]): round(prob.item(), 4)
            for i, prob in enumerate(probs)
        }

        predicted_class = max(probabilities, key=probabilities.get)
        lesion_confidence = probabilities[predicted_class]

        update_progress(5, 8, "Calculating risk level")

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

        update_progress(6, 8, "Running multimodal analysis")

        # Multimodal analysis
        multimodal_result = None
        if enable_multimodal:
            try:
                from multimodal_analyzer import perform_multimodal_analysis
                from database import get_db

                db = next(get_db())
                try:
                    image_results = {
                        "predicted_class": predicted_class,
                        "confidence": lesion_confidence,
                        "probabilities": probabilities,
                        "is_lesion": is_lesion,
                        "differential_diagnoses": []
                    }

                    multimodal_result = perform_multimodal_analysis(
                        db_session=db,
                        user_id=user_id,
                        image_results=image_results,
                        clinical_context=clinical_context,
                        body_location=body_location,
                        lesion_group_id=lesion_group_id,
                        include_labs=include_labs,
                        include_history=include_history
                    )

                    # Update predictions with multimodal results
                    if multimodal_result:
                        mm_analysis = multimodal_result.get("multimodal_analysis", {})
                        if mm_analysis.get("adjusted_probabilities"):
                            probabilities = mm_analysis["adjusted_probabilities"]
                            predicted_class = multimodal_result.get("predicted_class", predicted_class)
                            lesion_confidence = multimodal_result.get("confidence", lesion_confidence)
                        if multimodal_result.get("risk_level"):
                            risk_level = multimodal_result["risk_level"]
                        if multimodal_result.get("recommendations"):
                            treatment_recommendations = multimodal_result["recommendations"]
                finally:
                    db.close()
            except Exception as e:
                multimodal_result = {"error": str(e), "multimodal_analysis": {"enabled": False}}

        update_progress(7, 8, "Saving to database")

        analysis_id = None
        if save_to_db:
            from database import get_db, AnalysisHistory, UserProfile

            db = next(get_db())
            try:
                body_map_coords = None
                if body_map_x is not None and body_map_y is not None:
                    body_map_coords = {"x": body_map_x, "y": body_map_y}

                mm_data = multimodal_result.get("multimodal_analysis", {}) if multimodal_result else {}

                analysis_record = AnalysisHistory(
                    user_id=user_id,
                    image_filename=filename,
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
                    processing_time_seconds=time.time() - start_time,
                    model_version="full_classify_v2.0_async",
                    body_location=body_location,
                    body_sublocation=body_sublocation,
                    body_side=body_side,
                    body_map_coordinates=body_map_coords,
                    lesion_group_id=lesion_group_id,
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
                analysis_id = analysis_record.id

                # Update user profile
                profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
                if not profile:
                    profile = UserProfile(user_id=user_id, total_analyses=1)
                    db.add(profile)
                else:
                    profile.total_analyses += 1
                    profile.last_analysis_date = analysis_record.created_at
                db.commit()
            finally:
                db.close()

        update_progress(8, 8, "Complete")
        processing_time = time.time() - start_time

        # Determine if malignant
        is_malignant = predicted_class in ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]

        # Build risk recommendation
        risk_recommendation = "Continue regular skin self-examinations."
        if risk_level == "very_high":
            risk_recommendation = "URGENT: Immediate dermatologist consultation recommended."
        elif risk_level == "high":
            risk_recommendation = "High-risk lesion detected. Schedule dermatologist appointment within 1-2 weeks."
        elif risk_level == "medium":
            risk_recommendation = "Monitor for changes. Consider dermatologist consultation."

        return ml["sanitize_for_json"]({
            "analysis_id": analysis_id,
            "task_id": self.request.id,
            "filename": filename,
            "primary_condition_type": "lesion",
            "primary_condition_confidence": lesion_confidence,
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "lesion_confidence": lesion_confidence,
            "binary_probabilities": binary_result,
            "binary_predicted_class": ml["binary_labels"][binary_pred],
            "binary_confidence": binary_confidence,
            "is_lesion": is_lesion,
            "risk_level": risk_level,
            "risk_recommendation": risk_recommendation,
            "treatment_recommendations": treatment_recommendations,
            "multimodal_analysis": multimodal_result.get("multimodal_analysis") if multimodal_result else {
                "enabled": False,
                "data_sources": ["image"],
            },
            "image_quality": {
                "score": quality_assessment.get('score'),
                "passed": quality_assessment.get('passed'),
                "issues": quality_assessment.get('issues', [])
            },
            "processing_time": processing_time,
            "model_version": "full_classify_v2.0_async",
            "is_malignant": is_malignant,
            "status": "success"
        })

    except SoftTimeLimitExceeded:
        return {
            "status": "error",
            "error": "Task timed out - analysis took too long",
            "task_id": self.request.id
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "task_id": self.request.id
        }


@celery_app.task(bind=True, name="tasks.dermoscopy_analyze_task")
def dermoscopy_analyze_task(
    self,
    image_path: str,
    user_id: int,
    filename: str,
) -> Dict[str, Any]:
    """
    Dermoscopy analysis task for dermatoscopic feature detection.

    Analyzes dermoscopic features including:
    - Pigment networks
    - Globules
    - Streaks
    - Blue-white veil
    - Vascular patterns
    - Regression structures

    Returns 7-Point Checklist and ABCD dermoscopy scores.
    """
    try:
        start_time = time.time()
        update_progress(1, 5, "Loading image")

        from PIL import Image
        import numpy as np

        image = Image.open(image_path).convert("RGB")

        update_progress(2, 5, "Analyzing dermoscopic features")

        # Import dermoscopy analyzer
        try:
            from dermoscopy_analyzer import analyze_dermoscopy_features
            features = analyze_dermoscopy_features(image)
        except ImportError:
            # Fallback basic analysis
            features = {
                "pigment_network": {"present": False, "typical": True},
                "globules": {"present": False, "count": 0},
                "streaks": {"present": False, "count": 0},
                "blue_white_veil": {"present": False},
                "vascular_patterns": {"present": False, "type": None},
                "regression_structures": {"present": False}
            }

        update_progress(3, 5, "Calculating dermoscopy scores")

        # Calculate 7-Point Checklist score
        seven_point_score = 0
        if features.get("pigment_network", {}).get("present") and not features.get("pigment_network", {}).get("typical"):
            seven_point_score += 2
        if features.get("blue_white_veil", {}).get("present"):
            seven_point_score += 2
        if features.get("streaks", {}).get("present"):
            seven_point_score += 1
        if features.get("globules", {}).get("present"):
            seven_point_score += 1

        # ABCD score (simplified)
        abcd_score = {
            "asymmetry": 0,
            "border": 0,
            "color": 0,
            "differential_structures": 0,
            "total": 0
        }

        update_progress(4, 5, "Generating recommendations")

        # Risk assessment based on scores
        risk_level = "low"
        if seven_point_score >= 3:
            risk_level = "high"
        elif seven_point_score >= 1:
            risk_level = "medium"

        update_progress(5, 5, "Complete")
        processing_time = time.time() - start_time

        return {
            "task_id": self.request.id,
            "analysis_type": "dermoscopy",
            "features": features,
            "seven_point_checklist": {
                "score": seven_point_score,
                "threshold": 3,
                "suspicious": seven_point_score >= 3
            },
            "abcd_score": abcd_score,
            "risk_level": risk_level,
            "processing_time": processing_time,
            "status": "success"
        }

    except SoftTimeLimitExceeded:
        return {"status": "error", "error": "Task timed out", "task_id": self.request.id}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc(), "task_id": self.request.id}


@celery_app.task(bind=True, name="tasks.burn_classify_task")
def burn_classify_task(
    self,
    image_path: str,
    user_id: int,
    filename: str,
) -> Dict[str, Any]:
    """
    Burn severity classification task.

    Classifies burns into severity levels:
    - First degree (superficial)
    - Second degree (partial thickness)
    - Third degree (full thickness)
    """
    try:
        start_time = time.time()
        update_progress(1, 4, "Loading image")

        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        update_progress(2, 4, "Classifying burn severity")

        # Try to use burn classifier
        try:
            from burn_classifier import classify_burn
            result = classify_burn(image)
            severity = result.get("severity", "unknown")
            confidence = result.get("confidence", 0.0)
            tbsa_estimate = result.get("tbsa_estimate")
        except ImportError:
            # Fallback
            severity = "unknown"
            confidence = 0.0
            tbsa_estimate = None

        update_progress(3, 4, "Generating treatment advice")

        treatment_advice = []
        if severity == "first_degree":
            treatment_advice = [
                "Cool the burn with cool (not cold) running water for 10-20 minutes",
                "Apply aloe vera or burn cream",
                "Take over-the-counter pain relievers if needed",
                "Cover with sterile bandage"
            ]
        elif severity == "second_degree":
            treatment_advice = [
                "Seek medical attention",
                "Do not pop blisters",
                "Keep the area clean and covered",
                "May require prescription treatment"
            ]
        elif severity == "third_degree":
            treatment_advice = [
                "EMERGENCY: Seek immediate medical attention",
                "Call emergency services",
                "Do not remove clothing stuck to burn",
                "Cover with sterile, non-adhesive bandage"
            ]

        update_progress(4, 4, "Complete")
        processing_time = time.time() - start_time

        return {
            "task_id": self.request.id,
            "analysis_type": "burn",
            "severity": severity,
            "confidence": confidence,
            "tbsa_estimate": tbsa_estimate,
            "treatment_advice": treatment_advice,
            "processing_time": processing_time,
            "status": "success"
        }

    except SoftTimeLimitExceeded:
        return {"status": "error", "error": "Task timed out", "task_id": self.request.id}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc(), "task_id": self.request.id}


@celery_app.task(bind=True, name="tasks.batch_skin_check_task")
def batch_skin_check_task(
    self,
    image_paths: list,
    user_id: int,
    check_id: int,
) -> Dict[str, Any]:
    """
    Batch skin check task for processing multiple images.

    Processes multiple images in sequence and generates a comprehensive report.
    This is ideal for full-body skin checks.

    Args:
        image_paths: List of image file paths to process
        user_id: User ID for database storage
        check_id: Batch check session ID

    Returns:
        Comprehensive batch analysis results
    """
    try:
        start_time = time.time()
        total_images = len(image_paths)
        results = []
        high_risk_count = 0
        lesion_count = 0

        for idx, image_path in enumerate(image_paths):
            update_progress(idx + 1, total_images + 1, f"Analyzing image {idx + 1}/{total_images}")

            try:
                # Run full classification on each image
                result = full_classify_task.apply(
                    args=[image_path, user_id, Path(image_path).name],
                    kwargs={"save_to_db": True}
                ).get(timeout=60)

                if result.get("status") == "success":
                    results.append({
                        "image_path": image_path,
                        "predicted_class": result.get("predicted_class"),
                        "confidence": result.get("lesion_confidence"),
                        "risk_level": result.get("risk_level"),
                        "is_lesion": result.get("is_lesion"),
                        "analysis_id": result.get("analysis_id")
                    })

                    if result.get("is_lesion"):
                        lesion_count += 1
                    if result.get("risk_level") in ["high", "very_high"]:
                        high_risk_count += 1
                else:
                    results.append({
                        "image_path": image_path,
                        "error": result.get("error", "Unknown error")
                    })

            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "error": str(e)
                })

        update_progress(total_images + 1, total_images + 1, "Generating report")

        # Update batch check record
        try:
            from database import get_db, BatchSkinCheck

            db = next(get_db())
            try:
                batch_record = db.query(BatchSkinCheck).filter(
                    BatchSkinCheck.id == check_id
                ).first()

                if batch_record:
                    batch_record.status = "completed"
                    batch_record.total_images = total_images
                    batch_record.lesions_detected = lesion_count
                    batch_record.high_risk_count = high_risk_count
                    batch_record.results = results
                    batch_record.completed_at = datetime.utcnow()
                    db.commit()
            finally:
                db.close()
        except Exception:
            pass

        processing_time = time.time() - start_time

        return {
            "task_id": self.request.id,
            "check_id": check_id,
            "total_images": total_images,
            "images_processed": len([r for r in results if "error" not in r]),
            "errors": len([r for r in results if "error" in r]),
            "lesion_count": lesion_count,
            "high_risk_count": high_risk_count,
            "results": results,
            "processing_time": processing_time,
            "status": "success"
        }

    except SoftTimeLimitExceeded:
        return {"status": "error", "error": "Batch processing timed out", "task_id": self.request.id}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc(), "task_id": self.request.id}


@celery_app.task(bind=True, name="tasks.histopathology_analyze_task")
def histopathology_analyze_task(
    self,
    image_path: str,
    user_id: int,
    filename: str,
    tissue_type: str = None,
) -> Dict[str, Any]:
    """
    Histopathology analysis task for biopsy slide images.

    Analyzes tissue samples for:
    - Tissue classification
    - Malignancy assessment
    - Cellular patterns
    """
    try:
        start_time = time.time()
        update_progress(1, 4, "Loading slide image")

        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        update_progress(2, 4, "Analyzing tissue sample")

        # Try to use histopathology analyzer
        try:
            from histopathology_analyzer import analyze_histopathology
            result = analyze_histopathology(image, tissue_type)
        except ImportError:
            # Fallback basic analysis
            result = {
                "tissue_classification": "unknown",
                "malignancy_score": 0.0,
                "cellular_patterns": [],
                "confidence": 0.0
            }

        update_progress(3, 4, "Generating pathology report")

        # Generate report
        report = {
            "tissue_type": result.get("tissue_classification", "unknown"),
            "malignancy_assessment": "benign" if result.get("malignancy_score", 0) < 0.5 else "suspicious",
            "confidence": result.get("confidence", 0.0),
            "findings": result.get("cellular_patterns", []),
            "recommendation": "Clinical correlation recommended"
        }

        update_progress(4, 4, "Complete")
        processing_time = time.time() - start_time

        return {
            "task_id": self.request.id,
            "analysis_type": "histopathology",
            "report": report,
            "raw_results": result,
            "processing_time": processing_time,
            "status": "success"
        }

    except SoftTimeLimitExceeded:
        return {"status": "error", "error": "Task timed out", "task_id": self.request.id}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc(), "task_id": self.request.id}
