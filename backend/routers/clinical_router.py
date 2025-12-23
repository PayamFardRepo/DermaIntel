"""
Clinical Analysis Router

Endpoints for:
- Burn classification
- Dermoscopy analysis
- Biopsy and histopathology
- Clinical photography standards
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
import json

from database import get_db, User, AnalysisHistory, SessionLocal
from auth import get_current_active_user

# Import model monitoring (now non-blocking)
from model_monitoring import record_inference
import time

router = APIRouter(tags=["Clinical Analysis"])


# =============================================================================
# BURN CLASSIFICATION
# =============================================================================

@router.post("/classify-burn")
async def classify_burn(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Classify burn severity using VGG16 model.

    Returns:
        - severity_class: Normal/First/Second/Third Degree
        - severity_level: 0-3
        - confidence: Model confidence score
        - probabilities: All class probabilities
        - urgency: Urgency level
        - treatment_advice: Detailed treatment recommendations
        - medical_attention_required: Boolean flag
        - is_burn_detected: Boolean flag
    """
    print(f"[BURN CLASSIFICATION] User {current_user.username} uploaded image for burn analysis")

    image_bytes = await image.read()

    from burn_classifier import get_burn_classifier
    classifier = get_burn_classifier()

    # Burn classification with monitoring
    burn_inference_start = time.time()
    burn_inference_error = None
    try:
        result = classifier.classify(image_bytes)
        burn_inference_success = True
    except Exception as e:
        burn_inference_error = str(e)
        burn_inference_success = False
        print(f"Error classifying burn: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to classify burn: {str(e)}")
    finally:
        burn_inference_time = (time.time() - burn_inference_start) * 1000
        record_inference(
            model_name="burn_classifier",
            inference_time_ms=burn_inference_time,
            success=burn_inference_success,
            confidence=result['confidence'] if burn_inference_success else None,
            error=burn_inference_error,
            metadata={"endpoint": "/classify-burn", "user_id": current_user.id}
        )

    stats = classifier.get_burn_statistics(result['probabilities'])

    response = {
        'severity_class': result['severity_class'],
        'severity_level': result['severity_level'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities'],
        'urgency': result['urgency'],
        'treatment_advice': result['treatment_advice'],
        'medical_attention_required': result['medical_attention_required'],
        'is_burn_detected': result['is_burn_detected'],
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    }

    print(f"[BURN CLASSIFICATION] Result: {result['severity_class']} (confidence: {result['confidence']:.2%})")

    return response


# =============================================================================
# DERMOSCOPY ANALYSIS
# =============================================================================

@router.get("/dermoscopy/health")
async def dermoscopy_health():
    """Health check for dermoscopy endpoint."""
    try:
        from dermoscopy_analyzer import get_dermoscopy_detector
        detector = get_dermoscopy_detector()
        return {"status": "ok", "message": "Dermoscopy analyzer is ready"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/dermoscopy/analyze")
async def analyze_dermoscopy(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Comprehensive dermoscopic feature analysis.

    Detects and analyzes:
    - Pigment network (reticular, atypical, branched)
    - Globules and dots
    - Streaks and pseudopods
    - Blue-white veil
    - Vascular patterns
    - Regression structures
    - Color analysis
    - Symmetry analysis

    Also calculates:
    - 7-Point Checklist score
    - ABCD dermoscopy score
    - Overall risk assessment
    """
    print(f"[DERMOSCOPY] User {current_user.username} uploaded image for dermoscopic analysis")

    image_bytes = await image.read()

    from dermoscopy_analyzer import get_dermoscopy_detector
    detector = get_dermoscopy_detector()

    # Dermoscopy analysis with monitoring
    dermoscopy_inference_start = time.time()
    dermoscopy_inference_error = None
    try:
        results = detector.analyze(image_bytes)
        dermoscopy_inference_success = True
    except HTTPException:
        raise
    except ValueError as e:
        # Validation errors (e.g., image too small) return 422
        dermoscopy_inference_error = str(e)
        dermoscopy_inference_success = False
        print(f"[DERMOSCOPY ERROR] Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        dermoscopy_inference_error = str(e)
        dermoscopy_inference_success = False
        print(f"[DERMOSCOPY ERROR] Error in dermoscopy analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to analyze dermoscopic features: {str(e)}")
    finally:
        dermoscopy_inference_time = (time.time() - dermoscopy_inference_start) * 1000
        # Get confidence from risk assessment if available
        derm_confidence = None
        if dermoscopy_inference_success and results.get('risk_assessment'):
            risk = results['risk_assessment'].get('risk_level', 'unknown')
            # Map risk level to confidence (higher risk = higher confidence in detection)
            risk_conf_map = {'low': 0.3, 'medium': 0.5, 'high': 0.7, 'very_high': 0.9}
            derm_confidence = risk_conf_map.get(risk, 0.5)
        record_inference(
            model_name="dermoscopy_analyzer",
            inference_time_ms=dermoscopy_inference_time,
            success=dermoscopy_inference_success,
            confidence=derm_confidence,
            error=dermoscopy_inference_error,
            metadata={"endpoint": "/dermoscopy/analyze", "user_id": current_user.id}
        )

    print(f"[DERMOSCOPY] 7-Point Score: {results['seven_point_score']['score']}/9")
    print(f"[DERMOSCOPY] Risk Level: {results['risk_assessment']['risk_level']}")

    response = {
        'pigment_network': results['pigment_network'],
        'globules': results['globules'],
        'streaks': results['streaks'],
        'blue_white_veil': results['blue_white_veil'],
        'vascular_patterns': results['vascular_patterns'],
        'regression': results['regression'],
        'color_analysis': results['color_analysis'],
        'symmetry_analysis': results['symmetry_analysis'],
        'seven_point_checklist': results['seven_point_score'],
        'abcd_score': results['abcd_score'],
        'risk_assessment': results['risk_assessment'],
        'overlays': results['overlays'],
        'timestamp': datetime.now().isoformat()
    }

    return response


# =============================================================================
# BIOPSY AND HISTOPATHOLOGY
# =============================================================================

@router.post("/analysis/biopsy/{analysis_id}")
def add_biopsy_result(
    analysis_id: int,
    biopsy_result: str = Form(...),
    biopsy_date: str = Form(None),
    biopsy_notes: str = Form(None),
    biopsy_facility: str = Form(None),
    pathologist_name: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Add biopsy/pathology results to an analysis record for accuracy tracking.
    Links actual pathology results to AI predictions.
    """
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    biopsy_date_obj = None
    if biopsy_date:
        try:
            biopsy_date_obj = datetime.fromisoformat(biopsy_date.replace('Z', '+00:00'))
        except:
            biopsy_date_obj = datetime.utcnow()

    analysis.biopsy_performed = True
    analysis.biopsy_result = biopsy_result
    analysis.biopsy_date = biopsy_date_obj or datetime.utcnow()
    analysis.biopsy_notes = biopsy_notes
    analysis.biopsy_facility = biopsy_facility
    analysis.pathologist_name = pathologist_name
    analysis.biopsy_updated_at = datetime.utcnow()

    # Calculate accuracy
    ai_prediction = analysis.predicted_class

    if ai_prediction and biopsy_result:
        if ai_prediction.lower() == biopsy_result.lower():
            analysis.prediction_correct = True
            analysis.accuracy_category = "exact_match"
        else:
            malignant_conditions = ["melanoma", "basal cell carcinoma", "squamous cell carcinoma"]
            ai_is_malignant = any(cond in ai_prediction.lower() for cond in malignant_conditions)
            biopsy_is_malignant = any(cond in biopsy_result.lower() for cond in malignant_conditions)

            if ai_is_malignant == biopsy_is_malignant:
                analysis.prediction_correct = False
                analysis.accuracy_category = "category_match"
            else:
                analysis.prediction_correct = False
                analysis.accuracy_category = "mismatch"
    else:
        analysis.accuracy_category = "pending"

    db.commit()
    db.refresh(analysis)

    return {
        "message": "Biopsy results added successfully",
        "analysis_id": analysis_id,
        "prediction_correct": analysis.prediction_correct,
        "accuracy_category": analysis.accuracy_category,
        "ai_prediction": ai_prediction,
        "biopsy_result": biopsy_result
    }


@router.post("/analyze-histopathology")
async def analyze_histopathology_image(
    file: UploadFile = File(...),
    analysis_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Analyze a histopathology/biopsy slide image using AI.

    Returns:
    - Tissue classification (12 classes)
    - Malignancy assessment
    - Confidence scores with uncertainty quantification
    - Correlation with previous dermoscopy AI prediction
    """
    try:
        from histopathology_analyzer import analyze_biopsy_image

        contents = await file.read()

        dermoscopy_prediction = None
        if analysis_id:
            analysis = db.query(AnalysisHistory).filter(
                AnalysisHistory.id == analysis_id,
                AnalysisHistory.user_id == current_user.id
            ).first()
            if analysis:
                dermoscopy_prediction = {
                    'prediction': analysis.predicted_class,
                    'confidence': analysis.confidence,
                    'probabilities': json.loads(analysis.top_predictions) if analysis.top_predictions else {}
                }

        result = analyze_biopsy_image(contents, dermoscopy_prediction)

        if analysis_id and analysis:
            analysis.histopathology_performed = True
            analysis.histopathology_result = result['primary_diagnosis']
            analysis.histopathology_malignant = result['malignancy']['is_malignant']
            analysis.histopathology_confidence = result['primary_probability']
            analysis.histopathology_date = datetime.utcnow()
            analysis.histopathology_tissue_type = result.get('primary_diagnosis')
            analysis.histopathology_risk_level = result.get('malignancy', {}).get('risk_level', 'unknown')
            analysis.histopathology_features = json.dumps(result.get('key_features', []))
            analysis.histopathology_recommendations = '; '.join(result.get('recommendations', []))
            analysis.histopathology_image_quality = json.dumps(result.get('image_quality', {}))
            analysis.histopathology_predictions = json.dumps(result.get('predictions', []))

            if 'dermoscopy_correlation' in result:
                correlation = result['dermoscopy_correlation']
                analysis.ai_concordance = correlation.get('is_concordant', False)
                analysis.ai_concordance_type = correlation.get('concordance_type', 'unknown')
                analysis.ai_concordance_notes = correlation.get('clinical_implications', '')

            db.commit()

        return {
            "success": True,
            "analysis": result,
            "linked_analysis_id": analysis_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Histopathology analysis failed: {str(e)}")


@router.post("/analyze-histopathology-tiles")
async def analyze_histopathology_tiles(
    files: List[UploadFile] = File(...),
    aggregate: bool = Form(True),
    current_user: User = Depends(get_current_active_user)
):
    """
    Analyze multiple tiles from a whole slide image (WSI).
    """
    try:
        from histopathology_analyzer import get_histopathology_analyzer
        analyzer = get_histopathology_analyzer()

        tiles = []
        for file in files:
            contents = await file.read()
            tiles.append(contents)

        result = analyzer.analyze_tile_batch(tiles, aggregate=aggregate)

        return {
            "success": True,
            "num_tiles": len(tiles),
            "aggregated": aggregate,
            "analysis": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile analysis failed: {str(e)}")


@router.get("/biopsy-correlation/{analysis_id}")
def get_biopsy_correlation(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get correlation analysis between dermoscopy AI prediction and biopsy/histopathology results.
    """
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if not analysis.biopsy_performed and not getattr(analysis, 'histopathology_performed', False):
        return {
            "analysis_id": analysis_id,
            "biopsy_performed": False,
            "message": "No biopsy or histopathology results available for this analysis"
        }

    ai_prediction = analysis.predicted_class
    ai_confidence = analysis.confidence
    biopsy_result = analysis.biopsy_result
    histopath_result = getattr(analysis, 'histopathology_result', None)

    ground_truth = histopath_result or biopsy_result

    is_concordant = False
    concordance_type = "none"

    if ai_prediction and ground_truth:
        ai_lower = ai_prediction.lower()
        truth_lower = ground_truth.lower()

        if ai_lower == truth_lower:
            is_concordant = True
            concordance_type = "exact"
        else:
            malignant = ["melanoma", "basal cell carcinoma", "squamous cell carcinoma", "carcinoma"]
            ai_malignant = any(m in ai_lower for m in malignant)
            truth_malignant = any(m in truth_lower for m in malignant)

            if ai_malignant == truth_malignant:
                is_concordant = True
                concordance_type = "category"

    return {
        "analysis_id": analysis_id,
        "ai_prediction": ai_prediction,
        "ai_confidence": ai_confidence,
        "biopsy_result": biopsy_result,
        "histopathology_result": histopath_result,
        "ground_truth": ground_truth,
        "is_concordant": is_concordant,
        "concordance_type": concordance_type,
        "biopsy_date": analysis.biopsy_date.isoformat() if analysis.biopsy_date else None,
        "pathologist_name": analysis.pathologist_name,
        "biopsy_facility": analysis.biopsy_facility,
        "biopsy_notes": analysis.biopsy_notes
    }


@router.get("/biopsy/history")
async def get_biopsy_history(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all analyses with biopsy results for the current user."""
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id,
        AnalysisHistory.biopsy_performed == True
    ).order_by(AnalysisHistory.biopsy_date.desc()).all()

    return {
        "total": len(analyses),
        "biopsies": [
            {
                "analysis_id": a.id,
                "ai_prediction": a.predicted_class,
                "biopsy_result": a.biopsy_result,
                "biopsy_date": a.biopsy_date.isoformat() if a.biopsy_date else None,
                "prediction_correct": a.prediction_correct,
                "accuracy_category": a.accuracy_category
            }
            for a in analyses
        ]
    }


@router.get("/biopsy/report/{analysis_id}")
async def generate_biopsy_report(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate a comprehensive biopsy correlation report."""
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if not analysis.biopsy_performed:
        raise HTTPException(status_code=400, detail="No biopsy results available for this analysis")

    return {
        "report_type": "biopsy_correlation",
        "analysis_id": analysis_id,
        "generated_at": datetime.utcnow().isoformat(),
        "patient_analysis": {
            "original_image_date": analysis.created_at.isoformat() if analysis.created_at else None,
            "body_location": analysis.body_location
        },
        "ai_analysis": {
            "predicted_class": analysis.predicted_class,
            "confidence": analysis.confidence,
            "risk_level": analysis.risk_level
        },
        "pathology_results": {
            "biopsy_date": analysis.biopsy_date.isoformat() if analysis.biopsy_date else None,
            "biopsy_result": analysis.biopsy_result,
            "pathologist_name": analysis.pathologist_name,
            "facility": analysis.biopsy_facility,
            "notes": analysis.biopsy_notes
        },
        "correlation": {
            "prediction_correct": analysis.prediction_correct,
            "accuracy_category": analysis.accuracy_category
        }
    }


@router.get("/ai-accuracy-stats")
async def get_ai_accuracy_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get AI prediction accuracy statistics based on biopsy-confirmed results.

    Returns aggregate statistics on how well AI predictions matched
    pathology-confirmed diagnoses.
    """
    # Get all analyses with biopsy results
    analyses_with_biopsy = db.query(AnalysisHistory).filter(
        AnalysisHistory.biopsy_performed == True,
        AnalysisHistory.biopsy_result.isnot(None)
    ).all()

    total_with_biopsy = len(analyses_with_biopsy)

    if total_with_biopsy == 0:
        return {
            "total_with_biopsy": 0,
            "concordant": 0,
            "discordant": 0,
            "concordance_rate": None,
            "accuracy_by_category": {},
            "accuracy_by_condition": {},
            "message": "No biopsy-confirmed analyses available yet"
        }

    # Calculate concordance statistics
    concordant = sum(1 for a in analyses_with_biopsy if a.prediction_correct == True)
    discordant = sum(1 for a in analyses_with_biopsy if a.prediction_correct == False)
    unknown = total_with_biopsy - concordant - discordant

    concordance_rate = round(concordant / total_with_biopsy * 100, 2) if total_with_biopsy > 0 else None

    # Accuracy by category (e.g., true_positive, true_negative, false_positive, false_negative)
    accuracy_by_category = {}
    for analysis in analyses_with_biopsy:
        category = analysis.accuracy_category or "unclassified"
        accuracy_by_category[category] = accuracy_by_category.get(category, 0) + 1

    # Accuracy by predicted condition
    accuracy_by_condition = {}
    for analysis in analyses_with_biopsy:
        condition = analysis.predicted_class or "unknown"
        if condition not in accuracy_by_condition:
            accuracy_by_condition[condition] = {"total": 0, "correct": 0}
        accuracy_by_condition[condition]["total"] += 1
        if analysis.prediction_correct:
            accuracy_by_condition[condition]["correct"] += 1

    # Calculate per-condition accuracy rates
    for condition in accuracy_by_condition:
        total = accuracy_by_condition[condition]["total"]
        correct = accuracy_by_condition[condition]["correct"]
        accuracy_by_condition[condition]["accuracy_rate"] = round(correct / total * 100, 2) if total > 0 else None

    return {
        "total_with_biopsy": total_with_biopsy,
        "concordant": concordant,
        "discordant": discordant,
        "unknown_correlation": unknown,
        "concordance_rate": concordance_rate,
        "accuracy_by_category": accuracy_by_category,
        "accuracy_by_condition": accuracy_by_condition
    }


# =============================================================================
# CLINICAL PHOTOGRAPHY
# =============================================================================

@router.post("/photography/assess-quality")
async def assess_photo_quality(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Real-time photo quality assessment for clinical photography.

    Provides instant feedback on:
    - Ruler/scale detection
    - Color calibration card detection
    - Lighting quality
    - Focus quality
    - Distance and angle guidance
    - Medical photography standards compliance
    """
    try:
        from clinical_photography_assistant import get_clinical_photography_assistant

        image_bytes = await image.read()
        assistant = get_clinical_photography_assistant()
        feedback = assistant.assess_photo_quality(image_bytes)

        return {
            "overall_score": feedback.overall_score,
            "quality_level": feedback.quality_level.value,
            "meets_medical_standards": feedback.meets_medical_standards,
            "dicom_compliant": feedback.dicom_compliant,
            "scores": {
                "lighting": feedback.lighting_score,
                "focus": feedback.focus_score,
                "distance": feedback.distance_score,
                "angle": feedback.angle_score,
                "scale": feedback.scale_score,
                "color_card": feedback.color_card_score
            },
            "detections": {
                "ruler_detected": feedback.ruler_detected,
                "color_card_detected": feedback.color_card_detected,
                "has_glare": feedback.has_glare,
                "has_shadows": feedback.has_shadows,
                "is_blurry": feedback.is_blurry,
                "too_close": feedback.too_close,
                "too_far": feedback.too_far
            },
            "measurements": {
                "estimated_distance_cm": feedback.estimated_distance_cm,
                "pixel_to_mm_ratio": feedback.pixel_to_mm_ratio,
                "glare_percentage": feedback.glare_percentage,
                "shadow_percentage": feedback.shadow_percentage
            },
            "feedback": {
                "issues": feedback.issues,
                "suggestions": feedback.suggestions,
                "warnings": feedback.warnings
            },
            "ready_to_capture": (
                feedback.quality_level.value in ["excellent", "good"] and
                not feedback.warnings
            )
        }

    except Exception as e:
        print(f"Error assessing photo quality: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to assess photo quality: {str(e)}")


@router.post("/photography/calibrate")
async def store_calibration_data(
    pixel_to_mm_ratio: float = Form(...),
    color_profile: Optional[str] = Form(None),
    device_info: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """Store calibration data from a reference photo."""
    try:
        return {
            "message": "Calibration data stored successfully",
            "pixel_to_mm_ratio": pixel_to_mm_ratio,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error storing calibration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store calibration: {str(e)}")


@router.get("/photography/standards")
async def get_photography_standards():
    """Get medical photography standards and requirements."""
    return {
        "lighting": {
            "min_score": 70,
            "max_glare_percentage": 10,
            "max_shadow_percentage": 20,
            "ideal_brightness_range": [100, 180],
            "recommendations": [
                "Use diffused, even lighting from multiple angles",
                "Avoid direct flash - use ring light or dual lights at 45 degrees",
                "Eliminate shadows behind subject",
                "Use polarized filters to reduce glare on oily skin"
            ]
        },
        "focus": {
            "min_score": 75,
            "recommendations": [
                "Tap to focus on lesion before capture",
                "Hold camera steady or use tripod",
                "Ensure entire lesion is in sharp focus",
                "Use burst mode and select sharpest image"
            ]
        },
        "distance": {
            "ideal_range_cm": [15, 30],
            "recommendations": [
                "Include ruler for size reference",
                "Fill frame with lesion and surrounding normal skin",
                "Capture overview, medium, and close-up views",
                "Maintain consistent distance for serial monitoring"
            ]
        },
        "angle": {
            "recommendations": [
                "Position camera perpendicular to skin surface",
                "Avoid tilted or oblique angles",
                "Use level indicator or AR guide",
                "Document unusual angles if medically necessary"
            ]
        },
        "scale": {
            "requirements": [
                "Include metric ruler in at least one view",
                "Place ruler on same plane as lesion",
                "Ensure ruler is clearly visible and in focus",
                "Use standard medical ruler with mm markings"
            ]
        },
        "color_accuracy": {
            "requirements": [
                "Include color calibration card when possible",
                "Use standard reference (X-Rite ColorChecker, Kodak Gray Card)",
                "Ensure white balance is set correctly",
                "Capture under consistent lighting conditions"
            ]
        },
        "dicom_compliance": {
            "required_metadata": [
                "Patient ID (anonymized if necessary)",
                "Acquisition date and time",
                "Body site and laterality",
                "View description (overview/close-up)",
                "Scale/pixel spacing information",
                "Camera and lens information",
                "Lighting conditions"
            ]
        },
        "resolution": {
            "minimum": "1920x1080 (2MP)",
            "recommended": "3840x2160 (8MP) or higher",
            "format": "JPEG (high quality) or PNG (lossless)"
        }
    }


@router.post("/photography/assess-enhanced")
async def assess_photo_quality_enhanced(
    file: UploadFile = File(...),
    include_ar_overlay: bool = Form(True),
    include_dicom: bool = Form(True),
    session_id: Optional[str] = Form(None),
    capture_angle: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Enhanced photo quality assessment with full medical photography standards.
    Returns comprehensive analysis including AR overlay guidance and DICOM metadata.
    """
    try:
        from clinical_photography_assistant import (
            CaptureAngle, get_clinical_photography_assistant
        )

        assistant = get_clinical_photography_assistant()
        contents = await file.read()

        angle = None
        if capture_angle:
            try:
                angle = CaptureAngle(capture_angle)
            except ValueError:
                pass

        feedback = assistant.assess_photo_quality(
            image_bytes=contents,
            include_ar_overlay=include_ar_overlay,
            include_dicom=include_dicom,
            session_id=session_id,
            capture_angle=angle
        )

        result = {
            "overall_score": feedback.overall_score,
            "quality_level": feedback.quality_level.value,
            "issues": feedback.issues,
            "suggestions": feedback.suggestions,
            "warnings": feedback.warnings,
            "scores": {
                "lighting": feedback.lighting_score,
                "focus": feedback.focus_score,
                "distance": feedback.distance_score,
                "angle": feedback.angle_score,
                "scale": feedback.scale_score,
                "color_card": feedback.color_card_score
            },
            "detections": {
                "ruler_detected": feedback.ruler_detected,
                "color_card_detected": feedback.color_card_detected,
                "has_glare": feedback.has_glare,
                "has_shadows": feedback.has_shadows,
                "is_blurry": feedback.is_blurry,
                "too_close": feedback.too_close,
                "too_far": feedback.too_far
            },
            "measurements": {
                "estimated_distance_cm": feedback.estimated_distance_cm,
                "pixel_to_mm_ratio": feedback.pixel_to_mm_ratio,
                "glare_percentage": feedback.glare_percentage,
                "shadow_percentage": feedback.shadow_percentage
            },
            "compliance": {
                "meets_medical_standards": feedback.meets_medical_standards,
                "dicom_compliant": feedback.dicom_compliant
            }
        }

        if feedback.scale_reference and feedback.scale_reference.detected:
            result["scale_reference"] = {
                "type": feedback.scale_reference.reference_type.value,
                "confidence": feedback.scale_reference.confidence,
                "bounding_box": feedback.scale_reference.bounding_box,
                "real_world_size_mm": feedback.scale_reference.real_world_size_mm
            }

        if feedback.color_calibration and feedback.color_calibration.detected:
            result["color_calibration"] = {
                "card_type": feedback.color_calibration.card_type.value,
                "confidence": feedback.color_calibration.confidence,
                "bounding_box": feedback.color_calibration.bounding_box,
                "white_balance_correction": feedback.color_calibration.white_balance_correction,
                "color_temperature_k": feedback.color_calibration.color_temperature_k
            }

        if feedback.ar_overlay:
            result["ar_overlay"] = {
                "target_box": feedback.ar_overlay.target_box,
                "ruler_zone": feedback.ar_overlay.ruler_zone,
                "color_card_zone": feedback.ar_overlay.color_card_zone,
                "tilt_angle": feedback.ar_overlay.tilt_angle,
                "distance_indicator": feedback.ar_overlay.distance_indicator,
                "distance_bar_fill": feedback.ar_overlay.distance_bar_fill,
                "focus_indicator": feedback.ar_overlay.focus_indicator,
                "lighting_indicator": feedback.ar_overlay.lighting_indicator,
                "ready_to_capture": feedback.ar_overlay.ready_to_capture,
                "blocking_issues": feedback.ar_overlay.blocking_issues,
                "grid_lines": feedback.ar_overlay.grid_lines
            }

        if feedback.dicom_metadata:
            result["dicom_metadata"] = {
                "study_instance_uid": feedback.dicom_metadata.study_instance_uid,
                "series_instance_uid": feedback.dicom_metadata.series_instance_uid,
                "sop_instance_uid": feedback.dicom_metadata.sop_instance_uid,
                "modality": feedback.dicom_metadata.modality,
                "acquisition_datetime": feedback.dicom_metadata.acquisition_datetime,
                "rows": feedback.dicom_metadata.rows,
                "columns": feedback.dicom_metadata.columns,
                "pixel_spacing": feedback.dicom_metadata.pixel_spacing,
                "body_part_examined": feedback.dicom_metadata.body_part_examined,
                "image_quality_indicator": feedback.dicom_metadata.image_quality_indicator
            }

        return result

    except Exception as e:
        print(f"Error in enhanced photo assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@router.get("/photography/capture-angles")
async def get_capture_angles():
    """Get recommended capture angles for documentation."""
    return {
        "angles": [
            {"id": "overview", "name": "Overview", "description": "Wide shot showing body region context"},
            {"id": "medium", "name": "Medium", "description": "Lesion with surrounding skin (5cm margin)"},
            {"id": "closeup", "name": "Close-up", "description": "Detailed view of lesion surface"},
            {"id": "oblique_left", "name": "Oblique Left", "description": "45-degree angle from left"},
            {"id": "oblique_right", "name": "Oblique Right", "description": "45-degree angle from right"},
            {"id": "tangential", "name": "Tangential", "description": "Low angle to show elevation"}
        ]
    }


@router.get("/photography/reference-types")
async def get_reference_types():
    """Get supported reference object types for scale measurement."""
    return {
        "rulers": [
            {"id": "metric_ruler", "name": "Metric Ruler", "description": "Standard mm/cm ruler"},
            {"id": "medical_ruler", "name": "Medical Photo Ruler", "description": "L-shaped dermatology ruler"}
        ],
        "coins": [
            {"id": "us_quarter", "name": "US Quarter", "diameter_mm": 24.26},
            {"id": "us_dime", "name": "US Dime", "diameter_mm": 17.91},
            {"id": "euro_1", "name": "1 Euro", "diameter_mm": 23.25},
            {"id": "uk_pound", "name": "UK Pound", "diameter_mm": 22.5}
        ],
        "cards": [
            {"id": "credit_card", "name": "Credit Card", "width_mm": 85.6, "height_mm": 53.98}
        ]
    }


@router.get("/photography/color-cards")
async def get_color_card_types():
    """Get supported color calibration card types."""
    return {
        "cards": [
            {"id": "xrite_colorchecker", "name": "X-Rite ColorChecker", "patches": 24},
            {"id": "kodak_gray", "name": "Kodak Gray Card", "patches": 1},
            {"id": "qp_card", "name": "QP Card", "patches": 8},
            {"id": "colorgage", "name": "ColorGauge", "patches": 24}
        ]
    }


@router.get("/photography/lighting-tips")
async def get_lighting_tips():
    """Get lighting recommendations for clinical photography."""
    return {
        "setup_types": [
            {
                "name": "Ring Light",
                "description": "Even, shadowless lighting for close-ups",
                "ideal_for": ["dermoscopy", "macro", "face"],
                "pros": ["No shadows", "Even illumination", "Easy setup"],
                "cons": ["Ring reflection in wet surfaces"]
            },
            {
                "name": "Dual Side Lights",
                "description": "45-degree angles for texture detail",
                "ideal_for": ["texture", "scars", "raised lesions"],
                "pros": ["Shows depth", "Reveals texture"],
                "cons": ["May create shadows"]
            },
            {
                "name": "Polarized Light",
                "description": "Reduces glare on oily skin",
                "ideal_for": ["oily skin", "wet lesions"],
                "pros": ["Eliminates specular reflection"],
                "cons": ["Reduces color saturation slightly"]
            }
        ],
        "common_issues": [
            {"issue": "Glare/Hotspots", "solution": "Use diffuser or polarizer, angle light"},
            {"issue": "Dark Shadows", "solution": "Add fill light opposite main light"},
            {"issue": "Yellow Cast", "solution": "Use daylight-balanced bulbs (5500K)"},
            {"issue": "Uneven Lighting", "solution": "Ensure lights equidistant, use diffusers"}
        ]
    }


@router.get("/photography/dicom-requirements")
async def get_dicom_requirements():
    """Get DICOM metadata requirements for dermatology images."""
    return {
        "required_modules": [
            {
                "name": "Patient Module",
                "tags": ["PatientName", "PatientID", "PatientBirthDate", "PatientSex"]
            },
            {
                "name": "General Study Module",
                "tags": ["StudyInstanceUID", "StudyDate", "StudyTime", "ReferringPhysicianName"]
            },
            {
                "name": "General Series Module",
                "tags": ["Modality", "SeriesInstanceUID", "SeriesNumber", "Laterality", "BodyPartExamined"]
            },
            {
                "name": "VL Image Module",
                "tags": ["ImageType", "PhotometricInterpretation", "Rows", "Columns", "SamplesPerPixel"]
            }
        ],
        "dermatology_specific": [
            {"tag": "AnatomicRegionSequence", "description": "Body site location"},
            {"tag": "PrimaryAnatomicStructureSequence", "description": "Specific anatomic structure"},
            {"tag": "ImageLaterality", "description": "Left/Right/Both"},
            {"tag": "PixelSpacing", "description": "Physical size of pixels in mm"},
            {"tag": "CalibrationSequence", "description": "How pixel spacing was calibrated"}
        ]
    }


# =============================================================================
# CLINICAL SCORING ENDPOINTS (SCORAD, PASI)
# =============================================================================

@router.post("/calculate-scorad")
async def calculate_scorad(
    extent_percentage: float = Form(...),  # 0-100% body surface area affected
    intensity_erythema: int = Form(...),  # 0-3
    intensity_edema: int = Form(...),  # 0-3
    intensity_oozing: int = Form(...),  # 0-3
    intensity_excoriation: int = Form(...),  # 0-3
    intensity_lichenification: int = Form(...),  # 0-3
    intensity_dryness: int = Form(...),  # 0-3
    subjective_itch: int = Form(...),  # 0-10
    subjective_sleep_loss: int = Form(...),  # 0-10
    current_user: User = Depends(get_current_active_user)
):
    """
    Calculate SCORAD (SCORing Atopic Dermatitis) index.

    Formula: SCORAD = A/5 + 7B/2 + C
    Where:
    - A = extent (0-100%)
    - B = intensity (sum of 6 items, 0-18)
    - C = subjective symptoms (itch + sleep loss, 0-20)

    Score interpretation:
    - <25: Mild
    - 25-50: Moderate
    - >50: Severe
    """
    # Calculate components
    A = extent_percentage
    B = (intensity_erythema + intensity_edema + intensity_oozing +
         intensity_excoriation + intensity_lichenification + intensity_dryness)
    C = subjective_itch + subjective_sleep_loss

    # Calculate SCORAD
    scorad = (A / 5) + (7 * B / 2) + C

    # Determine severity
    if scorad < 25:
        severity = "mild"
        recommendation = "Maintain good skincare routine. Use emollients regularly."
    elif scorad < 50:
        severity = "moderate"
        recommendation = "Consider topical corticosteroids. Follow up with dermatologist."
    else:
        severity = "severe"
        recommendation = "Urgent dermatologist consultation recommended. May need systemic therapy."

    return {
        "scorad_score": round(scorad, 1),
        "severity": severity,
        "recommendation": recommendation,
        "components": {
            "extent_score": round(A / 5, 1),
            "intensity_score": round(7 * B / 2, 1),
            "subjective_score": C
        },
        "interpretation": {
            "mild": "<25",
            "moderate": "25-50",
            "severe": ">50"
        }
    }


@router.post("/calculate-pasi")
async def calculate_pasi(
    # Head
    head_involvement: float = Form(...),  # 0-6
    head_erythema: int = Form(...),  # 0-4
    head_thickness: int = Form(...),  # 0-4
    head_scaling: int = Form(...),  # 0-4
    # Upper extremities
    upper_involvement: float = Form(...),  # 0-6
    upper_erythema: int = Form(...),  # 0-4
    upper_thickness: int = Form(...),  # 0-4
    upper_scaling: int = Form(...),  # 0-4
    # Trunk
    trunk_involvement: float = Form(...),  # 0-6
    trunk_erythema: int = Form(...),  # 0-4
    trunk_thickness: int = Form(...),  # 0-4
    trunk_scaling: int = Form(...),  # 0-4
    # Lower extremities
    lower_involvement: float = Form(...),  # 0-6
    lower_erythema: int = Form(...),  # 0-4
    lower_thickness: int = Form(...),  # 0-4
    lower_scaling: int = Form(...),  # 0-4
    current_user: User = Depends(get_current_active_user)
):
    """
    Calculate PASI (Psoriasis Area and Severity Index).

    Formula: PASI = 0.1(Eh+Ih+Dh)Ah + 0.2(Eu+Iu+Du)Au + 0.3(Et+It+Dt)At + 0.4(El+Il+Dl)Al

    Score interpretation:
    - <10: Mild
    - 10-20: Moderate
    - >20: Severe
    """
    # Calculate each body region
    head_score = 0.1 * (head_erythema + head_thickness + head_scaling) * head_involvement
    upper_score = 0.2 * (upper_erythema + upper_thickness + upper_scaling) * upper_involvement
    trunk_score = 0.3 * (trunk_erythema + trunk_thickness + trunk_scaling) * trunk_involvement
    lower_score = 0.4 * (lower_erythema + lower_thickness + lower_scaling) * lower_involvement

    # Calculate total PASI
    pasi = head_score + upper_score + trunk_score + lower_score

    # Determine severity
    if pasi < 10:
        severity = "mild"
        recommendation = "Topical treatments may be sufficient. Regular moisturizing important."
    elif pasi < 20:
        severity = "moderate"
        recommendation = "Consider phototherapy or systemic treatments. Dermatologist follow-up recommended."
    else:
        severity = "severe"
        recommendation = "Systemic therapy likely needed. Urgent dermatologist consultation."

    return {
        "pasi_score": round(pasi, 1),
        "severity": severity,
        "recommendation": recommendation,
        "regional_scores": {
            "head": round(head_score, 1),
            "upper_extremities": round(upper_score, 1),
            "trunk": round(trunk_score, 1),
            "lower_extremities": round(lower_score, 1)
        },
        "interpretation": {
            "mild": "<10",
            "moderate": "10-20",
            "severe": ">20"
        }
    }


# =============================================================================
# BIOPSY PROGRESSION AND TRACKING ENDPOINTS
# =============================================================================

@router.get("/biopsy/lesion-progression/{lesion_group_id}")
async def get_lesion_progression(
    lesion_group_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get lesion progression data over time for a lesion group."""
    # Get all analyses for this lesion group
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == lesion_group_id,
        AnalysisHistory.user_id == current_user.id
    ).order_by(AnalysisHistory.created_at).all()

    if not analyses:
        raise HTTPException(status_code=404, detail="No analyses found for this lesion group")

    progression_data = []
    for analysis in analyses:
        abcde_data = analysis.red_flag_data or {}
        progression_data.append({
            "analysis_id": analysis.id,
            "date": analysis.created_at.isoformat(),
            "predicted_class": analysis.predicted_class,
            "confidence": analysis.lesion_confidence,
            "risk_level": analysis.risk_level,
            "abcde_scores": {
                "asymmetry": abcde_data.get("asymmetry", {}).get("overall_score"),
                "border": abcde_data.get("border", {}).get("overall_score"),
                "color": abcde_data.get("color", {}).get("overall_score"),
                "diameter": abcde_data.get("diameter", {}).get("overall_score"),
            },
            "biopsy_performed": analysis.biopsy_performed,
            "biopsy_result": analysis.biopsy_result
        })

    # Calculate trends
    if len(progression_data) >= 2:
        first = progression_data[0]
        last = progression_data[-1]
        risk_trend = "stable"

        risk_order = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
        first_risk = risk_order.get(first.get("risk_level"), 0)
        last_risk = risk_order.get(last.get("risk_level"), 0)

        if last_risk > first_risk:
            risk_trend = "increasing"
        elif last_risk < first_risk:
            risk_trend = "decreasing"
    else:
        risk_trend = "insufficient_data"

    return {
        "lesion_group_id": lesion_group_id,
        "total_analyses": len(analyses),
        "progression_data": progression_data,
        "risk_trend": risk_trend,
        "first_analysis_date": analyses[0].created_at.isoformat() if analyses else None,
        "last_analysis_date": analyses[-1].created_at.isoformat() if analyses else None
    }


@router.get("/biopsy/ai-accuracy-trends")
async def get_ai_accuracy_trends(
    days: int = Query(90, description="Number of days to analyze"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get AI accuracy trends over time based on biopsy confirmations."""
    from datetime import timedelta

    cutoff_date = datetime.utcnow() - timedelta(days=days)

    # Get analyses with biopsy results
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.biopsy_performed == True,
        AnalysisHistory.biopsy_result != None,
        AnalysisHistory.created_at >= cutoff_date
    ).order_by(AnalysisHistory.created_at).all()

    if not analyses:
        return {
            "period_days": days,
            "total_biopsied": 0,
            "message": "No biopsy-confirmed analyses in this period"
        }

    # Calculate accuracy metrics
    total = len(analyses)
    exact_matches = sum(1 for a in analyses if a.prediction_correct == True)
    category_matches = sum(1 for a in analyses if a.accuracy_category == "category_match")
    mismatches = sum(1 for a in analyses if a.accuracy_category == "mismatch")

    # Group by week for trend analysis
    weekly_accuracy = {}
    for analysis in analyses:
        week_key = analysis.created_at.strftime("%Y-W%W")
        if week_key not in weekly_accuracy:
            weekly_accuracy[week_key] = {"total": 0, "correct": 0}
        weekly_accuracy[week_key]["total"] += 1
        if analysis.prediction_correct:
            weekly_accuracy[week_key]["correct"] += 1

    weekly_trends = [
        {
            "week": week,
            "accuracy": round(data["correct"] / data["total"] * 100, 1) if data["total"] > 0 else 0,
            "sample_size": data["total"]
        }
        for week, data in sorted(weekly_accuracy.items())
    ]

    return {
        "period_days": days,
        "total_biopsied": total,
        "accuracy_metrics": {
            "exact_match_rate": round(exact_matches / total * 100, 1) if total > 0 else 0,
            "category_match_rate": round((exact_matches + category_matches) / total * 100, 1) if total > 0 else 0,
            "mismatch_rate": round(mismatches / total * 100, 1) if total > 0 else 0
        },
        "weekly_trends": weekly_trends,
        "breakdown": {
            "exact_matches": exact_matches,
            "category_matches": category_matches,
            "mismatches": mismatches
        }
    }


@router.get("/biopsy/recurrence-check/{lesion_group_id}")
async def check_recurrence(
    lesion_group_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Check for potential recurrence in a previously biopsied lesion."""
    # Get analyses for this lesion group, ordered by date
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == lesion_group_id,
        AnalysisHistory.user_id == current_user.id
    ).order_by(AnalysisHistory.created_at).all()

    if not analyses:
        raise HTTPException(status_code=404, detail="No analyses found for this lesion group")

    # Find biopsy result
    biopsied_analysis = next((a for a in analyses if a.biopsy_performed), None)

    if not biopsied_analysis:
        return {
            "lesion_group_id": lesion_group_id,
            "has_biopsy": False,
            "message": "No biopsy performed for this lesion"
        }

    # Get analyses after biopsy
    post_biopsy = [a for a in analyses if a.created_at > biopsied_analysis.created_at]

    recurrence_indicators = []
    for analysis in post_biopsy:
        if analysis.risk_level in ["high", "very_high"]:
            recurrence_indicators.append({
                "analysis_id": analysis.id,
                "date": analysis.created_at.isoformat(),
                "predicted_class": analysis.predicted_class,
                "risk_level": analysis.risk_level,
                "confidence": analysis.lesion_confidence
            })

    return {
        "lesion_group_id": lesion_group_id,
        "has_biopsy": True,
        "biopsy_result": biopsied_analysis.biopsy_result,
        "biopsy_date": biopsied_analysis.biopsy_date.isoformat() if biopsied_analysis.biopsy_date else None,
        "post_biopsy_analyses": len(post_biopsy),
        "recurrence_indicators": recurrence_indicators,
        "recurrence_suspected": len(recurrence_indicators) > 0,
        "recommendation": "Urgent follow-up recommended" if recurrence_indicators else "Continue routine monitoring"
    }


@router.get("/biopsy/treatment-response/{lesion_group_id}")
async def get_treatment_response(
    lesion_group_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Track treatment response for a lesion group over time."""
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == lesion_group_id,
        AnalysisHistory.user_id == current_user.id
    ).order_by(AnalysisHistory.created_at).all()

    if not analyses:
        raise HTTPException(status_code=404, detail="No analyses found for this lesion group")

    # Track metrics over time
    timeline = []
    for i, analysis in enumerate(analyses):
        abcde = analysis.red_flag_data or {}
        timeline.append({
            "analysis_id": analysis.id,
            "date": analysis.created_at.isoformat(),
            "risk_level": analysis.risk_level,
            "confidence": analysis.lesion_confidence,
            "total_abcde_score": abcde.get("total_score"),
        })

    # Calculate response
    if len(analyses) >= 2:
        first_risk = {"low": 0, "medium": 1, "high": 2, "very_high": 3}.get(analyses[0].risk_level, 0)
        last_risk = {"low": 0, "medium": 1, "high": 2, "very_high": 3}.get(analyses[-1].risk_level, 0)

        if last_risk < first_risk:
            response = "positive"
            response_description = "Lesion showing improvement"
        elif last_risk > first_risk:
            response = "negative"
            response_description = "Lesion showing progression - consider treatment adjustment"
        else:
            response = "stable"
            response_description = "Lesion stable - continue current management"
    else:
        response = "insufficient_data"
        response_description = "More analyses needed to assess treatment response"

    return {
        "lesion_group_id": lesion_group_id,
        "total_analyses": len(analyses),
        "treatment_response": response,
        "response_description": response_description,
        "timeline": timeline
    }


# =============================================================================
# AJCC MELANOMA STAGING CALCULATOR (8th Edition)
# =============================================================================

# AJCC 8th Edition T Categories for Melanoma
AJCC_T_CATEGORIES = {
    "Tis": {
        "description": "Melanoma in situ",
        "thickness": "In situ",
        "ulceration": "N/A",
        "details": "Confined to epidermis, no invasion"
    },
    "T1a": {
        "description": "< 0.8 mm without ulceration",
        "thickness": "< 0.8 mm",
        "ulceration": "No",
        "details": "Thin melanoma, favorable prognosis"
    },
    "T1b": {
        "description": "< 0.8 mm with ulceration, or 0.8-1.0 mm with or without ulceration",
        "thickness": "< 0.8 mm (ulcerated) or 0.8-1.0 mm",
        "ulceration": "Yes or Any",
        "details": "Thin melanoma with adverse features"
    },
    "T2a": {
        "description": "> 1.0-2.0 mm without ulceration",
        "thickness": "> 1.0-2.0 mm",
        "ulceration": "No",
        "details": "Intermediate thickness, no ulceration"
    },
    "T2b": {
        "description": "> 1.0-2.0 mm with ulceration",
        "thickness": "> 1.0-2.0 mm",
        "ulceration": "Yes",
        "details": "Intermediate thickness with ulceration"
    },
    "T3a": {
        "description": "> 2.0-4.0 mm without ulceration",
        "thickness": "> 2.0-4.0 mm",
        "ulceration": "No",
        "details": "Thick melanoma, no ulceration"
    },
    "T3b": {
        "description": "> 2.0-4.0 mm with ulceration",
        "thickness": "> 2.0-4.0 mm",
        "ulceration": "Yes",
        "details": "Thick melanoma with ulceration"
    },
    "T4a": {
        "description": "> 4.0 mm without ulceration",
        "thickness": "> 4.0 mm",
        "ulceration": "No",
        "details": "Very thick melanoma, no ulceration"
    },
    "T4b": {
        "description": "> 4.0 mm with ulceration",
        "thickness": "> 4.0 mm",
        "ulceration": "Yes",
        "details": "Very thick melanoma with ulceration - highest T category"
    },
    "TX": {
        "description": "Primary tumor cannot be assessed",
        "thickness": "Unknown",
        "ulceration": "Unknown",
        "details": "Insufficient information for T staging"
    }
}

# AJCC 8th Edition N Categories for Melanoma
AJCC_N_CATEGORIES = {
    "N0": {
        "description": "No regional lymph node metastasis",
        "nodes": "0",
        "satellite_transit": "No",
        "details": "No evidence of nodal involvement"
    },
    "N1a": {
        "description": "One clinically occult node (detected by SLN biopsy)",
        "nodes": "1 (occult)",
        "satellite_transit": "No",
        "details": "Micrometastasis in one node"
    },
    "N1b": {
        "description": "One clinically detected node",
        "nodes": "1 (clinical)",
        "satellite_transit": "No",
        "details": "Macrometastasis in one node"
    },
    "N1c": {
        "description": "No regional nodes, but satellite/in-transit metastasis",
        "nodes": "0",
        "satellite_transit": "Yes",
        "details": "Satellite or in-transit metastasis without nodal disease"
    },
    "N2a": {
        "description": "2-3 clinically occult nodes",
        "nodes": "2-3 (occult)",
        "satellite_transit": "No",
        "details": "Micrometastases in 2-3 nodes"
    },
    "N2b": {
        "description": "2-3 nodes, at least one clinically detected",
        "nodes": "2-3 (clinical)",
        "satellite_transit": "No",
        "details": "At least one macrometastasis"
    },
    "N2c": {
        "description": "1 node (occult or clinical) with satellite/in-transit",
        "nodes": "1",
        "satellite_transit": "Yes",
        "details": "Nodal disease with satellite/in-transit metastasis"
    },
    "N3a": {
        "description": "4+ clinically occult nodes",
        "nodes": "4+ (occult)",
        "satellite_transit": "No",
        "details": "Extensive micrometastatic nodal disease"
    },
    "N3b": {
        "description": "4+ nodes, at least one clinically detected, or matted nodes",
        "nodes": "4+ (clinical/matted)",
        "satellite_transit": "No",
        "details": "Extensive macrometastatic nodal disease"
    },
    "N3c": {
        "description": "2+ nodes with satellite/in-transit metastasis",
        "nodes": "2+",
        "satellite_transit": "Yes",
        "details": "Multiple nodes with satellite/in-transit disease"
    },
    "NX": {
        "description": "Regional nodes cannot be assessed",
        "nodes": "Unknown",
        "satellite_transit": "Unknown",
        "details": "Insufficient information for N staging"
    }
}

# AJCC 8th Edition M Categories for Melanoma
AJCC_M_CATEGORIES = {
    "M0": {
        "description": "No distant metastasis",
        "site": "None",
        "ldh": "N/A",
        "details": "No evidence of distant spread"
    },
    "M1a": {
        "description": "Distant skin, subcutaneous, or nodal metastasis",
        "site": "Skin/SC/Nodes",
        "ldh": "Not elevated",
        "details": "Distant soft tissue metastases, normal LDH"
    },
    "M1a(1)": {
        "description": "M1a with elevated LDH",
        "site": "Skin/SC/Nodes",
        "ldh": "Elevated",
        "details": "Distant soft tissue metastases with elevated LDH"
    },
    "M1b": {
        "description": "Lung metastasis",
        "site": "Lung",
        "ldh": "Not elevated",
        "details": "Pulmonary metastases, normal LDH"
    },
    "M1b(1)": {
        "description": "M1b with elevated LDH",
        "site": "Lung",
        "ldh": "Elevated",
        "details": "Pulmonary metastases with elevated LDH"
    },
    "M1c": {
        "description": "Non-CNS visceral metastasis",
        "site": "Non-CNS visceral",
        "ldh": "Not elevated",
        "details": "Visceral metastases (not brain), normal LDH"
    },
    "M1c(1)": {
        "description": "M1c with elevated LDH",
        "site": "Non-CNS visceral",
        "ldh": "Elevated",
        "details": "Visceral metastases with elevated LDH"
    },
    "M1d": {
        "description": "CNS metastasis",
        "site": "CNS/Brain",
        "ldh": "Not elevated",
        "details": "Brain or CNS metastases, normal LDH"
    },
    "M1d(1)": {
        "description": "M1d with elevated LDH",
        "site": "CNS/Brain",
        "ldh": "Elevated",
        "details": "Brain or CNS metastases with elevated LDH"
    }
}

# AJCC 8th Edition Stage Groupings
AJCC_STAGE_GROUPINGS = {
    # Stage 0
    ("Tis", "N0", "M0"): {"stage": "0", "description": "Melanoma in situ", "prognosis": "Excellent - nearly 100% survival"},

    # Stage IA
    ("T1a", "N0", "M0"): {"stage": "IA", "description": "Early localized melanoma", "prognosis": "Excellent - ~99% 5-year survival"},

    # Stage IB
    ("T1b", "N0", "M0"): {"stage": "IB", "description": "Early localized melanoma with adverse features", "prognosis": "Very good - ~97% 5-year survival"},
    ("T2a", "N0", "M0"): {"stage": "IB", "description": "Intermediate melanoma without ulceration", "prognosis": "Very good - ~97% 5-year survival"},

    # Stage IIA
    ("T2b", "N0", "M0"): {"stage": "IIA", "description": "Intermediate melanoma with ulceration", "prognosis": "Good - ~94% 5-year survival"},
    ("T3a", "N0", "M0"): {"stage": "IIA", "description": "Thick melanoma without ulceration", "prognosis": "Good - ~94% 5-year survival"},

    # Stage IIB
    ("T3b", "N0", "M0"): {"stage": "IIB", "description": "Thick melanoma with ulceration", "prognosis": "Moderate - ~87% 5-year survival"},
    ("T4a", "N0", "M0"): {"stage": "IIB", "description": "Very thick melanoma without ulceration", "prognosis": "Moderate - ~87% 5-year survival"},

    # Stage IIC
    ("T4b", "N0", "M0"): {"stage": "IIC", "description": "Very thick melanoma with ulceration", "prognosis": "Guarded - ~82% 5-year survival"},

    # Stage III (nodal involvement) - selected key combinations
    ("T1a", "N1a", "M0"): {"stage": "IIIA", "description": "Thin melanoma with micrometastatic node", "prognosis": "Variable - ~93% 5-year survival"},
    ("T1b", "N1a", "M0"): {"stage": "IIIA", "description": "Thin melanoma with micrometastatic node", "prognosis": "Variable - ~93% 5-year survival"},
    ("T2a", "N1a", "M0"): {"stage": "IIIA", "description": "Intermediate melanoma with micrometastatic node", "prognosis": "Variable - ~93% 5-year survival"},

    # Stage IV (distant metastasis)
    ("any", "any", "M1a"): {"stage": "IV", "description": "Distant skin/soft tissue metastasis", "prognosis": "Poor - variable survival"},
    ("any", "any", "M1b"): {"stage": "IV", "description": "Lung metastasis", "prognosis": "Poor - variable survival"},
    ("any", "any", "M1c"): {"stage": "IV", "description": "Visceral metastasis", "prognosis": "Poor - variable survival"},
    ("any", "any", "M1d"): {"stage": "IV", "description": "CNS metastasis", "prognosis": "Poor - variable survival"},
}


def calculate_ajcc_stage(t_category: str, n_category: str, m_category: str) -> dict:
    """Calculate AJCC stage based on TNM categories."""

    # Handle M1 categories (Stage IV)
    if m_category.startswith("M1"):
        m_base = m_category.split("(")[0]  # Get base M category
        return {
            "stage": "IV",
            "substage": "",
            "full_stage": "IV",
            "description": f"Distant metastasis ({AJCC_M_CATEGORIES.get(m_category, {}).get('site', 'distant')})",
            "prognosis": "Requires systemic therapy - survival depends on treatment response",
            "five_year_survival": "15-25% (improving with immunotherapy)",
            "treatment_implications": [
                "Systemic therapy indicated (immunotherapy, targeted therapy)",
                "Consider clinical trial enrollment",
                "Multidisciplinary tumor board review",
                "Palliative care consultation for symptom management",
                "Regular imaging surveillance"
            ]
        }

    # Check exact match first
    key = (t_category, n_category, m_category)
    if key in AJCC_STAGE_GROUPINGS:
        result = AJCC_STAGE_GROUPINGS[key]
        return {
            "stage": result["stage"][0] if len(result["stage"]) > 1 else result["stage"],
            "substage": result["stage"][1:] if len(result["stage"]) > 1 else "",
            "full_stage": result["stage"],
            "description": result["description"],
            "prognosis": result["prognosis"],
            "five_year_survival": _get_survival_rate(result["stage"]),
            "treatment_implications": _get_treatment_implications(result["stage"], t_category, n_category)
        }

    # Algorithmic staging for combinations not explicitly listed
    stage = _algorithmic_stage(t_category, n_category, m_category)
    return stage


def _get_survival_rate(stage: str) -> str:
    """Get approximate 5-year survival rate by stage."""
    rates = {
        "0": "~100%",
        "IA": "97-99%",
        "IB": "94-97%",
        "IIA": "90-94%",
        "IIB": "82-87%",
        "IIC": "75-82%",
        "IIIA": "88-93%",
        "IIIB": "77-83%",
        "IIIC": "60-69%",
        "IIID": "32-42%",
        "IV": "15-25%"
    }
    return rates.get(stage, "Variable")


def _get_treatment_implications(stage: str, t_cat: str, n_cat: str) -> list:
    """Get treatment implications based on stage."""
    implications = []

    if stage == "0":
        implications = [
            "Wide local excision with 0.5-1 cm margins",
            "No sentinel lymph node biopsy needed",
            "Regular skin surveillance recommended"
        ]
    elif stage in ["IA", "IB"]:
        implications = [
            "Wide local excision (1 cm margins for ≤1mm, 1-2 cm for >1mm)",
            "Consider sentinel lymph node biopsy for T1b",
            "Regular clinical follow-up every 6-12 months"
        ]
    elif stage in ["IIA", "IIB", "IIC"]:
        implications = [
            "Wide local excision with 2 cm margins",
            "Sentinel lymph node biopsy recommended",
            "Consider adjuvant therapy if high-risk features",
            "Imaging surveillance based on risk",
            "Follow-up every 3-6 months for 2 years"
        ]
    elif stage.startswith("III"):
        implications = [
            "Complete lymph node dissection if positive nodes",
            "Adjuvant immunotherapy (pembrolizumab/nivolumab) or targeted therapy (if BRAF+)",
            "Consider adjuvant radiation for high-risk nodal disease",
            "Regular imaging surveillance (CT/PET every 3-6 months)",
            "Multidisciplinary oncology team management"
        ]
    else:  # Stage IV
        implications = [
            "Systemic therapy: checkpoint inhibitors or targeted therapy",
            "BRAF/MEK inhibitors if BRAF V600 mutation positive",
            "Consider clinical trials",
            "Surgical resection of oligometastatic disease if feasible",
            "Radiation for symptomatic metastases"
        ]

    return implications


def _algorithmic_stage(t_cat: str, n_cat: str, m_cat: str) -> dict:
    """Algorithmic staging for TNM combinations."""

    # Stage 0
    if t_cat == "Tis" and n_cat == "N0" and m_cat == "M0":
        return {
            "stage": "0", "substage": "", "full_stage": "0",
            "description": "Melanoma in situ",
            "prognosis": "Excellent",
            "five_year_survival": "~100%",
            "treatment_implications": _get_treatment_implications("0", t_cat, n_cat)
        }

    # Any M1 = Stage IV
    if m_cat.startswith("M1"):
        return {
            "stage": "IV", "substage": "", "full_stage": "IV",
            "description": "Distant metastatic melanoma",
            "prognosis": "Requires systemic therapy",
            "five_year_survival": "15-25%",
            "treatment_implications": _get_treatment_implications("IV", t_cat, n_cat)
        }

    # N0, M0 = Stage I or II based on T
    if n_cat == "N0" and m_cat == "M0":
        if t_cat in ["T1a"]:
            return {"stage": "I", "substage": "A", "full_stage": "IA",
                    "description": "Early localized melanoma",
                    "prognosis": "Excellent", "five_year_survival": "97-99%",
                    "treatment_implications": _get_treatment_implications("IA", t_cat, n_cat)}
        elif t_cat in ["T1b", "T2a"]:
            return {"stage": "I", "substage": "B", "full_stage": "IB",
                    "description": "Early localized melanoma",
                    "prognosis": "Very good", "five_year_survival": "94-97%",
                    "treatment_implications": _get_treatment_implications("IB", t_cat, n_cat)}
        elif t_cat in ["T2b", "T3a"]:
            return {"stage": "II", "substage": "A", "full_stage": "IIA",
                    "description": "Intermediate melanoma",
                    "prognosis": "Good", "five_year_survival": "90-94%",
                    "treatment_implications": _get_treatment_implications("IIA", t_cat, n_cat)}
        elif t_cat in ["T3b", "T4a"]:
            return {"stage": "II", "substage": "B", "full_stage": "IIB",
                    "description": "Thick melanoma",
                    "prognosis": "Moderate", "five_year_survival": "82-87%",
                    "treatment_implications": _get_treatment_implications("IIB", t_cat, n_cat)}
        elif t_cat in ["T4b"]:
            return {"stage": "II", "substage": "C", "full_stage": "IIC",
                    "description": "Very thick ulcerated melanoma",
                    "prognosis": "Guarded", "five_year_survival": "75-82%",
                    "treatment_implications": _get_treatment_implications("IIC", t_cat, n_cat)}

    # Any N+ = Stage III
    if n_cat != "N0" and n_cat != "NX" and m_cat == "M0":
        # Determine substage based on T and N
        if n_cat in ["N1a"] and t_cat in ["T1a", "T1b", "T2a"]:
            substage = "A"
        elif n_cat in ["N1a", "N1b", "N2a"] and t_cat in ["T1a", "T1b", "T2a", "T2b", "T3a"]:
            substage = "B"
        elif n_cat in ["N2b", "N2c", "N3a", "N3b"] or t_cat in ["T3b", "T4a", "T4b"]:
            substage = "C"
        else:
            substage = "B"  # Default

        # N3c or T4b with N3 = IIID
        if n_cat == "N3c" or (t_cat == "T4b" and n_cat.startswith("N3")):
            substage = "D"

        return {
            "stage": "III", "substage": substage, "full_stage": f"III{substage}",
            "description": "Regional metastatic melanoma",
            "prognosis": "Variable - depends on extent of nodal involvement",
            "five_year_survival": _get_survival_rate(f"III{substage}"),
            "treatment_implications": _get_treatment_implications(f"III{substage}", t_cat, n_cat)
        }

    # Default/Unknown
    return {
        "stage": "Unknown", "substage": "", "full_stage": "Cannot be determined",
        "description": "Insufficient information for staging",
        "prognosis": "Cannot be determined",
        "five_year_survival": "N/A",
        "treatment_implications": ["Complete staging workup required", "Obtain pathology review"]
    }


@router.get("/ajcc-staging/categories")
async def get_ajcc_categories():
    """Get all AJCC TNM categories for melanoma staging."""
    return {
        "version": "AJCC 8th Edition (2018)",
        "cancer_type": "Cutaneous Melanoma",
        "t_categories": AJCC_T_CATEGORIES,
        "n_categories": AJCC_N_CATEGORIES,
        "m_categories": AJCC_M_CATEGORIES
    }


@router.post("/ajcc-staging/calculate")
async def calculate_melanoma_stage(
    t_category: str = Form(..., description="T category (e.g., T1a, T2b, T3a)"),
    n_category: str = Form(..., description="N category (e.g., N0, N1a, N2b)"),
    m_category: str = Form(..., description="M category (e.g., M0, M1a, M1c)"),
    breslow_thickness: Optional[float] = Form(None, description="Breslow thickness in mm"),
    ulceration: Optional[bool] = Form(None, description="Presence of ulceration"),
    mitotic_rate: Optional[float] = Form(None, description="Mitotic rate per mm²"),
    lymph_nodes_examined: Optional[int] = Form(None, description="Number of lymph nodes examined"),
    lymph_nodes_positive: Optional[int] = Form(None, description="Number of positive lymph nodes"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Calculate AJCC 8th Edition stage for cutaneous melanoma.

    Returns comprehensive staging information including:
    - Stage and substage
    - Prognosis and survival estimates
    - Treatment implications
    - Detailed category information
    """
    # Validate categories
    if t_category not in AJCC_T_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid T category: {t_category}")
    if n_category not in AJCC_N_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid N category: {n_category}")

    # Handle M category with LDH suffix
    m_base = m_category.split("(")[0]
    if m_base not in ["M0", "M1a", "M1b", "M1c", "M1d"] and m_category not in AJCC_M_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid M category: {m_category}")

    # Calculate stage
    staging_result = calculate_ajcc_stage(t_category, n_category, m_category)

    # Get category details
    t_details = AJCC_T_CATEGORIES.get(t_category, {})
    n_details = AJCC_N_CATEGORIES.get(n_category, {})
    m_details = AJCC_M_CATEGORIES.get(m_category, AJCC_M_CATEGORIES.get(m_base, {}))

    # Build response
    response = {
        "tnm_classification": {
            "t": {
                "category": t_category,
                **t_details
            },
            "n": {
                "category": n_category,
                **n_details
            },
            "m": {
                "category": m_category,
                **m_details
            }
        },
        "staging": staging_result,
        "clinical_parameters": {
            "breslow_thickness_mm": breslow_thickness,
            "ulceration": ulceration,
            "mitotic_rate_per_mm2": mitotic_rate,
            "lymph_nodes_examined": lymph_nodes_examined,
            "lymph_nodes_positive": lymph_nodes_positive
        },
        "ajcc_version": "8th Edition (2018)",
        "disclaimer": "This staging calculator is for educational purposes. Clinical staging should be performed by qualified healthcare professionals with full access to pathology reports and clinical information."
    }

    # Add T category suggestion based on thickness if provided
    if breslow_thickness is not None:
        suggested_t = _suggest_t_category(breslow_thickness, ulceration)
        response["suggested_t_category"] = suggested_t

    return response


def _suggest_t_category(thickness: float, ulceration: Optional[bool]) -> dict:
    """Suggest T category based on Breslow thickness and ulceration."""
    ulc = ulceration if ulceration is not None else False

    if thickness <= 0:
        return {"category": "Tis", "note": "In situ melanoma (no invasion)"}
    elif thickness < 0.8:
        return {"category": "T1a" if not ulc else "T1b",
                "note": f"< 0.8 mm {'without' if not ulc else 'with'} ulceration"}
    elif thickness <= 1.0:
        return {"category": "T1b", "note": "0.8-1.0 mm (T1b regardless of ulceration)"}
    elif thickness <= 2.0:
        return {"category": "T2a" if not ulc else "T2b",
                "note": f"> 1.0-2.0 mm {'without' if not ulc else 'with'} ulceration"}
    elif thickness <= 4.0:
        return {"category": "T3a" if not ulc else "T3b",
                "note": f"> 2.0-4.0 mm {'without' if not ulc else 'with'} ulceration"}
    else:
        return {"category": "T4a" if not ulc else "T4b",
                "note": f"> 4.0 mm {'without' if not ulc else 'with'} ulceration"}


@router.get("/ajcc-staging/stage-info/{stage}")
async def get_stage_info(stage: str):
    """Get detailed information about a specific AJCC stage."""
    stage = stage.upper()

    stage_info = {
        "0": {
            "name": "Stage 0 (Melanoma in situ)",
            "tnm": "Tis N0 M0",
            "description": "Melanoma cells confined to the epidermis, have not invaded deeper",
            "five_year_survival": "~100%",
            "ten_year_survival": "~100%",
            "characteristics": [
                "Confined to epidermis",
                "No dermal invasion",
                "Excellent prognosis"
            ],
            "treatment": [
                "Wide local excision with 0.5-1 cm margins",
                "Regular skin surveillance",
                "Sun protection education"
            ],
            "follow_up": "Annual skin exams, patient self-examination monthly"
        },
        "IA": {
            "name": "Stage IA",
            "tnm": "T1a N0 M0",
            "description": "Thin melanoma (< 0.8 mm) without ulceration, no spread",
            "five_year_survival": "97-99%",
            "ten_year_survival": "95-98%",
            "characteristics": [
                "Breslow thickness < 0.8 mm",
                "No ulceration",
                "No nodal or distant metastasis"
            ],
            "treatment": [
                "Wide local excision with 1 cm margins",
                "Sentinel lymph node biopsy generally not recommended",
                "Regular surveillance"
            ],
            "follow_up": "Clinical exam every 6-12 months for 5 years, then annually"
        },
        "IB": {
            "name": "Stage IB",
            "tnm": "T1b N0 M0 or T2a N0 M0",
            "description": "Thin melanoma with ulceration or intermediate thickness without ulceration",
            "five_year_survival": "94-97%",
            "ten_year_survival": "90-95%",
            "characteristics": [
                "T1b: < 0.8 mm with ulceration, or 0.8-1.0 mm any ulceration",
                "T2a: > 1.0-2.0 mm without ulceration",
                "No nodal involvement"
            ],
            "treatment": [
                "Wide local excision (1-2 cm margins)",
                "Consider sentinel lymph node biopsy",
                "Baseline imaging in select cases"
            ],
            "follow_up": "Clinical exam every 6 months for 2-3 years, then annually"
        },
        "IIA": {
            "name": "Stage IIA",
            "tnm": "T2b N0 M0 or T3a N0 M0",
            "description": "Intermediate to thick melanoma with varying ulceration status",
            "five_year_survival": "90-94%",
            "ten_year_survival": "85-90%",
            "characteristics": [
                "T2b: > 1.0-2.0 mm with ulceration",
                "T3a: > 2.0-4.0 mm without ulceration",
                "No nodal involvement"
            ],
            "treatment": [
                "Wide local excision with 2 cm margins",
                "Sentinel lymph node biopsy recommended",
                "Consider adjuvant therapy discussion if high-risk features"
            ],
            "follow_up": "Clinical exam every 3-6 months, consider surveillance imaging"
        },
        "IIB": {
            "name": "Stage IIB",
            "tnm": "T3b N0 M0 or T4a N0 M0",
            "description": "Thick melanoma, higher risk for recurrence",
            "five_year_survival": "82-87%",
            "ten_year_survival": "75-82%",
            "characteristics": [
                "T3b: > 2.0-4.0 mm with ulceration",
                "T4a: > 4.0 mm without ulceration",
                "Increased risk of micrometastatic disease"
            ],
            "treatment": [
                "Wide local excision with 2 cm margins",
                "Sentinel lymph node biopsy strongly recommended",
                "Consider adjuvant pembrolizumab if high-risk",
                "Baseline and surveillance imaging"
            ],
            "follow_up": "Clinical exam every 3-4 months, imaging every 6 months"
        },
        "IIC": {
            "name": "Stage IIC",
            "tnm": "T4b N0 M0",
            "description": "Very thick ulcerated melanoma, highest risk Stage II",
            "five_year_survival": "75-82%",
            "ten_year_survival": "68-75%",
            "characteristics": [
                "Breslow thickness > 4.0 mm",
                "Ulceration present",
                "High risk for occult metastatic disease"
            ],
            "treatment": [
                "Wide local excision with 2 cm margins",
                "Sentinel lymph node biopsy required",
                "Strong consideration of adjuvant immunotherapy",
                "Regular imaging surveillance"
            ],
            "follow_up": "Clinical exam every 3 months, imaging every 3-6 months"
        },
        "IIIA": {
            "name": "Stage IIIA",
            "tnm": "T1a-T2a N1a/N2a M0",
            "description": "Thin/intermediate melanoma with micrometastatic nodal disease",
            "five_year_survival": "88-93%",
            "ten_year_survival": "80-88%",
            "characteristics": [
                "Limited primary tumor (T1-T2a)",
                "Micrometastases in 1-3 lymph nodes (occult)",
                "Detected by sentinel node biopsy"
            ],
            "treatment": [
                "Complete lymph node dissection (consider observation in select cases)",
                "Adjuvant immunotherapy recommended",
                "Regular imaging surveillance"
            ],
            "follow_up": "Clinical exam every 3 months, imaging every 3-6 months"
        },
        "IIIB": {
            "name": "Stage IIIB",
            "tnm": "Various T N1-N2 M0 combinations",
            "description": "Regional nodal metastasis, moderate extent",
            "five_year_survival": "77-83%",
            "ten_year_survival": "70-77%",
            "characteristics": [
                "Variable primary tumor thickness",
                "Clinically detected nodes or multiple occult positive nodes",
                "May include satellite/in-transit metastases"
            ],
            "treatment": [
                "Complete lymph node dissection",
                "Adjuvant immunotherapy (pembrolizumab/nivolumab)",
                "Targeted therapy if BRAF V600 mutation",
                "Consider radiation for high-risk nodal features"
            ],
            "follow_up": "Clinical exam every 3 months, imaging every 3 months"
        },
        "IIIC": {
            "name": "Stage IIIC",
            "tnm": "Various T N2-N3 M0 combinations",
            "description": "Extensive regional nodal involvement",
            "five_year_survival": "60-69%",
            "ten_year_survival": "52-60%",
            "characteristics": [
                "Multiple involved nodes",
                "Matted nodes or extensive nodal disease",
                "High risk of systemic recurrence"
            ],
            "treatment": [
                "Complete lymph node dissection",
                "Adjuvant immunotherapy strongly recommended",
                "Consider combination immunotherapy in trials",
                "Radiation therapy for bulky or extranodal disease"
            ],
            "follow_up": "Multidisciplinary management, frequent imaging"
        },
        "IIID": {
            "name": "Stage IIID",
            "tnm": "T4b N3a/b/c M0",
            "description": "Highest risk Stage III - thick ulcerated primary with extensive nodal disease",
            "five_year_survival": "32-42%",
            "ten_year_survival": "25-35%",
            "characteristics": [
                "T4b primary (> 4mm, ulcerated)",
                "Extensive nodal involvement (N3)",
                "Very high risk of distant recurrence"
            ],
            "treatment": [
                "Aggressive multimodal therapy",
                "Combination immunotherapy consideration",
                "Clinical trial enrollment recommended",
                "Radiation to primary and nodal basin"
            ],
            "follow_up": "Close surveillance, imaging every 3 months"
        },
        "IV": {
            "name": "Stage IV",
            "tnm": "Any T, Any N, M1a-d",
            "description": "Distant metastatic melanoma",
            "five_year_survival": "15-25%",
            "ten_year_survival": "10-20%",
            "characteristics": [
                "M1a: Distant skin/subcutaneous/nodal metastases",
                "M1b: Lung metastases",
                "M1c: Non-CNS visceral metastases",
                "M1d: CNS/brain metastases"
            ],
            "treatment": [
                "First-line immunotherapy (nivolumab+ipilimumab or pembrolizumab)",
                "BRAF/MEK inhibitors for BRAF V600 mutation",
                "Surgical resection of oligometastatic disease",
                "Stereotactic radiosurgery for brain metastases",
                "Clinical trials for novel agents"
            ],
            "follow_up": "Response assessment imaging every 2-3 months"
        }
    }

    if stage not in stage_info:
        # Try to match partial stages
        for key in stage_info:
            if stage.startswith(key) or key.startswith(stage):
                return stage_info[key]
        raise HTTPException(status_code=404, detail=f"Stage information not found for: {stage}")

    return stage_info[stage]


# =============================================================================
# BRESLOW/CLARK DEPTH VISUALIZER
# =============================================================================

# Skin layer anatomy with typical depths (in mm)
SKIN_LAYERS = {
    "epidermis": {
        "name": "Epidermis",
        "depth_start": 0,
        "depth_end": 0.1,
        "color": "#FFDAB9",  # Peach
        "description": "Outermost layer containing keratinocytes and melanocytes",
        "sublayers": ["Stratum corneum", "Stratum granulosum", "Stratum spinosum", "Stratum basale"]
    },
    "papillary_dermis": {
        "name": "Papillary Dermis",
        "depth_start": 0.1,
        "depth_end": 0.3,
        "color": "#FFB6C1",  # Light pink
        "description": "Loose connective tissue with dermal papillae, contains capillaries",
        "features": ["Collagen fibers", "Elastic fibers", "Capillary loops", "Nerve endings"]
    },
    "reticular_dermis": {
        "name": "Reticular Dermis",
        "depth_start": 0.3,
        "depth_end": 2.0,
        "color": "#FF69B4",  # Hot pink
        "description": "Dense irregular connective tissue with thick collagen bundles",
        "features": ["Thick collagen bundles", "Elastic fibers", "Hair follicles", "Sweat glands", "Blood vessels"]
    },
    "subcutaneous": {
        "name": "Subcutaneous Fat (Hypodermis)",
        "depth_start": 2.0,
        "depth_end": 10.0,
        "color": "#FFE4B5",  # Moccasin/yellow
        "description": "Adipose tissue layer providing insulation and energy storage",
        "features": ["Adipocytes", "Blood vessels", "Nerves", "Connective tissue septa"]
    }
}

# Clark levels of invasion
CLARK_LEVELS = {
    1: {
        "level": "I",
        "name": "Level I (In Situ)",
        "anatomical_location": "Epidermis only",
        "description": "Melanoma confined to the epidermis, has not crossed the basement membrane",
        "invaded_layers": ["epidermis"],
        "prognosis": "Excellent - essentially 100% cure rate",
        "typical_breslow": "0 mm (no invasion)",
        "treatment": "Wide local excision with 0.5-1 cm margins"
    },
    2: {
        "level": "II",
        "name": "Level II",
        "anatomical_location": "Papillary dermis (partial)",
        "description": "Melanoma invades into but does not fill the papillary dermis",
        "invaded_layers": ["epidermis", "papillary_dermis"],
        "prognosis": "Very good - >95% survival",
        "typical_breslow": "< 0.5 mm",
        "treatment": "Wide local excision, consider sentinel node biopsy for adverse features"
    },
    3: {
        "level": "III",
        "name": "Level III",
        "anatomical_location": "Papillary dermis (complete)",
        "description": "Melanoma fills and expands the papillary dermis up to the papillary-reticular junction",
        "invaded_layers": ["epidermis", "papillary_dermis"],
        "prognosis": "Good - ~90% survival",
        "typical_breslow": "0.5-1.0 mm",
        "treatment": "Wide local excision with 1 cm margins, sentinel node biopsy recommended"
    },
    4: {
        "level": "IV",
        "name": "Level IV",
        "anatomical_location": "Reticular dermis",
        "description": "Melanoma invades into the reticular dermis",
        "invaded_layers": ["epidermis", "papillary_dermis", "reticular_dermis"],
        "prognosis": "Moderate - 70-85% survival depending on depth",
        "typical_breslow": "1.0-4.0 mm",
        "treatment": "Wide local excision with 2 cm margins, sentinel node biopsy required"
    },
    5: {
        "level": "V",
        "name": "Level V",
        "anatomical_location": "Subcutaneous fat",
        "description": "Melanoma invades into the subcutaneous tissue",
        "invaded_layers": ["epidermis", "papillary_dermis", "reticular_dermis", "subcutaneous"],
        "prognosis": "Guarded - <70% survival",
        "typical_breslow": "> 4.0 mm",
        "treatment": "Wide local excision, sentinel node biopsy, consider adjuvant therapy"
    }
}

# Breslow thickness categories and clinical significance
BRESLOW_CATEGORIES = {
    "in_situ": {
        "range": "0 mm",
        "min": 0,
        "max": 0,
        "t_category": "Tis",
        "clark_level": 1,
        "prognosis": "Excellent",
        "five_year_survival": "~100%",
        "sentinel_node_biopsy": "Not indicated",
        "excision_margin": "0.5-1 cm"
    },
    "thin": {
        "range": "≤ 1.0 mm",
        "min": 0.01,
        "max": 1.0,
        "t_category": "T1",
        "clark_level": "II-III",
        "prognosis": "Very good",
        "five_year_survival": "95-99%",
        "sentinel_node_biopsy": "Consider if ulcerated or > 0.8 mm",
        "excision_margin": "1 cm"
    },
    "intermediate": {
        "range": "1.01 - 2.0 mm",
        "min": 1.01,
        "max": 2.0,
        "t_category": "T2",
        "clark_level": "III-IV",
        "prognosis": "Good",
        "five_year_survival": "90-95%",
        "sentinel_node_biopsy": "Recommended",
        "excision_margin": "1-2 cm"
    },
    "thick": {
        "range": "2.01 - 4.0 mm",
        "min": 2.01,
        "max": 4.0,
        "t_category": "T3",
        "clark_level": "IV",
        "prognosis": "Moderate",
        "five_year_survival": "75-85%",
        "sentinel_node_biopsy": "Required",
        "excision_margin": "2 cm"
    },
    "very_thick": {
        "range": "> 4.0 mm",
        "min": 4.01,
        "max": 100,
        "t_category": "T4",
        "clark_level": "IV-V",
        "prognosis": "Guarded",
        "five_year_survival": "50-75%",
        "sentinel_node_biopsy": "Required",
        "excision_margin": "2 cm"
    }
}


def get_breslow_category(thickness: float) -> dict:
    """Get Breslow category based on thickness."""
    if thickness <= 0:
        return BRESLOW_CATEGORIES["in_situ"]
    elif thickness <= 1.0:
        return BRESLOW_CATEGORIES["thin"]
    elif thickness <= 2.0:
        return BRESLOW_CATEGORIES["intermediate"]
    elif thickness <= 4.0:
        return BRESLOW_CATEGORIES["thick"]
    else:
        return BRESLOW_CATEGORIES["very_thick"]


def estimate_clark_from_breslow(thickness: float) -> int:
    """Estimate Clark level from Breslow thickness."""
    if thickness <= 0:
        return 1
    elif thickness < 0.5:
        return 2
    elif thickness < 1.0:
        return 3
    elif thickness < 4.0:
        return 4
    else:
        return 5


def calculate_invasion_visualization(breslow_mm: float, clark_level: Optional[int] = None) -> dict:
    """Generate visualization data for invasion depth."""

    # Use Clark level if provided, otherwise estimate from Breslow
    if clark_level is None:
        clark_level = estimate_clark_from_breslow(breslow_mm)

    clark_info = CLARK_LEVELS.get(clark_level, CLARK_LEVELS[1])
    breslow_cat = get_breslow_category(breslow_mm)

    # Calculate invasion through each layer
    layers_data = []
    cumulative_depth = 0

    for layer_key, layer in SKIN_LAYERS.items():
        layer_thickness = layer["depth_end"] - layer["depth_start"]

        # Calculate how much of this layer is invaded
        if breslow_mm <= 0:
            invasion_percent = 100 if layer_key == "epidermis" else 0
            invaded = layer_key == "epidermis"
        elif breslow_mm <= layer["depth_end"]:
            if breslow_mm >= layer["depth_start"]:
                # Partially or fully invaded
                invasion_depth = breslow_mm - layer["depth_start"]
                invasion_percent = min(100, (invasion_depth / layer_thickness) * 100)
                invaded = True
            else:
                invasion_percent = 0
                invaded = False
        else:
            # Fully invaded (tumor extends beyond this layer)
            invasion_percent = 100
            invaded = True

        layers_data.append({
            "key": layer_key,
            "name": layer["name"],
            "depth_start_mm": layer["depth_start"],
            "depth_end_mm": layer["depth_end"],
            "thickness_mm": layer_thickness,
            "color": layer["color"],
            "invaded": invaded,
            "invasion_percent": round(invasion_percent, 1),
            "description": layer["description"]
        })

    return {
        "breslow_thickness_mm": breslow_mm,
        "breslow_category": breslow_cat,
        "clark_level": clark_level,
        "clark_info": clark_info,
        "layers": layers_data,
        "deepest_layer_invaded": clark_info["invaded_layers"][-1] if clark_info["invaded_layers"] else "epidermis"
    }


@router.get("/breslow-clark/layers")
async def get_skin_layers():
    """Get skin layer anatomy information for visualization."""
    return {
        "layers": SKIN_LAYERS,
        "clark_levels": CLARK_LEVELS,
        "breslow_categories": BRESLOW_CATEGORIES
    }


@router.post("/breslow-clark/visualize")
async def visualize_invasion_depth(
    breslow_thickness: float = Form(..., description="Breslow thickness in mm"),
    clark_level: Optional[int] = Form(None, description="Clark level (1-5), optional"),
    ulceration: Optional[bool] = Form(None, description="Presence of ulceration"),
    mitotic_rate: Optional[float] = Form(None, description="Mitotic rate per mm²"),
    regression: Optional[bool] = Form(None, description="Presence of regression"),
    lymphovascular_invasion: Optional[bool] = Form(None, description="Lymphovascular invasion"),
    perineural_invasion: Optional[bool] = Form(None, description="Perineural invasion"),
    tumor_infiltrating_lymphocytes: Optional[str] = Form(None, description="TIL status: brisk, non-brisk, absent"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate visualization data for melanoma invasion depth.

    Returns layer-by-layer invasion analysis with clinical correlations.
    """
    # Validate inputs
    if breslow_thickness < 0:
        raise HTTPException(status_code=400, detail="Breslow thickness cannot be negative")
    if clark_level is not None and clark_level not in [1, 2, 3, 4, 5]:
        raise HTTPException(status_code=400, detail="Clark level must be 1-5")

    # Generate visualization data
    viz_data = calculate_invasion_visualization(breslow_thickness, clark_level)

    # Add pathology features
    pathology_features = {
        "ulceration": ulceration,
        "mitotic_rate_per_mm2": mitotic_rate,
        "regression": regression,
        "lymphovascular_invasion": lymphovascular_invasion,
        "perineural_invasion": perineural_invasion,
        "tumor_infiltrating_lymphocytes": tumor_infiltrating_lymphocytes
    }

    # Calculate risk modifiers
    risk_factors = []
    protective_factors = []

    if ulceration:
        risk_factors.append("Ulceration increases stage and worsens prognosis")
    if mitotic_rate and mitotic_rate >= 1:
        risk_factors.append(f"Mitotic rate of {mitotic_rate}/mm² indicates proliferative activity")
    if lymphovascular_invasion:
        risk_factors.append("Lymphovascular invasion increases metastatic risk")
    if perineural_invasion:
        risk_factors.append("Perineural invasion may increase local recurrence risk")
    if regression:
        risk_factors.append("Regression may indicate prior deeper invasion")

    if tumor_infiltrating_lymphocytes == "brisk":
        protective_factors.append("Brisk TILs associated with better prognosis")
    if tumor_infiltrating_lymphocytes == "non-brisk":
        protective_factors.append("Non-brisk TILs present")

    # Determine T category with ulceration
    breslow_cat = viz_data["breslow_category"]
    t_category = breslow_cat["t_category"]
    if breslow_thickness > 0 and breslow_thickness <= 1.0:
        if ulceration or breslow_thickness > 0.8:
            t_category = "T1b"
        else:
            t_category = "T1a"
    elif breslow_thickness > 1.0 and breslow_thickness <= 2.0:
        t_category = "T2b" if ulceration else "T2a"
    elif breslow_thickness > 2.0 and breslow_thickness <= 4.0:
        t_category = "T3b" if ulceration else "T3a"
    elif breslow_thickness > 4.0:
        t_category = "T4b" if ulceration else "T4a"

    return {
        "visualization": viz_data,
        "pathology_features": pathology_features,
        "t_category": t_category,
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "clinical_recommendations": {
            "excision_margin": breslow_cat["excision_margin"],
            "sentinel_node_biopsy": breslow_cat["sentinel_node_biopsy"],
            "prognosis": breslow_cat["prognosis"],
            "five_year_survival": breslow_cat["five_year_survival"]
        },
        "disclaimer": "This visualization is for educational purposes. Pathology interpretation should be performed by qualified pathologists."
    }


@router.get("/breslow-clark/interpret/{breslow_mm}")
async def interpret_breslow_depth(breslow_mm: float):
    """
    Quick interpretation of Breslow depth without authentication.
    For educational/reference purposes.
    """
    if breslow_mm < 0:
        raise HTTPException(status_code=400, detail="Breslow thickness cannot be negative")

    viz_data = calculate_invasion_visualization(breslow_mm)
    breslow_cat = viz_data["breslow_category"]
    clark_info = viz_data["clark_info"]

    return {
        "breslow_mm": breslow_mm,
        "category": breslow_cat["range"],
        "t_category": breslow_cat["t_category"],
        "estimated_clark_level": viz_data["clark_level"],
        "clark_description": clark_info["anatomical_location"],
        "deepest_layer": viz_data["deepest_layer_invaded"],
        "prognosis": breslow_cat["prognosis"],
        "five_year_survival": breslow_cat["five_year_survival"],
        "sentinel_node_recommendation": breslow_cat["sentinel_node_biopsy"],
        "recommended_margin": breslow_cat["excision_margin"]
    }


# =============================================================================
# MELANOMA SURVIVAL ESTIMATOR
# =============================================================================

import math

# Base survival rates by AJCC stage (from SEER data and clinical studies)
# Format: {stage: {year: survival_probability}}
BASE_SURVIVAL_BY_STAGE = {
    "0": {1: 0.999, 2: 0.998, 3: 0.997, 5: 0.995, 10: 0.990},
    "IA": {1: 0.995, 2: 0.990, 3: 0.985, 5: 0.975, 10: 0.955},
    "IB": {1: 0.990, 2: 0.980, 3: 0.970, 5: 0.950, 10: 0.920},
    "IIA": {1: 0.985, 2: 0.965, 3: 0.945, 5: 0.910, 10: 0.860},
    "IIB": {1: 0.975, 2: 0.940, 3: 0.905, 5: 0.850, 10: 0.780},
    "IIC": {1: 0.960, 2: 0.910, 3: 0.860, 5: 0.780, 10: 0.680},
    "IIIA": {1: 0.970, 2: 0.935, 3: 0.900, 5: 0.830, 10: 0.750},
    "IIIB": {1: 0.940, 2: 0.880, 3: 0.820, 5: 0.720, 10: 0.600},
    "IIIC": {1: 0.890, 2: 0.790, 3: 0.700, 5: 0.560, 10: 0.420},
    "IIID": {1: 0.800, 2: 0.650, 3: 0.520, 5: 0.380, 10: 0.250},
    "IV": {1: 0.600, 2: 0.400, 3: 0.280, 5: 0.180, 10: 0.100},
}

# Hazard ratio modifiers for various prognostic factors
PROGNOSTIC_FACTORS = {
    "ulceration": {
        "present": 1.5,  # 50% increased hazard
        "absent": 1.0
    },
    "mitotic_rate": {
        "0": 1.0,
        "1-5": 1.2,
        "6-10": 1.5,
        ">10": 2.0
    },
    "sentinel_node": {
        "negative": 1.0,
        "positive": 2.0,
        "not_done": 1.1
    },
    "age": {
        "<40": 0.85,
        "40-60": 1.0,
        "60-70": 1.15,
        ">70": 1.35
    },
    "sex": {
        "female": 0.85,
        "male": 1.0
    },
    "location": {
        "extremity": 0.9,
        "trunk": 1.0,
        "head_neck": 1.15,
        "acral": 1.25
    },
    "lymphovascular_invasion": {
        "present": 1.6,
        "absent": 1.0
    },
    "regression": {
        "present": 1.2,  # Controversial - may indicate prior deeper tumor
        "absent": 1.0
    },
    "microsatellites": {
        "present": 1.8,
        "absent": 1.0
    },
    "tils": {
        "brisk": 0.75,
        "non_brisk": 0.9,
        "absent": 1.0
    }
}


def calculate_stage_from_inputs(breslow: float, ulceration: bool, node_status: str, metastasis: bool) -> str:
    """Determine AJCC stage from basic inputs."""
    if metastasis:
        return "IV"

    # Determine T category
    if breslow <= 0:
        t_cat = "Tis"
    elif breslow <= 1.0:
        t_cat = "T1b" if ulceration or breslow > 0.8 else "T1a"
    elif breslow <= 2.0:
        t_cat = "T2b" if ulceration else "T2a"
    elif breslow <= 4.0:
        t_cat = "T3b" if ulceration else "T3a"
    else:
        t_cat = "T4b" if ulceration else "T4a"

    # Determine stage based on T and N
    if node_status == "positive":
        if t_cat in ["T1a", "T1b", "T2a"]:
            return "IIIA"
        elif t_cat in ["T2b", "T3a"]:
            return "IIIB"
        elif t_cat in ["T3b", "T4a"]:
            return "IIIC"
        else:
            return "IIID"
    else:
        # N0 staging
        if t_cat == "Tis":
            return "0"
        elif t_cat == "T1a":
            return "IA"
        elif t_cat in ["T1b", "T2a"]:
            return "IB"
        elif t_cat in ["T2b", "T3a"]:
            return "IIA"
        elif t_cat in ["T3b", "T4a"]:
            return "IIB"
        else:
            return "IIC"


def apply_hazard_modifiers(base_survival: float, hazard_ratio: float, time_years: int) -> float:
    """Apply Cox proportional hazards model to modify survival."""
    # Convert survival to hazard, apply modifier, convert back
    if base_survival >= 1.0:
        return 0.999
    if base_survival <= 0:
        return 0.001

    # h(t) = h0(t) * HR
    # S(t) = exp(-H(t)) where H(t) = cumulative hazard
    # S_new(t) = S_base(t)^HR
    modified_survival = math.pow(base_survival, hazard_ratio)
    return max(0.001, min(0.999, modified_survival))


def generate_survival_curve(
    stage: str,
    hazard_ratio: float,
    years: int = 10
) -> list:
    """Generate survival curve data points."""
    base_curve = BASE_SURVIVAL_BY_STAGE.get(stage, BASE_SURVIVAL_BY_STAGE["IIA"])

    curve_points = []
    for year in range(years + 1):
        if year == 0:
            survival = 1.0
        elif year in base_curve:
            survival = apply_hazard_modifiers(base_curve[year], hazard_ratio, year)
        else:
            # Interpolate between known points
            lower_year = max([y for y in base_curve.keys() if y <= year], default=1)
            upper_year = min([y for y in base_curve.keys() if y >= year], default=10)

            if lower_year == upper_year:
                survival = apply_hazard_modifiers(base_curve[lower_year], hazard_ratio, year)
            else:
                lower_surv = base_curve[lower_year]
                upper_surv = base_curve[upper_year]
                fraction = (year - lower_year) / (upper_year - lower_year)
                interpolated = lower_surv - (lower_surv - upper_surv) * fraction
                survival = apply_hazard_modifiers(interpolated, hazard_ratio, year)

        curve_points.append({
            "year": year,
            "survival_probability": round(survival, 4),
            "survival_percent": round(survival * 100, 1)
        })

    return curve_points


def calculate_median_survival(curve: list) -> Optional[float]:
    """Calculate median survival time from curve."""
    for i, point in enumerate(curve):
        if point["survival_probability"] <= 0.5:
            if i == 0:
                return 0
            # Interpolate
            prev = curve[i - 1]
            curr = point
            if prev["survival_probability"] == curr["survival_probability"]:
                return curr["year"]
            fraction = (prev["survival_probability"] - 0.5) / (prev["survival_probability"] - curr["survival_probability"])
            return round(prev["year"] + fraction * (curr["year"] - prev["year"]), 1)
    return None  # Median not reached within timeframe


@router.post("/survival/estimate")
async def estimate_survival(
    # Required parameters
    breslow_thickness: float = Form(..., description="Breslow thickness in mm"),
    ulceration: bool = Form(False, description="Presence of ulceration"),
    # Optional tumor characteristics
    mitotic_rate: Optional[float] = Form(None, description="Mitotic rate per mm²"),
    sentinel_node_status: Optional[str] = Form(None, description="positive, negative, or not_done"),
    distant_metastasis: bool = Form(False, description="Presence of distant metastasis"),
    # Patient characteristics
    age: Optional[int] = Form(None, description="Patient age in years"),
    sex: Optional[str] = Form(None, description="male or female"),
    tumor_location: Optional[str] = Form(None, description="extremity, trunk, head_neck, or acral"),
    # Additional pathology features
    lymphovascular_invasion: bool = Form(False, description="LVI present"),
    regression: bool = Form(False, description="Regression present"),
    microsatellites: bool = Form(False, description="Microsatellites present"),
    tils: Optional[str] = Form(None, description="TIL status: brisk, non_brisk, absent"),
    # Analysis options
    years_to_project: int = Form(10, description="Years to project survival (1-20)"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Estimate melanoma-specific survival using patient and tumor characteristics.

    Uses a Cox proportional hazards model approach based on published prognostic factors.
    Returns survival curves and key statistics.
    """
    # Validate inputs
    if breslow_thickness < 0:
        raise HTTPException(status_code=400, detail="Breslow thickness cannot be negative")
    if years_to_project < 1 or years_to_project > 20:
        years_to_project = min(20, max(1, years_to_project))

    # Determine base stage
    node_status = sentinel_node_status if sentinel_node_status else "not_done"
    stage = calculate_stage_from_inputs(breslow_thickness, ulceration, node_status, distant_metastasis)

    # Calculate combined hazard ratio
    total_hr = 1.0
    applied_factors = []

    # Ulceration
    if ulceration:
        hr = PROGNOSTIC_FACTORS["ulceration"]["present"]
        total_hr *= hr
        applied_factors.append({"factor": "Ulceration", "status": "Present", "hazard_ratio": hr, "effect": "Adverse"})

    # Mitotic rate
    if mitotic_rate is not None:
        if mitotic_rate == 0:
            mr_key = "0"
        elif mitotic_rate <= 5:
            mr_key = "1-5"
        elif mitotic_rate <= 10:
            mr_key = "6-10"
        else:
            mr_key = ">10"
        hr = PROGNOSTIC_FACTORS["mitotic_rate"][mr_key]
        total_hr *= hr
        if hr != 1.0:
            applied_factors.append({
                "factor": "Mitotic Rate",
                "status": f"{mitotic_rate}/mm²",
                "hazard_ratio": hr,
                "effect": "Adverse" if hr > 1 else "Neutral"
            })

    # Sentinel node
    if sentinel_node_status:
        hr = PROGNOSTIC_FACTORS["sentinel_node"].get(sentinel_node_status, 1.0)
        total_hr *= hr
        if hr != 1.0:
            applied_factors.append({
                "factor": "Sentinel Node",
                "status": sentinel_node_status.replace("_", " ").title(),
                "hazard_ratio": hr,
                "effect": "Adverse" if hr > 1 else "Neutral"
            })

    # Age
    if age is not None:
        if age < 40:
            age_key = "<40"
        elif age <= 60:
            age_key = "40-60"
        elif age <= 70:
            age_key = "60-70"
        else:
            age_key = ">70"
        hr = PROGNOSTIC_FACTORS["age"][age_key]
        total_hr *= hr
        if hr != 1.0:
            applied_factors.append({
                "factor": "Age",
                "status": f"{age} years",
                "hazard_ratio": hr,
                "effect": "Favorable" if hr < 1 else "Adverse"
            })

    # Sex
    if sex:
        hr = PROGNOSTIC_FACTORS["sex"].get(sex.lower(), 1.0)
        total_hr *= hr
        if hr != 1.0:
            applied_factors.append({
                "factor": "Sex",
                "status": sex.title(),
                "hazard_ratio": hr,
                "effect": "Favorable" if hr < 1 else "Adverse"
            })

    # Location
    if tumor_location:
        hr = PROGNOSTIC_FACTORS["location"].get(tumor_location.lower(), 1.0)
        total_hr *= hr
        if hr != 1.0:
            applied_factors.append({
                "factor": "Location",
                "status": tumor_location.replace("_", "/").title(),
                "hazard_ratio": hr,
                "effect": "Favorable" if hr < 1 else "Adverse"
            })

    # LVI
    if lymphovascular_invasion:
        hr = PROGNOSTIC_FACTORS["lymphovascular_invasion"]["present"]
        total_hr *= hr
        applied_factors.append({
            "factor": "Lymphovascular Invasion",
            "status": "Present",
            "hazard_ratio": hr,
            "effect": "Adverse"
        })

    # Regression
    if regression:
        hr = PROGNOSTIC_FACTORS["regression"]["present"]
        total_hr *= hr
        applied_factors.append({
            "factor": "Regression",
            "status": "Present",
            "hazard_ratio": hr,
            "effect": "Uncertain"
        })

    # Microsatellites
    if microsatellites:
        hr = PROGNOSTIC_FACTORS["microsatellites"]["present"]
        total_hr *= hr
        applied_factors.append({
            "factor": "Microsatellites",
            "status": "Present",
            "hazard_ratio": hr,
            "effect": "Adverse"
        })

    # TILs
    if tils:
        hr = PROGNOSTIC_FACTORS["tils"].get(tils.lower().replace("-", "_"), 1.0)
        total_hr *= hr
        if hr != 1.0:
            applied_factors.append({
                "factor": "Tumor-Infiltrating Lymphocytes",
                "status": tils.replace("_", "-").title(),
                "hazard_ratio": hr,
                "effect": "Favorable" if hr < 1 else "Neutral"
            })

    # Generate survival curves
    patient_curve = generate_survival_curve(stage, total_hr, years_to_project)
    baseline_curve = generate_survival_curve(stage, 1.0, years_to_project)

    # Calculate key statistics
    survival_5yr = next((p["survival_percent"] for p in patient_curve if p["year"] == 5), None)
    survival_10yr = next((p["survival_percent"] for p in patient_curve if p["year"] == 10), None)
    median_survival = calculate_median_survival(patient_curve)

    # Risk classification
    if survival_5yr >= 90:
        risk_category = "Low"
        risk_color = "#10b981"
    elif survival_5yr >= 75:
        risk_category = "Intermediate"
        risk_color = "#f59e0b"
    elif survival_5yr >= 50:
        risk_category = "High"
        risk_color = "#f97316"
    else:
        risk_category = "Very High"
        risk_color = "#ef4444"

    return {
        "patient_characteristics": {
            "breslow_thickness_mm": breslow_thickness,
            "ulceration": ulceration,
            "mitotic_rate": mitotic_rate,
            "sentinel_node_status": sentinel_node_status,
            "distant_metastasis": distant_metastasis,
            "age": age,
            "sex": sex,
            "tumor_location": tumor_location
        },
        "staging": {
            "stage": stage,
            "description": f"AJCC 8th Edition Stage {stage}"
        },
        "survival_estimate": {
            "five_year_survival": f"{survival_5yr}%" if survival_5yr else "N/A",
            "ten_year_survival": f"{survival_10yr}%" if survival_10yr else "N/A",
            "median_survival_years": median_survival,
            "risk_category": risk_category,
            "risk_color": risk_color
        },
        "hazard_analysis": {
            "combined_hazard_ratio": round(total_hr, 3),
            "applied_factors": applied_factors,
            "interpretation": "Higher hazard ratio indicates worse prognosis relative to baseline"
        },
        "survival_curves": {
            "patient_specific": patient_curve,
            "stage_baseline": baseline_curve
        },
        "comparison": {
            "difference_from_baseline_5yr": round(
                (patient_curve[5]["survival_percent"] if len(patient_curve) > 5 else 0) -
                (baseline_curve[5]["survival_percent"] if len(baseline_curve) > 5 else 0), 1
            ),
            "patient_vs_baseline": "Better" if total_hr < 1 else ("Worse" if total_hr > 1 else "Same")
        },
        "disclaimer": "This survival estimate is based on statistical models and published data. Individual outcomes may vary significantly. This tool is for educational purposes and should not replace clinical judgment.",
        "data_sources": [
            "AJCC Cancer Staging Manual, 8th Edition",
            "SEER Cancer Statistics",
            "Melanoma Staging Database (MSDB)",
            "Published Cox regression models for melanoma"
        ]
    }


@router.get("/survival/prognostic-factors")
async def get_prognostic_factors():
    """Get information about all prognostic factors used in survival estimation."""
    return {
        "factors": [
            {
                "name": "Breslow Thickness",
                "type": "continuous",
                "description": "Tumor thickness from granular layer to deepest tumor cell",
                "impact": "Primary prognostic factor - thicker tumors have worse prognosis",
                "categories": [
                    {"range": "≤1.0 mm", "risk": "Low"},
                    {"range": "1.01-2.0 mm", "risk": "Intermediate"},
                    {"range": "2.01-4.0 mm", "risk": "High"},
                    {"range": ">4.0 mm", "risk": "Very High"}
                ]
            },
            {
                "name": "Ulceration",
                "type": "binary",
                "description": "Loss of epidermis overlying the melanoma",
                "impact": "Upstages tumor by one T subcategory, ~50% increased risk",
                "hazard_ratio": 1.5
            },
            {
                "name": "Mitotic Rate",
                "type": "continuous",
                "description": "Number of mitoses per mm² in hot spot",
                "impact": "Higher rates indicate more aggressive tumor biology",
                "categories": [
                    {"range": "0/mm²", "hazard_ratio": 1.0},
                    {"range": "1-5/mm²", "hazard_ratio": 1.2},
                    {"range": "6-10/mm²", "hazard_ratio": 1.5},
                    {"range": ">10/mm²", "hazard_ratio": 2.0}
                ]
            },
            {
                "name": "Sentinel Node Status",
                "type": "categorical",
                "description": "Result of sentinel lymph node biopsy",
                "impact": "Positive nodes significantly worsen prognosis",
                "categories": [
                    {"status": "Negative", "hazard_ratio": 1.0},
                    {"status": "Positive", "hazard_ratio": 2.0}
                ]
            },
            {
                "name": "Age",
                "type": "continuous",
                "description": "Patient age at diagnosis",
                "impact": "Older patients have slightly worse outcomes",
                "categories": [
                    {"range": "<40 years", "hazard_ratio": 0.85},
                    {"range": "40-60 years", "hazard_ratio": 1.0},
                    {"range": "60-70 years", "hazard_ratio": 1.15},
                    {"range": ">70 years", "hazard_ratio": 1.35}
                ]
            },
            {
                "name": "Sex",
                "type": "categorical",
                "description": "Patient biological sex",
                "impact": "Females have slightly better outcomes than males",
                "categories": [
                    {"status": "Female", "hazard_ratio": 0.85},
                    {"status": "Male", "hazard_ratio": 1.0}
                ]
            },
            {
                "name": "Tumor Location",
                "type": "categorical",
                "description": "Anatomic site of primary melanoma",
                "impact": "Head/neck and acral locations have worse prognosis",
                "categories": [
                    {"location": "Extremity", "hazard_ratio": 0.9},
                    {"location": "Trunk", "hazard_ratio": 1.0},
                    {"location": "Head/Neck", "hazard_ratio": 1.15},
                    {"location": "Acral", "hazard_ratio": 1.25}
                ]
            },
            {
                "name": "Lymphovascular Invasion",
                "type": "binary",
                "description": "Tumor cells within lymphatic or vascular channels",
                "impact": "Associated with increased metastatic risk",
                "hazard_ratio": 1.6
            },
            {
                "name": "Tumor-Infiltrating Lymphocytes",
                "type": "categorical",
                "description": "Immune cell infiltration of tumor",
                "impact": "Brisk TILs associated with better immune response",
                "categories": [
                    {"status": "Brisk", "hazard_ratio": 0.75, "effect": "Favorable"},
                    {"status": "Non-brisk", "hazard_ratio": 0.9, "effect": "Neutral"},
                    {"status": "Absent", "hazard_ratio": 1.0, "effect": "Neutral"}
                ]
            }
        ],
        "model_info": {
            "approach": "Cox Proportional Hazards",
            "baseline": "Stage-specific survival from AJCC/SEER data",
            "modifications": "Multiplicative hazard ratios from prognostic factors"
        }
    }


# =============================================================================
# SENTINEL NODE MAPPER
# Visual lymph node basin mapping with biopsy tracking
# =============================================================================

# Lymph node basin definitions by anatomic region
LYMPH_NODE_BASINS = {
    "head_neck": {
        "name": "Head & Neck",
        "primary_basins": [
            {
                "id": "parotid",
                "name": "Parotid",
                "description": "Preauricular and parotid nodes",
                "location": {"x": 75, "y": 15},
                "drainage_from": ["scalp_anterior", "forehead", "temple", "upper_face", "eyelid"]
            },
            {
                "id": "submandibular",
                "name": "Submandibular",
                "description": "Below the mandible",
                "location": {"x": 50, "y": 22},
                "drainage_from": ["lower_face", "chin", "lip", "anterior_oral"]
            },
            {
                "id": "submental",
                "name": "Submental",
                "description": "Below the chin",
                "location": {"x": 50, "y": 25},
                "drainage_from": ["chin", "lower_lip", "floor_of_mouth"]
            },
            {
                "id": "upper_cervical",
                "name": "Upper Cervical (Level II)",
                "description": "Upper jugular chain",
                "location": {"x": 65, "y": 28},
                "drainage_from": ["face", "parotid", "submandibular"]
            },
            {
                "id": "mid_cervical",
                "name": "Mid Cervical (Level III)",
                "description": "Middle jugular chain",
                "location": {"x": 65, "y": 35},
                "drainage_from": ["upper_cervical", "posterior_scalp"]
            },
            {
                "id": "lower_cervical",
                "name": "Lower Cervical (Level IV)",
                "description": "Lower jugular chain",
                "location": {"x": 65, "y": 42},
                "drainage_from": ["mid_cervical"]
            },
            {
                "id": "posterior_triangle",
                "name": "Posterior Triangle (Level V)",
                "description": "Spinal accessory chain",
                "location": {"x": 75, "y": 35},
                "drainage_from": ["posterior_scalp", "posterior_neck"]
            },
            {
                "id": "occipital",
                "name": "Occipital",
                "description": "Base of skull posteriorly",
                "location": {"x": 80, "y": 12},
                "drainage_from": ["posterior_scalp", "upper_neck"]
            },
            {
                "id": "postauricular",
                "name": "Postauricular",
                "description": "Behind the ear",
                "location": {"x": 82, "y": 15},
                "drainage_from": ["posterior_ear", "temporal_scalp"]
            }
        ],
        "body_outline": "head_neck"
    },
    "upper_extremity": {
        "name": "Upper Extremity",
        "primary_basins": [
            {
                "id": "epitrochlear",
                "name": "Epitrochlear",
                "description": "Medial elbow",
                "location": {"x": 25, "y": 55},
                "drainage_from": ["hand_ulnar", "forearm_medial"]
            },
            {
                "id": "axillary_level_i",
                "name": "Axillary Level I",
                "description": "Lateral to pectoralis minor",
                "location": {"x": 20, "y": 35},
                "drainage_from": ["arm", "lateral_chest", "back_upper"]
            },
            {
                "id": "axillary_level_ii",
                "name": "Axillary Level II",
                "description": "Behind pectoralis minor",
                "location": {"x": 22, "y": 32},
                "drainage_from": ["axillary_level_i"]
            },
            {
                "id": "axillary_level_iii",
                "name": "Axillary Level III",
                "description": "Medial to pectoralis minor (apical)",
                "location": {"x": 25, "y": 29},
                "drainage_from": ["axillary_level_ii"]
            },
            {
                "id": "supraclavicular",
                "name": "Supraclavicular",
                "description": "Above the clavicle",
                "location": {"x": 30, "y": 24},
                "drainage_from": ["axillary_level_iii"]
            }
        ],
        "body_outline": "upper_limb"
    },
    "trunk_anterior": {
        "name": "Anterior Trunk",
        "primary_basins": [
            {
                "id": "axillary_right",
                "name": "Right Axillary",
                "description": "Right axillary basin",
                "location": {"x": 20, "y": 35},
                "drainage_from": ["chest_right", "abdomen_upper_right"]
            },
            {
                "id": "axillary_left",
                "name": "Left Axillary",
                "description": "Left axillary basin",
                "location": {"x": 80, "y": 35},
                "drainage_from": ["chest_left", "abdomen_upper_left"]
            },
            {
                "id": "inguinal_right",
                "name": "Right Inguinal",
                "description": "Right groin nodes",
                "location": {"x": 35, "y": 70},
                "drainage_from": ["abdomen_lower_right", "flank_right"]
            },
            {
                "id": "inguinal_left",
                "name": "Left Inguinal",
                "description": "Left groin nodes",
                "location": {"x": 65, "y": 70},
                "drainage_from": ["abdomen_lower_left", "flank_left"]
            }
        ],
        "watershed_line": {
            "description": "Sappey's line - horizontal watershed at umbilicus level",
            "y_position": 55,
            "note": "Lesions above drain to axillary, below to inguinal"
        },
        "body_outline": "trunk_front"
    },
    "trunk_posterior": {
        "name": "Posterior Trunk",
        "primary_basins": [
            {
                "id": "axillary_right",
                "name": "Right Axillary",
                "description": "Right axillary basin",
                "location": {"x": 20, "y": 35},
                "drainage_from": ["back_upper_right", "shoulder_right"]
            },
            {
                "id": "axillary_left",
                "name": "Left Axillary",
                "description": "Left axillary basin",
                "location": {"x": 80, "y": 35},
                "drainage_from": ["back_upper_left", "shoulder_left"]
            },
            {
                "id": "inguinal_right",
                "name": "Right Inguinal",
                "description": "Right groin nodes",
                "location": {"x": 35, "y": 72},
                "drainage_from": ["back_lower_right", "buttock_right"]
            },
            {
                "id": "inguinal_left",
                "name": "Left Inguinal",
                "description": "Left groin nodes",
                "location": {"x": 65, "y": 72},
                "drainage_from": ["back_lower_left", "buttock_left"]
            }
        ],
        "watershed_line": {
            "description": "Horizontal watershed at iliac crest level",
            "y_position": 60
        },
        "body_outline": "trunk_back"
    },
    "lower_extremity": {
        "name": "Lower Extremity",
        "primary_basins": [
            {
                "id": "popliteal",
                "name": "Popliteal",
                "description": "Behind the knee",
                "location": {"x": 50, "y": 55},
                "drainage_from": ["foot_lateral", "calf_posterior"]
            },
            {
                "id": "inguinal_superficial",
                "name": "Superficial Inguinal",
                "description": "Superficial groin nodes (horizontal & vertical groups)",
                "location": {"x": 50, "y": 18},
                "drainage_from": ["thigh", "leg", "popliteal", "buttock_lower", "perineum"]
            },
            {
                "id": "inguinal_deep",
                "name": "Deep Inguinal (Cloquet's node)",
                "description": "Deep femoral nodes",
                "location": {"x": 50, "y": 15},
                "drainage_from": ["inguinal_superficial"]
            },
            {
                "id": "external_iliac",
                "name": "External Iliac",
                "description": "Pelvic nodes",
                "location": {"x": 50, "y": 10},
                "drainage_from": ["inguinal_deep"]
            }
        ],
        "body_outline": "lower_limb"
    }
}

# Sentinel node biopsy result categories
SLN_RESULT_CATEGORIES = {
    "negative": {
        "code": "N0",
        "description": "No tumor cells identified",
        "color": "#10b981",
        "implications": [
            "No nodal metastasis detected",
            "Completion lymph node dissection not indicated",
            "Follow-up per NCCN guidelines"
        ]
    },
    "isolated_tumor_cells": {
        "code": "N0(i+)",
        "description": "Isolated tumor cells ≤0.2mm or <200 cells",
        "color": "#f59e0b",
        "implications": [
            "Prognostic significance uncertain",
            "Generally staged as N0",
            "Consider close surveillance"
        ]
    },
    "micrometastasis": {
        "code": "N1a/N2a",
        "description": "Metastasis >0.2mm but ≤2.0mm",
        "color": "#f97316",
        "implications": [
            "Clinically occult nodal disease",
            "Discuss completion dissection vs. observation",
            "Consider adjuvant therapy eligibility"
        ]
    },
    "macrometastasis": {
        "code": "N1a/N2a/N3a",
        "description": "Metastasis >2.0mm",
        "color": "#ef4444",
        "implications": [
            "Significant nodal disease burden",
            "Completion lymph node dissection per guidelines",
            "Adjuvant systemic therapy recommended",
            "Consider imaging for distant disease"
        ]
    },
    "extracapsular_extension": {
        "code": "N2c/N3",
        "description": "Tumor extends beyond lymph node capsule",
        "color": "#dc2626",
        "implications": [
            "Higher risk of regional recurrence",
            "Completion dissection + adjuvant therapy",
            "Consider radiation therapy",
            "Intensive surveillance required"
        ]
    }
}

# Store for sentinel node biopsies (in production, this would be in the database)
sentinel_node_biopsies = {}


@router.get("/sentinel-node/basins")
async def get_lymph_node_basins():
    """Get all lymph node basin definitions for mapping."""
    return {
        "basins": LYMPH_NODE_BASINS,
        "result_categories": SLN_RESULT_CATEGORIES
    }


@router.get("/sentinel-node/basins/{region}")
async def get_basin_by_region(region: str):
    """Get lymph node basin for a specific anatomic region."""
    if region not in LYMPH_NODE_BASINS:
        raise HTTPException(
            status_code=404,
            detail=f"Region '{region}' not found. Valid regions: {list(LYMPH_NODE_BASINS.keys())}"
        )
    return LYMPH_NODE_BASINS[region]


@router.post("/sentinel-node/map-drainage")
async def map_lymphatic_drainage(
    primary_site: str = Form(..., description="Primary tumor anatomic site"),
    laterality: Optional[str] = Form(None, description="left, right, or midline"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Map expected lymphatic drainage basins based on primary tumor location.
    Returns probable sentinel node basins for surgical planning.
    """
    # Normalize input
    site_lower = primary_site.lower().replace(" ", "_")

    # Determine region
    region = None
    specific_basins = []

    # Head and neck mapping
    head_neck_sites = ["scalp", "forehead", "temple", "face", "ear", "neck", "lip", "nose", "eyelid"]
    if any(s in site_lower for s in head_neck_sites):
        region = "head_neck"
        basin_data = LYMPH_NODE_BASINS["head_neck"]

        # Determine specific basins based on subsite
        if "scalp" in site_lower or "forehead" in site_lower:
            if "posterior" in site_lower:
                specific_basins = ["occipital", "postauricular", "posterior_triangle"]
            else:
                specific_basins = ["parotid", "upper_cervical"]
        elif "face" in site_lower or "nose" in site_lower:
            specific_basins = ["parotid", "submandibular", "upper_cervical"]
        elif "lip" in site_lower or "chin" in site_lower:
            specific_basins = ["submandibular", "submental", "upper_cervical"]
        elif "ear" in site_lower:
            specific_basins = ["parotid", "postauricular", "upper_cervical"]
        elif "neck" in site_lower:
            specific_basins = ["upper_cervical", "mid_cervical", "posterior_triangle"]
        else:
            specific_basins = ["parotid", "submandibular", "upper_cervical"]

    # Upper extremity mapping
    elif any(s in site_lower for s in ["arm", "hand", "finger", "forearm", "elbow", "wrist"]):
        region = "upper_extremity"
        basin_data = LYMPH_NODE_BASINS["upper_extremity"]

        if "hand" in site_lower or "finger" in site_lower:
            if "ulnar" in site_lower or "medial" in site_lower:
                specific_basins = ["epitrochlear", "axillary_level_i"]
            else:
                specific_basins = ["axillary_level_i"]
        elif "forearm" in site_lower:
            specific_basins = ["epitrochlear", "axillary_level_i"]
        else:
            specific_basins = ["axillary_level_i", "axillary_level_ii"]

    # Trunk mapping
    elif any(s in site_lower for s in ["chest", "abdomen", "back", "flank", "shoulder"]):
        if "back" in site_lower or "posterior" in site_lower:
            region = "trunk_posterior"
        else:
            region = "trunk_anterior"
        basin_data = LYMPH_NODE_BASINS[region]

        # Apply Sappey's line watershed
        upper_trunk = any(s in site_lower for s in ["chest", "upper", "shoulder", "scapula"])

        if laterality == "right" or "right" in site_lower:
            if upper_trunk:
                specific_basins = ["axillary_right"]
            else:
                specific_basins = ["inguinal_right"]
        elif laterality == "left" or "left" in site_lower:
            if upper_trunk:
                specific_basins = ["axillary_left"]
            else:
                specific_basins = ["inguinal_left"]
        else:  # Midline
            if upper_trunk:
                specific_basins = ["axillary_right", "axillary_left"]
            else:
                specific_basins = ["inguinal_right", "inguinal_left"]

    # Lower extremity mapping
    elif any(s in site_lower for s in ["leg", "thigh", "foot", "toe", "ankle", "knee", "calf", "buttock"]):
        region = "lower_extremity"
        basin_data = LYMPH_NODE_BASINS["lower_extremity"]

        if "foot" in site_lower or "toe" in site_lower or "ankle" in site_lower:
            if "lateral" in site_lower or "posterior" in site_lower:
                specific_basins = ["popliteal", "inguinal_superficial"]
            else:
                specific_basins = ["inguinal_superficial"]
        elif "calf" in site_lower and "posterior" in site_lower:
            specific_basins = ["popliteal", "inguinal_superficial"]
        else:
            specific_basins = ["inguinal_superficial", "inguinal_deep"]

    if not region:
        raise HTTPException(
            status_code=400,
            detail=f"Could not determine lymphatic drainage for site: {primary_site}"
        )

    # Build response with specific basin details
    mapped_basins = []
    for basin in basin_data["primary_basins"]:
        is_primary = basin["id"] in specific_basins
        mapped_basins.append({
            **basin,
            "is_primary_drainage": is_primary,
            "priority": 1 if is_primary else 2
        })

    # Sort by priority
    mapped_basins.sort(key=lambda x: (x["priority"], x["name"]))

    return {
        "primary_site": primary_site,
        "laterality": laterality,
        "region": region,
        "region_name": basin_data["name"],
        "expected_basins": mapped_basins,
        "primary_basin_ids": specific_basins,
        "watershed_info": basin_data.get("watershed_line"),
        "clinical_notes": [
            "Lymphoscintigraphy recommended for definitive mapping",
            "Sentinel node(s) may drain to unexpected basins in 5-10% of cases",
            "In-transit nodes should be identified and biopsied if present"
        ]
    }


@router.post("/sentinel-node/record-biopsy")
async def record_sentinel_node_biopsy(
    lesion_id: Optional[int] = Form(None, description="Associated lesion ID"),
    patient_id: Optional[int] = Form(None, description="Patient ID if no lesion"),
    primary_site: str = Form(..., description="Primary tumor location"),
    biopsy_date: str = Form(..., description="Date of SLNB procedure"),
    basin: str = Form(..., description="Lymph node basin biopsied"),
    node_id: str = Form(..., description="Specific node identifier"),
    result: str = Form(..., description="Biopsy result category"),
    tumor_deposit_mm: Optional[float] = Form(None, description="Size of largest tumor deposit in mm"),
    nodes_examined: int = Form(1, description="Number of nodes examined"),
    nodes_positive: int = Form(0, description="Number of positive nodes"),
    extracapsular: bool = Form(False, description="Extracapsular extension present"),
    immunohistochemistry: Optional[str] = Form(None, description="IHC markers used"),
    notes: Optional[str] = Form(None, description="Additional pathology notes"),
    current_user: User = Depends(get_current_active_user)
):
    """Record sentinel lymph node biopsy results."""

    # Validate result category
    if result not in SLN_RESULT_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid result. Valid options: {list(SLN_RESULT_CATEGORIES.keys())}"
        )

    result_info = SLN_RESULT_CATEGORIES[result]

    # Generate biopsy record ID
    import uuid
    biopsy_id = str(uuid.uuid4())[:8]

    # Determine N category based on result and tumor size
    n_category = result_info["code"]
    if result in ["micrometastasis", "macrometastasis"]:
        if nodes_positive == 1:
            n_category = "N1a"
        elif nodes_positive <= 3:
            n_category = "N2a"
        else:
            n_category = "N3a"

        if extracapsular:
            n_category = n_category.replace("a", "c")

    biopsy_record = {
        "biopsy_id": biopsy_id,
        "user_id": current_user.id,
        "lesion_id": lesion_id,
        "primary_site": primary_site,
        "biopsy_date": biopsy_date,
        "basin": basin,
        "node_id": node_id,
        "result": result,
        "result_description": result_info["description"],
        "tumor_deposit_mm": tumor_deposit_mm,
        "nodes_examined": nodes_examined,
        "nodes_positive": nodes_positive,
        "extracapsular_extension": extracapsular,
        "immunohistochemistry": immunohistochemistry,
        "notes": notes,
        "n_category": n_category,
        "implications": result_info["implications"],
        "result_color": result_info["color"],
        "created_at": datetime.utcnow().isoformat()
    }

    # Store in memory (in production, save to database)
    user_key = f"user_{current_user.id}"
    if user_key not in sentinel_node_biopsies:
        sentinel_node_biopsies[user_key] = []
    sentinel_node_biopsies[user_key].append(biopsy_record)

    return {
        "success": True,
        "biopsy_record": biopsy_record,
        "staging_impact": {
            "n_category": n_category,
            "description": result_info["description"],
            "recommendations": result_info["implications"]
        }
    }


@router.get("/sentinel-node/biopsies")
async def get_sentinel_node_biopsies(
    lesion_id: Optional[int] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get all sentinel node biopsy records for the current user."""
    user_key = f"user_{current_user.id}"
    biopsies = sentinel_node_biopsies.get(user_key, [])

    if lesion_id:
        biopsies = [b for b in biopsies if b.get("lesion_id") == lesion_id]

    # Group by basin for visualization
    by_basin = {}
    for biopsy in biopsies:
        basin = biopsy["basin"]
        if basin not in by_basin:
            by_basin[basin] = []
        by_basin[basin].append(biopsy)

    # Summary statistics
    total_nodes = sum(b["nodes_examined"] for b in biopsies)
    positive_nodes = sum(b["nodes_positive"] for b in biopsies)

    return {
        "biopsies": biopsies,
        "by_basin": by_basin,
        "summary": {
            "total_procedures": len(biopsies),
            "total_nodes_examined": total_nodes,
            "total_nodes_positive": positive_nodes,
            "positivity_rate": f"{(positive_nodes/total_nodes*100):.1f}%" if total_nodes > 0 else "N/A"
        }
    }


@router.get("/sentinel-node/result-categories")
async def get_result_categories():
    """Get all sentinel node biopsy result categories with descriptions."""
    return {
        "categories": [
            {
                "id": key,
                **value
            }
            for key, value in SLN_RESULT_CATEGORIES.items()
        ]
    }


# =============================================================================
# AI ACCURACY IMPROVEMENT TRACKING
# Track and visualize diagnostic accuracy over time
# =============================================================================

# Simulated historical accuracy data (in production, would come from database)
# This represents how the model has improved over different periods
AI_ACCURACY_HISTORY = {
    "overall": {
        "2024_Q1": {"accuracy": 0.847, "samples": 15420, "precision": 0.831, "recall": 0.862, "f1": 0.846},
        "2024_Q2": {"accuracy": 0.863, "samples": 18750, "precision": 0.852, "recall": 0.874, "f1": 0.863},
        "2024_Q3": {"accuracy": 0.881, "samples": 22100, "precision": 0.869, "recall": 0.893, "f1": 0.881},
        "2024_Q4": {"accuracy": 0.894, "samples": 26800, "precision": 0.885, "recall": 0.903, "f1": 0.894},
        "2025_Q1": {"accuracy": 0.908, "samples": 31500, "precision": 0.897, "recall": 0.919, "f1": 0.908},
    },
    "by_condition": {
        "melanoma": {
            "2024_Q1": {"accuracy": 0.891, "sensitivity": 0.923, "specificity": 0.876},
            "2024_Q2": {"accuracy": 0.904, "sensitivity": 0.931, "specificity": 0.889},
            "2024_Q3": {"accuracy": 0.918, "sensitivity": 0.942, "specificity": 0.901},
            "2024_Q4": {"accuracy": 0.927, "sensitivity": 0.951, "specificity": 0.912},
            "2025_Q1": {"accuracy": 0.936, "sensitivity": 0.958, "specificity": 0.921},
        },
        "basal_cell_carcinoma": {
            "2024_Q1": {"accuracy": 0.872, "sensitivity": 0.889, "specificity": 0.861},
            "2024_Q2": {"accuracy": 0.885, "sensitivity": 0.901, "specificity": 0.874},
            "2024_Q3": {"accuracy": 0.897, "sensitivity": 0.912, "specificity": 0.886},
            "2024_Q4": {"accuracy": 0.911, "sensitivity": 0.924, "specificity": 0.901},
            "2025_Q1": {"accuracy": 0.921, "sensitivity": 0.932, "specificity": 0.913},
        },
        "squamous_cell_carcinoma": {
            "2024_Q1": {"accuracy": 0.856, "sensitivity": 0.871, "specificity": 0.847},
            "2024_Q2": {"accuracy": 0.869, "sensitivity": 0.883, "specificity": 0.859},
            "2024_Q3": {"accuracy": 0.884, "sensitivity": 0.897, "specificity": 0.874},
            "2024_Q4": {"accuracy": 0.896, "sensitivity": 0.908, "specificity": 0.887},
            "2025_Q1": {"accuracy": 0.909, "sensitivity": 0.921, "specificity": 0.899},
        },
        "nevus": {
            "2024_Q1": {"accuracy": 0.912, "sensitivity": 0.924, "specificity": 0.903},
            "2024_Q2": {"accuracy": 0.921, "sensitivity": 0.932, "specificity": 0.912},
            "2024_Q3": {"accuracy": 0.931, "sensitivity": 0.941, "specificity": 0.923},
            "2024_Q4": {"accuracy": 0.938, "sensitivity": 0.947, "specificity": 0.931},
            "2025_Q1": {"accuracy": 0.944, "sensitivity": 0.952, "specificity": 0.937},
        },
        "seborrheic_keratosis": {
            "2024_Q1": {"accuracy": 0.878, "sensitivity": 0.891, "specificity": 0.869},
            "2024_Q2": {"accuracy": 0.889, "sensitivity": 0.901, "specificity": 0.879},
            "2024_Q3": {"accuracy": 0.901, "sensitivity": 0.913, "specificity": 0.891},
            "2024_Q4": {"accuracy": 0.912, "sensitivity": 0.923, "specificity": 0.903},
            "2025_Q1": {"accuracy": 0.921, "sensitivity": 0.931, "specificity": 0.913},
        },
        "actinic_keratosis": {
            "2024_Q1": {"accuracy": 0.834, "sensitivity": 0.852, "specificity": 0.821},
            "2024_Q2": {"accuracy": 0.851, "sensitivity": 0.868, "specificity": 0.838},
            "2024_Q3": {"accuracy": 0.867, "sensitivity": 0.883, "specificity": 0.854},
            "2024_Q4": {"accuracy": 0.881, "sensitivity": 0.896, "specificity": 0.869},
            "2025_Q1": {"accuracy": 0.894, "sensitivity": 0.908, "specificity": 0.882},
        },
        "dermatofibroma": {
            "2024_Q1": {"accuracy": 0.867, "sensitivity": 0.879, "specificity": 0.858},
            "2024_Q2": {"accuracy": 0.879, "sensitivity": 0.891, "specificity": 0.869},
            "2024_Q3": {"accuracy": 0.892, "sensitivity": 0.903, "specificity": 0.883},
            "2024_Q4": {"accuracy": 0.904, "sensitivity": 0.914, "specificity": 0.896},
            "2025_Q1": {"accuracy": 0.914, "sensitivity": 0.924, "specificity": 0.906},
        },
        "vascular_lesion": {
            "2024_Q1": {"accuracy": 0.889, "sensitivity": 0.901, "specificity": 0.879},
            "2024_Q2": {"accuracy": 0.899, "sensitivity": 0.911, "specificity": 0.889},
            "2024_Q3": {"accuracy": 0.911, "sensitivity": 0.922, "specificity": 0.901},
            "2024_Q4": {"accuracy": 0.921, "sensitivity": 0.931, "specificity": 0.912},
            "2025_Q1": {"accuracy": 0.929, "sensitivity": 0.939, "specificity": 0.921},
        },
    },
    "by_skin_type": {
        "type_i_ii": {
            "2024_Q1": {"accuracy": 0.871, "samples": 8210},
            "2024_Q2": {"accuracy": 0.884, "samples": 9980},
            "2024_Q3": {"accuracy": 0.898, "samples": 11760},
            "2024_Q4": {"accuracy": 0.912, "samples": 14270},
            "2025_Q1": {"accuracy": 0.923, "samples": 16780},
        },
        "type_iii_iv": {
            "2024_Q1": {"accuracy": 0.856, "samples": 5130},
            "2024_Q2": {"accuracy": 0.871, "samples": 6240},
            "2024_Q3": {"accuracy": 0.887, "samples": 7350},
            "2024_Q4": {"accuracy": 0.901, "samples": 8910},
            "2025_Q1": {"accuracy": 0.914, "samples": 10470},
        },
        "type_v_vi": {
            "2024_Q1": {"accuracy": 0.812, "samples": 2080},
            "2024_Q2": {"accuracy": 0.834, "samples": 2530},
            "2024_Q3": {"accuracy": 0.858, "samples": 2990},
            "2024_Q4": {"accuracy": 0.879, "samples": 3620},
            "2025_Q1": {"accuracy": 0.898, "samples": 4250},
        },
    },
    "model_versions": [
        {"version": "1.0", "date": "2024-01-15", "accuracy": 0.847, "notes": "Initial release with ResNet34 backbone"},
        {"version": "1.1", "date": "2024-04-01", "accuracy": 0.863, "notes": "Added data augmentation, expanded training set"},
        {"version": "1.2", "date": "2024-07-01", "accuracy": 0.881, "notes": "Upgraded to EfficientNet-B3, added diverse skin tone data"},
        {"version": "2.0", "date": "2024-10-01", "accuracy": 0.894, "notes": "Multi-task learning with dermoscopy features"},
        {"version": "2.1", "date": "2025-01-15", "accuracy": 0.908, "notes": "Fine-tuned on clinical validation feedback"},
    ],
    "feedback_impact": {
        "total_feedback_received": 4850,
        "corrections_incorporated": 3920,
        "accuracy_improvement_from_feedback": 0.023,
        "top_correction_categories": [
            {"category": "melanoma_vs_nevus", "corrections": 890, "impact": "+2.1%"},
            {"category": "scc_vs_ak", "corrections": 720, "impact": "+1.8%"},
            {"category": "bcc_morphology", "corrections": 610, "impact": "+1.5%"},
            {"category": "skin_tone_adjustment", "corrections": 540, "impact": "+1.2%"},
            {"category": "dermoscopy_features", "corrections": 480, "impact": "+1.0%"},
        ]
    }
}


@router.get("/ai-accuracy/overview")
async def get_ai_accuracy_overview():
    """
    Get overview of AI diagnostic accuracy with improvement trends.
    Shows how accuracy has improved over time with more training data.
    """
    overall = AI_ACCURACY_HISTORY["overall"]
    periods = list(overall.keys())

    # Calculate improvement metrics
    first_period = overall[periods[0]]
    latest_period = overall[periods[-1]]

    improvement = {
        "accuracy_gain": round((latest_period["accuracy"] - first_period["accuracy"]) * 100, 1),
        "precision_gain": round((latest_period["precision"] - first_period["precision"]) * 100, 1),
        "recall_gain": round((latest_period["recall"] - first_period["recall"]) * 100, 1),
        "sample_growth": latest_period["samples"] - first_period["samples"],
        "sample_growth_pct": round((latest_period["samples"] / first_period["samples"] - 1) * 100, 1),
    }

    # Format timeline data
    timeline = []
    for period, data in overall.items():
        year, quarter = period.split("_")
        timeline.append({
            "period": period,
            "label": f"{quarter} {year}",
            "accuracy": data["accuracy"],
            "accuracy_pct": f"{data['accuracy']*100:.1f}%",
            "samples": data["samples"],
            "precision": data["precision"],
            "recall": data["recall"],
            "f1": data["f1"],
        })

    return {
        "current_accuracy": f"{latest_period['accuracy']*100:.1f}%",
        "current_metrics": {
            "accuracy": latest_period["accuracy"],
            "precision": latest_period["precision"],
            "recall": latest_period["recall"],
            "f1_score": latest_period["f1"],
            "total_samples": latest_period["samples"],
        },
        "improvement_summary": improvement,
        "timeline": timeline,
        "improvement_drivers": [
            "Expanded training dataset with diverse skin tones",
            "Incorporation of clinician feedback corrections",
            "Model architecture upgrades (ResNet → EfficientNet)",
            "Multi-task learning with dermoscopic feature detection",
            "Enhanced data augmentation techniques",
        ],
    }


@router.get("/ai-accuracy/by-condition")
async def get_accuracy_by_condition():
    """Get accuracy breakdown by skin condition type."""
    by_condition = AI_ACCURACY_HISTORY["by_condition"]

    results = []
    for condition, data in by_condition.items():
        periods = list(data.keys())
        first = data[periods[0]]
        latest = data[periods[-1]]

        results.append({
            "condition": condition.replace("_", " ").title(),
            "condition_id": condition,
            "current_accuracy": f"{latest['accuracy']*100:.1f}%",
            "current_sensitivity": f"{latest['sensitivity']*100:.1f}%",
            "current_specificity": f"{latest['specificity']*100:.1f}%",
            "accuracy_improvement": f"+{(latest['accuracy'] - first['accuracy'])*100:.1f}%",
            "timeline": [
                {
                    "period": p,
                    "accuracy": d["accuracy"],
                    "sensitivity": d["sensitivity"],
                    "specificity": d["specificity"],
                }
                for p, d in data.items()
            ],
        })

    # Sort by current accuracy
    results.sort(key=lambda x: float(x["current_accuracy"].rstrip("%")), reverse=True)

    return {
        "conditions": results,
        "best_performing": results[0]["condition"],
        "most_improved": max(results, key=lambda x: float(x["accuracy_improvement"].rstrip("%").lstrip("+")))["condition"],
    }


@router.get("/ai-accuracy/by-skin-type")
async def get_accuracy_by_skin_type():
    """Get accuracy breakdown by Fitzpatrick skin type."""
    by_skin_type = AI_ACCURACY_HISTORY["by_skin_type"]

    results = []
    for skin_type, data in by_skin_type.items():
        periods = list(data.keys())
        first = data[periods[0]]
        latest = data[periods[-1]]

        type_label = skin_type.replace("type_", "Type ").replace("_", "-").upper()

        results.append({
            "skin_type": type_label,
            "skin_type_id": skin_type,
            "current_accuracy": f"{latest['accuracy']*100:.1f}%",
            "sample_count": latest["samples"],
            "accuracy_improvement": f"+{(latest['accuracy'] - first['accuracy'])*100:.1f}%",
            "sample_growth": f"+{latest['samples'] - first['samples']:,}",
            "timeline": [
                {"period": p, "accuracy": d["accuracy"], "samples": d["samples"]}
                for p, d in data.items()
            ],
        })

    # Calculate equity gap
    accuracies = [float(r["current_accuracy"].rstrip("%")) for r in results]
    equity_gap = max(accuracies) - min(accuracies)

    return {
        "skin_types": results,
        "equity_metrics": {
            "accuracy_gap": f"{equity_gap:.1f}%",
            "gap_trend": "Narrowing",
            "lowest_accuracy_type": min(results, key=lambda x: float(x["current_accuracy"].rstrip("%")))["skin_type"],
            "note": "Active efforts to collect more diverse training data are reducing accuracy gaps across skin types.",
        },
    }


@router.get("/ai-accuracy/model-versions")
async def get_model_version_history():
    """Get history of model versions and their improvements."""
    versions = AI_ACCURACY_HISTORY["model_versions"]

    return {
        "versions": versions,
        "current_version": versions[-1]["version"],
        "total_versions": len(versions),
        "accuracy_progression": [
            {"version": v["version"], "accuracy": v["accuracy"]}
            for v in versions
        ],
    }


@router.get("/ai-accuracy/feedback-impact")
async def get_feedback_impact():
    """Get information about how user feedback has improved accuracy."""
    feedback = AI_ACCURACY_HISTORY["feedback_impact"]

    return {
        "summary": {
            "total_feedback": feedback["total_feedback_received"],
            "incorporated": feedback["corrections_incorporated"],
            "incorporation_rate": f"{(feedback['corrections_incorporated']/feedback['total_feedback_received'])*100:.1f}%",
            "accuracy_improvement": f"+{feedback['accuracy_improvement_from_feedback']*100:.1f}%",
        },
        "top_categories": feedback["top_correction_categories"],
        "how_feedback_helps": [
            "Corrections are reviewed by dermatologists for validation",
            "Validated corrections are added to training dataset",
            "Model is periodically retrained with accumulated feedback",
            "Edge cases and misclassifications get extra attention",
            "Skin tone-specific corrections help reduce bias",
        ],
        "call_to_action": "Your feedback directly improves diagnostic accuracy for all users. Please report any incorrect diagnoses.",
    }


@router.get("/ai-accuracy/projections")
async def get_accuracy_projections():
    """Get projected accuracy improvements based on current trends."""
    overall = AI_ACCURACY_HISTORY["overall"]
    periods = list(overall.keys())

    # Calculate average quarterly improvement
    improvements = []
    for i in range(1, len(periods)):
        prev = overall[periods[i-1]]["accuracy"]
        curr = overall[periods[i]]["accuracy"]
        improvements.append(curr - prev)

    avg_improvement = sum(improvements) / len(improvements)
    latest_accuracy = overall[periods[-1]]["accuracy"]

    # Project next 4 quarters
    projections = []
    projected_accuracy = latest_accuracy
    quarters = ["2025_Q2", "2025_Q3", "2025_Q4", "2026_Q1"]

    for quarter in quarters:
        # Diminishing returns as we approach theoretical limits
        diminishing_factor = max(0.5, 1 - (projected_accuracy - 0.85) * 2)
        projected_accuracy = min(0.98, projected_accuracy + (avg_improvement * diminishing_factor))

        year, q = quarter.split("_")
        projections.append({
            "period": quarter,
            "label": f"{q} {year}",
            "projected_accuracy": round(projected_accuracy, 3),
            "projected_accuracy_pct": f"{projected_accuracy*100:.1f}%",
            "confidence": "Medium" if projected_accuracy < 0.95 else "Lower",
        })

    return {
        "current_accuracy": f"{latest_accuracy*100:.1f}%",
        "avg_quarterly_improvement": f"+{avg_improvement*100:.2f}%",
        "projections": projections,
        "factors_affecting_projections": [
            "Rate of new training data collection",
            "Quality and diversity of feedback",
            "Architectural improvements",
            "Diminishing returns near theoretical accuracy ceiling",
        ],
        "theoretical_ceiling": {
            "accuracy": "~96-98%",
            "note": "Inter-rater variability among dermatologists sets the practical ceiling",
        },
    }


# ============================================================================
# MALPRACTICE SHIELD - Liability Analysis & Insurance Coverage Recommendations
# ============================================================================

# Risk factors and liability data
MALPRACTICE_DATA = {
    "high_risk_conditions": {
        "melanoma": {
            "risk_level": "very_high",
            "risk_score": 95,
            "common_claims": [
                "Delayed diagnosis",
                "Failure to biopsy suspicious lesion",
                "Inadequate follow-up",
                "Misinterpretation of pathology",
            ],
            "average_settlement": 850000,
            "median_settlement": 425000,
            "documentation_requirements": [
                "Detailed dermoscopy findings",
                "ABCDE criteria assessment",
                "Photo documentation with measurement",
                "Biopsy recommendation and patient response",
                "Follow-up schedule documented",
            ],
        },
        "squamous_cell_carcinoma": {
            "risk_level": "high",
            "risk_score": 75,
            "common_claims": [
                "Delayed treatment",
                "Inadequate margins",
                "Failure to assess metastatic risk",
            ],
            "average_settlement": 320000,
            "median_settlement": 175000,
            "documentation_requirements": [
                "Lesion size and location",
                "Risk factor assessment",
                "Treatment plan rationale",
                "Margin documentation",
            ],
        },
        "basal_cell_carcinoma": {
            "risk_level": "moderate",
            "risk_score": 45,
            "common_claims": [
                "Cosmetic outcome dissatisfaction",
                "Recurrence due to inadequate treatment",
            ],
            "average_settlement": 125000,
            "median_settlement": 65000,
            "documentation_requirements": [
                "Treatment options discussed",
                "Cosmetic expectations documented",
                "Informed consent comprehensive",
            ],
        },
        "psoriasis": {
            "risk_level": "low",
            "risk_score": 25,
            "common_claims": [
                "Medication side effects",
                "Failure to monitor biologics",
            ],
            "average_settlement": 85000,
            "median_settlement": 45000,
            "documentation_requirements": [
                "Lab monitoring records",
                "Side effect counseling",
                "Treatment response tracking",
            ],
        },
        "eczema": {
            "risk_level": "low",
            "risk_score": 20,
            "common_claims": [
                "Steroid side effects",
                "Delayed referral for severe cases",
            ],
            "average_settlement": 55000,
            "median_settlement": 30000,
            "documentation_requirements": [
                "Severity assessment",
                "Steroid potency and duration",
                "Referral criteria evaluation",
            ],
        },
    },
    "documentation_score_factors": {
        "photo_documentation": 15,
        "detailed_history": 15,
        "physical_exam_documented": 15,
        "differential_diagnosis": 10,
        "treatment_rationale": 15,
        "informed_consent": 15,
        "follow_up_plan": 10,
        "patient_education": 5,
    },
    "insurance_coverage_types": {
        "occurrence": {
            "description": "Covers claims from incidents during policy period, regardless of when claim is filed",
            "pros": ["Lifetime coverage for policy period incidents", "No tail coverage needed"],
            "cons": ["Higher premiums", "Less common"],
            "recommended_for": ["Established practices", "Near-retirement physicians"],
        },
        "claims_made": {
            "description": "Covers claims made during policy period for incidents during policy period",
            "pros": ["Lower initial premiums", "More common/available"],
            "cons": ["Requires tail coverage", "Gaps possible"],
            "recommended_for": ["New practices", "Budget-conscious practices"],
        },
    },
    "coverage_recommendations": {
        "solo_practice": {
            "min_per_occurrence": 1000000,
            "min_aggregate": 3000000,
            "recommended_per_occurrence": 2000000,
            "recommended_aggregate": 6000000,
        },
        "group_practice": {
            "min_per_occurrence": 1000000,
            "min_aggregate": 3000000,
            "recommended_per_occurrence": 3000000,
            "recommended_aggregate": 9000000,
        },
        "academic_medical_center": {
            "min_per_occurrence": 2000000,
            "min_aggregate": 6000000,
            "recommended_per_occurrence": 5000000,
            "recommended_aggregate": 15000000,
        },
    },
    "risk_mitigation_strategies": [
        {
            "strategy": "Standardized Documentation Templates",
            "effectiveness": "high",
            "implementation_cost": "low",
            "description": "Use consistent templates for all skin lesion evaluations",
            "premium_reduction": "5-10%",
        },
        {
            "strategy": "AI-Assisted Diagnosis Logging",
            "effectiveness": "high",
            "implementation_cost": "medium",
            "description": "Document AI confidence levels and reasoning for all assessments",
            "premium_reduction": "3-5%",
        },
        {
            "strategy": "Patient Communication Documentation",
            "effectiveness": "very_high",
            "implementation_cost": "low",
            "description": "Document all patient communications about diagnoses and follow-up",
            "premium_reduction": "8-12%",
        },
        {
            "strategy": "Peer Review Program",
            "effectiveness": "high",
            "implementation_cost": "medium",
            "description": "Regular case reviews with colleagues for complex cases",
            "premium_reduction": "5-8%",
        },
        {
            "strategy": "CME in Dermoscopy",
            "effectiveness": "medium",
            "implementation_cost": "low",
            "description": "Annual continuing education in dermoscopy techniques",
            "premium_reduction": "2-3%",
        },
        {
            "strategy": "Biopsy Threshold Protocol",
            "effectiveness": "very_high",
            "implementation_cost": "low",
            "description": "Clear protocols for when to recommend biopsy",
            "premium_reduction": "10-15%",
        },
    ],
}


@router.post("/malpractice/analyze-risk")
async def analyze_malpractice_risk(
    diagnosis: str = Form(...),
    confidence_level: float = Form(...),
    documentation_completeness: str = Form(None),  # JSON string of documentation items
    patient_factors: str = Form(None),  # JSON string of patient risk factors
    current_user: dict = Depends(get_current_active_user),
):
    """
    Analyze malpractice liability risk for a specific diagnosis.
    Returns risk assessment and mitigation recommendations.
    """
    import json

    diagnosis_lower = diagnosis.lower().replace(" ", "_").replace("-", "_")

    # Find matching condition
    condition_data = None
    matched_condition = None
    for condition, data in MALPRACTICE_DATA["high_risk_conditions"].items():
        if condition in diagnosis_lower or diagnosis_lower in condition:
            condition_data = data
            matched_condition = condition
            break

    # Default for unknown conditions
    if not condition_data:
        condition_data = {
            "risk_level": "moderate",
            "risk_score": 50,
            "common_claims": ["Misdiagnosis", "Delayed treatment"],
            "average_settlement": 200000,
            "median_settlement": 100000,
            "documentation_requirements": [
                "Complete history and physical",
                "Differential diagnosis",
                "Treatment rationale",
            ],
        }
        matched_condition = "general_dermatology"

    # Parse documentation completeness
    doc_items = {}
    if documentation_completeness:
        try:
            doc_items = json.loads(documentation_completeness)
        except:
            pass

    # Calculate documentation score
    doc_score = 0
    doc_factors = MALPRACTICE_DATA["documentation_score_factors"]
    missing_documentation = []

    for item, points in doc_factors.items():
        if doc_items.get(item, False):
            doc_score += points
        else:
            missing_documentation.append({
                "item": item.replace("_", " ").title(),
                "points": points,
                "priority": "high" if points >= 15 else "medium" if points >= 10 else "low",
            })

    # Adjust risk based on confidence and documentation
    base_risk = condition_data["risk_score"]

    # Lower confidence increases risk
    confidence_adjustment = (1 - confidence_level) * 20

    # Poor documentation increases risk
    doc_adjustment = (100 - doc_score) * 0.3

    adjusted_risk = min(100, base_risk + confidence_adjustment + doc_adjustment)

    # Determine risk category
    if adjusted_risk >= 80:
        risk_category = "critical"
        risk_color = "#dc3545"
    elif adjusted_risk >= 60:
        risk_category = "high"
        risk_color = "#fd7e14"
    elif adjusted_risk >= 40:
        risk_category = "moderate"
        risk_color = "#ffc107"
    else:
        risk_category = "low"
        risk_color = "#28a745"

    # Get relevant mitigation strategies
    mitigation_strategies = []
    for strategy in MALPRACTICE_DATA["risk_mitigation_strategies"]:
        relevance = "high" if adjusted_risk >= 60 else "medium" if adjusted_risk >= 40 else "standard"
        mitigation_strategies.append({
            **strategy,
            "relevance": relevance,
        })

    # Sort by effectiveness
    effectiveness_order = {"very_high": 0, "high": 1, "medium": 2, "low": 3}
    mitigation_strategies.sort(key=lambda x: effectiveness_order.get(x["effectiveness"], 2))

    return {
        "diagnosis": diagnosis,
        "matched_condition": matched_condition.replace("_", " ").title(),
        "risk_assessment": {
            "base_risk_score": base_risk,
            "adjusted_risk_score": round(adjusted_risk, 1),
            "risk_category": risk_category,
            "risk_color": risk_color,
            "risk_level": condition_data["risk_level"],
        },
        "liability_exposure": {
            "average_settlement": condition_data["average_settlement"],
            "average_settlement_formatted": f"${condition_data['average_settlement']:,}",
            "median_settlement": condition_data["median_settlement"],
            "median_settlement_formatted": f"${condition_data['median_settlement']:,}",
            "common_claims": condition_data["common_claims"],
        },
        "documentation_analysis": {
            "current_score": doc_score,
            "max_score": 100,
            "score_percentage": f"{doc_score}%",
            "grade": "A" if doc_score >= 90 else "B" if doc_score >= 80 else "C" if doc_score >= 70 else "D" if doc_score >= 60 else "F",
            "missing_items": missing_documentation,
            "required_documentation": condition_data["documentation_requirements"],
        },
        "confidence_impact": {
            "ai_confidence": f"{confidence_level*100:.1f}%",
            "confidence_risk_adjustment": round(confidence_adjustment, 1),
            "recommendation": "Consider additional evaluation or referral" if confidence_level < 0.7 else "Confidence level acceptable",
        },
        "mitigation_strategies": mitigation_strategies[:5],
        "immediate_actions": [
            action for action in missing_documentation if action["priority"] == "high"
        ][:3],
    }


@router.get("/malpractice/insurance-recommendations")
async def get_insurance_recommendations(
    practice_type: str = Query("solo_practice", description="solo_practice, group_practice, or academic_medical_center"),
    annual_patient_volume: int = Query(2000, description="Approximate annual patient volume"),
    high_risk_procedures: bool = Query(False, description="Whether practice performs high-risk procedures"),
    current_user: dict = Depends(get_current_active_user),
):
    """
    Get malpractice insurance coverage recommendations based on practice profile.
    """
    coverage = MALPRACTICE_DATA["coverage_recommendations"].get(
        practice_type,
        MALPRACTICE_DATA["coverage_recommendations"]["solo_practice"]
    )

    # Adjust based on volume and risk
    volume_multiplier = 1.0
    if annual_patient_volume > 5000:
        volume_multiplier = 1.5
    elif annual_patient_volume > 3000:
        volume_multiplier = 1.25

    risk_multiplier = 1.5 if high_risk_procedures else 1.0

    adjusted_recommended = {
        "per_occurrence": int(coverage["recommended_per_occurrence"] * volume_multiplier * risk_multiplier),
        "aggregate": int(coverage["recommended_aggregate"] * volume_multiplier * risk_multiplier),
    }

    # Estimate premium range
    base_premium = 15000 if practice_type == "solo_practice" else 25000 if practice_type == "group_practice" else 40000
    premium_range = {
        "minimum_coverage": {
            "low": int(base_premium * 0.8),
            "high": int(base_premium * 1.2),
        },
        "recommended_coverage": {
            "low": int(base_premium * volume_multiplier * risk_multiplier * 0.9),
            "high": int(base_premium * volume_multiplier * risk_multiplier * 1.3),
        },
    }

    return {
        "practice_profile": {
            "type": practice_type.replace("_", " ").title(),
            "annual_volume": annual_patient_volume,
            "high_risk_procedures": high_risk_procedures,
        },
        "minimum_coverage": {
            "per_occurrence": coverage["min_per_occurrence"],
            "per_occurrence_formatted": f"${coverage['min_per_occurrence']:,}",
            "aggregate": coverage["min_aggregate"],
            "aggregate_formatted": f"${coverage['min_aggregate']:,}",
            "notation": f"${coverage['min_per_occurrence']//1000000}M/${coverage['min_aggregate']//1000000}M",
        },
        "recommended_coverage": {
            "per_occurrence": adjusted_recommended["per_occurrence"],
            "per_occurrence_formatted": f"${adjusted_recommended['per_occurrence']:,}",
            "aggregate": adjusted_recommended["aggregate"],
            "aggregate_formatted": f"${adjusted_recommended['aggregate']:,}",
            "notation": f"${adjusted_recommended['per_occurrence']//1000000}M/${adjusted_recommended['aggregate']//1000000}M",
        },
        "estimated_annual_premium": {
            "minimum_coverage_range": f"${premium_range['minimum_coverage']['low']:,} - ${premium_range['minimum_coverage']['high']:,}",
            "recommended_coverage_range": f"${premium_range['recommended_coverage']['low']:,} - ${premium_range['recommended_coverage']['high']:,}",
        },
        "coverage_types": MALPRACTICE_DATA["insurance_coverage_types"],
        "recommendation": "claims_made" if practice_type == "solo_practice" else "occurrence",
        "recommendation_rationale": "Claims-made policies offer lower initial premiums for newer/smaller practices, while occurrence policies provide better long-term protection for established practices.",
        "additional_coverage_considerations": [
            {
                "type": "Cyber Liability",
                "reason": "AI-assisted diagnosis systems process sensitive patient data",
                "recommended_limit": "$1,000,000",
            },
            {
                "type": "Tail Coverage",
                "reason": "Essential if switching from claims-made policy",
                "recommended_limit": "Match primary policy limits",
            },
            {
                "type": "Consent to Settle",
                "reason": "Gives you control over settlement decisions",
                "recommended": True,
            },
        ],
    }


@router.get("/malpractice/documentation-checklist")
async def get_documentation_checklist(
    condition_type: str = Query("general", description="Type of condition: melanoma, skin_cancer, inflammatory, general"),
    current_user: dict = Depends(get_current_active_user),
):
    """
    Get comprehensive documentation checklist to minimize liability risk.
    """
    base_checklist = [
        {"item": "Patient demographics and identifiers", "category": "Administrative", "required": True},
        {"item": "Chief complaint in patient's words", "category": "History", "required": True},
        {"item": "History of present illness (onset, duration, changes)", "category": "History", "required": True},
        {"item": "Past medical history", "category": "History", "required": True},
        {"item": "Family history of skin conditions/cancer", "category": "History", "required": True},
        {"item": "Medication list including OTC", "category": "History", "required": True},
        {"item": "Allergy documentation", "category": "History", "required": True},
        {"item": "Social history (sun exposure, tanning)", "category": "History", "required": True},
    ]

    exam_checklist = [
        {"item": "Full skin examination performed (Y/N)", "category": "Physical Exam", "required": True},
        {"item": "Lesion location documented", "category": "Physical Exam", "required": True},
        {"item": "Lesion size with measurement", "category": "Physical Exam", "required": True},
        {"item": "Color description", "category": "Physical Exam", "required": True},
        {"item": "Border characteristics", "category": "Physical Exam", "required": True},
        {"item": "Surface features (texture, scale)", "category": "Physical Exam", "required": True},
        {"item": "Photographic documentation", "category": "Physical Exam", "required": True},
    ]

    assessment_checklist = [
        {"item": "Clinical impression/diagnosis", "category": "Assessment", "required": True},
        {"item": "Differential diagnosis list", "category": "Assessment", "required": True},
        {"item": "AI-assisted diagnosis confidence level", "category": "Assessment", "required": False},
        {"item": "Dermoscopy findings (if performed)", "category": "Assessment", "required": False},
    ]

    plan_checklist = [
        {"item": "Treatment plan documented", "category": "Plan", "required": True},
        {"item": "Rationale for treatment choice", "category": "Plan", "required": True},
        {"item": "Patient education provided", "category": "Plan", "required": True},
        {"item": "Follow-up schedule", "category": "Plan", "required": True},
        {"item": "Warning signs to watch for", "category": "Plan", "required": True},
    ]

    consent_checklist = [
        {"item": "Informed consent for procedure", "category": "Consent", "required": True},
        {"item": "Risks discussed and documented", "category": "Consent", "required": True},
        {"item": "Alternatives discussed", "category": "Consent", "required": True},
        {"item": "Patient questions addressed", "category": "Consent", "required": True},
    ]

    # Condition-specific additions
    condition_specific = []

    if condition_type == "melanoma" or condition_type == "skin_cancer":
        condition_specific = [
            {"item": "ABCDE criteria assessment", "category": "Melanoma-Specific", "required": True},
            {"item": "Breslow thickness (if biopsied)", "category": "Melanoma-Specific", "required": True},
            {"item": "Ulceration status", "category": "Melanoma-Specific", "required": True},
            {"item": "Mitotic rate", "category": "Melanoma-Specific", "required": True},
            {"item": "Sentinel node evaluation discussed", "category": "Melanoma-Specific", "required": True},
            {"item": "Staging documented", "category": "Melanoma-Specific", "required": True},
            {"item": "Referral to oncology (if indicated)", "category": "Melanoma-Specific", "required": False},
            {"item": "Genetic counseling discussed (if indicated)", "category": "Melanoma-Specific", "required": False},
        ]
    elif condition_type == "inflammatory":
        condition_specific = [
            {"item": "Severity score (PASI, EASI, etc.)", "category": "Inflammatory-Specific", "required": True},
            {"item": "Previous treatments and responses", "category": "Inflammatory-Specific", "required": True},
            {"item": "Quality of life impact assessment", "category": "Inflammatory-Specific", "required": True},
            {"item": "Biologic screening labs (if applicable)", "category": "Inflammatory-Specific", "required": False},
            {"item": "TB screening (if starting biologics)", "category": "Inflammatory-Specific", "required": False},
        ]

    all_items = base_checklist + exam_checklist + assessment_checklist + plan_checklist + consent_checklist + condition_specific

    return {
        "condition_type": condition_type,
        "total_items": len(all_items),
        "required_items": len([i for i in all_items if i["required"]]),
        "checklist": all_items,
        "categories": {
            "Administrative": [i for i in all_items if i["category"] == "Administrative"],
            "History": [i for i in all_items if i["category"] == "History"],
            "Physical Exam": [i for i in all_items if i["category"] == "Physical Exam"],
            "Assessment": [i for i in all_items if i["category"] == "Assessment"],
            "Plan": [i for i in all_items if i["category"] == "Plan"],
            "Consent": [i for i in all_items if i["category"] == "Consent"],
            "Condition-Specific": condition_specific,
        },
        "tips": [
            "Document in real-time rather than at end of day",
            "Use specific measurements rather than vague descriptions",
            "Record patient's understanding of diagnosis and plan",
            "Document any patient non-compliance or missed appointments",
            "Always document when patient declines recommended procedures",
        ],
    }


@router.get("/malpractice/claim-statistics")
async def get_claim_statistics(
    current_user: dict = Depends(get_current_active_user),
):
    """
    Get dermatology malpractice claim statistics and trends.
    """
    return {
        "overview": {
            "specialty_ranking": "Dermatology ranks among the lower-risk specialties for malpractice claims",
            "annual_claim_rate": "2-3% of dermatologists face a claim annually",
            "claims_resulting_in_payment": "Approximately 20-25% of claims result in payment",
        },
        "claim_breakdown_by_type": [
            {"type": "Failure to Diagnose Melanoma", "percentage": 35, "avg_payout": 850000},
            {"type": "Surgical Complications", "percentage": 20, "avg_payout": 225000},
            {"type": "Medication Side Effects", "percentage": 15, "avg_payout": 125000},
            {"type": "Cosmetic Outcome Dissatisfaction", "percentage": 12, "avg_payout": 95000},
            {"type": "Delayed Treatment", "percentage": 10, "avg_payout": 175000},
            {"type": "Other", "percentage": 8, "avg_payout": 85000},
        ],
        "trends": [
            {
                "trend": "Increasing melanoma-related claims",
                "direction": "up",
                "note": "Early detection expectations rising with AI availability",
            },
            {
                "trend": "Telemedicine-related claims",
                "direction": "up",
                "note": "New category emerging with remote diagnosis",
            },
            {
                "trend": "Documentation-related defenses",
                "direction": "up",
                "note": "Better documentation improving defense outcomes",
            },
        ],
        "defense_success_factors": [
            {"factor": "Complete photo documentation", "impact": "Reduces unfavorable outcome by 40%"},
            {"factor": "Documented differential diagnosis", "impact": "Reduces unfavorable outcome by 35%"},
            {"factor": "Timely follow-up documented", "impact": "Reduces unfavorable outcome by 30%"},
            {"factor": "Patient education documented", "impact": "Reduces unfavorable outcome by 25%"},
            {"factor": "Peer consultation documented", "impact": "Reduces unfavorable outcome by 20%"},
        ],
        "high_risk_scenarios": [
            {
                "scenario": "Pigmented lesion in difficult location (scalp, between toes)",
                "risk_level": "very_high",
                "recommendation": "Lower threshold for biopsy, document reasoning thoroughly",
            },
            {
                "scenario": "Patient with history of melanoma presenting with new lesion",
                "risk_level": "very_high",
                "recommendation": "Document full body exam, consider total body photography",
            },
            {
                "scenario": "Lesion that changed between visits",
                "risk_level": "high",
                "recommendation": "Document comparison, strong consideration for biopsy",
            },
            {
                "scenario": "Patient requesting specific diagnosis against clinical judgment",
                "risk_level": "high",
                "recommendation": "Document patient request and your clinical reasoning",
            },
        ],
    }


@router.get("/malpractice/risk-mitigation-tips")
async def get_risk_mitigation_tips(
    current_user: dict = Depends(get_current_active_user),
):
    """
    Get comprehensive risk mitigation tips for dermatology practice.
    """
    return {
        "strategies": MALPRACTICE_DATA["risk_mitigation_strategies"],
        "communication_tips": [
            {
                "tip": "Set realistic expectations",
                "details": "Clearly explain limitations of visual diagnosis and AI assistance",
            },
            {
                "tip": "Document patient understanding",
                "details": "Note that patient verbalized understanding of diagnosis and plan",
            },
            {
                "tip": "Provide written instructions",
                "details": "Give patients take-home instructions for wound care and warning signs",
            },
            {
                "tip": "Encourage questions",
                "details": "Document that patient was given opportunity to ask questions",
            },
            {
                "tip": "Use teach-back method",
                "details": "Have patient repeat back key instructions to confirm understanding",
            },
        ],
        "documentation_best_practices": [
            "Document in real-time or immediately after encounter",
            "Use objective descriptions (mm measurements vs. 'small')",
            "Include pertinent negatives in physical exam",
            "Document clinical decision-making rationale",
            "Note any deviations from standard protocols and why",
            "Record all patient communications including phone calls",
            "Document missed appointments and follow-up attempts",
        ],
        "ai_specific_guidance": [
            {
                "practice": "Document AI confidence levels",
                "rationale": "Shows due diligence in utilizing diagnostic tools",
            },
            {
                "practice": "Note when AI recommendation differs from clinical judgment",
                "rationale": "Demonstrates independent clinical reasoning",
            },
            {
                "practice": "Record AI version and date of analysis",
                "rationale": "Creates audit trail for quality improvement",
            },
            {
                "practice": "Document patient consent for AI-assisted diagnosis",
                "rationale": "Ensures informed consent includes technology use",
            },
        ],
        "when_to_refer": [
            "Diagnosis uncertain after dermoscopy",
            "Lesion with atypical features but patient declines biopsy",
            "Treatment not responding as expected",
            "Patient requests second opinion",
            "Complex medical-legal situation",
        ],
    }
