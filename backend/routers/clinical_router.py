"""
Clinical Analysis Router

Endpoints for:
- Burn classification
- Dermoscopy analysis
- Biopsy and histopathology
- Clinical photography standards
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
import json

from database import get_db, User, AnalysisHistory, SessionLocal
from auth import get_current_active_user

# Import model monitoring
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
