"""
Enhanced ML Capabilities Router

API endpoints for:
1. U-Net Segmentation - Lesion boundary detection
2. Temporal Prediction - Growth forecasting
3. Federated Learning - Privacy-preserving updates
4. Multi-Scale Feature Extraction
"""

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import base64
import io

from PIL import Image

# Import auth
from auth import get_current_user
from database import User

# Import enhanced ML modules
from ml_enhanced import (
    lesion_segmenter,
    temporal_predictor,
    federated_manager,
    feature_extractor
)

router = APIRouter(prefix="/api/ml", tags=["Enhanced ML"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SegmentationRequest(BaseModel):
    image_base64: str
    threshold: Optional[float] = 0.5


class TemporalMeasurement(BaseModel):
    date: str
    area_mm2: float
    diameter_mm: float
    asymmetry: Optional[float] = 0.0
    border_irregularity: Optional[float] = 0.0
    color_variation: Optional[float] = 0.0
    elevation: Optional[float] = 0.0


class GrowthPredictionRequest(BaseModel):
    lesion_id: Optional[int] = None
    historical_measurements: List[TemporalMeasurement]
    prediction_days: Optional[int] = 90


class FederatedGradientRequest(BaseModel):
    local_data_count: int
    gradient_summary: Optional[Dict[str, float]] = None


class FeatureExtractionRequest(BaseModel):
    image_base64: str


# =============================================================================
# SEGMENTATION ENDPOINTS
# =============================================================================

@router.post("/segmentation/analyze")
async def analyze_lesion_segmentation(
    request: SegmentationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Segment a lesion from an image using U-Net.

    Returns:
    - Binary segmentation mask
    - Boundary points
    - Area and geometric features
    - Asymmetry and border irregularity scores
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Run segmentation
        result = lesion_segmenter.segment_lesion(image, threshold=request.threshold)

        return {
            "success": True,
            "segmentation": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@router.post("/segmentation/upload")
async def segment_uploaded_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    current_user: User = Depends(get_current_user)
):
    """
    Segment a lesion from an uploaded image file.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run segmentation
        result = lesion_segmenter.segment_lesion(image, threshold=threshold)

        return {
            "success": True,
            "filename": file.filename,
            "segmentation": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@router.get("/segmentation/info")
async def get_segmentation_info():
    """Get information about the segmentation model."""
    return {
        "model": "U-Net",
        "input_size": "256x256 (auto-resized)",
        "output": "Binary segmentation mask",
        "features_extracted": [
            "Lesion boundary",
            "Area (pixels and percentage)",
            "Centroid location",
            "Bounding box",
            "Asymmetry score",
            "Border irregularity",
            "Compactness",
            "Eccentricity"
        ],
        "threshold_range": "0.0 - 1.0 (default: 0.5)"
    }


# =============================================================================
# TEMPORAL PREDICTION ENDPOINTS
# =============================================================================

@router.post("/temporal/predict")
async def predict_lesion_growth(
    request: GrowthPredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Predict future lesion growth based on historical measurements.

    Returns:
    - Growth trajectory predictions
    - Risk assessment
    - Recommended follow-up schedule
    """
    try:
        # Convert measurements to dict format
        historical_data = [
            {
                "date": m.date,
                "area_mm2": m.area_mm2,
                "diameter_mm": m.diameter_mm,
                "asymmetry": m.asymmetry,
                "border_irregularity": m.border_irregularity,
                "color_variation": m.color_variation,
                "elevation": m.elevation
            }
            for m in request.historical_measurements
        ]

        # Run prediction
        result = temporal_predictor.predict_growth(
            historical_data,
            prediction_days=request.prediction_days
        )

        return {
            "success": True,
            "lesion_id": request.lesion_id,
            "prediction_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/temporal/risk-assessment")
async def assess_growth_risk(
    request: GrowthPredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Get a detailed risk assessment based on historical measurements.
    """
    try:
        historical_data = [
            {
                "date": m.date,
                "area_mm2": m.area_mm2,
                "diameter_mm": m.diameter_mm,
                "asymmetry": m.asymmetry,
                "border_irregularity": m.border_irregularity,
                "color_variation": m.color_variation,
                "elevation": m.elevation
            }
            for m in request.historical_measurements
        ]

        # Get full prediction with risk assessment
        result = temporal_predictor.predict_growth(historical_data, prediction_days=30)

        return {
            "success": True,
            "risk_assessment": result.get("risk_assessment", {}),
            "recommended_followup": result.get("recommended_followup", {}),
            "historical_summary": result.get("historical_summary", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")


@router.get("/temporal/info")
async def get_temporal_model_info():
    """Get information about the temporal prediction model."""
    return {
        "model": "LSTM (Long Short-Term Memory)",
        "architecture": {
            "input_size": 6,
            "hidden_size": 64,
            "num_layers": 2
        },
        "input_features": [
            "area_mm2",
            "diameter_mm",
            "asymmetry",
            "border_irregularity",
            "color_variation",
            "elevation"
        ],
        "output": {
            "predictions": "Weekly growth forecasts",
            "confidence_intervals": "Upper and lower bounds",
            "risk_assessment": "Low/Moderate/High risk level",
            "trend": "stable/slow_growth/moderate_growth/rapid_growth/shrinking"
        },
        "minimum_measurements": 2,
        "recommended_measurements": 4
    }


# =============================================================================
# FEDERATED LEARNING ENDPOINTS
# =============================================================================

@router.get("/federated/status")
async def get_federated_status(
    current_user: User = Depends(get_current_user)
):
    """Get current federated learning status and model info."""
    model_info = federated_manager.get_model_info()
    privacy_budget = federated_manager.check_privacy_budget()

    return {
        "model_info": model_info,
        "privacy_budget": privacy_budget,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/federated/compute-gradients")
async def compute_local_gradients(
    request: FederatedGradientRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Compute local gradients for federated learning.
    Gradients are computed with differential privacy.
    """
    # Simulate local data based on count
    local_data = [{"sample": i} for i in range(request.local_data_count)]

    result = federated_manager.compute_local_gradients(local_data)

    return {
        "success": True,
        "result": result,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/federated/privacy-budget")
async def check_privacy_budget(
    current_user: User = Depends(get_current_user)
):
    """Check remaining differential privacy budget."""
    budget = federated_manager.check_privacy_budget()

    return {
        "privacy_budget": budget,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/federated/info")
async def get_federated_info():
    """Get information about the federated learning framework."""
    return {
        "framework": "Federated Learning with Differential Privacy",
        "features": [
            "Privacy-preserving model updates",
            "Differential privacy noise injection",
            "Secure gradient aggregation",
            "Privacy budget tracking"
        ],
        "privacy_mechanism": "Laplacian noise",
        "epsilon_per_update": 0.1,
        "total_privacy_budget": 10.0,
        "benefits": [
            "No raw data leaves device",
            "Model improves from distributed data",
            "Mathematically guaranteed privacy",
            "Compliant with HIPAA/GDPR"
        ]
    }


# =============================================================================
# FEATURE EXTRACTION ENDPOINTS
# =============================================================================

@router.post("/features/extract")
async def extract_multi_scale_features(
    request: FeatureExtractionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Extract multi-scale features from an image.

    Returns features at multiple resolutions for comprehensive analysis.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Extract features
        result = feature_extractor.extract_features(image)

        return {
            "success": True,
            "features": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")


@router.post("/features/upload")
async def extract_features_from_upload(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Extract multi-scale features from an uploaded image file."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = feature_extractor.extract_features(image)

        return {
            "success": True,
            "filename": file.filename,
            "features": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")


@router.get("/features/info")
async def get_feature_extraction_info():
    """Get information about the multi-scale feature extractor."""
    return {
        "extractor": "Multi-Scale Feature Extractor",
        "scales": [1.0, 0.75, 0.5, 0.25],
        "features_per_scale": {
            "color": ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"],
            "texture": ["gradient_mean", "gradient_std"],
            "size": ["width", "height"]
        },
        "output_dimension": 24,
        "applications": [
            "Comprehensive lesion characterization",
            "Multi-resolution pattern detection",
            "Feature input for downstream models"
        ]
    }


# =============================================================================
# COMBINED ANALYSIS ENDPOINT
# =============================================================================

@router.post("/analyze/comprehensive")
async def comprehensive_lesion_analysis(
    file: UploadFile = File(...),
    include_segmentation: bool = Form(True),
    include_features: bool = Form(True),
    current_user: User = Depends(get_current_user)
):
    """
    Perform comprehensive lesion analysis including:
    - U-Net segmentation
    - Multi-scale feature extraction
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        results = {
            "success": True,
            "filename": file.filename,
            "image_size": {"width": image.width, "height": image.height},
            "timestamp": datetime.now().isoformat()
        }

        if include_segmentation:
            results["segmentation"] = lesion_segmenter.segment_lesion(image)

        if include_features:
            results["multi_scale_features"] = feature_extractor.extract_features(image)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")
