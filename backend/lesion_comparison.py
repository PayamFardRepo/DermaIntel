"""
Lesion Comparison & Change Detection Module

AI-powered analysis to detect changes in skin lesions over time.
Critical for early melanoma detection and tracking lesion evolution.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from scipy.spatial.distance import cosine
import io
import base64


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def extract_feature_vector(image: Image.Image, model, processor, device) -> np.ndarray:
    """
    Extract deep feature vector from lesion image using the AI model.

    Args:
        image: PIL Image
        model: The lesion classification model
        processor: Image processor
        device: torch device (cuda/cpu)

    Returns:
        Feature vector as numpy array
    """
    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Extract features from penultimate layer (before final classification)
    model.eval()
    with torch.no_grad():
        # Get the output of the model before the classification head
        outputs = model(**inputs, output_hidden_states=True)

        # Use the last hidden state as feature vector
        # This represents the learned visual features
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            feature_vector = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        else:
            # Fallback: use logits as feature representation
            feature_vector = outputs.logits.cpu().numpy()

    return feature_vector.flatten()


def calculate_visual_similarity(baseline_image: Image.Image, current_image: Image.Image,
                                model, processor, device) -> Dict:
    """
    Calculate visual similarity between two lesion images using deep learning features.

    Returns:
        Dictionary with similarity metrics
    """
    # Extract feature vectors
    baseline_features = extract_feature_vector(baseline_image, model, processor, device)
    current_features = extract_feature_vector(current_image, model, processor, device)

    # Calculate cosine distance (0 = identical, 2 = opposite)
    feature_distance = cosine(baseline_features, current_features)

    # Convert to similarity score (0 = different, 1 = identical)
    visual_similarity = 1 - (feature_distance / 2)

    return {
        "feature_vector_distance": float(feature_distance),
        "visual_similarity_score": float(visual_similarity),
        "baseline_features": baseline_features.tolist(),
        "current_features": current_features.tolist()
    }


def analyze_size_change(baseline_image: Image.Image, current_image: Image.Image,
                       baseline_measurements: Optional[Dict] = None,
                       current_measurements: Optional[Dict] = None) -> Dict:
    """
    Analyze changes in lesion size between two images.

    Args:
        baseline_image: Earlier image
        current_image: Later image
        baseline_measurements: Optional calibrated measurements from first analysis
        current_measurements: Optional calibrated measurements from current analysis

    Returns:
        Dictionary with size change metrics
    """
    result = {
        "size_changed": False,
        "size_change_percent": 0.0,
        "size_change_mm": None,
        "size_trend": "stable"
    }

    # If calibrated measurements available, use them for precise size comparison
    if baseline_measurements and current_measurements:
        baseline_area = baseline_measurements.get('area_mm2', 0)
        current_area = current_measurements.get('area_mm2', 0)

        if baseline_area > 0 and current_area > 0:
            change_percent = ((current_area - baseline_area) / baseline_area) * 100
            result["size_change_percent"] = float(change_percent)
            result["size_change_mm"] = float(current_area - baseline_area)

            # Threshold: >20% change is significant
            if abs(change_percent) > 20:
                result["size_changed"] = True
                result["size_trend"] = "growing" if change_percent > 0 else "shrinking"

    else:
        # Fallback: Estimate size change using image segmentation
        baseline_gray = cv2.cvtColor(np.array(baseline_image), cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2GRAY)

        # Simple thresholding to segment lesion (this is a rough estimate)
        _, baseline_mask = cv2.threshold(baseline_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, current_mask = cv2.threshold(current_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        baseline_pixels = np.sum(baseline_mask > 0)
        current_pixels = np.sum(current_mask > 0)

        if baseline_pixels > 0:
            change_percent = ((current_pixels - baseline_pixels) / baseline_pixels) * 100
            result["size_change_percent"] = float(change_percent)

            if abs(change_percent) > 25:  # Higher threshold for pixel-based estimate
                result["size_changed"] = True
                result["size_trend"] = "growing" if change_percent > 0 else "shrinking"

    return result


def analyze_color_change(baseline_image: Image.Image, current_image: Image.Image) -> Dict:
    """
    Analyze changes in lesion color between two images.
    Uses color histograms and statistical measures.
    """
    # Convert to numpy arrays
    baseline_arr = np.array(baseline_image)
    current_arr = np.array(current_image)

    # Resize to same size for comparison
    target_size = (224, 224)
    baseline_resized = cv2.resize(baseline_arr, target_size)
    current_resized = cv2.resize(current_arr, target_size)

    # Convert to LAB color space for perceptual color difference
    baseline_lab = cv2.cvtColor(baseline_resized, cv2.COLOR_RGB2LAB)
    current_lab = cv2.cvtColor(current_resized, cv2.COLOR_RGB2LAB)

    # Calculate mean color in each channel
    baseline_mean = np.mean(baseline_lab, axis=(0, 1))
    current_mean = np.mean(current_lab, axis=(0, 1))

    # Calculate color difference (Delta E)
    color_diff = np.linalg.norm(baseline_mean - current_mean)

    # Normalize to 0-1 scale (100 is perceptually very different)
    color_change_score = min(color_diff / 100, 1.0)

    # Calculate color histogram comparison
    baseline_hist = [cv2.calcHist([baseline_resized], [i], None, [256], [0, 256]) for i in range(3)]
    current_hist = [cv2.calcHist([current_resized], [i], None, [256], [0, 256]) for i in range(3)]

    # Normalize histograms
    baseline_hist = [cv2.normalize(h, h).flatten() for h in baseline_hist]
    current_hist = [cv2.normalize(h, h).flatten() for h in current_hist]

    # Calculate histogram correlation for each channel
    correlations = [cv2.compareHist(baseline_hist[i], current_hist[i], cv2.HISTCMP_CORREL) for i in range(3)]
    avg_correlation = np.mean(correlations)

    # Detect new colors (check for bimodality changes)
    new_colors = False
    if avg_correlation < 0.85:  # Significant distribution change
        new_colors = True

    result = {
        "color_changed": color_change_score > 0.15,  # Threshold for noticeable change
        "color_change_score": float(color_change_score),
        "color_description": _generate_color_change_description(color_change_score),
        "new_colors_appeared": new_colors,
        "histogram_correlation": float(avg_correlation)
    }

    return result


def _generate_color_change_description(score: float) -> str:
    """Generate human-readable description of color change."""
    if score < 0.1:
        return "No significant color change detected"
    elif score < 0.25:
        return "Minimal color variation observed"
    elif score < 0.5:
        return "Moderate color change detected"
    else:
        return "Significant color change observed - may indicate concerning evolution"


def analyze_shape_change(baseline_image: Image.Image, current_image: Image.Image) -> Dict:
    """
    Analyze changes in lesion shape and border regularity.
    Important for ABCDE criteria (Asymmetry and Border).
    """
    # Convert to grayscale
    baseline_gray = cv2.cvtColor(np.array(baseline_image), cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(np.array(current_image), cv2.COLOR_RGB2GRAY)

    # Resize to same size
    target_size = (224, 224)
    baseline_gray = cv2.resize(baseline_gray, target_size)
    current_gray = cv2.resize(current_gray, target_size)

    # Segment lesions using thresholding
    _, baseline_mask = cv2.threshold(baseline_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, current_mask = cv2.threshold(current_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    baseline_contours, _ = cv2.findContours(baseline_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not baseline_contours or not current_contours:
        return {
            "shape_changed": False,
            "asymmetry_increased": False,
            "border_irregularity_increased": False,
            "shape_change_score": 0.0
        }

    # Get largest contour
    baseline_contour = max(baseline_contours, key=cv2.contourArea)
    current_contour = max(current_contours, key=cv2.contourArea)

    # Calculate shape features
    def get_shape_features(contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return 0, 0, 0

        # Compactness (circularity)
        compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # Calculate moments for asymmetry
        moments = cv2.moments(contour)
        if moments['m00'] > 0:
            hu_moments = cv2.HuMoments(moments)
            asymmetry = float(np.sum(np.abs(hu_moments)))
        else:
            asymmetry = 0

        # Border irregularity (inverse of compactness)
        irregularity = 1 - compactness

        return compactness, asymmetry, irregularity

    baseline_comp, baseline_asym, baseline_irreg = get_shape_features(baseline_contour)
    current_comp, current_asym, current_irreg = get_shape_features(current_contour)

    # Calculate changes
    asymmetry_change = current_asym - baseline_asym
    irregularity_change = current_irreg - baseline_irreg

    # Shape change score (combining multiple metrics)
    shape_change_score = (abs(asymmetry_change) + abs(irregularity_change)) / 2
    shape_change_score = min(shape_change_score, 1.0)

    result = {
        "shape_changed": shape_change_score > 0.2,
        "asymmetry_increased": asymmetry_change > 0.15,
        "border_irregularity_increased": irregularity_change > 0.15,
        "shape_change_score": float(shape_change_score)
    }

    return result


def generate_change_heatmap(baseline_image: Image.Image, current_image: Image.Image) -> str:
    """
    Generate a visual heatmap showing differences between two images.

    Returns:
        Base64-encoded PNG image
    """
    # Resize both images to same size
    target_size = (400, 400)
    baseline_resized = cv2.resize(np.array(baseline_image), target_size)
    current_resized = cv2.resize(np.array(current_image), target_size)

    # Convert to LAB color space for perceptual difference
    baseline_lab = cv2.cvtColor(baseline_resized, cv2.COLOR_RGB2LAB).astype(float)
    current_lab = cv2.cvtColor(current_resized, cv2.COLOR_RGB2LAB).astype(float)

    # Calculate pixel-wise difference
    diff = np.linalg.norm(baseline_lab - current_lab, axis=2)

    # Normalize to 0-255
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply colormap (hot = high difference)
    heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)

    # Overlay heatmap on current image
    overlay = cv2.addWeighted(current_resized, 0.6, heatmap, 0.4, 0)

    # Convert to PIL Image
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    # Convert to base64
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format='PNG')
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return base64_image


def compare_symptoms(baseline_symptoms: Dict, current_symptoms: Dict) -> Dict:
    """
    Compare symptom data between two time points.
    """
    symptom_changes = []

    # Check for new symptoms
    new_symptoms = False
    if not baseline_symptoms.get('symptom_itching') and current_symptoms.get('symptom_itching'):
        new_symptoms = True
        symptom_changes.append("New itching reported")

    if not baseline_symptoms.get('symptom_pain') and current_symptoms.get('symptom_pain'):
        new_symptoms = True
        symptom_changes.append("New pain reported")

    if not baseline_symptoms.get('symptom_bleeding') and current_symptoms.get('symptom_bleeding'):
        new_symptoms = True
        symptom_changes.append("New bleeding reported")

    # Check for worsening symptoms
    symptom_worsening = False
    if (baseline_symptoms.get('symptom_itching_severity', 0) < current_symptoms.get('symptom_itching_severity', 0)):
        symptom_worsening = True
        symptom_changes.append(f"Itching severity increased from {baseline_symptoms.get('symptom_itching_severity')} to {current_symptoms.get('symptom_itching_severity')}")

    if (baseline_symptoms.get('symptom_pain_severity', 0) < current_symptoms.get('symptom_pain_severity', 0)):
        symptom_worsening = True
        symptom_changes.append(f"Pain severity increased from {baseline_symptoms.get('symptom_pain_severity')} to {current_symptoms.get('symptom_pain_severity')}")

    return {
        "new_symptoms": new_symptoms,
        "symptom_worsening": symptom_worsening,
        "symptom_changes_list": symptom_changes
    }


def compare_abcde_criteria(baseline_abcde: Dict, current_abcde: Dict) -> Dict:
    """
    Compare ABCDE criteria between two analyses to detect worsening.
    """
    if not baseline_abcde or not current_abcde:
        return {
            "abcde_worsening": False,
            "abcde_comparison": {}
        }

    worsening = False
    comparison = {}

    for criterion in ['asymmetry', 'border', 'color', 'diameter', 'evolution']:
        baseline_value = baseline_abcde.get(criterion, {})
        current_value = current_abcde.get(criterion, {})

        baseline_concern = baseline_value.get('concerning', False)
        current_concern = current_value.get('concerning', False)

        comparison[criterion] = {
            "baseline": baseline_value,
            "current": current_value,
            "worsened": not baseline_concern and current_concern
        }

        if not baseline_concern and current_concern:
            worsening = True

    return {
        "abcde_worsening": worsening,
        "abcde_comparison": comparison
    }


def determine_change_severity(metrics: Dict) -> Tuple[str, float]:
    """
    Determine overall severity of detected changes.

    Returns:
        (severity_level, change_score)
        severity_level: "none", "minimal", "moderate", "significant", "concerning"
        change_score: 0-1 score of change magnitude
    """
    score = 0.0
    weights = []

    # Visual similarity (inverse - low similarity = high change)
    if 'visual_similarity_score' in metrics:
        visual_change = 1 - metrics['visual_similarity_score']
        score += visual_change * 0.3
        weights.append(0.3)

    # Size change
    if metrics.get('size_changed'):
        size_impact = min(abs(metrics.get('size_change_percent', 0)) / 100, 1.0)
        score += size_impact * 0.25
        weights.append(0.25)

    # Color change
    if 'color_change_score' in metrics:
        score += metrics['color_change_score'] * 0.2
        weights.append(0.2)

    # Shape change
    if 'shape_change_score' in metrics:
        score += metrics['shape_change_score'] * 0.15
        weights.append(0.15)

    # New symptoms
    if metrics.get('new_symptoms'):
        score += 0.5 * 0.1
        weights.append(0.1)

    # Normalize score
    if weights:
        score = score / sum(weights)

    # Determine severity level
    if score < 0.1:
        severity = "none"
    elif score < 0.25:
        severity = "minimal"
    elif score < 0.5:
        severity = "moderate"
    elif score < 0.75:
        severity = "significant"
    else:
        severity = "concerning"

    return severity, float(score)


def generate_recommendation(comparison_result: Dict) -> Tuple[str, str, bool]:
    """
    Generate clinical recommendation based on comparison results.

    Returns:
        (recommendation_text, urgency_level, action_required)
    """
    severity = comparison_result.get('change_severity', 'none')
    risk_escalated = comparison_result.get('risk_escalated', False)
    abcde_worsening = comparison_result.get('abcde_worsening', False)
    new_symptoms = comparison_result.get('new_symptoms', False)
    size_trend = comparison_result.get('size_trend', 'stable')

    # Determine urgency
    if severity == "concerning" or (risk_escalated and abcde_worsening):
        urgency = "urgent"
        action_required = True
        recommendation = "URGENT: Significant concerning changes detected. Schedule dermatologist appointment within 1-2 weeks. Potential signs of malignancy require professional evaluation."

    elif severity == "significant" or risk_escalated or abcde_worsening:
        urgency = "soon"
        action_required = True
        recommendation = "Noticeable changes detected that warrant professional evaluation. Schedule dermatologist appointment within 2-4 weeks for assessment."

    elif severity == "moderate" or new_symptoms or (size_trend == "growing" and comparison_result.get('size_change_percent', 0) > 30):
        urgency = "routine"
        action_required = True
        recommendation = "Moderate changes observed. Consider scheduling a dermatologist appointment within 1-2 months. Continue monitoring this lesion closely."

    elif severity == "minimal":
        urgency = "routine"
        action_required = False
        recommendation = "Minor changes detected. Continue regular monitoring. Re-check in 3 months. If changes accelerate or new symptoms develop, consult a dermatologist."

    else:  # none
        urgency = "routine"
        action_required = False
        recommendation = "No significant changes detected. Lesion appears stable. Continue monitoring according to your regular schedule."

    return recommendation, urgency, action_required


def calculate_growth_rate(comparisons: List[Dict]) -> Optional[float]:
    """
    Calculate lesion growth rate based on multiple comparisons over time.

    Returns:
        Growth rate in mm²/month, or None if insufficient data
    """
    if len(comparisons) < 2:
        return None

    # Extract size changes over time
    size_changes = []
    for comp in comparisons:
        if comp.get('size_change_mm') and comp.get('time_difference_days'):
            # Convert to mm²/month
            growth_per_month = (comp['size_change_mm'] / comp['time_difference_days']) * 30
            size_changes.append(growth_per_month)

    if not size_changes:
        return None

    # Calculate average growth rate
    avg_growth = np.mean(size_changes)
    return float(avg_growth)


def compare_lesions_full(
    baseline_image_path: str,
    current_image_path: str,
    baseline_diagnosis: str = "",
    current_diagnosis: str = "",
    baseline_confidence: float = None,
    current_confidence: float = None,
    baseline_risk: str = "",
    current_risk: str = ""
) -> Dict:
    """
    Comprehensive lesion comparison orchestrating all analysis functions.

    Args:
        baseline_image_path: Path to earlier image
        current_image_path: Path to later image
        baseline_diagnosis: AI diagnosis from baseline analysis
        current_diagnosis: AI diagnosis from current analysis
        baseline_confidence: Confidence score of baseline
        current_confidence: Confidence score of current
        baseline_risk: Risk level of baseline
        current_risk: Risk level of current

    Returns:
        Complete comparison results dictionary
    """
    try:
        # Load images
        baseline_img = Image.open(baseline_image_path).convert('RGB')
        current_img = Image.open(current_image_path).convert('RGB')

        # Initialize results
        comparison_result = {
            'success': True,
            'baseline_diagnosis': baseline_diagnosis,
            'current_diagnosis': current_diagnosis,
            'diagnosis_changed': baseline_diagnosis != current_diagnosis if baseline_diagnosis and current_diagnosis else False
        }

        # 1. Size analysis
        size_result = analyze_size_change(baseline_img, current_img)
        comparison_result.update({
            'size_changed': size_result.get('size_changed', False),
            'size_change_percent': size_result.get('size_change_percent', 0.0),
            'size_change_mm': size_result.get('size_change_mm'),
            'size_trend': size_result.get('size_trend', 'stable'),
            'size_details': size_result
        })

        # 2. Color analysis
        color_result = analyze_color_change(baseline_img, current_img)
        comparison_result.update({
            'color_changed': color_result.get('color_changed', False),
            'color_change_score': color_result.get('color_change_score', 0.0),
            'new_colors_appeared': color_result.get('new_colors_appeared', False),
            'color_details': color_result
        })

        # 3. Shape/border analysis
        shape_result = analyze_shape_change(baseline_img, current_img)
        comparison_result.update({
            'border_changed': shape_result.get('border_changed', False),
            'asymmetry_increased': shape_result.get('asymmetry_increased', False),
            'border_irregularity_increased': shape_result.get('border_irregularity_increased', False),
            'shape_details': shape_result
        })

        # 4. Generate change heatmap
        try:
            heatmap = generate_change_heatmap(baseline_img, current_img)
            comparison_result['change_heatmap'] = heatmap
        except Exception as e:
            print(f"Could not generate heatmap: {e}")
            comparison_result['change_heatmap'] = None

        # 5. Visual similarity (requires model - optional)
        try:
            # This would require loading the model - skip for now if not available
            comparison_result['visual_similarity_score'] = None
            comparison_result['feature_vector_distance'] = None
        except Exception as e:
            print(f"Visual similarity calculation skipped: {e}")

        # 6. Determine overall change severity
        metrics = {
            'size_changed': comparison_result['size_changed'],
            'size_change_percent': abs(comparison_result['size_change_percent']),
            'color_changed': comparison_result['color_changed'],
            'color_change_score': comparison_result['color_change_score'],
            'border_changed': comparison_result['border_changed'],
            'diagnosis_changed': comparison_result['diagnosis_changed'],
            'risk_escalated': baseline_risk in ['low', 'medium'] and current_risk in ['high', 'very_high']
        }

        severity, change_score = determine_change_severity(metrics)
        comparison_result.update({
            'change_severity': severity,
            'change_score': change_score,
            'change_detected': change_score > 0.15  # Threshold for "change detected"
        })

        # 7. Generate recommendations
        recommendation, urgency, action_required = generate_recommendation(comparison_result)
        comparison_result.update({
            'recommendation': recommendation,
            'urgency_level': urgency,
            'action_required': action_required
        })

        # 8. Risk escalation check
        comparison_result['risk_escalated'] = metrics['risk_escalated']
        comparison_result['baseline_risk'] = baseline_risk
        comparison_result['current_risk'] = current_risk

        # 9. Confidence comparison
        if baseline_confidence is not None and current_confidence is not None:
            comparison_result['confidence_decreased'] = current_confidence < baseline_confidence - 0.1
            comparison_result['confidence_change'] = current_confidence - baseline_confidence
        else:
            comparison_result['confidence_decreased'] = False
            comparison_result['confidence_change'] = 0.0

        # Convert all numpy types to Python native types for JSON serialization
        return convert_numpy_types(comparison_result)

    except Exception as e:
        print(f"Error in compare_lesions_full: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'change_detected': False,
            'change_severity': 'error'
        }
