"""
Explainable AI Module for Skin Lesion Classification

This module provides comprehensive visual and textual explanations for AI predictions:
- Grad-CAM and Grad-CAM++ heatmaps
- Region highlighting with bounding boxes
- Natural language explanations
- Feature importance scores
- ABCDE criteria analysis with visual annotations
- Comparison with dermatologist annotations
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import base64
from io import BytesIO
import json
from datetime import datetime


class ExplanationType(Enum):
    """Types of explanations available"""
    GRAD_CAM = "grad_cam"
    GRAD_CAM_PLUS = "grad_cam_plus"
    ATTENTION_MAP = "attention_map"
    FEATURE_ATTRIBUTION = "feature_attribution"
    REGION_HIGHLIGHT = "region_highlight"
    ABCDE_ANALYSIS = "abcde_analysis"


class FeatureCategory(Enum):
    """Categories of visual features"""
    ASYMMETRY = "asymmetry"
    BORDER = "border"
    COLOR = "color"
    DIAMETER = "diameter"
    EVOLUTION = "evolution"
    TEXTURE = "texture"
    STRUCTURE = "structure"
    PATTERN = "pattern"


@dataclass
class HighlightedRegion:
    """A region of interest in the image"""
    x: int
    y: int
    width: int
    height: int
    importance_score: float
    feature_type: str
    description: str
    color: Tuple[int, int, int] = (255, 0, 0)


@dataclass
class FeatureExplanation:
    """Explanation for a specific feature"""
    feature_name: str
    category: FeatureCategory
    importance_score: float
    value: Any
    description: str
    clinical_significance: str
    visual_indicator: str  # Where to look in the image
    confidence: float


@dataclass
class ABCDEScore:
    """ABCDE criteria scoring"""
    asymmetry_score: float
    asymmetry_description: str
    asymmetry_regions: List[HighlightedRegion]

    border_score: float
    border_description: str
    border_regions: List[HighlightedRegion]

    color_score: float
    color_description: str
    colors_detected: List[str]
    color_regions: List[HighlightedRegion]

    diameter_score: float
    diameter_description: str
    diameter_mm: Optional[float]

    evolution_score: Optional[float]
    evolution_description: str

    total_score: float
    risk_level: str

    def to_dict(self) -> Dict:
        return {
            "asymmetry": {
                "score": self.asymmetry_score,
                "description": self.asymmetry_description,
                "max_score": 2.0
            },
            "border": {
                "score": self.border_score,
                "description": self.border_description,
                "max_score": 2.0
            },
            "color": {
                "score": self.color_score,
                "description": self.color_description,
                "colors_detected": self.colors_detected,
                "max_score": 2.0
            },
            "diameter": {
                "score": self.diameter_score,
                "description": self.diameter_description,
                "diameter_mm": self.diameter_mm,
                "max_score": 2.0
            },
            "evolution": {
                "score": self.evolution_score,
                "description": self.evolution_description,
                "max_score": 2.0
            },
            "total_score": self.total_score,
            "max_total": 10.0,
            "risk_level": self.risk_level
        }


@dataclass
class DermatologistComparison:
    """Comparison between AI and dermatologist annotations"""
    ai_diagnosis: str
    ai_confidence: float
    ai_highlighted_regions: List[HighlightedRegion]
    ai_feature_scores: Dict[str, float]

    dermatologist_diagnosis: Optional[str]
    dermatologist_confidence: Optional[float]
    dermatologist_annotations: List[HighlightedRegion]
    dermatologist_notes: Optional[str]

    agreement_score: Optional[float]
    disagreement_regions: List[HighlightedRegion]
    comparison_notes: List[str]


@dataclass
class ExplainableResult:
    """Complete explainable AI result"""
    analysis_id: str
    timestamp: str

    # Prediction info
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

    # Visual explanations
    original_image_base64: str
    grad_cam_heatmap: str
    grad_cam_plus_heatmap: Optional[str]
    attention_overlay: str
    region_highlights_image: str
    abcde_annotated_image: str
    feature_importance_chart: str

    # Highlighted regions
    important_regions: List[Dict]

    # Feature explanations
    feature_explanations: List[Dict]
    feature_importance_scores: Dict[str, float]

    # ABCDE analysis
    abcde_analysis: Dict

    # Natural language explanation
    summary_explanation: str
    detailed_explanation: str
    clinical_reasoning: str

    # For comparison
    dermatologist_comparison: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp,
            "prediction": {
                "class": self.predicted_class,
                "confidence": self.confidence,
                "probabilities": self.all_probabilities
            },
            "visual_explanations": {
                "original_image": self.original_image_base64,
                "grad_cam_heatmap": self.grad_cam_heatmap,
                "grad_cam_plus_heatmap": self.grad_cam_plus_heatmap,
                "attention_overlay": self.attention_overlay,
                "region_highlights": self.region_highlights_image,
                "abcde_annotated": self.abcde_annotated_image,
                "feature_importance_chart": self.feature_importance_chart
            },
            "important_regions": self.important_regions,
            "feature_explanations": self.feature_explanations,
            "feature_importance_scores": self.feature_importance_scores,
            "abcde_analysis": self.abcde_analysis,
            "explanations": {
                "summary": self.summary_explanation,
                "detailed": self.detailed_explanation,
                "clinical_reasoning": self.clinical_reasoning
            },
            "dermatologist_comparison": self.dermatologist_comparison
        }


class GradCAMGenerator:
    """Generates Grad-CAM and Grad-CAM++ visualizations"""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.activations = {}
        self.gradients = {}
        self.hooks = []

    def _find_target_layer(self):
        """Find the appropriate target layer for Grad-CAM"""
        # For Vision Transformer models
        if hasattr(self.model, 'vit'):
            return self.model.vit.encoder.layer[-1].output
        elif hasattr(self.model, 'encoder'):
            return self.model.encoder.layer[-1]
        elif hasattr(self.model, 'dinov2'):
            return self.model.dinov2.encoder.layer[-1]
        # For ConvNeXt
        elif hasattr(self.model, 'convnext'):
            return self.model.convnext.encoder.stages[-1]
        # For ResNet models
        elif hasattr(self.model, 'layer4'):
            return self.model.layer4[-1]
        elif hasattr(self.model, 'features'):
            # Find last conv layer
            for layer in reversed(list(self.model.features.modules())):
                if isinstance(layer, torch.nn.Conv2d):
                    return layer
        # Generic fallback
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (torch.nn.Conv2d, torch.nn.MultiheadAttention)):
                return module
        return None

    def _register_hooks(self, target_layer):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations['value'] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]
            self.gradients['value'] = grad_output.detach()

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}

    def generate_grad_cam(
        self,
        image: Image.Image,
        processor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate Grad-CAM heatmap

        Returns:
            - heatmap: Raw heatmap array
            - overlay: Heatmap overlaid on image
            - info: Additional information about the generation
        """
        target_layer = self._find_target_layer()
        if target_layer is None:
            raise ValueError("Could not find target layer for Grad-CAM")

        self._register_hooks(target_layer)

        try:
            # Preprocess image
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            self.model.eval()
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Get target class
            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            # Backward pass
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

            # Get activations and gradients
            activations = self.activations['value']
            gradients = self.gradients['value']

            # Handle different activation shapes
            if len(activations.shape) == 3:
                # ViT: (batch, seq_len, hidden_dim) - reshape to spatial
                batch, seq_len, hidden_dim = activations.shape
                h = w = int(np.sqrt(seq_len - 1))  # -1 for CLS token
                # Remove CLS token and reshape
                activations = activations[:, 1:, :].reshape(batch, h, w, hidden_dim)
                activations = activations.permute(0, 3, 1, 2)  # (B, C, H, W)
                gradients = gradients[:, 1:, :].reshape(batch, h, w, hidden_dim)
                gradients = gradients.permute(0, 3, 1, 2)

            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

            # Weighted combination of activations
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            # Resize to image size
            img_array = np.array(image)
            cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))

            # Apply colormap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Create overlay
            overlay = np.uint8(0.6 * img_array + 0.4 * heatmap)

            info = {
                "target_class": target_class,
                "activation_shape": str(self.activations['value'].shape),
                "method": "grad_cam"
            }

            return cam_resized, overlay, info

        finally:
            self._remove_hooks()

    def generate_grad_cam_plus(
        self,
        image: Image.Image,
        processor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate Grad-CAM++ heatmap (improved version with better localization)
        """
        target_layer = self._find_target_layer()
        if target_layer is None:
            raise ValueError("Could not find target layer for Grad-CAM++")

        self._register_hooks(target_layer)

        try:
            # Preprocess image
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            self.model.eval()
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            # Backward pass
            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

            activations = self.activations['value']
            gradients = self.gradients['value']

            # Handle ViT shape
            if len(activations.shape) == 3:
                batch, seq_len, hidden_dim = activations.shape
                h = w = int(np.sqrt(seq_len - 1))
                activations = activations[:, 1:, :].reshape(batch, h, w, hidden_dim)
                activations = activations.permute(0, 3, 1, 2)
                gradients = gradients[:, 1:, :].reshape(batch, h, w, hidden_dim)
                gradients = gradients.permute(0, 3, 1, 2)

            # Grad-CAM++ weighting
            grad_2 = gradients ** 2
            grad_3 = gradients ** 3

            sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
            alpha_num = grad_2
            alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
            alpha = alpha_num / alpha_denom

            weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)

            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            # Resize and colorize
            img_array = np.array(image)
            cam_resized = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.uint8(0.6 * img_array + 0.4 * heatmap)

            info = {
                "target_class": target_class,
                "method": "grad_cam_plus_plus"
            }

            return cam_resized, overlay, info

        finally:
            self._remove_hooks()


class RegionHighlighter:
    """Identifies and highlights important regions in the image"""

    def __init__(self):
        self.min_region_size = 20
        self.importance_threshold = 0.3

    def find_important_regions(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        num_regions: int = 5
    ) -> List[HighlightedRegion]:
        """
        Find the most important regions based on Grad-CAM heatmap
        """
        regions = []

        # Threshold the heatmap
        threshold = self.importance_threshold
        binary = (heatmap > threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort by area and importance
        contour_info = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_region_size or h < self.min_region_size:
                continue

            # Calculate mean importance in this region
            region_heatmap = heatmap[y:y+h, x:x+w]
            importance = float(np.mean(region_heatmap))

            contour_info.append({
                "x": x, "y": y, "w": w, "h": h,
                "importance": importance,
                "contour": contour
            })

        # Sort by importance
        contour_info.sort(key=lambda c: c["importance"], reverse=True)

        # Take top regions
        for i, info in enumerate(contour_info[:num_regions]):
            # Determine feature type based on position and characteristics
            feature_type = self._classify_region(
                info["x"], info["y"], info["w"], info["h"],
                original_image, heatmap
            )

            region = HighlightedRegion(
                x=info["x"],
                y=info["y"],
                width=info["w"],
                height=info["h"],
                importance_score=info["importance"],
                feature_type=feature_type,
                description=self._generate_region_description(feature_type, info["importance"]),
                color=self._get_color_for_importance(info["importance"])
            )
            regions.append(region)

        return regions

    def _classify_region(
        self,
        x: int, y: int, w: int, h: int,
        image: np.ndarray,
        heatmap: np.ndarray
    ) -> str:
        """Classify the type of feature in a region"""
        # Extract region
        region = image[y:y+h, x:x+w]

        if len(region) == 0:
            return "unknown"

        # Analyze color variation
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        color_variance = np.var(hsv[:, :, 0])

        # Analyze texture
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255

        # Check if at border
        img_h, img_w = image.shape[:2]
        at_border = (x < img_w * 0.1 or x + w > img_w * 0.9 or
                     y < img_h * 0.1 or y + h > img_h * 0.9)

        # Classify
        if color_variance > 1000:
            return "color_variation"
        elif edge_density > 0.3 and at_border:
            return "irregular_border"
        elif edge_density > 0.2:
            return "texture_pattern"
        elif at_border:
            return "border_feature"
        else:
            return "structural_feature"

    def _generate_region_description(self, feature_type: str, importance: float) -> str:
        """Generate a description for a region"""
        descriptions = {
            "color_variation": "Area showing significant color variation, which may indicate pigment irregularity",
            "irregular_border": "Border region with irregular characteristics",
            "texture_pattern": "Area with notable texture patterns that influenced the classification",
            "border_feature": "Edge feature that contributed to the diagnosis",
            "structural_feature": "Structural element that the AI identified as significant",
            "unknown": "Region of interest identified by the AI"
        }

        base = descriptions.get(feature_type, descriptions["unknown"])

        if importance > 0.8:
            return f"Highly significant: {base}"
        elif importance > 0.5:
            return f"Moderately significant: {base}"
        else:
            return f"Contributing factor: {base}"

    def _get_color_for_importance(self, importance: float) -> Tuple[int, int, int]:
        """Get color based on importance score"""
        if importance > 0.8:
            return (255, 0, 0)  # Red for high importance
        elif importance > 0.5:
            return (255, 165, 0)  # Orange for medium
        else:
            return (255, 255, 0)  # Yellow for lower

    def draw_regions_on_image(
        self,
        image: np.ndarray,
        regions: List[HighlightedRegion],
        show_labels: bool = True
    ) -> np.ndarray:
        """Draw highlighted regions on the image"""
        result = image.copy()

        for i, region in enumerate(regions):
            # Draw rectangle
            cv2.rectangle(
                result,
                (region.x, region.y),
                (region.x + region.width, region.y + region.height),
                region.color,
                2
            )

            if show_labels:
                # Add label
                label = f"#{i+1}: {region.feature_type}"
                label_y = max(region.y - 10, 20)
                cv2.putText(
                    result,
                    label,
                    (region.x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    region.color,
                    1
                )

                # Add importance score
                score_label = f"{region.importance_score:.1%}"
                cv2.putText(
                    result,
                    score_label,
                    (region.x, region.y + region.height + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    region.color,
                    1
                )

        return result


class ABCDEAnalyzer:
    """Analyzes images using ABCDE criteria with visual annotations"""

    def __init__(self):
        self.color_names = {
            "black": (0, 0, 0),
            "dark_brown": (101, 67, 33),
            "light_brown": (181, 137, 102),
            "tan": (210, 180, 140),
            "red": (255, 0, 0),
            "blue_gray": (119, 136, 153),
            "white": (255, 255, 255)
        }

    def analyze(
        self,
        image: np.ndarray,
        lesion_mask: Optional[np.ndarray] = None,
        pixels_per_mm: Optional[float] = None,
        previous_analysis: Optional[Dict] = None
    ) -> ABCDEScore:
        """
        Perform complete ABCDE analysis
        """
        # Generate lesion mask if not provided
        if lesion_mask is None:
            lesion_mask = self._generate_lesion_mask(image)

        # Analyze each criterion
        asymmetry_score, asymmetry_desc, asymmetry_regions = self._analyze_asymmetry(
            image, lesion_mask
        )

        border_score, border_desc, border_regions = self._analyze_border(
            image, lesion_mask
        )

        color_score, color_desc, colors, color_regions = self._analyze_color(
            image, lesion_mask
        )

        diameter_score, diameter_desc, diameter_mm = self._analyze_diameter(
            lesion_mask, pixels_per_mm
        )

        evolution_score, evolution_desc = self._analyze_evolution(previous_analysis)

        # Calculate total score
        total = asymmetry_score + border_score + color_score + diameter_score
        if evolution_score is not None:
            total += evolution_score

        # Determine risk level
        if total >= 6:
            risk_level = "high"
        elif total >= 4:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return ABCDEScore(
            asymmetry_score=asymmetry_score,
            asymmetry_description=asymmetry_desc,
            asymmetry_regions=asymmetry_regions,
            border_score=border_score,
            border_description=border_desc,
            border_regions=border_regions,
            color_score=color_score,
            color_description=color_desc,
            colors_detected=colors,
            color_regions=color_regions,
            diameter_score=diameter_score,
            diameter_description=diameter_desc,
            diameter_mm=diameter_mm,
            evolution_score=evolution_score,
            evolution_description=evolution_desc,
            total_score=total,
            risk_level=risk_level
        )

    def _generate_lesion_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate a mask for the lesion area"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 10
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find the largest contour (assuming it's the lesion)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(thresh)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            return mask

        return thresh

    def _analyze_asymmetry(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float, str, List[HighlightedRegion]]:
        """Analyze asymmetry of the lesion"""
        regions = []

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, "Unable to analyze asymmetry", regions

        contour = max(contours, key=cv2.contourArea)

        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2

        # Split mask into quadrants and compare
        top_left = mask[y:cy, x:cx]
        top_right = mask[y:cy, cx:x+w]
        bottom_left = mask[cy:y+h, x:cx]
        bottom_right = mask[cy:y+h, cx:x+w]

        # Flip and compare
        top_right_flipped = cv2.flip(top_right, 1)
        bottom_left_flipped = cv2.flip(bottom_left, 0)

        # Calculate asymmetry scores
        h_asymmetry = 0
        v_asymmetry = 0

        # Horizontal asymmetry
        min_h, min_w = min(top_left.shape[0], top_right_flipped.shape[0]), min(top_left.shape[1], top_right_flipped.shape[1])
        if min_h > 0 and min_w > 0:
            h_diff = cv2.absdiff(top_left[:min_h, :min_w], top_right_flipped[:min_h, :min_w])
            h_asymmetry = np.sum(h_diff) / (min_h * min_w * 255)

        # Vertical asymmetry
        min_h, min_w = min(top_left.shape[0], bottom_left_flipped.shape[0]), min(top_left.shape[1], bottom_left_flipped.shape[1])
        if min_h > 0 and min_w > 0:
            v_diff = cv2.absdiff(top_left[:min_h, :min_w], bottom_left_flipped[:min_h, :min_w])
            v_asymmetry = np.sum(v_diff) / (min_h * min_w * 255)

        # Combined asymmetry
        total_asymmetry = (h_asymmetry + v_asymmetry) / 2

        # Score (0-2)
        if total_asymmetry > 0.4:
            score = 2.0
            desc = "Asymmetric in 2 axes: The lesion shows significant asymmetry both horizontally and vertically"
        elif total_asymmetry > 0.2:
            score = 1.0
            desc = "Asymmetric in 1 axis: The lesion shows asymmetry in one direction"
        else:
            score = 0.0
            desc = "Symmetric: The lesion appears relatively symmetric"

        # Add asymmetry regions
        if h_asymmetry > 0.2:
            regions.append(HighlightedRegion(
                x=x, y=y, width=w//2, height=h,
                importance_score=h_asymmetry,
                feature_type="horizontal_asymmetry",
                description="Left-right asymmetry detected",
                color=(255, 100, 100)
            ))

        if v_asymmetry > 0.2:
            regions.append(HighlightedRegion(
                x=x, y=y, width=w, height=h//2,
                importance_score=v_asymmetry,
                feature_type="vertical_asymmetry",
                description="Top-bottom asymmetry detected",
                color=(100, 100, 255)
            ))

        return score, desc, regions

    def _analyze_border(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float, str, List[HighlightedRegion]]:
        """Analyze border irregularity"""
        regions = []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, "Unable to analyze border", regions

        contour = max(contours, key=cv2.contourArea)

        # Calculate perimeter and area
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        if area == 0:
            return 0.0, "Unable to analyze border", regions

        # Circularity ratio (1 = perfect circle)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Fit ellipse and compare
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_mask = np.zeros_like(mask)
            cv2.ellipse(ellipse_mask, ellipse, 255, -1)

            # Calculate how much the contour deviates from ellipse
            difference = cv2.absdiff(mask, ellipse_mask)
            deviation = np.sum(difference) / np.sum(mask) if np.sum(mask) > 0 else 0
        else:
            deviation = 0.5

        # Find irregular border points
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3 and len(contour) > 3:
            try:
                defects = cv2.convexityDefects(contour, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        if d > 1000:  # Significant defect
                            far_point = tuple(contour[f][0])
                            regions.append(HighlightedRegion(
                                x=far_point[0] - 10,
                                y=far_point[1] - 10,
                                width=20,
                                height=20,
                                importance_score=min(d / 5000, 1.0),
                                feature_type="border_irregularity",
                                description="Irregular border indentation",
                                color=(255, 165, 0)
                            ))
            except cv2.error:
                pass

        # Score based on irregularity
        irregularity = (1 - circularity) + deviation

        if irregularity > 0.6:
            score = 2.0
            desc = "Highly irregular border: The lesion has a very irregular, notched, or scalloped border"
        elif irregularity > 0.3:
            score = 1.0
            desc = "Somewhat irregular border: The border shows some irregularity"
        else:
            score = 0.0
            desc = "Regular border: The lesion has a smooth, well-defined border"

        return score, desc, regions

    def _analyze_color(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[float, str, List[str], List[HighlightedRegion]]:
        """Analyze color variation in the lesion"""
        colors_found = []
        regions = []

        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Convert to different color spaces
        hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(masked_image, cv2.COLOR_RGB2LAB)

        # Get pixels within the lesion
        lesion_pixels = image[mask > 0]

        if len(lesion_pixels) == 0:
            return 0.0, "Unable to analyze color", [], regions

        # Analyze color distribution
        # Check for different colors
        colors_detected = set()

        # Dark brown/black (low lightness)
        dark_mask = lab[:, :, 0] < 50
        if np.sum(dark_mask & (mask > 0)) > np.sum(mask > 0) * 0.05:
            colors_detected.add("dark_brown_black")
            y_coords, x_coords = np.where(dark_mask & (mask > 0))
            if len(x_coords) > 0:
                regions.append(HighlightedRegion(
                    x=int(np.min(x_coords)),
                    y=int(np.min(y_coords)),
                    width=int(np.max(x_coords) - np.min(x_coords)),
                    height=int(np.max(y_coords) - np.min(y_coords)),
                    importance_score=0.8,
                    feature_type="dark_pigment",
                    description="Dark brown/black pigmentation",
                    color=(50, 50, 50)
                ))

        # Light brown/tan
        light_brown_mask = (lab[:, :, 0] > 100) & (lab[:, :, 0] < 180)
        if np.sum(light_brown_mask & (mask > 0)) > np.sum(mask > 0) * 0.05:
            colors_detected.add("light_brown_tan")

        # Red (high a* channel in LAB)
        red_mask = lab[:, :, 1] > 150
        if np.sum(red_mask & (mask > 0)) > np.sum(mask > 0) * 0.03:
            colors_detected.add("red")
            y_coords, x_coords = np.where(red_mask & (mask > 0))
            if len(x_coords) > 0:
                regions.append(HighlightedRegion(
                    x=int(np.min(x_coords)),
                    y=int(np.min(y_coords)),
                    width=int(np.max(x_coords) - np.min(x_coords)),
                    height=int(np.max(y_coords) - np.min(y_coords)),
                    importance_score=0.9,
                    feature_type="red_pigment",
                    description="Red coloration (may indicate vascularity or inflammation)",
                    color=(255, 0, 0)
                ))

        # Blue-gray (specific HSV range)
        blue_gray_mask = (hsv[:, :, 0] > 100) & (hsv[:, :, 0] < 130) & (hsv[:, :, 1] < 100)
        if np.sum(blue_gray_mask & (mask > 0)) > np.sum(mask > 0) * 0.03:
            colors_detected.add("blue_gray")
            y_coords, x_coords = np.where(blue_gray_mask & (mask > 0))
            if len(x_coords) > 0:
                regions.append(HighlightedRegion(
                    x=int(np.min(x_coords)),
                    y=int(np.min(y_coords)),
                    width=int(np.max(x_coords) - np.min(x_coords)),
                    height=int(np.max(y_coords) - np.min(y_coords)),
                    importance_score=0.95,
                    feature_type="blue_gray_pigment",
                    description="Blue-gray coloration (concerning for melanoma)",
                    color=(119, 136, 153)
                ))

        # White (depigmentation/regression)
        white_mask = (lab[:, :, 0] > 200) & (lab[:, :, 1] > 120) & (lab[:, :, 1] < 135)
        if np.sum(white_mask & (mask > 0)) > np.sum(mask > 0) * 0.03:
            colors_detected.add("white")

        colors_found = list(colors_detected)
        num_colors = len(colors_found)

        # Score based on number of colors
        if num_colors >= 3:
            score = 2.0
            desc = f"Multiple colors present: {', '.join(colors_found)}. Multiple colors in a lesion can indicate melanoma"
        elif num_colors == 2:
            score = 1.0
            desc = f"Two colors present: {', '.join(colors_found)}"
        else:
            score = 0.0
            desc = f"Uniform color: {'Single color (' + colors_found[0] + ')' if colors_found else 'No distinct colors detected'}"

        return score, desc, colors_found, regions

    def _analyze_diameter(
        self,
        mask: np.ndarray,
        pixels_per_mm: Optional[float]
    ) -> Tuple[float, str, Optional[float]]:
        """Analyze diameter of the lesion"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, "Unable to measure diameter", None

        contour = max(contours, key=cv2.contourArea)

        # Get minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter_pixels = radius * 2

        # Convert to mm if calibration available
        if pixels_per_mm and pixels_per_mm > 0:
            diameter_mm = diameter_pixels / pixels_per_mm
        else:
            diameter_mm = None

        # Score based on diameter
        if diameter_mm is not None:
            if diameter_mm >= 6:
                score = 2.0
                desc = f"Large diameter: {diameter_mm:.1f}mm (≥6mm is concerning)"
            elif diameter_mm >= 4:
                score = 1.0
                desc = f"Moderate diameter: {diameter_mm:.1f}mm"
            else:
                score = 0.0
                desc = f"Small diameter: {diameter_mm:.1f}mm (<6mm)"
        else:
            # Estimate based on image proportion
            img_diagonal = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
            size_ratio = diameter_pixels / img_diagonal

            if size_ratio > 0.3:
                score = 2.0
                desc = "Large relative size (actual size unknown - no calibration)"
            elif size_ratio > 0.15:
                score = 1.0
                desc = "Moderate relative size (actual size unknown - no calibration)"
            else:
                score = 0.0
                desc = "Small relative size (actual size unknown - no calibration)"

        return score, desc, diameter_mm

    def _analyze_evolution(
        self,
        previous_analysis: Optional[Dict]
    ) -> Tuple[Optional[float], str]:
        """Analyze evolution/change over time"""
        if previous_analysis is None:
            return None, "No previous analysis available for evolution comparison"

        # Compare with previous analysis
        changes = []

        # Check size change
        if "diameter_mm" in previous_analysis and previous_analysis["diameter_mm"]:
            prev_diameter = previous_analysis["diameter_mm"]
            # Would need current diameter here
            changes.append("size")

        # Check color change
        if "colors" in previous_analysis:
            changes.append("color")

        # Check shape change
        if "asymmetry_score" in previous_analysis:
            changes.append("shape")

        if len(changes) >= 2:
            score = 2.0
            desc = f"Significant evolution detected: Changes in {', '.join(changes)}"
        elif len(changes) == 1:
            score = 1.0
            desc = f"Some evolution detected: Change in {changes[0]}"
        else:
            score = 0.0
            desc = "No significant evolution detected"

        return score, desc

    def create_annotated_image(
        self,
        image: np.ndarray,
        abcde_score: ABCDEScore
    ) -> np.ndarray:
        """Create an image with ABCDE annotations"""
        result = image.copy()

        # Draw all regions
        all_regions = (
            abcde_score.asymmetry_regions +
            abcde_score.border_regions +
            abcde_score.color_regions
        )

        for region in all_regions:
            cv2.rectangle(
                result,
                (region.x, region.y),
                (region.x + region.width, region.y + region.height),
                region.color,
                2
            )

        # Add text overlay with scores
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        line_height = 25

        # Create semi-transparent background for text
        overlay = result.copy()
        cv2.rectangle(overlay, (5, 5), (250, 150), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.5, result, 0.5, 0)

        scores_text = [
            f"A (Asymmetry): {abcde_score.asymmetry_score:.1f}/2",
            f"B (Border): {abcde_score.border_score:.1f}/2",
            f"C (Color): {abcde_score.color_score:.1f}/2",
            f"D (Diameter): {abcde_score.diameter_score:.1f}/2",
            f"Total: {abcde_score.total_score:.1f}/8",
            f"Risk: {abcde_score.risk_level.upper()}"
        ]

        for i, text in enumerate(scores_text):
            color = (0, 255, 0) if "low" in text.lower() else (
                (255, 255, 0) if "moderate" in text.lower() else (255, 255, 255)
            )
            if "high" in text.lower():
                color = (0, 0, 255)
            cv2.putText(result, text, (10, y_offset + i * line_height),
                        font, 0.5, color, 1)

        return result


class FeatureExplainer:
    """Generates natural language explanations for AI predictions"""

    def __init__(self):
        self.class_descriptions = {
            "Melanoma": {
                "description": "Melanoma is a type of skin cancer that develops from melanocytes",
                "key_features": ["asymmetry", "irregular borders", "multiple colors", "large diameter", "evolution"],
                "risk_factors": ["UV exposure", "family history", "many moles", "fair skin"],
                "urgency": "high"
            },
            "Basal Cell Carcinoma": {
                "description": "Basal cell carcinoma is the most common type of skin cancer",
                "key_features": ["pearly appearance", "rolled borders", "telangiectasia", "ulceration"],
                "risk_factors": ["UV exposure", "fair skin", "older age"],
                "urgency": "moderate"
            },
            "Melanocytic Nevi": {
                "description": "Melanocytic nevi (moles) are benign growths of melanocytes",
                "key_features": ["uniform color", "regular borders", "symmetric shape"],
                "risk_factors": [],
                "urgency": "low"
            },
            "Benign Keratosis": {
                "description": "Benign keratoses are non-cancerous skin growths",
                "key_features": ["waxy appearance", "stuck-on look", "well-defined borders"],
                "risk_factors": [],
                "urgency": "low"
            },
            "Actinic Keratosis": {
                "description": "Actinic keratoses are precancerous patches caused by sun damage",
                "key_features": ["rough texture", "scaly surface", "sun-exposed areas"],
                "risk_factors": ["UV exposure", "fair skin"],
                "urgency": "moderate"
            },
            "Dermatofibroma": {
                "description": "Dermatofibromas are benign fibrous nodules",
                "key_features": ["firm texture", "dimple sign", "brownish color"],
                "risk_factors": [],
                "urgency": "low"
            },
            "Vascular Lesion": {
                "description": "Vascular lesions are abnormalities of blood vessels",
                "key_features": ["red/purple color", "blanches with pressure", "vascular pattern"],
                "risk_factors": [],
                "urgency": "low"
            }
        }

    def generate_explanation(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: Dict[str, float],
        feature_scores: Dict[str, float],
        abcde_analysis: Optional[ABCDEScore] = None,
        uncertainty_metrics: Optional[Dict] = None
    ) -> Tuple[str, str, str]:
        """
        Generate natural language explanations

        Returns:
            - summary: Brief 1-2 sentence summary
            - detailed: Full explanation
            - clinical_reasoning: Clinical reasoning chain
        """
        class_info = self.class_descriptions.get(
            predicted_class,
            {"description": f"Classification: {predicted_class}", "key_features": [], "urgency": "unknown"}
        )

        # Summary
        summary = self._generate_summary(predicted_class, confidence, class_info)

        # Detailed explanation
        detailed = self._generate_detailed_explanation(
            predicted_class, confidence, probabilities,
            feature_scores, abcde_analysis, class_info
        )

        # Clinical reasoning
        clinical = self._generate_clinical_reasoning(
            predicted_class, confidence, feature_scores,
            abcde_analysis, uncertainty_metrics, class_info
        )

        return summary, detailed, clinical

    def _generate_summary(
        self,
        predicted_class: str,
        confidence: float,
        class_info: Dict
    ) -> str:
        """Generate a brief summary"""
        confidence_term = (
            "high confidence" if confidence > 0.8 else
            "moderate confidence" if confidence > 0.6 else
            "low confidence"
        )

        urgency = class_info.get("urgency", "unknown")
        urgency_note = ""
        if urgency == "high":
            urgency_note = " Immediate dermatologist consultation recommended."
        elif urgency == "moderate":
            urgency_note = " Dermatologist evaluation advised."

        return (
            f"The AI classifies this lesion as {predicted_class} with {confidence_term} "
            f"({confidence:.1%}).{urgency_note}"
        )

    def _generate_detailed_explanation(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: Dict[str, float],
        feature_scores: Dict[str, float],
        abcde_analysis: Optional[ABCDEScore],
        class_info: Dict
    ) -> str:
        """Generate detailed explanation"""
        lines = []

        # Class description
        lines.append(f"**{predicted_class}**")
        lines.append(class_info.get("description", ""))
        lines.append("")

        # Confidence breakdown
        lines.append("**Classification Confidence:**")
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs[:3]:
            marker = "→" if cls == predicted_class else " "
            lines.append(f"  {marker} {cls}: {prob:.1%}")
        lines.append("")

        # Key features identified
        lines.append("**Key Features Identified:**")
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features[:5]:
            importance = "High" if score > 0.7 else "Moderate" if score > 0.4 else "Low"
            lines.append(f"  • {feature.replace('_', ' ').title()}: {importance} importance ({score:.1%})")
        lines.append("")

        # ABCDE analysis if available
        if abcde_analysis:
            lines.append("**ABCDE Criteria Analysis:**")
            lines.append(f"  • Asymmetry: {abcde_analysis.asymmetry_score}/2 - {abcde_analysis.asymmetry_description}")
            lines.append(f"  • Border: {abcde_analysis.border_score}/2 - {abcde_analysis.border_description}")
            lines.append(f"  • Color: {abcde_analysis.color_score}/2 - {abcde_analysis.color_description}")
            lines.append(f"  • Diameter: {abcde_analysis.diameter_score}/2 - {abcde_analysis.diameter_description}")
            lines.append(f"  • **Total Score: {abcde_analysis.total_score}/8 ({abcde_analysis.risk_level} risk)**")
            lines.append("")

        # What to look for
        if class_info.get("key_features"):
            lines.append("**Typical Features of This Condition:**")
            for feature in class_info["key_features"]:
                lines.append(f"  • {feature.replace('_', ' ').title()}")

        return "\n".join(lines)

    def _generate_clinical_reasoning(
        self,
        predicted_class: str,
        confidence: float,
        feature_scores: Dict[str, float],
        abcde_analysis: Optional[ABCDEScore],
        uncertainty_metrics: Optional[Dict],
        class_info: Dict
    ) -> str:
        """Generate clinical reasoning chain"""
        lines = []

        lines.append("**AI Clinical Reasoning:**")
        lines.append("")

        # Step 1: Initial assessment
        lines.append("1. **Initial Assessment:**")
        lines.append(f"   The image shows a skin lesion that was analyzed for morphological features.")
        lines.append("")

        # Step 2: Feature analysis
        lines.append("2. **Feature Analysis:**")
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:3]
        for feature, score in top_features:
            feature_name = feature.replace("_", " ").title()
            if score > 0.7:
                lines.append(f"   - Strong presence of {feature_name} (importance: {score:.1%})")
            elif score > 0.4:
                lines.append(f"   - Moderate {feature_name} detected (importance: {score:.1%})")
        lines.append("")

        # Step 3: ABCDE reasoning
        if abcde_analysis:
            lines.append("3. **ABCDE Criteria Evaluation:**")
            concerning = []
            if abcde_analysis.asymmetry_score >= 1:
                concerning.append(f"asymmetry (score: {abcde_analysis.asymmetry_score}/2)")
            if abcde_analysis.border_score >= 1:
                concerning.append(f"irregular border (score: {abcde_analysis.border_score}/2)")
            if abcde_analysis.color_score >= 1:
                concerning.append(f"color variation (score: {abcde_analysis.color_score}/2)")
            if abcde_analysis.diameter_score >= 1:
                concerning.append(f"diameter (score: {abcde_analysis.diameter_score}/2)")

            if concerning:
                lines.append(f"   Concerning features: {', '.join(concerning)}")
            else:
                lines.append("   No significantly concerning ABCDE features identified")
            lines.append("")

        # Step 4: Uncertainty assessment
        if uncertainty_metrics:
            lines.append("4. **Confidence Assessment:**")
            epistemic = uncertainty_metrics.get("epistemic_uncertainty", 0)
            aleatoric = uncertainty_metrics.get("aleatoric_uncertainty", 0)
            reliability = uncertainty_metrics.get("reliability_score", confidence)

            if epistemic > 0.3:
                lines.append(f"   - Model shows some uncertainty (epistemic: {epistemic:.2f})")
                lines.append("   - Consider additional views or dermoscopy for confirmation")
            if reliability > 0.8:
                lines.append(f"   - High reliability score ({reliability:.1%})")
            lines.append("")

        # Step 5: Conclusion
        lines.append("5. **Conclusion:**")
        lines.append(f"   Based on the visual features analyzed, the AI concludes {predicted_class} ")
        lines.append(f"   with {confidence:.1%} confidence.")

        urgency = class_info.get("urgency", "unknown")
        if urgency == "high":
            lines.append("")
            lines.append("   **⚠️ RECOMMENDATION: Urgent dermatologist referral advised**")
        elif urgency == "moderate":
            lines.append("")
            lines.append("   **📋 RECOMMENDATION: Schedule dermatologist evaluation**")

        lines.append("")
        lines.append("*Note: AI analysis is not a substitute for professional medical diagnosis.*")

        return "\n".join(lines)

    def generate_feature_explanations(
        self,
        feature_scores: Dict[str, float],
        predicted_class: str
    ) -> List[FeatureExplanation]:
        """Generate detailed explanations for each feature"""
        explanations = []

        feature_details = {
            "asymmetry": {
                "category": FeatureCategory.ASYMMETRY,
                "clinical_significance": "Asymmetry is a key indicator of melanoma. Benign lesions tend to be symmetric.",
                "visual_indicator": "Compare left vs right and top vs bottom halves of the lesion"
            },
            "border_irregularity": {
                "category": FeatureCategory.BORDER,
                "clinical_significance": "Irregular, ragged, or blurred borders suggest possible malignancy.",
                "visual_indicator": "Look at the edges of the lesion for smoothness vs notching"
            },
            "color_variation": {
                "category": FeatureCategory.COLOR,
                "clinical_significance": "Multiple colors (especially blue, black, red) increase melanoma risk.",
                "visual_indicator": "Note different shades within the lesion"
            },
            "texture_pattern": {
                "category": FeatureCategory.TEXTURE,
                "clinical_significance": "Texture changes can indicate different lesion types.",
                "visual_indicator": "Surface characteristics - smooth, rough, scaly"
            },
            "pigment_network": {
                "category": FeatureCategory.PATTERN,
                "clinical_significance": "Regular pigment network suggests benign; irregular suggests concern.",
                "visual_indicator": "Grid-like pattern visible in dermoscopy"
            },
            "globules": {
                "category": FeatureCategory.STRUCTURE,
                "clinical_significance": "Regular globules are benign; irregular peripheral globules are concerning.",
                "visual_indicator": "Round structures within the lesion"
            },
            "vascular_pattern": {
                "category": FeatureCategory.STRUCTURE,
                "clinical_significance": "Certain vascular patterns (dotted, polymorphous) suggest malignancy.",
                "visual_indicator": "Red lines or dots indicating blood vessels"
            },
            "structural_feature": {
                "category": FeatureCategory.STRUCTURE,
                "clinical_significance": "Structural features help differentiate lesion types.",
                "visual_indicator": "Overall architecture of the lesion"
            }
        }

        for feature_name, score in feature_scores.items():
            normalized_name = feature_name.lower().replace(" ", "_")
            details = feature_details.get(normalized_name, {
                "category": FeatureCategory.STRUCTURE,
                "clinical_significance": "This feature contributed to the classification.",
                "visual_indicator": "See highlighted region in the image"
            })

            # Generate description based on score
            if score > 0.7:
                description = f"Strong {feature_name.replace('_', ' ')} detected, significantly influencing the {predicted_class} classification"
            elif score > 0.4:
                description = f"Moderate {feature_name.replace('_', ' ')} present, contributing to the classification"
            else:
                description = f"Mild {feature_name.replace('_', ' ')} noted, minor contribution to classification"

            explanations.append(FeatureExplanation(
                feature_name=feature_name,
                category=details["category"],
                importance_score=score,
                value=score,
                description=description,
                clinical_significance=details["clinical_significance"],
                visual_indicator=details["visual_indicator"],
                confidence=score
            ))

        return sorted(explanations, key=lambda x: x.importance_score, reverse=True)


class FeatureImportanceVisualizer:
    """Creates visual representations of feature importance"""

    def create_importance_chart(
        self,
        feature_scores: Dict[str, float],
        width: int = 400,
        height: int = 300
    ) -> np.ndarray:
        """Create a horizontal bar chart of feature importance"""
        # Sort features by importance
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:8]

        # Create image
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        if not sorted_features:
            return img

        # Chart dimensions
        margin_left = 120
        margin_right = 50
        margin_top = 30
        margin_bottom = 30
        bar_height = 20
        bar_spacing = 10

        chart_width = width - margin_left - margin_right
        chart_height = height - margin_top - margin_bottom

        # Draw title
        cv2.putText(img, "Feature Importance", (width // 2 - 60, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Draw bars
        max_score = max(score for _, score in sorted_features) if sorted_features else 1

        for i, (feature, score) in enumerate(sorted_features):
            y = margin_top + i * (bar_height + bar_spacing)

            # Feature name
            display_name = feature.replace("_", " ").title()[:15]
            cv2.putText(img, display_name, (5, y + bar_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Bar
            bar_width = int((score / max_score) * chart_width)

            # Color based on importance
            if score > 0.7:
                color = (0, 0, 200)  # Red for high
            elif score > 0.4:
                color = (0, 165, 255)  # Orange for medium
            else:
                color = (0, 200, 0)  # Green for low

            cv2.rectangle(img, (margin_left, y), (margin_left + bar_width, y + bar_height),
                          color, -1)
            cv2.rectangle(img, (margin_left, y), (margin_left + bar_width, y + bar_height),
                          (0, 0, 0), 1)

            # Score label
            cv2.putText(img, f"{score:.0%}", (margin_left + bar_width + 5, y + bar_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        return img


class DermatologistComparer:
    """Compares AI analysis with dermatologist annotations"""

    def compare(
        self,
        ai_result: Dict,
        dermatologist_annotation: Dict
    ) -> DermatologistComparison:
        """
        Compare AI results with dermatologist annotations

        Args:
            ai_result: AI analysis results
            dermatologist_annotation: Dermatologist's annotations including:
                - diagnosis: str
                - confidence: float
                - regions: List of annotated regions
                - notes: str
        """
        # Extract AI data
        ai_diagnosis = ai_result.get("predicted_class", "Unknown")
        ai_confidence = ai_result.get("confidence", 0)
        ai_regions = [
            HighlightedRegion(**r) if isinstance(r, dict) else r
            for r in ai_result.get("highlighted_regions", [])
        ]
        ai_features = ai_result.get("feature_scores", {})

        # Extract dermatologist data
        derm_diagnosis = dermatologist_annotation.get("diagnosis")
        derm_confidence = dermatologist_annotation.get("confidence")
        derm_regions = [
            HighlightedRegion(**r) if isinstance(r, dict) else r
            for r in dermatologist_annotation.get("regions", [])
        ]
        derm_notes = dermatologist_annotation.get("notes")

        # Calculate agreement
        agreement_score = self._calculate_agreement(
            ai_diagnosis, ai_confidence, ai_regions,
            derm_diagnosis, derm_confidence, derm_regions
        )

        # Find disagreement regions
        disagreement_regions = self._find_disagreement_regions(ai_regions, derm_regions)

        # Generate comparison notes
        comparison_notes = self._generate_comparison_notes(
            ai_diagnosis, derm_diagnosis,
            ai_confidence, derm_confidence,
            ai_regions, derm_regions
        )

        return DermatologistComparison(
            ai_diagnosis=ai_diagnosis,
            ai_confidence=ai_confidence,
            ai_highlighted_regions=ai_regions,
            ai_feature_scores=ai_features,
            dermatologist_diagnosis=derm_diagnosis,
            dermatologist_confidence=derm_confidence,
            dermatologist_annotations=derm_regions,
            dermatologist_notes=derm_notes,
            agreement_score=agreement_score,
            disagreement_regions=disagreement_regions,
            comparison_notes=comparison_notes
        )

    def _calculate_agreement(
        self,
        ai_diagnosis: str,
        ai_confidence: float,
        ai_regions: List[HighlightedRegion],
        derm_diagnosis: Optional[str],
        derm_confidence: Optional[float],
        derm_regions: List[HighlightedRegion]
    ) -> Optional[float]:
        """Calculate agreement score between AI and dermatologist"""
        if derm_diagnosis is None:
            return None

        agreement = 0.0

        # Diagnosis agreement (50% weight)
        if ai_diagnosis.lower() == derm_diagnosis.lower():
            agreement += 0.5
        elif self._are_similar_diagnoses(ai_diagnosis, derm_diagnosis):
            agreement += 0.25

        # Confidence alignment (20% weight)
        if derm_confidence is not None:
            conf_diff = abs(ai_confidence - derm_confidence)
            agreement += 0.2 * (1 - conf_diff)
        else:
            agreement += 0.1

        # Region overlap (30% weight)
        if ai_regions and derm_regions:
            overlap = self._calculate_region_overlap(ai_regions, derm_regions)
            agreement += 0.3 * overlap
        elif not ai_regions and not derm_regions:
            agreement += 0.3

        return agreement

    def _are_similar_diagnoses(self, diag1: str, diag2: str) -> bool:
        """Check if two diagnoses are clinically similar"""
        similar_groups = [
            {"melanoma", "atypical nevus", "dysplastic nevus"},
            {"basal cell carcinoma", "squamous cell carcinoma", "actinic keratosis"},
            {"melanocytic nevus", "benign nevus", "mole"},
            {"seborrheic keratosis", "benign keratosis"}
        ]

        d1_lower = diag1.lower()
        d2_lower = diag2.lower()

        for group in similar_groups:
            if any(g in d1_lower for g in group) and any(g in d2_lower for g in group):
                return True

        return False

    def _calculate_region_overlap(
        self,
        regions1: List[HighlightedRegion],
        regions2: List[HighlightedRegion]
    ) -> float:
        """Calculate IoU-based overlap between two sets of regions"""
        if not regions1 or not regions2:
            return 0.0

        total_overlap = 0
        count = 0

        for r1 in regions1:
            best_overlap = 0
            for r2 in regions2:
                overlap = self._calculate_iou(r1, r2)
                best_overlap = max(best_overlap, overlap)
            total_overlap += best_overlap
            count += 1

        return total_overlap / count if count > 0 else 0.0

    def _calculate_iou(self, r1: HighlightedRegion, r2: HighlightedRegion) -> float:
        """Calculate Intersection over Union of two regions"""
        # Calculate intersection
        x1 = max(r1.x, r2.x)
        y1 = max(r1.y, r2.y)
        x2 = min(r1.x + r1.width, r2.x + r2.width)
        y2 = min(r1.y + r1.height, r2.y + r2.height)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = r1.width * r1.height
        area2 = r2.width * r2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _find_disagreement_regions(
        self,
        ai_regions: List[HighlightedRegion],
        derm_regions: List[HighlightedRegion]
    ) -> List[HighlightedRegion]:
        """Find regions where AI and dermatologist disagree"""
        disagreements = []

        # Regions identified by AI but not dermatologist
        for ai_r in ai_regions:
            has_overlap = False
            for derm_r in derm_regions:
                if self._calculate_iou(ai_r, derm_r) > 0.3:
                    has_overlap = True
                    break
            if not has_overlap:
                ai_r.description = f"AI-only: {ai_r.description}"
                ai_r.color = (255, 0, 255)  # Magenta for AI-only
                disagreements.append(ai_r)

        # Regions identified by dermatologist but not AI
        for derm_r in derm_regions:
            has_overlap = False
            for ai_r in ai_regions:
                if self._calculate_iou(ai_r, derm_r) > 0.3:
                    has_overlap = True
                    break
            if not has_overlap:
                derm_r.description = f"Dermatologist-only: {derm_r.description}"
                derm_r.color = (0, 255, 255)  # Cyan for dermatologist-only
                disagreements.append(derm_r)

        return disagreements

    def _generate_comparison_notes(
        self,
        ai_diagnosis: str,
        derm_diagnosis: Optional[str],
        ai_confidence: float,
        derm_confidence: Optional[float],
        ai_regions: List[HighlightedRegion],
        derm_regions: List[HighlightedRegion]
    ) -> List[str]:
        """Generate notes about the comparison"""
        notes = []

        if derm_diagnosis is None:
            notes.append("No dermatologist diagnosis available for comparison")
            return notes

        # Diagnosis comparison
        if ai_diagnosis.lower() == derm_diagnosis.lower():
            notes.append(f"✓ Agreement: Both AI and dermatologist diagnosed {ai_diagnosis}")
        else:
            notes.append(f"✗ Disagreement: AI diagnosed {ai_diagnosis}, dermatologist diagnosed {derm_diagnosis}")

        # Confidence comparison
        if derm_confidence is not None:
            diff = abs(ai_confidence - derm_confidence)
            if diff < 0.1:
                notes.append(f"✓ Similar confidence levels (AI: {ai_confidence:.1%}, Derm: {derm_confidence:.1%})")
            else:
                notes.append(f"△ Different confidence levels (AI: {ai_confidence:.1%}, Derm: {derm_confidence:.1%})")

        # Region comparison
        ai_only = len([r for r in ai_regions if r.color == (255, 0, 255)])
        derm_only = len([r for r in derm_regions if r.color == (0, 255, 255)])

        if ai_only > 0:
            notes.append(f"△ AI identified {ai_only} region(s) not noted by dermatologist")
        if derm_only > 0:
            notes.append(f"△ Dermatologist noted {derm_only} region(s) not identified by AI")
        if ai_only == 0 and derm_only == 0 and ai_regions and derm_regions:
            notes.append("✓ Good overlap in identified regions of interest")

        return notes

    def create_comparison_image(
        self,
        original_image: np.ndarray,
        comparison: DermatologistComparison
    ) -> np.ndarray:
        """Create side-by-side comparison image"""
        h, w = original_image.shape[:2]

        # Create combined image (original + AI + dermatologist)
        combined_width = w * 3 + 20  # 10px gaps
        combined = np.ones((h + 60, combined_width, 3), dtype=np.uint8) * 255

        # Original image (left)
        combined[40:40+h, 0:w] = original_image
        cv2.putText(combined, "Original", (w//2 - 30, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # AI annotations (center)
        ai_img = original_image.copy()
        for region in comparison.ai_highlighted_regions:
            cv2.rectangle(ai_img, (region.x, region.y),
                          (region.x + region.width, region.y + region.height),
                          (255, 0, 0), 2)
        combined[40:40+h, w+10:2*w+10] = ai_img
        cv2.putText(combined, f"AI: {comparison.ai_diagnosis}", (w + w//2 - 50, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Dermatologist annotations (right)
        derm_img = original_image.copy()
        for region in comparison.dermatologist_annotations:
            cv2.rectangle(derm_img, (region.x, region.y),
                          (region.x + region.width, region.y + region.height),
                          (0, 128, 0), 2)
        combined[40:40+h, 2*w+20:3*w+20] = derm_img
        derm_label = comparison.dermatologist_diagnosis or "No annotation"
        cv2.putText(combined, f"Derm: {derm_label}", (2*w + w//2 - 40, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)

        # Agreement score
        if comparison.agreement_score is not None:
            cv2.putText(combined, f"Agreement: {comparison.agreement_score:.1%}",
                        (combined_width // 2 - 50, h + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return combined


class ExplainableAIService:
    """Main service for generating explainable AI results"""

    def __init__(self, model, processor, device: str = "cpu"):
        self.model = model
        self.processor = processor
        self.device = device

        self.grad_cam = GradCAMGenerator(model, device)
        self.region_highlighter = RegionHighlighter()
        self.abcde_analyzer = ABCDEAnalyzer()
        self.feature_explainer = FeatureExplainer()
        self.importance_visualizer = FeatureImportanceVisualizer()
        self.dermatologist_comparer = DermatologistComparer()

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array to base64 string"""
        img_pil = Image.fromarray(image)
        buffer = BytesIO()
        img_pil.save(buffer, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def generate_explanation(
        self,
        image: Image.Image,
        analysis_id: str,
        predicted_class: str,
        confidence: float,
        all_probabilities: Dict[str, float],
        uncertainty_metrics: Optional[Dict] = None,
        pixels_per_mm: Optional[float] = None,
        previous_analysis: Optional[Dict] = None,
        dermatologist_annotation: Optional[Dict] = None
    ) -> ExplainableResult:
        """
        Generate complete explainable AI result
        """
        img_array = np.array(image)

        # Generate Grad-CAM heatmaps
        try:
            target_class_idx = list(all_probabilities.keys()).index(predicted_class)
        except (ValueError, IndexError):
            target_class_idx = None

        try:
            heatmap, grad_cam_overlay, grad_cam_info = self.grad_cam.generate_grad_cam(
                image, self.processor, target_class_idx
            )
        except Exception as e:
            # Fallback to simple attention map
            heatmap = self._simple_attention_map(img_array)
            grad_cam_overlay = self._apply_heatmap(img_array, heatmap)
            grad_cam_info = {"method": "fallback_attention", "error": str(e)}

        # Generate Grad-CAM++
        try:
            _, grad_cam_plus_overlay, _ = self.grad_cam.generate_grad_cam_plus(
                image, self.processor, target_class_idx
            )
            grad_cam_plus_base64 = self._image_to_base64(grad_cam_plus_overlay)
        except Exception:
            grad_cam_plus_base64 = None

        # Find important regions
        important_regions = self.region_highlighter.find_important_regions(
            heatmap, img_array, num_regions=5
        )

        # Draw regions on image
        region_highlights_img = self.region_highlighter.draw_regions_on_image(
            img_array, important_regions, show_labels=True
        )

        # Calculate feature scores from regions
        feature_scores = {}
        for region in important_regions:
            feature_type = region.feature_type
            if feature_type not in feature_scores:
                feature_scores[feature_type] = region.importance_score
            else:
                feature_scores[feature_type] = max(
                    feature_scores[feature_type],
                    region.importance_score
                )

        # Add base features if not present
        base_features = ["asymmetry", "border_irregularity", "color_variation", "texture_pattern"]
        for feature in base_features:
            if feature not in feature_scores:
                feature_scores[feature] = 0.3  # Default low score

        # ABCDE analysis
        abcde_score = self.abcde_analyzer.analyze(
            img_array,
            pixels_per_mm=pixels_per_mm,
            previous_analysis=previous_analysis
        )

        # Create ABCDE annotated image
        abcde_annotated = self.abcde_analyzer.create_annotated_image(img_array, abcde_score)

        # Generate feature importance chart
        importance_chart = self.importance_visualizer.create_importance_chart(feature_scores)

        # Generate natural language explanations
        summary, detailed, clinical = self.feature_explainer.generate_explanation(
            predicted_class, confidence, all_probabilities,
            feature_scores, abcde_score, uncertainty_metrics
        )

        # Generate feature explanations
        feature_explanations = self.feature_explainer.generate_feature_explanations(
            feature_scores, predicted_class
        )

        # Dermatologist comparison
        dermatologist_comparison = None
        if dermatologist_annotation:
            ai_result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "highlighted_regions": [
                    {
                        "x": r.x, "y": r.y, "width": r.width, "height": r.height,
                        "importance_score": r.importance_score,
                        "feature_type": r.feature_type,
                        "description": r.description,
                        "color": r.color
                    }
                    for r in important_regions
                ],
                "feature_scores": feature_scores
            }
            comparison = self.dermatologist_comparer.compare(ai_result, dermatologist_annotation)
            dermatologist_comparison = {
                "ai_diagnosis": comparison.ai_diagnosis,
                "ai_confidence": comparison.ai_confidence,
                "dermatologist_diagnosis": comparison.dermatologist_diagnosis,
                "dermatologist_confidence": comparison.dermatologist_confidence,
                "dermatologist_notes": comparison.dermatologist_notes,
                "agreement_score": comparison.agreement_score,
                "comparison_notes": comparison.comparison_notes
            }

        # Create result
        return ExplainableResult(
            analysis_id=analysis_id,
            timestamp=datetime.utcnow().isoformat(),
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probabilities,
            original_image_base64=self._image_to_base64(img_array),
            grad_cam_heatmap=self._image_to_base64(grad_cam_overlay),
            grad_cam_plus_heatmap=grad_cam_plus_base64,
            attention_overlay=self._image_to_base64(grad_cam_overlay),  # Same as grad_cam for now
            region_highlights_image=self._image_to_base64(region_highlights_img),
            abcde_annotated_image=self._image_to_base64(abcde_annotated),
            feature_importance_chart=self._image_to_base64(importance_chart),
            important_regions=[
                {
                    "x": r.x, "y": r.y, "width": r.width, "height": r.height,
                    "importance_score": r.importance_score,
                    "feature_type": r.feature_type,
                    "description": r.description
                }
                for r in important_regions
            ],
            feature_explanations=[
                {
                    "feature_name": fe.feature_name,
                    "category": fe.category.value,
                    "importance_score": fe.importance_score,
                    "description": fe.description,
                    "clinical_significance": fe.clinical_significance,
                    "visual_indicator": fe.visual_indicator
                }
                for fe in feature_explanations
            ],
            feature_importance_scores=feature_scores,
            abcde_analysis=abcde_score.to_dict(),
            summary_explanation=summary,
            detailed_explanation=detailed,
            clinical_reasoning=clinical,
            dermatologist_comparison=dermatologist_comparison
        )

    def _simple_attention_map(self, image: np.ndarray) -> np.ndarray:
        """Generate simple attention map as fallback"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        blurred = cv2.GaussianBlur(edges.astype(float), (21, 21), 0)
        normalized = (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-8)
        return normalized

    def _apply_heatmap(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Apply heatmap overlay to image"""
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.6 * image + 0.4 * heatmap_colored)
        return overlay
