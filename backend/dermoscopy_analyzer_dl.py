"""
Enhanced Dermoscopic Feature Detector using Deep Learning.

This module integrates the trained U-Net + Attention models
with the existing computer vision-based analyzer for improved accuracy.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import io
import base64
from pathlib import Path

from dermoscopy_analyzer import DermoscopicFeatureDetector as CVDetector
from dermoscopy_models import DermoscopyNet, get_model
from torchvision import transforms


class DeepLearningDermoscopyDetector:
    """
    Enhanced dermoscopy detector using deep learning models.

    Combines classical computer vision with trained neural networks for:
    - More accurate feature segmentation
    - Better classification of feature types
    - Improved risk assessment
    """

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)

        # Initialize classical CV detector as fallback
        self.cv_detector = CVDetector()

        # Load deep learning model if available
        self.dl_model = None
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            print(f"[DL DERMOSCOPY] Loaded deep learning model from {model_path}")
        else:
            print("[DL DERMOSCOPY] No trained model found, using classical CV only")

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        try:
            # Create model
            self.dl_model = get_model('full', device=self.device)

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.dl_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.dl_model.load_state_dict(checkpoint)

            self.dl_model.eval()
            print(f"✓ Model loaded successfully")

        except Exception as e:
            print(f"Error loading model: {e}")
            self.dl_model = None

    def analyze(self, image_bytes: bytes) -> Dict:
        """
        Comprehensive dermoscopic feature analysis using both CV and DL.

        Args:
            image_bytes: Image bytes

        Returns:
            Dictionary with all detected features and risk assessment
        """
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image.convert('RGB'))

        # Run classical CV analysis (always available)
        cv_results = self.cv_detector.analyze(image_bytes)

        # If DL model is available, enhance results
        if self.dl_model is not None:
            dl_results = self._analyze_with_dl(image)

            # Merge CV and DL results
            results = self._merge_results(cv_results, dl_results)
        else:
            results = cv_results

        return results

    def _analyze_with_dl(self, image: Image.Image) -> Dict:
        """
        Run deep learning analysis.

        Args:
            image: PIL Image

        Returns:
            Dictionary with DL predictions
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.dl_model(img_tensor, return_segmentation=True)

        # Process segmentation
        seg_mask = outputs['segmentation'][0]  # (6, H, W)
        seg_probs = F.softmax(seg_mask, dim=0)
        seg_pred = torch.argmax(seg_probs, dim=0)  # (H, W)

        # Convert to numpy
        seg_pred_np = seg_pred.cpu().numpy()
        seg_probs_np = seg_probs.cpu().numpy()

        # Process classification predictions
        classifications = {}
        for task_name, task_output in outputs['classification'].items():
            probs = F.softmax(task_output[0], dim=0)
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()

            classifications[task_name] = {
                'class': pred_class,
                'confidence': confidence,
                'probabilities': probs.cpu().numpy().tolist()
            }

        return {
            'segmentation': seg_pred_np,
            'segmentation_probs': seg_probs_np,
            'classification': classifications
        }

    def _merge_results(self, cv_results: Dict, dl_results: Dict) -> Dict:
        """
        Merge classical CV and deep learning results.

        Strategy:
        - Use DL segmentation for precise feature localization
        - Use CV for additional metadata (contour counts, etc.)
        - Average confidences for final risk assessment
        """
        merged = cv_results.copy()

        # Extract feature maps from DL segmentation
        # Channels: 0=background, 1=pigment_network, 2=globules, 3=streaks,
        #           4=blue_white_veil, 5=vascular, 6=regression
        seg_probs = dl_results['segmentation_probs']

        # Update pigment network
        if seg_probs[1].max() > 0.3:  # Threshold for detection
            merged['pigment_network']['detected'] = True
            merged['pigment_network']['dl_confidence'] = float(seg_probs[1].max())

            # Use DL classification for type
            dl_class = dl_results['classification']['pigment_network']
            types = ['absent', 'reticular', 'atypical', 'branched']
            if dl_class['class'] > 0:
                merged['pigment_network']['type'] = types[dl_class['class']]
                merged['pigment_network']['dl_type_confidence'] = dl_class['confidence']

        # Update globules
        if seg_probs[2].max() > 0.3:
            merged['globules']['detected'] = True
            merged['globules']['dl_confidence'] = float(seg_probs[2].max())

        # Update streaks
        if seg_probs[3].max() > 0.3:
            merged['streaks']['detected'] = True
            merged['streaks']['dl_confidence'] = float(seg_probs[3].max())

        # Update blue-white veil
        if seg_probs[4].max() > 0.4:  # Higher threshold for this important feature
            merged['blue_white_veil']['detected'] = True
            merged['blue_white_veil']['dl_confidence'] = float(seg_probs[4].max())
            # Calculate coverage from segmentation
            coverage = (seg_probs[4] > 0.4).sum() / seg_probs[4].size * 100
            merged['blue_white_veil']['coverage_percentage'] = float(coverage)

        # Update vascular patterns
        if seg_probs[5].max() > 0.3:
            merged['vascular_patterns']['detected'] = True
            merged['vascular_patterns']['dl_confidence'] = float(seg_probs[5].max())

        # Update regression
        if len(seg_probs) > 6 and seg_probs[6].max() > 0.3:
            merged['regression']['detected'] = True
            merged['regression']['dl_confidence'] = float(seg_probs[6].max())

        # Recalculate risk assessment with DL confidence
        merged['risk_assessment'] = self._enhanced_risk_assessment(merged)

        # Add DL metadata
        merged['dl_enhanced'] = True
        merged['dl_model_used'] = 'UNet_with_Attention'

        return merged

    def _enhanced_risk_assessment(self, features: Dict) -> Dict:
        """
        Enhanced risk assessment incorporating DL confidences.
        """
        risk_factors = []
        risk_score = 0

        # High-risk features (weighted by DL confidence if available)
        if features['blue_white_veil']['detected']:
            confidence_mult = features['blue_white_veil'].get('dl_confidence', 1.0)
            risk_factors.append(f"Blue-white veil present (confidence: {confidence_mult:.2f})")
            risk_score += 3 * confidence_mult

        if features['pigment_network']['type'] in ['atypical', 'branched']:
            confidence_mult = features['pigment_network'].get('dl_type_confidence', 1.0)
            risk_factors.append(f"Atypical pigment network (confidence: {confidence_mult:.2f})")
            risk_score += 2 * confidence_mult

        if features['streaks']['type'] == 'radial':
            risk_factors.append('Radial streaming present')
            risk_score += 2

        if features['globules']['type'] in ['irregular', 'peripheral']:
            risk_factors.append(f"Irregular globules ({features['globules']['type']})")
            risk_score += 2

        if features['vascular_patterns']['type'] == 'linear_irregular':
            risk_factors.append('Atypical vascular pattern')
            risk_score += 2

        # Moderate risk features
        if features['regression']['detected']:
            risk_factors.append('Regression structures present')
            risk_score += 1

        if features['color_analysis']['distinct_colors'] >= 4:
            risk_factors.append(f"{features['color_analysis']['distinct_colors']} distinct colors")
            risk_score += 1

        if features['symmetry_analysis']['classification'] == 'highly_asymmetric':
            risk_factors.append('Highly asymmetric lesion')
            risk_score += 1

        # Overall risk level
        if risk_score >= 6:
            overall_risk = 'HIGH'
            urgency = 'Urgent dermatologist evaluation recommended'
        elif risk_score >= 3:
            overall_risk = 'MODERATE'
            urgency = 'Dermatologist evaluation recommended within 2 weeks'
        elif risk_score >= 1:
            overall_risk = 'LOW-MODERATE'
            urgency = 'Routine dermatologist follow-up recommended'
        else:
            overall_risk = 'LOW'
            urgency = 'Routine monitoring'

        return {
            'risk_level': overall_risk,
            'risk_score': round(risk_score, 2),
            'risk_factors': risk_factors,
            'recommendation': urgency
        }


# Global instance
_dl_detector = None


def get_dl_dermoscopy_detector(
    model_path: Optional[str] = './checkpoints/dermoscopy/best_model.pth',
    device: str = 'cpu'
) -> DeepLearningDermoscopyDetector:
    """
    Get or create global deep learning dermoscopy detector instance.

    Args:
        model_path: Path to trained model checkpoint
        device: Device to run on

    Returns:
        DeepLearningDermoscopyDetector instance
    """
    global _dl_detector

    if _dl_detector is None:
        # Auto-detect CUDA
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        _dl_detector = DeepLearningDermoscopyDetector(
            model_path=model_path,
            device=device
        )
        print(f"[DL DERMOSCOPY] Initialized detector on {device}")

    return _dl_detector


if __name__ == '__main__':
    # Test the enhanced detector
    print("Testing Deep Learning Dermoscopy Detector...")

    detector = get_dl_dermoscopy_detector(
        model_path='./checkpoints/dermoscopy/best_model.pth',
        device='auto'
    )

    print("\n✓ Detector initialized successfully")
    print(f"  - Classical CV: Available")
    print(f"  - Deep Learning: {'Available' if detector.dl_model else 'Not loaded'}")

    # Test with a dummy image
    dummy_image = Image.new('RGB', (512, 512), color='red')
    img_bytes = io.BytesIO()
    dummy_image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    print("\nRunning test analysis...")
    results = detector.analyze(img_bytes)

    print("\n✓ Analysis complete")
    print(f"  - DL Enhanced: {results.get('dl_enhanced', False)}")
    print(f"  - Features detected: {sum(1 for k, v in results.items() if isinstance(v, dict) and v.get('detected', False))}")
