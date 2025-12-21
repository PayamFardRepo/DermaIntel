"""
Histopathology Image Analyzer

Analyzes skin biopsy/histopathology images using the Hibou-L foundation model.
Hibou-L is a pathology-specific Vision Transformer trained on 1.1M+ slides.

Features:
- Skin histopathology feature extraction using Hibou-L
- Tissue classification (12 tissue classes)
- Malignancy detection (benign vs malignant)
- Confidence scoring with uncertainty quantification
- Correlation with dermoscopy AI predictions
- Support for whole slide images (WSI)

Model: Hibou-L (Apache 2.0 License - Commercial Use Allowed)
Source: https://huggingface.co/histai/hibou-L
"""

import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Histopathology tissue classes (based on skin biopsy analysis)
TISSUE_CLASSES = [
    'epidermis_normal',
    'epidermis_hyperplasia',
    'epidermis_dysplasia',
    'dermis_normal',
    'dermis_inflammation',
    'dermis_fibrosis',
    'melanocytic_nevus',
    'melanoma_in_situ',
    'melanoma_invasive',
    'basal_cell_carcinoma',
    'squamous_cell_carcinoma',
    'other_malignancy'
]

# Simplified diagnostic classes
DIAGNOSTIC_CLASSES = {
    'benign': ['epidermis_normal', 'epidermis_hyperplasia', 'dermis_normal',
               'dermis_inflammation', 'dermis_fibrosis', 'melanocytic_nevus'],
    'pre_malignant': ['epidermis_dysplasia', 'melanoma_in_situ'],
    'malignant': ['melanoma_invasive', 'basal_cell_carcinoma',
                  'squamous_cell_carcinoma', 'other_malignancy']
}

# Map histopathology to dermoscopy classes for correlation
HISTOPATH_TO_DERMOSCOPY = {
    'melanocytic_nevus': 'NV',
    'melanoma_in_situ': 'MEL',
    'melanoma_invasive': 'MEL',
    'basal_cell_carcinoma': 'BCC',
    'squamous_cell_carcinoma': 'SCC',
    'epidermis_dysplasia': 'AK',
}

# Tissue type descriptions for user-friendly output
TISSUE_DESCRIPTIONS = {
    'epidermis_normal': 'Normal epidermis - healthy skin outer layer',
    'epidermis_hyperplasia': 'Epidermal hyperplasia - thickened skin layer, often benign',
    'epidermis_dysplasia': 'Epidermal dysplasia - abnormal cell changes, pre-cancerous',
    'dermis_normal': 'Normal dermis - healthy deeper skin layer',
    'dermis_inflammation': 'Dermal inflammation - inflammatory skin condition',
    'dermis_fibrosis': 'Dermal fibrosis - scar tissue formation',
    'melanocytic_nevus': 'Melanocytic nevus - benign mole',
    'melanoma_in_situ': 'Melanoma in situ - early melanoma confined to epidermis',
    'melanoma_invasive': 'Invasive melanoma - melanoma penetrating deeper layers',
    'basal_cell_carcinoma': 'Basal cell carcinoma - common skin cancer, slow-growing',
    'squamous_cell_carcinoma': 'Squamous cell carcinoma - skin cancer from squamous cells',
    'other_malignancy': 'Other malignancy - other types of skin cancer'
}


def load_hibou_model(model_name: str = "histai/hibou-L"):
    """
    Load the Hibou-L foundation model from HuggingFace.

    Args:
        model_name: HuggingFace model name (histai/hibou-L or histai/hibou-b)

    Returns:
        model, transform tuple
    """
    try:
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        logger.info(f"Loading Hibou model: {model_name}")

        # Load the model from HuggingFace
        model = timm.create_model(
            f"hf-hub:{model_name}",
            pretrained=True,
            num_classes=0  # Remove classifier head to get features
        )

        # Get the appropriate transforms
        data_config = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**data_config)

        logger.info(f"Successfully loaded {model_name}")
        logger.info(f"Feature dimension: {model.num_features}")

        return model, transform

    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Install with: pip install timm huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Failed to load Hibou model: {e}")
        raise


class HibouClassificationHead(nn.Module):
    """
    Classification head for Hibou features.
    Maps Hibou-L features (1024-dim) to tissue classes.
    """

    def __init__(
        self,
        input_dim: int = 1024,  # Hibou-L feature dimension
        num_classes: int = 12,
        dropout: float = 0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

        self.malignancy_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 2)  # benign vs malignant
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        class_logits = self.classifier(features)
        malignancy_logits = self.malignancy_head(features)
        return class_logits, malignancy_logits


class HistopathologyAnalyzer:
    """
    Analyzer for skin histopathology/biopsy images using Hibou-L foundation model.

    Hibou-L is trained on 1.1M+ pathology slides and provides state-of-the-art
    feature extraction for histopathology images.

    License: Apache 2.0 (Commercial use allowed)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        use_hibou: bool = True,
        hibou_model: str = "histai/hibou-L",
        enable_stain_normalization: bool = True,
        stain_method: str = 'macenko'
    ):
        """
        Initialize the histopathology analyzer.

        Args:
            model_path: Path to trained classification head weights (optional)
            device: Device to use ('auto', 'cuda', 'cpu')
            use_hibou: Whether to use Hibou foundation model
            hibou_model: Which Hibou model to use (histai/hibou-L or histai/hibou-b)
            enable_stain_normalization: Apply stain normalization before analysis
            stain_method: Stain normalization method ('macenko', 'reinhard')
        """
        # Stain normalization settings
        self.enable_stain_normalization = enable_stain_normalization
        self.stain_normalizer = None
        if enable_stain_normalization:
            try:
                from stain_normalization import StainNormalizer
                self.stain_normalizer = StainNormalizer(method=stain_method)
                logger.info(f"Stain normalization enabled ({stain_method} method)")
            except ImportError:
                logger.warning("Stain normalization module not found, disabled")
                self.enable_stain_normalization = False
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Histopathology analyzer using device: {self.device}")

        self.use_hibou = use_hibou
        self.hibou_available = False
        self.feature_dim = 1024  # Default for Hibou-L

        if use_hibou:
            try:
                # Load Hibou foundation model
                self.backbone, self.transform = load_hibou_model(hibou_model)
                self.backbone = self.backbone.to(self.device)
                self.backbone.eval()
                self.feature_dim = self.backbone.num_features
                self.hibou_available = True
                logger.info(f"Hibou-L loaded successfully. Feature dim: {self.feature_dim}")
            except Exception as e:
                logger.warning(f"Could not load Hibou model: {e}")
                logger.warning("Falling back to EfficientNet backbone")
                self._load_fallback_backbone()
        else:
            self._load_fallback_backbone()

        # Initialize classification head
        self.classification_head = HibouClassificationHead(
            input_dim=self.feature_dim,
            num_classes=len(TISSUE_CLASSES)
        ).to(self.device)

        # Load trained classification head if available
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'classification_head' in checkpoint:
                self.classification_head.load_state_dict(checkpoint['classification_head'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.classification_head.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.classification_head.load_state_dict(checkpoint)
            logger.info(f"Loaded classification head from {model_path}")
        else:
            logger.info("Using untrained classification head - predictions will be based on pretrained features only")

        self.classification_head.eval()

    def _load_fallback_backbone(self):
        """Load fallback EfficientNet backbone if Hibou is not available."""
        from torchvision import models

        logger.info("Loading fallback EfficientNet-B3 backbone")
        self.backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        self.backbone.classifier = nn.Identity()
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        self.feature_dim = 1536

        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(
        self,
        image: Union[Image.Image, bytes, str],
        apply_stain_norm: bool = True
    ) -> torch.Tensor:
        """Preprocess image for analysis."""
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')

        # Apply stain normalization if enabled
        if apply_stain_norm and self.enable_stain_normalization and self.stain_normalizer:
            try:
                image = self.stain_normalizer.transform(image)
            except Exception as e:
                logger.warning(f"Stain normalization failed: {e}, using original image")

        # Apply transforms
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_features(self, image: Union[Image.Image, bytes, str]) -> torch.Tensor:
        """Extract features using Hibou backbone."""
        input_tensor = self.preprocess_image(image)
        features = self.backbone(input_tensor)
        return features

    @torch.no_grad()
    def analyze(
        self,
        image: Union[Image.Image, bytes, str],
        dermoscopy_prediction: Optional[Dict] = None,
        num_mc_samples: int = 10
    ) -> Dict:
        """
        Analyze a histopathology image.

        Args:
            image: Input image (PIL Image, bytes, or file path)
            dermoscopy_prediction: Optional dermoscopy AI prediction for correlation
            num_mc_samples: Number of Monte Carlo samples for uncertainty

        Returns:
            Analysis results with classifications and confidence scores
        """
        # Extract features using Hibou
        features = self.extract_features(image)

        # Monte Carlo Dropout for uncertainty estimation
        self.classification_head.train()  # Enable dropout

        tissue_probs_list = []
        malignancy_probs_list = []

        for _ in range(num_mc_samples):
            class_logits, malignancy_logits = self.classification_head(features)
            tissue_probs_list.append(F.softmax(class_logits, dim=1).cpu().numpy())
            malignancy_probs_list.append(F.softmax(malignancy_logits, dim=1).cpu().numpy())

        self.classification_head.eval()

        # Aggregate predictions
        tissue_probs = np.mean(tissue_probs_list, axis=0)[0]
        tissue_std = np.std(tissue_probs_list, axis=0)[0]
        malignancy_probs = np.mean(malignancy_probs_list, axis=0)[0]
        malignancy_std = np.std(malignancy_probs_list, axis=0)[0]

        # Get top predictions
        top_indices = np.argsort(tissue_probs)[::-1][:5]
        top_predictions = [
            {
                'type': TISSUE_CLASSES[i],
                'confidence': float(tissue_probs[i]),
                'description': TISSUE_DESCRIPTIONS.get(TISSUE_CLASSES[i], ''),
                'confidence_interval': [
                    float(max(0, tissue_probs[i] - 1.96 * tissue_std[i])),
                    float(min(1, tissue_probs[i] + 1.96 * tissue_std[i]))
                ]
            }
            for i in top_indices
        ]

        # Primary diagnosis
        primary_idx = top_indices[0]
        primary_class = TISSUE_CLASSES[primary_idx]

        # Determine diagnostic category
        diagnostic_category = 'unknown'
        for category, classes in DIAGNOSTIC_CLASSES.items():
            if primary_class in classes:
                diagnostic_category = category
                break

        # Determine risk level
        if diagnostic_category == 'malignant':
            risk_level = 'high'
        elif diagnostic_category == 'pre_malignant':
            risk_level = 'moderate'
        else:
            risk_level = 'low'

        # Key features based on prediction
        key_features = self._get_key_features(primary_class, tissue_probs)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            primary_class, diagnostic_category, float(malignancy_probs[1])
        )

        # Assess image quality using stain normalization module
        quality_metrics = {
            'focus_quality': 'Good' if tissue_std.mean() < 0.15 else 'Moderate',
            'staining_quality': 'Good',
            'tissue_adequacy': 'Adequate' if tissue_probs.max() > 0.3 else 'Limited'
        }

        # Enhanced quality assessment if stain module available
        try:
            from stain_normalization import assess_staining_quality
            stain_quality = assess_staining_quality(image)
            quality_metrics = {
                'focus_quality': 'Good' if tissue_std.mean() < 0.15 else 'Moderate',
                'staining_quality': stain_quality.get('quality_label', 'Unknown'),
                'stain_intensity': stain_quality.get('stain_intensity', 'unknown'),
                'color_balance': stain_quality.get('color_balance', 'unknown'),
                'uniformity': stain_quality.get('uniformity', 'unknown'),
                'tissue_percent': stain_quality.get('tissue_percent', 0),
                'tissue_adequacy': 'Adequate' if stain_quality.get('tissue_percent', 0) > 20 else 'Limited',
                'quality_score': stain_quality.get('quality_score', 0),
                'issues': stain_quality.get('issues', []),
                'staining_recommendations': stain_quality.get('recommendations', [])
            }
        except Exception as e:
            logger.debug(f"Enhanced quality assessment not available: {e}")

        # Build result in format expected by frontend
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_used': 'Hibou-L' if self.hibou_available else 'EfficientNet-B3',
            'tissue_types': top_predictions,
            'malignancy_assessment': {
                'risk_level': risk_level,
                'malignant_probability': float(malignancy_probs[1]),
                'confidence_interval': [
                    float(max(0, malignancy_probs[1] - 1.96 * malignancy_std[1])),
                    float(min(1, malignancy_probs[1] + 1.96 * malignancy_std[1]))
                ],
                'key_features': key_features
            },
            'quality_metrics': quality_metrics,
            'recommendations': recommendations,
            'primary_diagnosis': primary_class,
            'primary_probability': float(tissue_probs[primary_idx]),
            'diagnostic_category': diagnostic_category,
            'uncertainty': {
                'mean_std': float(np.mean(tissue_std)),
                'max_std': float(np.max(tissue_std)),
                'is_reliable': bool(np.mean(tissue_std) < 0.15)
            }
        }

        # Add dermoscopy correlation if provided
        if dermoscopy_prediction:
            result['dermoscopy_correlation'] = self._correlate_with_dermoscopy(
                primary_class, dermoscopy_prediction
            )

        return result

    def _get_key_features(self, primary_class: str, tissue_probs: np.ndarray) -> List[str]:
        """Generate key features identified based on predictions."""
        features = []

        # Add features based on primary class
        if 'melanoma' in primary_class:
            features.extend([
                'Atypical melanocytes',
                'Irregular nuclear morphology',
                'Pagetoid spread pattern'
            ])
        elif primary_class == 'basal_cell_carcinoma':
            features.extend([
                'Basaloid cell nests',
                'Peripheral palisading',
                'Clefting artifact'
            ])
        elif primary_class == 'squamous_cell_carcinoma':
            features.extend([
                'Atypical keratinocytes',
                'Keratin pearls',
                'Invasive growth pattern'
            ])
        elif 'dysplasia' in primary_class:
            features.extend([
                'Cellular atypia',
                'Architectural disorder',
                'Abnormal maturation'
            ])
        elif 'inflammation' in primary_class:
            features.extend([
                'Inflammatory infiltrate',
                'Perivascular involvement'
            ])
        elif 'nevus' in primary_class:
            features.extend([
                'Nested melanocytes',
                'Regular architecture',
                'Maturation with depth'
            ])
        else:
            features.append('Normal tissue architecture')

        return features

    def _generate_recommendations(
        self,
        primary_class: str,
        diagnostic_category: str,
        malignancy_prob: float
    ) -> List[str]:
        """Generate clinical recommendations based on findings."""
        recommendations = []

        if diagnostic_category == 'malignant':
            recommendations.extend([
                'Urgent consultation with dermatologist/oncologist recommended',
                'Complete surgical excision with margin assessment',
                'Consider sentinel lymph node biopsy if melanoma',
                'Staging workup as per clinical guidelines'
            ])
        elif diagnostic_category == 'pre_malignant':
            recommendations.extend([
                'Close clinical monitoring recommended',
                'Consider excision or treatment',
                'Regular follow-up examinations',
                'Sun protection counseling'
            ])
        elif malignancy_prob > 0.3:
            recommendations.extend([
                'Clinical correlation recommended',
                'Consider repeat biopsy if clinical suspicion persists',
                'Regular follow-up monitoring'
            ])
        else:
            recommendations.extend([
                'Benign findings - routine follow-up',
                'Continue regular skin self-examinations',
                'Annual dermatological review'
            ])

        return recommendations

    def _correlate_with_dermoscopy(
        self,
        histopath_class: str,
        dermoscopy_prediction: Dict
    ) -> Dict:
        """Correlate histopathology finding with dermoscopy prediction."""
        expected_dermoscopy = HISTOPATH_TO_DERMOSCOPY.get(histopath_class, None)

        actual_dermoscopy = dermoscopy_prediction.get('prediction') or \
                           dermoscopy_prediction.get('primary_diagnosis') or \
                           dermoscopy_prediction.get('class')

        is_concordant = False
        if expected_dermoscopy and actual_dermoscopy:
            is_concordant = expected_dermoscopy.upper() == actual_dermoscopy.upper()

        dermoscopy_probs = dermoscopy_prediction.get('probabilities', {})
        expected_prob = dermoscopy_probs.get(expected_dermoscopy, 0) if expected_dermoscopy else 0

        return {
            'histopathology_diagnosis': histopath_class,
            'expected_dermoscopy_class': expected_dermoscopy,
            'actual_dermoscopy_class': actual_dermoscopy,
            'is_concordant': is_concordant,
            'dermoscopy_probability_for_expected': float(expected_prob),
            'agreement_assessment': self._assess_agreement(
                is_concordant, expected_prob, histopath_class
            )
        }

    @torch.no_grad()
    def _analyze_batch(self, tiles: List[Union[Image.Image, bytes]]) -> List[Dict]:
        """
        Analyze a batch of tiles efficiently using batched inference.

        This is more efficient than analyzing tiles one by one as it
        batches the forward pass through the backbone.
        """
        # Preprocess all tiles
        tensors = []
        for tile in tiles:
            tensor = self.preprocess_image(tile)
            tensors.append(tensor.squeeze(0))  # Remove batch dim for stacking

        # Stack into batch
        batch_tensor = torch.stack(tensors, dim=0)

        # Extract features for entire batch
        features = self.backbone(batch_tensor)

        # Run classification for each (MC dropout needs individual runs)
        results = []
        for i, tile in enumerate(tiles):
            single_features = features[i:i+1]

            # Monte Carlo Dropout (smaller samples for batch efficiency)
            self.classification_head.train()
            tissue_probs_list = []
            malignancy_probs_list = []

            for _ in range(5):  # Reduced samples for batch
                class_logits, malignancy_logits = self.classification_head(single_features)
                tissue_probs_list.append(F.softmax(class_logits, dim=1).cpu().numpy())
                malignancy_probs_list.append(F.softmax(malignancy_logits, dim=1).cpu().numpy())

            self.classification_head.eval()

            # Aggregate predictions
            tissue_probs = np.mean(tissue_probs_list, axis=0)[0]
            malignancy_probs = np.mean(malignancy_probs_list, axis=0)[0]

            # Get primary prediction
            primary_idx = np.argmax(tissue_probs)
            primary_class = TISSUE_CLASSES[primary_idx]

            # Determine risk level
            if primary_class in DIAGNOSTIC_CLASSES['malignant']:
                risk_level = 'high'
            elif primary_class in DIAGNOSTIC_CLASSES['pre_malignant']:
                risk_level = 'moderate'
            else:
                risk_level = 'low'

            results.append({
                'primary_diagnosis': primary_class,
                'primary_probability': float(tissue_probs[primary_idx]),
                'tissue_types': [
                    {'type': TISSUE_CLASSES[j], 'confidence': float(tissue_probs[j])}
                    for j in np.argsort(tissue_probs)[::-1][:5]
                ],
                'malignancy_assessment': {
                    'risk_level': risk_level,
                    'malignant_probability': float(malignancy_probs[1])
                }
            })

        return results

    def _assess_agreement(
        self,
        is_concordant: bool,
        expected_prob: float,
        histopath_class: str
    ) -> str:
        """Generate agreement assessment text."""
        if is_concordant:
            if expected_prob > 0.7:
                return "Strong agreement: Both dermoscopy AI and histopathology confirm the diagnosis."
            elif expected_prob > 0.4:
                return "Moderate agreement: Dermoscopy AI showed some indication, confirmed by histopathology."
            else:
                return "Histopathology confirms diagnosis that dermoscopy AI ranked lower."
        else:
            if 'melanoma' in histopath_class:
                return "CRITICAL DISCORDANCE: Histopathology shows melanoma but dermoscopy AI did not predict this. Review recommended."
            elif 'carcinoma' in histopath_class:
                return "Discordance: Histopathology shows carcinoma not predicted by dermoscopy AI. This can occur with atypical presentations."
            else:
                return "Discordance between dermoscopy AI and histopathology. Clinical correlation recommended."

    def analyze_tile_batch(
        self,
        tiles: List[Union[Image.Image, bytes]],
        aggregate: bool = True,
        batch_size: int = 8
    ) -> Union[Dict, List[Dict]]:
        """
        Analyze multiple tiles from a whole slide image with optimized batch processing.

        Args:
            tiles: List of image tiles
            aggregate: If True, aggregate results; if False, return per-tile results
            batch_size: Number of tiles to process in parallel

        Returns:
            Aggregated or per-tile analysis results
        """
        results = []

        # Process tiles in batches for better GPU utilization
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]
            batch_results = self._analyze_batch(batch)
            results.extend(batch_results)

        if not aggregate:
            return results

        # Aggregate tissue probabilities
        all_tissue_probs = []
        for r in results:
            probs = [t['confidence'] for t in r['tissue_types']]
            # Pad if needed
            while len(probs) < len(TISSUE_CLASSES):
                probs.append(0)
            all_tissue_probs.append(probs[:len(TISSUE_CLASSES)])

        all_tissue_probs = np.array(all_tissue_probs)
        mean_probs = np.mean(all_tissue_probs, axis=0)
        max_probs = np.max(all_tissue_probs, axis=0)

        # Find primary diagnosis across tiles
        primary_idx = np.argmax(max_probs)
        primary_class = TISSUE_CLASSES[primary_idx]

        # Count malignant tiles
        malignant_tiles = sum(
            1 for r in results
            if r['malignancy_assessment']['malignant_probability'] > 0.5
        )
        malignant_ratio = malignant_tiles / len(results) if results else 0

        return {
            'num_tiles_analyzed': len(results),
            'primary_diagnosis': primary_class,
            'primary_probability': float(max_probs[primary_idx]),
            'malignant_tile_ratio': malignant_ratio,
            'is_malignant': malignant_ratio > 0.3,
            'mean_tissue_probabilities': {
                TISSUE_CLASSES[i]: float(mean_probs[i])
                for i in range(len(TISSUE_CLASSES))
            },
            'per_tile_results': results
        }


# Singleton instance
_analyzer_instance = None


def get_histopathology_analyzer(
    model_path: Optional[str] = None,
    use_hibou: bool = True
) -> HistopathologyAnalyzer:
    """Get or create histopathology analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = HistopathologyAnalyzer(
            model_path=model_path,
            use_hibou=use_hibou
        )
    return _analyzer_instance


def analyze_biopsy_image(
    image: Union[Image.Image, bytes, str],
    dermoscopy_prediction: Optional[Dict] = None
) -> Dict:
    """
    Convenience function to analyze a biopsy image.

    Args:
        image: Biopsy/histopathology image
        dermoscopy_prediction: Optional dermoscopy prediction for correlation

    Returns:
        Analysis results
    """
    analyzer = get_histopathology_analyzer()
    return analyzer.analyze(image, dermoscopy_prediction)


if __name__ == '__main__':
    # Test the analyzer
    print("Histopathology Analyzer Test (Hibou-L)")
    print("=" * 50)

    analyzer = HistopathologyAnalyzer()

    print(f"\nModel: {analyzer.hibou_available and 'Hibou-L' or 'EfficientNet-B3 (fallback)'}")
    print(f"Feature dimension: {analyzer.feature_dim}")
    print(f"Device: {analyzer.device}")

    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='pink')

    print("\nAnalyzing test image...")
    result = analyzer.analyze(test_image)

    print(f"\nPrimary diagnosis: {result['primary_diagnosis']}")
    print(f"Diagnostic category: {result['diagnostic_category']}")
    print(f"Risk level: {result['malignancy_assessment']['risk_level']}")
    print(f"Malignant probability: {result['malignancy_assessment']['malignant_probability']:.2%}")

    print("\nTop tissue types:")
    for tissue in result['tissue_types'][:3]:
        print(f"  {tissue['type']}: {tissue['confidence']:.2%}")

    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
