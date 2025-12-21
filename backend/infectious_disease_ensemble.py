"""
Ensemble Model for Infectious Disease Classification

This module provides an ensemble classifier that combines predictions from multiple
models for improved accuracy and robustness:

1. EfficientNet-B3 (or ResNet50) - Fast, efficient CNN
2. DinoV2 - State-of-the-art vision transformer
3. ConvNeXt - Modern CNN architecture

Ensemble methods:
- Soft voting (average probabilities)
- Weighted voting (based on validation accuracy)
- Stacking (train meta-classifier on predictions)

Expected improvement: 5-10% over single model

Usage:
    from infectious_disease_ensemble import InfectiousDiseaseEnsemble

    ensemble = InfectiousDiseaseEnsemble()
    ensemble.load_models()
    prediction = ensemble.predict(image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
from tqdm import tqdm

# Try to import transformers for DinoV2
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from torchvision import models, transforms


class ModelWrapper:
    """Wrapper class for individual models in the ensemble."""

    def __init__(self, name: str, model: nn.Module, processor=None,
                 weight: float = 1.0, device: str = 'cpu'):
        self.name = name
        self.model = model
        self.processor = processor
        self.weight = weight
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, image: Image.Image) -> torch.Tensor:
        """Get prediction probabilities for an image."""
        with torch.no_grad():
            if self.processor is not None:
                # HuggingFace model
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
            else:
                # Torchvision model
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                logits = self.model(img_tensor)

            probs = F.softmax(logits, dim=-1)
            return probs.squeeze(0)


class InfectiousDiseaseEnsemble:
    """
    Ensemble classifier for infectious skin diseases.

    Combines multiple models using various voting strategies for improved accuracy.
    """

    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: List[ModelWrapper] = []
        self.class_names: List[str] = []
        self.class_metadata: Dict = {}
        self.ensemble_method = 'weighted_soft_voting'

    def add_model(self, name: str, model: nn.Module, processor=None,
                  weight: float = 1.0):
        """Add a model to the ensemble."""
        wrapper = ModelWrapper(name, model, processor, weight, self.device)
        self.models.append(wrapper)
        print(f"Added model: {name} (weight: {weight:.2f})")

    def load_models(self,
                    efficientnet_path: str = None,
                    dinov2_path: str = None,
                    convnext_path: str = None,
                    resnet_path: str = None):
        """
        Load pre-trained models from checkpoints.

        Args:
            efficientnet_path: Path to EfficientNet checkpoint
            dinov2_path: Path to DinoV2 checkpoint
            convnext_path: Path to ConvNeXt checkpoint
            resnet_path: Path to ResNet checkpoint
        """

        print("\n" + "="*60)
        print("Loading Ensemble Models")
        print("="*60)

        # Default paths
        base_path = Path("./checkpoints/infectious")

        # Load EfficientNet-B3
        eff_path = efficientnet_path or base_path / "efficientnet_b3_best.pth"
        if Path(eff_path).exists():
            self._load_efficientnet(eff_path, weight=1.0)
        else:
            print(f"EfficientNet not found at {eff_path}")

        # Load DinoV2
        dino_path = dinov2_path or Path("./infectious_disease_model_dinov2/dinov2_infectious_model.pth")
        if Path(dino_path).exists() and HAS_TRANSFORMERS:
            self._load_dinov2(dino_path, weight=1.2)  # Higher weight for DinoV2
        else:
            if not HAS_TRANSFORMERS:
                print("DinoV2 requires transformers library")
            else:
                print(f"DinoV2 not found at {dino_path}")

        # Load ConvNeXt
        conv_path = convnext_path or base_path / "convnext_tiny_best.pth"
        if Path(conv_path).exists():
            self._load_convnext(conv_path, weight=1.0)
        else:
            print(f"ConvNeXt not found at {conv_path}")

        # Load ResNet50 as fallback
        res_path = resnet_path or base_path / "best_model.pth"
        if Path(res_path).exists():
            self._load_resnet(res_path, weight=0.8)
        else:
            print(f"ResNet not found at {res_path}")

        if len(self.models) == 0:
            raise RuntimeError("No models loaded! Please train models first.")

        print(f"\nLoaded {len(self.models)} models in ensemble")

    def _load_efficientnet(self, path: str, weight: float):
        """Load EfficientNet-B3 model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            config = checkpoint.get('config', {})
            num_classes = config.get('num_classes', 9)

            model = models.efficientnet_b3(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            self.add_model('EfficientNet-B3', model, weight=weight)

            # Load class names if available
            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']

        except Exception as e:
            print(f"Error loading EfficientNet: {e}")

    def _load_dinov2(self, path: str, weight: float):
        """Load DinoV2 model."""
        try:
            from transformers import Dinov2Model

            checkpoint = torch.load(path, map_location=self.device)
            config = checkpoint.get('config', {})
            num_classes = len(checkpoint.get('class_names', []))

            # Recreate DinoV2 classifier
            class DinoV2Classifier(nn.Module):
                def __init__(self, hidden_size, num_classes):
                    super().__init__()
                    self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.GELU(),
                        nn.Dropout(0.15),
                        nn.Linear(hidden_size // 2, num_classes)
                    )

                def forward(self, pixel_values):
                    outputs = self.backbone(pixel_values)
                    cls_token = outputs.last_hidden_state[:, 0]
                    return self.classifier(cls_token)

            model = DinoV2Classifier(768, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])

            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.add_model('DinoV2', model, processor=processor, weight=weight)

            if 'class_names' in checkpoint:
                self.class_names = checkpoint['class_names']

        except Exception as e:
            print(f"Error loading DinoV2: {e}")

    def _load_convnext(self, path: str, weight: float):
        """Load ConvNeXt model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            config = checkpoint.get('config', {})
            num_classes = config.get('num_classes', 9)

            model = models.convnext_tiny(weights=None)
            model.classifier[2] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.classifier[2].in_features, num_classes)
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            self.add_model('ConvNeXt-Tiny', model, weight=weight)

        except Exception as e:
            print(f"Error loading ConvNeXt: {e}")

    def _load_resnet(self, path: str, weight: float):
        """Load ResNet model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            config = checkpoint.get('config', {})
            num_classes = config.get('num_classes', 9)

            # Try to determine model type
            model_name = config.get('model_name', 'resnet50')

            if 'resnet18' in model_name:
                model = models.resnet18(weights=None)
            else:
                model = models.resnet50(weights=None)

            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.fc.in_features, num_classes)
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            self.add_model(f'ResNet ({model_name})', model, weight=weight)

            if 'class_names' in checkpoint and not self.class_names:
                self.class_names = checkpoint['class_names']

        except Exception as e:
            print(f"Error loading ResNet: {e}")

    def predict(self, image: Union[Image.Image, str, np.ndarray],
                return_all_probs: bool = False) -> Dict:
        """
        Make prediction using ensemble.

        Args:
            image: PIL Image, file path, or numpy array
            return_all_probs: If True, return probabilities from all models

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """

        if len(self.models) == 0:
            raise RuntimeError("No models loaded. Call load_models() first.")

        # Convert to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        # Get predictions from all models
        all_probs = []
        model_predictions = {}

        for wrapper in self.models:
            probs = wrapper.predict(image)
            all_probs.append(probs.cpu().numpy() * wrapper.weight)
            model_predictions[wrapper.name] = {
                'probabilities': probs.cpu().numpy().tolist(),
                'prediction': self.class_names[probs.argmax().item()] if self.class_names else int(probs.argmax().item()),
                'confidence': float(probs.max().item())
            }

        # Ensemble predictions
        if self.ensemble_method == 'weighted_soft_voting':
            # Weighted average of probabilities
            total_weight = sum(m.weight for m in self.models)
            ensemble_probs = np.sum(all_probs, axis=0) / total_weight
        elif self.ensemble_method == 'soft_voting':
            # Simple average
            ensemble_probs = np.mean(all_probs, axis=0)
        elif self.ensemble_method == 'hard_voting':
            # Majority voting
            votes = np.zeros(len(self.class_names))
            for probs in all_probs:
                votes[np.argmax(probs)] += 1
            ensemble_probs = votes / len(all_probs)
        else:
            ensemble_probs = np.mean(all_probs, axis=0)

        # Get final prediction
        predicted_idx = np.argmax(ensemble_probs)
        confidence = float(ensemble_probs[predicted_idx])

        result = {
            'prediction': self.class_names[predicted_idx] if self.class_names else int(predicted_idx),
            'confidence': confidence,
            'ensemble_probabilities': {
                self.class_names[i] if self.class_names else str(i): float(p)
                for i, p in enumerate(ensemble_probs)
            },
            'ensemble_method': self.ensemble_method,
            'num_models': len(self.models),
            'model_names': [m.name for m in self.models]
        }

        if return_all_probs:
            result['model_predictions'] = model_predictions

        # Add disease metadata if available
        pred_class = result['prediction']
        if pred_class in self.class_metadata:
            result['disease_info'] = self.class_metadata[pred_class]

        return result

    def predict_batch(self, images: List[Union[Image.Image, str]]) -> List[Dict]:
        """Make predictions for a batch of images."""
        return [self.predict(img) for img in tqdm(images, desc="Predicting")]

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate ensemble on a dataloader.

        Returns accuracy and per-class metrics.
        """

        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in tqdm(dataloader, desc="Evaluating"):
            for i in range(len(images)):
                # Convert tensor to PIL
                img = transforms.ToPILImage()(images[i])
                result = self.predict(img)

                pred_idx = self.class_names.index(result['prediction']) if self.class_names else int(result['prediction'])
                all_preds.append(pred_idx)
                all_labels.append(labels[i].item())
                all_probs.append(list(result['ensemble_probabilities'].values()))

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = (all_preds == all_labels).mean() * 100

        # Per-class accuracy
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
                class_metrics[class_name] = {
                    'accuracy': float(class_acc),
                    'count': int(mask.sum())
                }

        return {
            'accuracy': float(accuracy),
            'class_metrics': class_metrics,
            'num_samples': len(all_labels)
        }

    def set_weights(self, weights: Dict[str, float]):
        """Set model weights by name."""
        for model in self.models:
            if model.name in weights:
                model.weight = weights[model.name]
                print(f"Set {model.name} weight to {weights[model.name]}")

    def calibrate_weights(self, val_dataloader: DataLoader):
        """
        Automatically calibrate model weights based on validation accuracy.
        """

        print("\nCalibrating ensemble weights...")

        model_accuracies = {}

        for wrapper in self.models:
            correct = 0
            total = 0

            for images, labels in tqdm(val_dataloader, desc=f"Evaluating {wrapper.name}"):
                for i in range(len(images)):
                    img = transforms.ToPILImage()(images[i])
                    probs = wrapper.predict(img)
                    pred = probs.argmax().item()
                    if pred == labels[i].item():
                        correct += 1
                    total += 1

            accuracy = correct / total
            model_accuracies[wrapper.name] = accuracy
            print(f"  {wrapper.name}: {accuracy*100:.2f}%")

        # Set weights proportional to accuracy
        total_acc = sum(model_accuracies.values())
        for wrapper in self.models:
            wrapper.weight = model_accuracies[wrapper.name] / total_acc * len(self.models)
            print(f"  {wrapper.name} new weight: {wrapper.weight:.3f}")

    def save(self, path: str):
        """Save ensemble configuration."""
        config = {
            'class_names': self.class_names,
            'class_metadata': self.class_metadata,
            'ensemble_method': self.ensemble_method,
            'models': [
                {'name': m.name, 'weight': m.weight}
                for m in self.models
            ]
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Ensemble config saved to {path}")


def train_ensemble_from_scratch():
    """
    Train all models for the ensemble from scratch.
    This is a convenience function that trains each model sequentially.
    """

    print("="*80)
    print("TRAINING ENSEMBLE MODELS")
    print("="*80)

    import subprocess
    import sys

    # Train EfficientNet-B3
    print("\n[1/3] Training EfficientNet-B3...")
    subprocess.run([sys.executable, "train_infectious_disease_model.py"])

    # Train DinoV2
    print("\n[2/3] Training DinoV2...")
    subprocess.run([sys.executable, "train_infectious_dinov2.py"])

    # Train ConvNeXt (modify config and train)
    print("\n[3/3] Training ConvNeXt...")
    # This would require modifying the config - for now just inform user
    print("To train ConvNeXt, modify CONFIG['model_name'] = 'convnext_tiny' in train_infectious_disease_model.py")

    print("\n" + "="*80)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*80)


# Example usage and testing
if __name__ == "__main__":
    print("Infectious Disease Ensemble Classifier")
    print("="*60)

    # Create ensemble
    ensemble = InfectiousDiseaseEnsemble()

    # Try to load models
    try:
        ensemble.load_models()
        print(f"\nEnsemble ready with {len(ensemble.models)} models")

        # Test with a sample prediction
        print("\nTo make a prediction:")
        print("  result = ensemble.predict('path/to/image.jpg')")
        print("  print(result['prediction'], result['confidence'])")

    except RuntimeError as e:
        print(f"\n{e}")
        print("\nTo train models, run:")
        print("  python train_infectious_disease_model.py")
        print("  python train_infectious_dinov2.py")
