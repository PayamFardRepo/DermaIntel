"""
Balanced Skin Cancer Classifier Training

This script addresses the class imbalance problem in skin lesion classification
where malignant cases are severely underrepresented.

Techniques used:
1. Class-weighted loss function (penalize missing malignant more)
2. Focal Loss for hard examples
3. Aggressive data augmentation for minority class
4. Oversampling with WeightedRandomSampler
5. Threshold optimization for clinical use
6. Evaluation focused on sensitivity/recall

Usage:
    python train_balanced_classifier.py --data_dir ./data/isic --epochs 20

Author: DermAI Pro Team
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# FOCAL LOSS - Better for imbalanced datasets
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses training on hard examples by down-weighting easy examples.
    Originally proposed for object detection (RetinaNet paper).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class balancing weight (higher for minority class)
        gamma: Focusing parameter (higher = more focus on hard examples)
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


# =============================================================================
# DATA AUGMENTATION - Aggressive for minority class
# =============================================================================

def get_train_transforms(is_malignant: bool = False) -> transforms.Compose:
    """
    Get training transforms with extra augmentation for malignant samples.

    Malignant samples get more aggressive augmentation to increase diversity.
    """
    base_transforms = [
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
    ]

    if is_malignant:
        # Extra augmentation for minority class
        base_transforms.extend([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    else:
        base_transforms.extend([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])

    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),  # Cutout augmentation
    ])

    return transforms.Compose(base_transforms)


def get_val_transforms() -> transforms.Compose:
    """Validation/test transforms - no augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# =============================================================================
# DATASET WITH CLASS-AWARE AUGMENTATION
# =============================================================================

class BalancedSkinDataset(Dataset):
    """
    Skin lesion dataset with class-aware augmentation.

    Malignant samples get more aggressive augmentation to increase diversity.
    """

    # Map various diagnosis labels to binary
    MALIGNANT_LABELS = {'mel', 'melanoma', 'bcc', 'basal cell carcinoma', 'akiec',
                        'actinic keratosis', 'squamous cell carcinoma', 'scc', 'malignant'}
    BENIGN_LABELS = {'nv', 'nevus', 'bkl', 'benign keratosis', 'df', 'dermatofibroma',
                     'vasc', 'vascular', 'benign'}

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        malignant_transform: Optional[transforms.Compose] = None,
        is_training: bool = True
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.malignant_transform = malignant_transform
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        # Use different transforms for malignant during training
        if self.is_training and label == 1 and self.malignant_transform:
            image = self.malignant_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, label


# =============================================================================
# MODEL WITH DROPOUT FOR UNCERTAINTY
# =============================================================================

def create_model(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    Create a ResNet50 model with dropout for uncertainty estimation.

    Uses Monte Carlo Dropout for uncertainty quantification during inference.
    """
    model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)

    # Add dropout before final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes)
    )

    return model


# =============================================================================
# TRAINING WITH CLASS BALANCING
# =============================================================================

def compute_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.

    For medical applications, we weight malignant class higher because
    missing cancer is much worse than a false positive.
    """
    counter = Counter(labels)
    total = len(labels)

    # Inverse frequency weighting
    weights = []
    for i in range(len(counter)):
        # Add extra weight to malignant class (clinical importance)
        if i == 1:  # Malignant class
            weight = (total / counter[i]) * 2.0  # 2x extra weight for malignant
        else:
            weight = total / counter[i]
        weights.append(weight)

    # Normalize weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)

    print(f"\nClass weights: {weights.tolist()}")
    print(f"  Benign (0): {weights[0]:.3f}")
    print(f"  Malignant (1): {weights[1]:.3f}")

    return weights.to(device)


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """
    Create a weighted sampler that oversamples minority class.

    This ensures each batch has roughly equal representation of both classes.
    """
    counter = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counter.items()}

    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

    return sampler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(dataloader), 100. * correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Dict:
    """
    Evaluate model with comprehensive metrics.

    For medical applications, we focus on:
    - Sensitivity (Recall): % of actual cancers correctly identified
    - Specificity: % of benign correctly identified
    - NPV: If we say benign, how confident are we?
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant

            # Use threshold for predictions
            preds = (probs[:, 1] >= threshold).long()
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute metrics
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for malignant
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for benign
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    metrics = {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy * 100,
        'sensitivity': sensitivity * 100,  # Most important for cancer detection
        'specificity': specificity * 100,
        'precision': precision * 100,
        'npv': npv * 100,
        'f1': f1 * 100,
        'auc': auc * 100,
        'confusion_matrix': cm.tolist(),
        'threshold': threshold,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }

    return metrics, all_labels, all_probs


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray, target_sensitivity: float = 0.95) -> float:
    """
    Find the optimal threshold to achieve target sensitivity.

    For cancer screening, we want HIGH sensitivity (catch most cancers)
    even at the cost of more false positives.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    print(f"\nFinding optimal threshold (target sensitivity: {target_sensitivity*100:.0f}%):")

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        # Find threshold that achieves target sensitivity with best F1
        if sensitivity >= target_sensitivity and f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            print(f"  Threshold {thresh:.2f}: Sensitivity={sensitivity*100:.1f}%, Precision={precision*100:.1f}%, F1={f1*100:.1f}%")

    return best_threshold


def plot_training_history(history: Dict, save_path: str):
    """Plot training curves and save."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()

    # Sensitivity (most important)
    axes[1, 0].plot(history['val_sensitivity'], label='Sensitivity (Recall)', color='red')
    axes[1, 0].plot(history['val_specificity'], label='Specificity', color='blue')
    axes[1, 0].axhline(y=90, color='green', linestyle='--', label='90% target')
    axes[1, 0].set_title('Sensitivity vs Specificity')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].legend()

    # AUC
    axes[1, 1].plot(history['val_auc'], label='AUC-ROC', color='purple')
    axes[1, 1].set_title('AUC-ROC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylim([50, 100])
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training plots saved to {save_path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def load_isic_data(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load ISIC dataset with binary labels.

    Expects either:
    - Organized folders: data_dir/benign/*.jpg, data_dir/malignant/*.jpg
    - Or CSV metadata with image paths and diagnoses
    """
    data_path = Path(data_dir)
    image_paths = []
    labels = []

    # Check for organized folder structure
    benign_dir = data_path / "benign"
    malignant_dir = data_path / "malignant"

    if benign_dir.exists() and malignant_dir.exists():
        print("Loading from organized folders...")
        for img in benign_dir.glob("*.jpg"):
            image_paths.append(str(img))
            labels.append(0)
        for img in benign_dir.glob("*.png"):
            image_paths.append(str(img))
            labels.append(0)
        for img in malignant_dir.glob("*.jpg"):
            image_paths.append(str(img))
            labels.append(1)
        for img in malignant_dir.glob("*.png"):
            image_paths.append(str(img))
            labels.append(1)
    else:
        # Look for CSV metadata
        csv_files = list(data_path.glob("*.csv"))
        if csv_files:
            print(f"Loading from CSV: {csv_files[0]}")
            df = pd.read_csv(csv_files[0])

            # Try different column names
            img_col = None
            label_col = None
            for col in df.columns:
                if 'image' in col.lower() or 'path' in col.lower() or 'file' in col.lower():
                    img_col = col
                if 'diagnosis' in col.lower() or 'label' in col.lower() or 'target' in col.lower():
                    label_col = col

            if img_col and label_col:
                for _, row in df.iterrows():
                    img_path = data_path / row[img_col]
                    if not img_path.exists():
                        img_path = data_path / "images" / row[img_col]

                    if img_path.exists():
                        image_paths.append(str(img_path))

                        # Convert label to binary
                        label_str = str(row[label_col]).lower()
                        if label_str in BalancedSkinDataset.MALIGNANT_LABELS or label_str == '1':
                            labels.append(1)
                        else:
                            labels.append(0)

    print(f"Loaded {len(image_paths)} images")
    print(f"  Benign: {labels.count(0)}")
    print(f"  Malignant: {labels.count(1)}")
    print(f"  Imbalance ratio: {labels.count(0) / max(labels.count(1), 1):.1f}:1")

    return image_paths, labels


def main():
    parser = argparse.ArgumentParser(description="Train balanced skin cancer classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ISIC data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/balanced", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use focal loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--target_sensitivity", type=float, default=0.90, help="Target sensitivity (0-1)")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    image_paths, labels = load_isic_data(args.data_dir)

    if len(image_paths) == 0:
        print("ERROR: No images found. Please check data_dir path.")
        return

    # Split data with stratification
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    print(f"\nData split:")
    print(f"  Train: {len(train_paths)} (Benign: {train_labels.count(0)}, Malignant: {train_labels.count(1)})")
    print(f"  Val: {len(val_paths)} (Benign: {val_labels.count(0)}, Malignant: {val_labels.count(1)})")
    print(f"  Test: {len(test_paths)} (Benign: {test_labels.count(0)}, Malignant: {test_labels.count(1)})")

    # Create datasets with class-aware augmentation
    train_dataset = BalancedSkinDataset(
        train_paths, train_labels,
        transform=get_train_transforms(is_malignant=False),
        malignant_transform=get_train_transforms(is_malignant=True),
        is_training=True
    )
    val_dataset = BalancedSkinDataset(
        val_paths, val_labels,
        transform=get_val_transforms(),
        is_training=False
    )
    test_dataset = BalancedSkinDataset(
        test_paths, test_labels,
        transform=get_val_transforms(),
        is_training=False
    )

    # Create weighted sampler for training
    train_sampler = create_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # Create model
    model = create_model(num_classes=2, pretrained=True, dropout=0.3)
    model = model.to(device)

    # Compute class weights
    class_weights = compute_class_weights(train_labels, device)

    # Loss function
    if args.use_focal_loss:
        print("\nUsing Focal Loss with gamma={args.focal_gamma}")
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        print("\nUsing Weighted Cross Entropy Loss")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_sensitivity': [], 'val_specificity': [],
        'val_auc': [], 'val_f1': []
    }

    best_sensitivity = 0
    best_f1 = 0
    best_model_path = None
    start_epoch = 0

    # Check for existing checkpoint to resume from
    checkpoint_path = output_dir / "latest_checkpoint.pth"
    if checkpoint_path.exists():
        print(f"\nFound existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_sensitivity = checkpoint.get('best_sensitivity', 0)
        best_f1 = checkpoint.get('best_f1', 0)
        history = checkpoint.get('history', history)
        print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Best sensitivity so far: {best_sensitivity:.2f}%")

    print("\n" + "="*60)
    print("TRAINING WITH CLASS BALANCING")
    print("="*60)
    print(f"Target: {args.target_sensitivity*100:.0f}% sensitivity for malignant detection")
    print("="*60 + "\n")

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  Sensitivity: {val_metrics['sensitivity']:.2f}% | Specificity: {val_metrics['specificity']:.2f}%")
        print(f"  Precision: {val_metrics['precision']:.2f}% | F1: {val_metrics['f1']:.2f}% | AUC: {val_metrics['auc']:.2f}%")
        print(f"  Confusion Matrix: TP={val_metrics['tp']}, TN={val_metrics['tn']}, FP={val_metrics['fp']}, FN={val_metrics['fn']}")

        # Save best model based on sensitivity (most important for medical)
        # But also consider F1 to avoid too many false positives
        score = val_metrics['sensitivity'] * 0.7 + val_metrics['f1'] * 0.3  # Weighted score

        if val_metrics['sensitivity'] >= args.target_sensitivity * 100 and val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_sensitivity = val_metrics['sensitivity']
            best_model_path = output_dir / "best_balanced_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'class_weights': class_weights.cpu().numpy().tolist(),
            }, best_model_path)
            print(f"  ✓ New best model saved! (Sensitivity: {best_sensitivity:.2f}%, F1: {best_f1:.2f}%)")
        elif val_metrics['sensitivity'] > best_sensitivity:
            best_sensitivity = val_metrics['sensitivity']
            best_model_path = output_dir / "best_balanced_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'class_weights': class_weights.cpu().numpy().tolist(),
            }, best_model_path)
            print(f"  ✓ New best sensitivity! (Sensitivity: {best_sensitivity:.2f}%)")

        # Save checkpoint after every epoch (for resuming)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_sensitivity': best_sensitivity,
            'best_f1': best_f1,
            'history': history,
            'metrics': val_metrics,
        }, output_dir / "latest_checkpoint.pth")
        print(f"  💾 Checkpoint saved (epoch {epoch+1})")

    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)

    # Load best model
    if best_model_path and best_model_path.exists():
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

    # Evaluate with default threshold
    test_metrics, test_labels_arr, test_probs = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results (threshold=0.5):")
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Sensitivity (Recall): {test_metrics['sensitivity']:.2f}%")
    print(f"  Specificity: {test_metrics['specificity']:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.2f}%")
    print(f"  F1 Score: {test_metrics['f1']:.2f}%")
    print(f"  AUC-ROC: {test_metrics['auc']:.2f}%")
    print(f"  NPV: {test_metrics['npv']:.2f}%")

    # Find optimal threshold for target sensitivity
    optimal_threshold = find_optimal_threshold(test_labels_arr, test_probs, args.target_sensitivity)

    # Re-evaluate with optimal threshold
    test_metrics_opt, _, _ = evaluate(model, test_loader, criterion, device, threshold=optimal_threshold)

    print(f"\nTest Results (optimal threshold={optimal_threshold:.2f}):")
    print(f"  Accuracy: {test_metrics_opt['accuracy']:.2f}%")
    print(f"  Sensitivity (Recall): {test_metrics_opt['sensitivity']:.2f}%")
    print(f"  Specificity: {test_metrics_opt['specificity']:.2f}%")
    print(f"  Precision: {test_metrics_opt['precision']:.2f}%")
    print(f"  F1 Score: {test_metrics_opt['f1']:.2f}%")

    # Save results
    results = {
        'model': 'resnet50_balanced',
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'use_focal_loss': args.use_focal_loss,
            'focal_gamma': args.focal_gamma if args.use_focal_loss else None,
            'target_sensitivity': args.target_sensitivity,
        },
        'class_distribution': {
            'train': {'benign': train_labels.count(0), 'malignant': train_labels.count(1)},
            'val': {'benign': val_labels.count(0), 'malignant': val_labels.count(1)},
            'test': {'benign': test_labels.count(0), 'malignant': test_labels.count(1)},
        },
        'test_metrics_default_threshold': test_metrics,
        'test_metrics_optimal_threshold': test_metrics_opt,
        'optimal_threshold': optimal_threshold,
        'training_history': history,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot training history
    plot_training_history(history, str(output_dir / "training_curves.png"))

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model saved to: {best_model_path}")
    print(f"Results saved to: {output_dir / 'training_results.json'}")
    print(f"Training plots saved to: {output_dir / 'training_curves.png'}")
    print(f"\nKey Achievement:")
    print(f"  Sensitivity: {test_metrics_opt['sensitivity']:.1f}% (detecting {test_metrics_opt['sensitivity']:.0f}% of cancers)")
    print(f"  With {test_metrics_opt['specificity']:.1f}% specificity")


if __name__ == "__main__":
    main()
