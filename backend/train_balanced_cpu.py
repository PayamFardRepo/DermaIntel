"""
Balanced Skin Cancer Classifier Training - CPU Optimized Version

Optimizations for CPU training:
1. Smaller batch size (8) to reduce memory usage
2. ResNet34 instead of ResNet50 (faster, less memory)
3. Smaller image size (192x192 instead of 224x224)
4. Fewer data loader workers
5. Gradient accumulation for effective larger batch size
6. Mixed precision disabled (CPU doesn't benefit)
7. Checkpointing after every epoch

Usage:
    python train_balanced_cpu.py --data_dir ./data/isic/organized --epochs 20
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
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# CPU optimizations
torch.set_num_threads(os.cpu_count())  # Use all CPU cores


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

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


def get_train_transforms(is_malignant: bool = False, img_size: int = 192) -> transforms.Compose:
    """Training transforms - smaller image size for CPU."""
    base_transforms = [
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
    ]

    if is_malignant:
        base_transforms.extend([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    else:
        base_transforms.extend([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])

    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transforms.Compose(base_transforms)


def get_val_transforms(img_size: int = 192) -> transforms.Compose:
    """Validation transforms."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class BalancedSkinDataset(Dataset):
    """Skin lesion dataset with class-aware augmentation."""

    MALIGNANT_LABELS = {'mel', 'melanoma', 'bcc', 'basal cell carcinoma', 'akiec',
                        'actinic keratosis', 'squamous cell carcinoma', 'scc', 'malignant'}

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform=None, malignant_transform=None, is_training: bool = True):
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

        if self.is_training and label == 1 and self.malignant_transform:
            image = self.malignant_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, label


def create_model(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Create ResNet34 model (lighter than ResNet50 for CPU)."""
    model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes)
    )

    return model


def compute_class_weights(labels: List[int], device: torch.device) -> torch.Tensor:
    """Compute class weights."""
    counter = Counter(labels)
    total = len(labels)

    weights = []
    for i in range(len(counter)):
        if i == 1:  # Malignant
            weight = (total / counter[i]) * 2.0
        else:
            weight = total / counter[i]
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)

    print(f"\nClass weights: Benign={weights[0]:.3f}, Malignant={weights[1]:.3f}")
    return weights.to(device)


def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create weighted sampler for balanced batches."""
    counter = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counter.items()}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, accumulation_steps=4):
    """Train one epoch with gradient accumulation."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

    # Final step if batches don't divide evenly
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = F.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            preds = (probs[:, 1] >= threshold).long()
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        'loss': running_loss / len(dataloader),
        'accuracy': (tp + tn) / (tp + tn + fp + fn) * 100,
        'sensitivity': sensitivity * 100,
        'specificity': specificity * 100,
        'precision': precision * 100,
        'f1': f1 * 100,
        'auc': auc * 100,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }, all_labels, all_probs


def load_isic_data(data_dir: str) -> Tuple[List[str], List[int]]:
    """Load ISIC dataset."""
    data_path = Path(data_dir)
    image_paths, labels = [], []

    benign_dir = data_path / "benign"
    malignant_dir = data_path / "malignant"

    if benign_dir.exists() and malignant_dir.exists():
        print("Loading from organized folders...")
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            for img in benign_dir.glob(ext):
                image_paths.append(str(img))
                labels.append(0)
            for img in malignant_dir.glob(ext):
                image_paths.append(str(img))
                labels.append(1)

    print(f"Loaded {len(image_paths)} images")
    print(f"  Benign: {labels.count(0)}, Malignant: {labels.count(1)}")
    print(f"  Imbalance ratio: {labels.count(0) / max(labels.count(1), 1):.1f}:1")

    return image_paths, labels


def plot_history(history: Dict, save_path: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Val')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()

    axes[1, 0].plot(history['val_sensitivity'], 'r-', label='Sensitivity')
    axes[1, 0].plot(history['val_specificity'], 'b-', label='Specificity')
    axes[1, 0].axhline(y=90, color='g', linestyle='--', label='90% target')
    axes[1, 0].set_title('Sensitivity vs Specificity')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].legend()

    axes[1, 1].plot(history['val_auc'], 'purple', label='AUC')
    axes[1, 1].set_title('AUC-ROC')
    axes[1, 1].set_ylim([50, 100])
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train skin cancer classifier (CPU optimized)")
    parser.add_argument("--data_dir", type=str, default="./data/isic/organized")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/balanced_cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)  # Small for CPU
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=192)  # Smaller images
    parser.add_argument("--accumulation_steps", type=int, default=4)  # Effective batch = 32
    parser.add_argument("--target_sensitivity", type=float, default=0.90)
    args = parser.parse_args()

    # Setup
    device = torch.device("cpu")
    print(f"Training on CPU with {os.cpu_count()} threads")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    image_paths, labels = load_isic_data(args.data_dir)
    if len(image_paths) == 0:
        print("ERROR: No images found!")
        return

    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    print(f"\nData split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    # Create datasets
    train_dataset = BalancedSkinDataset(
        train_paths, train_labels,
        transform=get_train_transforms(False, args.img_size),
        malignant_transform=get_train_transforms(True, args.img_size),
        is_training=True
    )
    val_dataset = BalancedSkinDataset(val_paths, val_labels, get_val_transforms(args.img_size), is_training=False)
    test_dataset = BalancedSkinDataset(test_paths, test_labels, get_val_transforms(args.img_size), is_training=False)

    # DataLoaders - fewer workers for CPU
    train_sampler = create_weighted_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = create_model(num_classes=2, pretrained=True, dropout=0.3)
    print(f"\nModel: ResNet34 (CPU optimized)")

    # Loss & optimizer
    class_weights = compute_class_weights(train_labels, device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'val_sensitivity': [], 'val_specificity': [], 'val_auc': [], 'val_f1': []
    }

    best_sensitivity = 0
    best_f1 = 0
    best_model_path = None
    start_epoch = 0

    # Resume from checkpoint
    checkpoint_path = output_dir / "latest_checkpoint.pth"
    if checkpoint_path.exists():
        print(f"\nResuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_sensitivity = checkpoint.get('best_sensitivity', 0)
        best_f1 = checkpoint.get('best_f1', 0)
        history = checkpoint.get('history', history)
        print(f"Resuming from epoch {start_epoch + 1}, best sensitivity: {best_sensitivity:.1f}%")

    # Training loop
    print(f"\n{'='*50}")
    print(f"TRAINING - Target: {args.target_sensitivity*100:.0f}% sensitivity")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.accumulation_steps
        )
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
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

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  Sensitivity: {val_metrics['sensitivity']:.1f}% | Specificity: {val_metrics['specificity']:.1f}%")
        print(f"  F1: {val_metrics['f1']:.1f}% | AUC: {val_metrics['auc']:.1f}%")

        # Save best model
        if val_metrics['sensitivity'] >= args.target_sensitivity * 100 and val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_sensitivity = val_metrics['sensitivity']
            best_model_path = output_dir / "best_model.pth"
            torch.save({'model_state_dict': model.state_dict(), 'metrics': val_metrics}, best_model_path)
            print(f"  * New best model! (Sens: {best_sensitivity:.1f}%, F1: {best_f1:.1f}%)")
        elif val_metrics['sensitivity'] > best_sensitivity:
            best_sensitivity = val_metrics['sensitivity']
            best_model_path = output_dir / "best_model.pth"
            torch.save({'model_state_dict': model.state_dict(), 'metrics': val_metrics}, best_model_path)
            print(f"  * New best sensitivity: {best_sensitivity:.1f}%")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_sensitivity': best_sensitivity,
            'best_f1': best_f1,
            'history': history,
        }, checkpoint_path)
        print(f"  Checkpoint saved")

    # Final evaluation
    print(f"\n{'='*50}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*50}")

    if best_model_path and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])

    test_metrics, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"Sensitivity: {test_metrics['sensitivity']:.1f}%")
    print(f"Specificity: {test_metrics['specificity']:.1f}%")
    print(f"F1 Score: {test_metrics['f1']:.1f}%")
    print(f"AUC: {test_metrics['auc']:.1f}%")

    # Save results
    plot_history(history, str(output_dir / "training_curves.png"))

    with open(output_dir / "results.json", "w") as f:
        json.dump({'config': vars(args), 'test_metrics': test_metrics, 'history': history}, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
