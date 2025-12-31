"""
Improved Infectious Disease Model Training
==========================================
Uses the same techniques that achieved 94.3% AUC on ISIC:
- Heavy augmentation for minority classes
- Weighted sampling
- Focal loss with label smoothing
- Low learning rate with warmup
- ResNet34 or EfficientNet-B3

Usage:
    python train_infectious_improved.py --data ./data/infectious_merged --epochs 20

Requirements:
    pip install torch torchvision albumentations tqdm scikit-learn
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class InfectiousDataset(Dataset):
    """Dataset with heavy augmentation for minority classes."""

    def __init__(self, image_paths, labels, class_names, transform=None,
                 augment_minority=True, minority_threshold=500):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.augment_minority = augment_minority
        self.minority_threshold = minority_threshold

        # Calculate class counts
        self.class_counts = Counter(labels)
        self.minority_classes = {
            cls for cls, count in self.class_counts.items()
            if count < minority_threshold
        }

        # Heavy augmentation transforms for minority classes
        self.heavy_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        # Apply heavy augmentation to minority classes
        if self.augment_minority and label in self.minority_classes:
            image = self.heavy_augment(image)

        # Apply standard transform
        if self.transform:
            image = self.transform(image)

        return image, label


def load_data(data_dir: Path):
    """Load image paths and labels from organized directory structure."""
    image_paths = []
    labels = []
    class_names = []

    # Get class names from directory structure
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    print(f"Found {len(class_names)} classes:")
    for idx, class_dir in enumerate(class_dirs):
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        print(f"  {idx}: {class_dir.name} - {len(images)} images")

        for img_path in images:
            image_paths.append(img_path)
            labels.append(idx)

    return image_paths, labels, class_names


def create_weighted_sampler(labels, minority_weight=0.4):
    """Create weighted sampler to oversample minority classes."""
    class_counts = Counter(labels)
    total = len(labels)

    # Calculate weights to achieve desired minority representation
    max_count = max(class_counts.values())

    weights = []
    for label in labels:
        # Higher weight for minority classes
        class_weight = max_count / class_counts[label]
        weights.append(class_weight)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler


def create_model(num_classes, architecture='resnet34', pretrained=True):
    """Create model with specified architecture."""
    if architecture == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, num_classes)
        )
    elif architecture == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, num_classes)
        )
    elif architecture == 'efficientnet_b3':
        model = models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def train_epoch(model, dataloader, criterion, optimizer, device, accum_steps=2):
    """Train for one epoch with gradient accumulation."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels) / accum_steps

        loss.backward()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'loss': f"{running_loss/(i+1):.4f}",
            'acc': f"{100*accuracy_score(all_labels, all_preds):.1f}%"
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, class_names):
    """Evaluate model and return detailed metrics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # AUC (one-vs-rest)
    try:
        all_probs = np.array(all_probs)
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        auc = 0.0

    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'auc': auc * 100,
    }

    return metrics, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train improved infectious disease model")
    parser.add_argument("--data", type=str, required=True, help="Path to organized data directory")
    parser.add_argument("--output", type=str, default="./checkpoints/infectious_improved",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--architecture", type=str, default="resnet34",
                        choices=['resnet34', 'resnet50', 'efficientnet_b3'],
                        help="Model architecture")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs")

    args = parser.parse_args()

    # Setup
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cpu':
        num_threads = os.cpu_count()
        torch.set_num_threads(num_threads)
        print(f"Using {num_threads} CPU threads")

    # Load data
    print("\nLoading data...")
    image_paths, labels, class_names = load_data(data_dir)
    num_classes = len(class_names)

    print(f"\nTotal images: {len(image_paths)}")
    print(f"Number of classes: {num_classes}")

    # Split data (80% train, 10% val, 10% test)
    indices = list(range(len(image_paths)))
    random.shuffle(indices)

    train_size = int(0.8 * len(indices))
    val_size = int(0.1 * len(indices))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size + 32, args.img_size + 32)),
        transforms.RandomCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = InfectiousDataset(
        train_paths, train_labels, class_names,
        transform=train_transform, augment_minority=True
    )
    val_dataset = InfectiousDataset(
        val_paths, val_labels, class_names,
        transform=val_transform, augment_minority=False
    )
    test_dataset = InfectiousDataset(
        test_paths, test_labels, class_names,
        transform=val_transform, augment_minority=False
    )

    # Weighted sampler for training
    train_sampler = create_weighted_sampler(train_labels)

    # DataLoaders
    num_workers = 0 if device.type == 'cpu' else 4
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Model
    print(f"\nCreating {args.architecture} model...")
    model = create_model(num_classes, args.architecture)
    model = model.to(device)

    # Calculate class weights for focal loss
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = torch.tensor([
        total_samples / (num_classes * class_counts[i])
        for i in range(num_classes)
    ], dtype=torch.float32).to(device)

    # Loss and optimizer
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    best_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.accum_steps
        )

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, class_names)

        # Update scheduler
        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_metrics'].append(val_metrics)

        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc*100:.1f}%")
        print(f"  Val   - Acc: {val_metrics['accuracy']:.1f}% | F1: {val_metrics['f1']:.1f}% | AUC: {val_metrics['auc']:.1f}%")

        # Save checkpoint for every epoch
        checkpoint_path = output_dir / f"epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'class_names': class_names,
        }, checkpoint_path)

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_path = output_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'class_names': class_names,
                'architecture': args.architecture,
            }, best_path)
            print(f"  *** NEW BEST! F1: {best_f1:.1f}%")

    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)

    # Load best model
    best_checkpoint = torch.load(output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_metrics, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, class_names
    )

    print(f"Accuracy:  {test_metrics['accuracy']:.1f}%")
    print(f"Precision: {test_metrics['precision']:.1f}%")
    print(f"Recall:    {test_metrics['recall']:.1f}%")
    print(f"F1 Score:  {test_metrics['f1']:.1f}%")
    print(f"AUC:       {test_metrics['auc']:.1f}%")

    # Classification report
    print("\nPer-class results:")
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    # Save results
    results = {
        'config': vars(args),
        'class_names': class_names,
        'best_epoch': best_checkpoint['epoch'] + 1,
        'test_metrics': test_metrics,
        'history': history,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Training complete!")
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")
    print(f"Results saved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
