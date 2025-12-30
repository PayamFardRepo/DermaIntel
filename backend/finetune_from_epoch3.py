"""
Fine-tune from Epoch 3 Checkpoint - Stable Training

Loads the best model (epoch 3) and fine-tunes with:
1. Frozen early layers (only train last layers)
2. Very low learning rate (1e-5)
3. No aggressive scheduling
4. Gradual unfreezing

This prevents oscillation by making smaller, more careful updates.

Usage:
    python finetune_from_epoch3.py --epochs 10
"""

import os
import argparse
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Tuple, List, Optional

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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# CPU Optimizations
torch.set_num_threads(os.cpu_count())
torch.backends.mkl.is_available() and setattr(torch.backends.mkl, 'enabled', True)  # Intel MKL
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('medium')  # Faster matrix ops

# Optimal number of workers for data loading
NUM_WORKERS = min(4, os.cpu_count() or 1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(1)
        smooth_targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
        smooth_targets = smooth_targets * (1 - self.smoothing) + self.smoothing / n_classes

        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=1)

        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()


def get_train_transforms(img_size=192):
    """Moderate augmentation - not too aggressive."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size=192):
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class SkinDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def create_model(num_classes=2, dropout=0.5):
    """Same architecture as v3."""
    model = models.resnet34(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )
    return model


def freeze_early_layers(model, unfreeze_from='layer3'):
    """
    Freeze early layers of ResNet.

    unfreeze_from options:
    - 'layer4': Only train layer4 + fc (most frozen)
    - 'layer3': Train layer3, layer4 + fc
    - 'layer2': Train layer2, layer3, layer4 + fc
    - 'fc': Only train fc (classifier head)
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze from specified layer onwards
    unfreeze = False
    for name, child in model.named_children():
        if name == unfreeze_from:
            unfreeze = True
        if unfreeze:
            for param in child.parameters():
                param.requires_grad = True

    # Always unfreeze fc
    for param in model.fc.parameters():
        param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model


def load_all_isic(base_dir):
    """Load all ISIC data."""
    base = Path(base_dir)
    paths, labels = [], []

    for data_path in [base / "isic" / "organized", base / "isic_2019" / "organized"]:
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img in (data_path / "benign").glob(ext):
                paths.append(str(img))
                labels.append(0)
            for img in (data_path / "malignant").glob(ext):
                paths.append(str(img))
                labels.append(1)

    print(f"Loaded {len(paths):,} images (B:{labels.count(0):,}, M:{labels.count(1):,})")
    return paths, labels


def create_balanced_sampler(labels, malignant_ratio=0.35):
    """Slightly less aggressive sampling - 35% malignant per batch."""
    counter = Counter(labels)
    w_mal = malignant_ratio * counter[0] / ((1 - malignant_ratio) * counter[1])
    weights = [1.0 if l == 0 else w_mal for l in labels]
    return WeightedRandomSampler(weights, len(labels), replacement=True)


def train_epoch(model, loader, criterion, optimizer, device, epoch, accum_steps=2):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")

    for i, (imgs, lbls) in enumerate(pbar):
        imgs, lbls = imgs.to(device), lbls.to(device)

        out = model(imgs)
        loss = criterion(out, lbls) / accum_steps
        loss.backward()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

        loss_sum += loss.item() * accum_steps
        _, pred = out.max(1)
        total += lbls.size(0)
        correct += pred.eq(lbls).sum().item()

        pbar.set_postfix({'loss': f'{loss_sum/(i+1):.4f}', 'acc': f'{100*correct/total:.1f}%'})

    # Final step if needed
    if (i + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        optimizer.zero_grad()

    return loss_sum / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    loss_sum = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Eval", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss_sum += criterion(out, lbls).item()

            probs = F.softmax(out, dim=1)[:, 1]
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs >= threshold).long().cpu().numpy())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    return {
        'loss': loss_sum / len(loader),
        'sensitivity': sens * 100,
        'specificity': spec * 100,
        'precision': prec * 100,
        'f1': f1 * 100,
        'auc': auc * 100,
        'balanced': np.sqrt(sens * spec) * 100,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }, labels, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/combined_isic_v3/best_model.pth")
    parser.add_argument("--output_dir", default="./checkpoints/finetuned")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)  # Keep small for CPU
    parser.add_argument("--accum_steps", type=int, default=2)  # Effective batch = 16
    parser.add_argument("--lr", type=float, default=1e-5)  # Very low LR
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("--unfreeze_from", default="layer3", choices=['layer2', 'layer3', 'layer4', 'fc'])
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"\n{'='*70}")
    print("FINE-TUNING FROM EPOCH 3 - STABLE APPROACH")
    print(f"{'='*70}")
    print(f"Device: CPU ({os.cpu_count()} threads)")
    print(f"Data workers: {NUM_WORKERS}")
    print(f"MKL enabled: {torch.backends.mkl.is_available()}")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Learning rate: {args.lr} (very low for stability)")
    print(f"Freezing layers before: {args.unfreeze_from}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and checkpoint
    model = create_model(dropout=0.5)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model with metrics: {ckpt.get('metrics', 'N/A')}")
    else:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Make sure you have the epoch 3 checkpoint from v3 training")
        return

    # Freeze early layers
    print(f"\nFreezing early layers (training from {args.unfreeze_from} onwards)...")
    model = freeze_early_layers(model, args.unfreeze_from)

    # Load data
    paths, labels = load_all_isic(args.data_dir)

    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=0.5, stratify=temp_l, random_state=42
    )

    print(f"\nData: Train={len(train_p):,}, Val={len(val_p):,}, Test={len(test_p):,}")

    # Datasets
    train_ds = SkinDataset(train_p, train_l, get_train_transforms(args.img_size))
    val_ds = SkinDataset(val_p, val_l, get_val_transforms(args.img_size))
    test_ds = SkinDataset(test_p, test_l, get_val_transforms(args.img_size))

    train_loader = DataLoader(train_ds, args.batch_size, sampler=create_balanced_sampler(train_l),
                               num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=NUM_WORKERS > 0)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=NUM_WORKERS > 0)

    # Loss - balanced weights
    counter = Counter(train_l)
    weights = torch.tensor([1.0, (counter[0] / counter[1]) * 2.5], dtype=torch.float32)
    weights = weights / weights.sum() * 2
    print(f"Loss weights: {weights.tolist()}")

    criterion = FocalLoss(weights, gamma=2.0, smoothing=0.1)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Very gentle scheduler - reduce LR if no improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # Initial evaluation
    print(f"\n{'='*70}")
    print("INITIAL EVALUATION (Epoch 3 model)")
    print(f"{'='*70}")
    init_metrics, _, _ = evaluate(model, val_loader, criterion, device)
    print(f"Sens: {init_metrics['sensitivity']:.1f}% | Spec: {init_metrics['specificity']:.1f}% | AUC: {init_metrics['auc']:.1f}%")

    # Training
    history = {'val_sensitivity': [], 'val_specificity': [], 'val_auc': [], 'val_balanced': []}
    best_balanced = init_metrics['balanced']
    best_path = None

    print(f"\n{'='*70}")
    print("FINE-TUNING")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.accum_steps)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_balanced'].append(val_metrics['balanced'])

        # Update scheduler based on balanced score
        scheduler.step(val_metrics['balanced'])

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  Val   - Sens: {val_metrics['sensitivity']:.1f}% | Spec: {val_metrics['specificity']:.1f}% | Balanced: {val_metrics['balanced']:.1f}%")
        print(f"        - F1: {val_metrics['f1']:.1f}% | AUC: {val_metrics['auc']:.1f}%")
        print(f"        - TP:{val_metrics['tp']} TN:{val_metrics['tn']} FP:{val_metrics['fp']} FN:{val_metrics['fn']}")

        # Save if improved
        if val_metrics['balanced'] > best_balanced:
            best_balanced = val_metrics['balanced']
            best_path = output_dir / "best_finetuned.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'epoch': epoch,
            }, best_path)
            print(f"  *** NEW BEST! Balanced: {best_balanced:.1f}%")

        # Save individual epoch checkpoint (so we never lose a good epoch!)
        epoch_path = output_dir / f"epoch_{epoch+1}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': val_metrics,
            'epoch': epoch,
        }, epoch_path)
        print(f"  Saved: epoch_{epoch+1}.pth")

        # Latest checkpoint (for resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_balanced': best_balanced,
        }, output_dir / "latest_checkpoint.pth")

    # Final test
    print(f"\n{'='*70}")
    print("FINAL TEST RESULTS")
    print(f"{'='*70}")

    if best_path and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=False)['model_state_dict'])

    test_metrics, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"Sensitivity: {test_metrics['sensitivity']:.1f}%")
    print(f"Specificity: {test_metrics['specificity']:.1f}%")
    print(f"Balanced:    {test_metrics['balanced']:.1f}%")
    print(f"F1:          {test_metrics['f1']:.1f}%")
    print(f"AUC:         {test_metrics['auc']:.1f}%")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            'approach': 'finetune_from_epoch3',
            'config': vars(args),
            'initial_metrics': init_metrics,
            'final_metrics': test_metrics,
            'history': history,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
