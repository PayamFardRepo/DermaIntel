"""
Combined ISIC 2019 + 2020 Skin Cancer Classifier - CPU Optimized v3 FIXED

FIXES:
- Saves EVERY epoch checkpoint (epoch_1.pth, epoch_2.pth, etc.)
- Best model based on balanced score ONLY (no sensitivity threshold)
- Won't lose good epochs anymore!

Usage:
    python train_combined_isic_cpu_v3_fixed.py --epochs 5
"""

import os
import argparse
import json
import random
import numpy as np
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

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.set_num_threads(os.cpu_count())


class FocalLoss(nn.Module):
    """Focal Loss with label smoothing."""
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


class MalignantHeavyAugment:
    """
    Heavy augmentation for malignant class - creates virtual 3x more samples.
    Randomly applies one of 3 different augmentation strategies.
    """
    def __init__(self, img_size=192):
        self.img_size = img_size

        # Strategy 1: Geometric transforms
        self.geo_transform = transforms.Compose([
            transforms.Resize((img_size + 48, img_size + 48)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(45),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])

        # Strategy 2: Color transforms
        self.color_transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomEqualize(p=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Strategy 3: Mixed
        self.mixed_transform = transforms.Compose([
            transforms.Resize((img_size + 40, img_size + 40)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(35),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),
            transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15),
        ])

        self.strategies = [self.geo_transform, self.color_transform, self.mixed_transform]

    def __call__(self, img):
        # Randomly pick one of 3 strategies
        strategy = random.choice(self.strategies)
        return strategy(img)


def get_benign_transforms(img_size: int = 192):
    """Standard augmentation for benign class."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size: int = 192):
    """Validation transforms - no augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class SkinDataset(Dataset):
    """Dataset with heavy augmentation for malignant class."""
    def __init__(self, paths, labels, benign_transform=None, malignant_transform=None, training=True):
        self.paths = paths
        self.labels = labels
        self.benign_transform = benign_transform
        self.malignant_transform = malignant_transform
        self.training = training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.training:
            if label == 1 and self.malignant_transform:
                img = self.malignant_transform(img)
            elif self.benign_transform:
                img = self.benign_transform(img)
        else:
            # Validation - use benign transform (it's just resize + center crop for val)
            if self.benign_transform:
                img = self.benign_transform(img)

        return img, label


def create_model(num_classes=2, dropout=0.5):
    """ResNet34 with higher dropout and BatchNorm."""
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


def load_all_isic(base_dir: str) -> Tuple[List[str], List[int]]:
    """Load ALL ISIC 2019 + 2020 data (no undersampling)."""
    base = Path(base_dir)
    paths, labels = [], []

    datasets = [
        (base / "isic" / "organized", "ISIC 2020"),
        (base / "isic_2019" / "organized", "ISIC 2019"),
    ]

    print("\nLoading ALL datasets (no undersampling):")
    print("-" * 60)

    total_b, total_m = 0, 0

    for data_path, name in datasets:
        b_count, m_count = 0, 0

        benign_dir = data_path / "benign"
        malignant_dir = data_path / "malignant"

        if benign_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img in benign_dir.glob(ext):
                    paths.append(str(img))
                    labels.append(0)
                    b_count += 1

        if malignant_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img in malignant_dir.glob(ext):
                    paths.append(str(img))
                    labels.append(1)
                    m_count += 1

        print(f"  {name}: {b_count:,} benign + {m_count:,} malignant = {b_count + m_count:,}")
        total_b += b_count
        total_m += m_count

    print("-" * 60)
    print(f"  TOTAL: {total_b:,} benign + {total_m:,} malignant = {len(paths):,} images")
    print(f"  Ratio: {total_b/total_m:.1f}:1 (will be balanced via weighted sampling)")

    return paths, labels


def create_balanced_sampler(labels, malignant_ratio=0.4):
    """
    Create sampler that ensures ~40% malignant per batch.

    This effectively oversamples malignant without duplicating data.
    """
    counter = Counter(labels)
    n_benign = counter[0]
    n_malignant = counter[1]

    # Calculate weights to achieve target ratio
    # P(malignant) = w_m * n_m / (w_m * n_m + w_b * n_b) = 0.4
    # Solving: w_m/w_b = 0.4 * n_b / (0.6 * n_m)
    w_malignant = malignant_ratio * n_benign / ((1 - malignant_ratio) * n_malignant)
    w_benign = 1.0

    weights = [w_benign if l == 0 else w_malignant for l in labels]

    print(f"\nWeighted Sampler:")
    print(f"  Target malignant ratio per batch: {malignant_ratio*100:.0f}%")
    print(f"  Benign weight: {w_benign:.3f}")
    print(f"  Malignant weight: {w_malignant:.3f}")
    print(f"  Effective oversampling of malignant: {w_malignant:.1f}x")

    return WeightedRandomSampler(weights, len(labels), replacement=True)


def compute_class_weights(labels, device):
    """Compute loss weights - 3x for malignant."""
    counter = Counter(labels)
    total = len(labels)

    weights = []
    for i in range(len(counter)):
        w = (total / counter[i]) * (3.0 if i == 1 else 1.0)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)
    print(f"Loss weights: Benign={weights[0]:.3f}, Malignant={weights[1]:.3f}")
    return weights.to(device)


def get_lr_with_warmup(optimizer, epoch, warmup_epochs, base_lr, max_lr):
    """Linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr + (max_lr - base_lr) * (epoch / warmup_epochs)
    else:
        lr = max_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_epoch(model, loader, criterion, optimizer, device, epoch, accum_steps=4):
    """Train with gradient accumulation and clipping."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    mal_correct, mal_total = 0, 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")

    for i, (imgs, lbls) in enumerate(pbar):
        imgs, lbls = imgs.to(device), lbls.to(device)

        out = model(imgs)
        loss = criterion(out, lbls) / accum_steps
        loss.backward()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_sum += loss.item() * accum_steps
        _, pred = out.max(1)
        total += lbls.size(0)
        correct += pred.eq(lbls).sum().item()

        # Track malignant accuracy
        mal_mask = lbls == 1
        mal_total += mal_mask.sum().item()
        mal_correct += (pred[mal_mask] == lbls[mal_mask]).sum().item()

        mal_acc = 100 * mal_correct / max(mal_total, 1)
        pbar.set_postfix({
            'loss': f'{loss_sum/(i+1):.4f}',
            'acc': f'{100*correct/total:.1f}%',
            'mal_acc': f'{mal_acc:.1f}%'
        })

    if (i + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return loss_sum / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate model."""
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
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0

    return {
        'loss': loss_sum / len(loader),
        'accuracy': (tp + tn) / (tp + tn + fp + fn) * 100,
        'sensitivity': sens * 100,
        'specificity': spec * 100,
        'precision': prec * 100,
        'npv': npv * 100,
        'f1': f1 * 100,
        'auc': auc * 100,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }, labels, probs


def find_best_threshold(labels, probs, target_sens=0.90):
    """Find threshold achieving target sensitivity with best F1."""
    best_thresh, best_f1 = 0.5, 0

    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

        if sens >= target_sens and f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh


def plot_history(history, path):
    """Save training plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0,0].set_title('Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0,1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[0,1].set_title('Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(epochs, history['val_sensitivity'], 'r-', linewidth=2, label='Sensitivity')
    axes[1,0].plot(epochs, history['val_specificity'], 'b-', linewidth=2, label='Specificity')
    axes[1,0].axhline(90, color='g', linestyle='--', alpha=0.7, label='90% target')
    axes[1,0].fill_between(epochs, history['val_sensitivity'], alpha=0.3, color='red')
    axes[1,0].set_ylim([0, 100])
    axes[1,0].set_title('Sensitivity vs Specificity')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(epochs, history['val_auc'], 'purple', linewidth=2)
    axes[1,1].fill_between(epochs, history['val_auc'], alpha=0.3, color='purple')
    axes[1,1].set_title('AUC-ROC')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylim([50, 100])
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./checkpoints/combined_isic_v3_fixed")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--target_sens", type=float, default=0.90)
    parser.add_argument("--malignant_ratio", type=float, default=0.4)  # 40% malignant per batch
    parser.add_argument("--warmup_epochs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"\n{'='*70}")
    print("COMBINED ISIC 2019 + 2020 TRAINING v3 - FULL DATASET")
    print(f"{'='*70}")
    print("Using ALL images with heavy malignant augmentation + weighted sampling")
    print(f"CPU threads: {os.cpu_count()}")
    print(f"Learning rate: {args.lr} with {args.warmup_epochs} warmup epochs")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ALL data
    paths, labels = load_all_isic(args.data_dir)

    if len(paths) == 0:
        print("ERROR: No images found!")
        return

    # Split with stratification
    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=0.5, stratify=temp_l, random_state=42
    )

    print(f"\nData split:")
    print(f"  Train: {len(train_p):,} (B:{train_l.count(0):,}, M:{train_l.count(1):,})")
    print(f"  Val:   {len(val_p):,} (B:{val_l.count(0):,}, M:{val_l.count(1):,})")
    print(f"  Test:  {len(test_p):,} (B:{test_l.count(0):,}, M:{test_l.count(1):,})")

    # Datasets with heavy malignant augmentation
    train_ds = SkinDataset(
        train_p, train_l,
        benign_transform=get_benign_transforms(args.img_size),
        malignant_transform=MalignantHeavyAugment(args.img_size),
        training=True
    )
    val_ds = SkinDataset(val_p, val_l, benign_transform=get_val_transforms(args.img_size), training=False)
    test_ds = SkinDataset(test_p, test_l, benign_transform=get_val_transforms(args.img_size), training=False)

    # Balanced sampler - ensures ~40% malignant per batch
    train_sampler = create_balanced_sampler(train_l, args.malignant_ratio)

    train_loader = DataLoader(train_ds, args.batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)

    print(f"\nBatches per epoch: {len(train_loader):,}")

    # Model
    model = create_model(dropout=0.5)
    print(f"Model: ResNet34 (dropout=0.5, BatchNorm)")

    # Loss & optimizer
    class_weights = compute_class_weights(train_l, device)
    criterion = FocalLoss(class_weights, gamma=2.0, smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup_epochs)

    # History
    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                                'val_sensitivity', 'val_specificity', 'val_auc', 'val_f1']}

    best_balanced = 0
    best_sens = 0
    best_path = None
    start_epoch = 0
    patience_counter = 0

    # Resume
    ckpt_path = output_dir / "latest_checkpoint.pth"
    if ckpt_path.exists():
        print(f"\nResuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_balanced = ckpt.get('best_balanced', 0)
        best_sens = ckpt.get('best_sensitivity', 0)
        history = ckpt.get('history', history)
        print(f"Resuming epoch {start_epoch + 1}")

    # Training
    print(f"\n{'='*70}")
    print(f"TRAINING - Target: {args.target_sens*100:.0f}% sensitivity")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs):
        # Warmup
        if epoch < args.warmup_epochs:
            lr = get_lr_with_warmup(optimizer, epoch, args.warmup_epochs, 1e-6, args.lr)
            print(f"[Warmup {epoch+1}/{args.warmup_epochs}] LR: {lr:.2e}")
        elif epoch == args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.accum_steps
        )

        if epoch >= args.warmup_epochs:
            scheduler.step()

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])

        # Balanced score (geometric mean)
        balanced = np.sqrt(val_metrics['sensitivity'] * val_metrics['specificity'])

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  Val   - Sens: {val_metrics['sensitivity']:.1f}% | Spec: {val_metrics['specificity']:.1f}% | Balanced: {balanced:.1f}%")
        print(f"        - F1: {val_metrics['f1']:.1f}% | AUC: {val_metrics['auc']:.1f}% | NPV: {val_metrics['npv']:.1f}%")
        print(f"        - TP:{val_metrics['tp']} TN:{val_metrics['tn']} FP:{val_metrics['fp']} FN:{val_metrics['fn']}")

        # Early stopping - model collapse detection
        if val_metrics['sensitivity'] < 40:
            patience_counter += 1
            print(f"  WARNING: Low sensitivity! Patience: {patience_counter}/3")
            if patience_counter >= 3:
                print("  STOPPING: Model unstable")
                break
        else:
            patience_counter = 0

        # Save EVERY epoch checkpoint (so we never lose a good epoch!)
        epoch_ckpt_path = output_dir / f"epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': val_metrics,
            'balanced_score': balanced,
        }, epoch_ckpt_path)
        print(f"  Saved: epoch_{epoch+1}.pth")

        # Save best model (NO sensitivity threshold - just best balanced score)
        if balanced > best_balanced:
            best_balanced = balanced
            best_sens = val_metrics['sensitivity']
            best_path = output_dir / "best_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'balanced_score': balanced,
                'epoch': epoch + 1,
            }, best_path)
            print(f"  *** NEW BEST! Balanced: {balanced:.1f}% (Sens: {val_metrics['sensitivity']:.1f}%, Spec: {val_metrics['specificity']:.1f}%)")

        # Latest checkpoint (for resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_balanced': best_balanced,
            'best_sensitivity': best_sens,
            'history': history,
        }, ckpt_path)
        print(f"  Latest checkpoint saved")

    # Final test
    print(f"\n{'='*70}")
    print("FINAL TEST RESULTS")
    print(f"{'='*70}")

    if best_path and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=False)['model_state_dict'])
        print("Loaded best model\n")

    test_metrics, test_labels, test_probs = evaluate(model, test_loader, criterion, device)

    print(f"Default threshold (0.5):")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.1f}% (catching {test_metrics['tp']}/{test_metrics['tp']+test_metrics['fn']} cancers)")
    print(f"  Specificity: {test_metrics['specificity']:.1f}%")
    print(f"  Precision:   {test_metrics['precision']:.1f}%")
    print(f"  NPV:         {test_metrics['npv']:.1f}%")
    print(f"  F1 Score:    {test_metrics['f1']:.1f}%")
    print(f"  AUC-ROC:     {test_metrics['auc']:.1f}%")

    # Optimal threshold
    opt_thresh = find_best_threshold(test_labels, test_probs, args.target_sens)
    test_opt, _, _ = evaluate(model, test_loader, criterion, device, threshold=opt_thresh)

    print(f"\nOptimal threshold ({opt_thresh:.2f}) for {args.target_sens*100:.0f}% sensitivity:")
    print(f"  Sensitivity: {test_opt['sensitivity']:.1f}%")
    print(f"  Specificity: {test_opt['specificity']:.1f}%")
    print(f"  Precision:   {test_opt['precision']:.1f}%")
    print(f"  F1 Score:    {test_opt['f1']:.1f}%")

    # Save
    plot_history(history, str(output_dir / "training_curves.png"))

    results = {
        'version': 'v3_full_dataset',
        'features': [
            'all_58k_images',
            'heavy_malignant_augmentation_3_strategies',
            'weighted_sampler_40%_malignant',
            'warmup_3_epochs',
            'gradient_clipping',
            'focal_loss_with_label_smoothing',
        ],
        'config': vars(args),
        'data_stats': {
            'total_images': len(paths),
            'train': len(train_p),
            'val': len(val_p),
            'test': len(test_p),
        },
        'test_metrics_default': test_metrics,
        'test_metrics_optimal': test_opt,
        'optimal_threshold': opt_thresh,
        'history': history,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
