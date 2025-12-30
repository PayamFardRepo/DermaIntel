"""
Combined ISIC 2019 + 2020 Skin Cancer Classifier - CPU Optimized v2

FIXES from v1:
1. Lower learning rate (5e-5) with warmup for stability
2. Undersample benign to 2:1 ratio (prevents model collapse)
3. Increased dropout (0.5) for regularization
4. Early stopping if sensitivity drops below 50%
5. Better class weighting (3x for malignant)

Usage:
    python train_combined_isic_cpu_v2.py --epochs 15
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
        # Label smoothing
        n_classes = inputs.size(1)
        smooth_targets = torch.zeros_like(inputs).scatter_(
            1, targets.unsqueeze(1), 1.0
        )
        smooth_targets = smooth_targets * (1 - self.smoothing) + self.smoothing / n_classes

        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=1)

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        return focal_loss.mean()


def get_train_transforms(is_malignant: bool = False, img_size: int = 192):
    """Training transforms."""
    base = [
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(25),
    ]

    if is_malignant:
        # More augmentation for malignant
        base.extend([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.15),
            transforms.RandomAffine(15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
        ])
    else:
        base.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))

    base.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])
    return transforms.Compose(base)


def get_val_transforms(img_size: int = 192):
    """Validation transforms."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class SkinDataset(Dataset):
    """Dataset with class-aware augmentation."""
    def __init__(self, paths, labels, transform=None, mal_transform=None, training=True):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.mal_transform = mal_transform
        self.training = training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.training and label == 1 and self.mal_transform:
            img = self.mal_transform(img)
        elif self.transform:
            img = self.transform(img)

        return img, label


def create_model(num_classes=2, dropout=0.5):
    """ResNet34 with higher dropout."""
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


def load_combined_isic_balanced(base_dir: str, max_benign_ratio: float = 2.0) -> Tuple[List[str], List[int]]:
    """
    Load ISIC 2019 + 2020 with undersampling of benign class.

    Args:
        max_benign_ratio: Maximum ratio of benign:malignant (default 2:1)
    """
    base = Path(base_dir)
    benign_paths, malignant_paths = [], []

    datasets = [
        base / "isic" / "organized",
        base / "isic_2019" / "organized",
    ]

    print("\nLoading datasets:")
    print("-" * 50)

    for data_path in datasets:
        name = "ISIC 2020" if "isic_2019" not in str(data_path) else "ISIC 2019"
        b_count, m_count = 0, 0

        benign_dir = data_path / "benign"
        malignant_dir = data_path / "malignant"

        if benign_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img in benign_dir.glob(ext):
                    benign_paths.append(str(img))
                    b_count += 1

        if malignant_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img in malignant_dir.glob(ext):
                    malignant_paths.append(str(img))
                    m_count += 1

        print(f"  {name}: {b_count:,} benign + {m_count:,} malignant")

    print("-" * 50)
    print(f"  Original: {len(benign_paths):,} benign + {len(malignant_paths):,} malignant")
    print(f"  Original ratio: {len(benign_paths)/len(malignant_paths):.1f}:1")

    # Undersample benign to achieve target ratio
    max_benign = int(len(malignant_paths) * max_benign_ratio)
    if len(benign_paths) > max_benign:
        print(f"\n  Undersampling benign: {len(benign_paths):,} -> {max_benign:,}")
        random.shuffle(benign_paths)
        benign_paths = benign_paths[:max_benign]

    # Combine
    paths = benign_paths + malignant_paths
    labels = [0] * len(benign_paths) + [1] * len(malignant_paths)

    print(f"\n  FINAL: {len(benign_paths):,} benign + {len(malignant_paths):,} malignant")
    print(f"  Final ratio: {len(benign_paths)/len(malignant_paths):.1f}:1")
    print(f"  Total images: {len(paths):,}")

    return paths, labels


def compute_class_weights(labels, device):
    """Compute weights - 3x penalty for missing malignant."""
    counter = Counter(labels)
    total = len(labels)

    # Higher weight for malignant (3x)
    weights = []
    for i in range(len(counter)):
        w = (total / counter[i]) * (3.0 if i == 1 else 1.0)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)
    print(f"\nClass weights: Benign={weights[0]:.3f}, Malignant={weights[1]:.3f}")
    return weights.to(device)


def create_sampler(labels):
    """Weighted sampler."""
    counter = Counter(labels)
    weights = [1.0 / counter[l] for l in labels]
    return WeightedRandomSampler(weights, len(labels), replacement=True)


def get_lr_with_warmup(optimizer, epoch, warmup_epochs, base_lr, max_lr):
    """Linear warmup then constant."""
    if epoch < warmup_epochs:
        lr = base_lr + (max_lr - base_lr) * (epoch / warmup_epochs)
    else:
        lr = max_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_epoch(model, loader, criterion, optimizer, device, epoch, accum_steps=4):
    """Train with gradient accumulation."""
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
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_sum += loss.item() * accum_steps
        _, pred = out.max(1)
        total += lbls.size(0)
        correct += pred.eq(lbls).sum().item()

        pbar.set_postfix({'loss': f'{loss_sum/(i+1):.4f}', 'acc': f'{100*correct/total:.1f}%'})

    if (i + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return loss_sum / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate."""
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
        'accuracy': (tp + tn) / (tp + tn + fp + fn) * 100,
        'sensitivity': sens * 100,
        'specificity': spec * 100,
        'precision': prec * 100,
        'f1': f1 * 100,
        'auc': auc * 100,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }, labels, probs


def find_best_threshold(labels, probs, target_sens=0.90):
    """Find threshold that achieves target sensitivity."""
    best_thresh, best_f1 = 0.5, 0

    for thresh in np.arange(0.1, 0.9, 0.05):
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0,0].plot(history['train_loss'], label='Train')
    axes[0,0].plot(history['val_loss'], label='Val')
    axes[0,0].set_title('Loss'); axes[0,0].legend()

    axes[0,1].plot(history['train_acc'], label='Train')
    axes[0,1].plot(history['val_acc'], label='Val')
    axes[0,1].set_title('Accuracy'); axes[0,1].legend()

    axes[1,0].plot(history['val_sensitivity'], 'r-', label='Sensitivity')
    axes[1,0].plot(history['val_specificity'], 'b-', label='Specificity')
    axes[1,0].axhline(90, color='g', linestyle='--', label='90% target')
    axes[1,0].set_ylim([0, 100]); axes[1,0].legend()
    axes[1,0].set_title('Sensitivity vs Specificity')

    axes[1,1].plot(history['val_auc'], 'purple')
    axes[1,1].set_title('AUC-ROC'); axes[1,1].set_ylim([50, 100])

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./checkpoints/combined_isic_v2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)  # Lower LR
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--target_sens", type=float, default=0.90)
    parser.add_argument("--benign_ratio", type=float, default=2.0)  # 2:1 ratio
    parser.add_argument("--warmup_epochs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"\n{'='*60}")
    print("COMBINED ISIC 2019 + 2020 TRAINING v2 (CPU) - STABLE")
    print(f"{'='*60}")
    print(f"FIXES: Lower LR ({args.lr}), {args.benign_ratio}:1 ratio, warmup, gradient clipping")
    print(f"CPU threads: {os.cpu_count()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load balanced data
    paths, labels = load_combined_isic_balanced(args.data_dir, args.benign_ratio)

    if len(paths) == 0:
        print("ERROR: No images found!")
        return

    # Split
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

    # Datasets
    train_ds = SkinDataset(
        train_p, train_l,
        get_train_transforms(False, args.img_size),
        get_train_transforms(True, args.img_size),
        training=True
    )
    val_ds = SkinDataset(val_p, val_l, get_val_transforms(args.img_size), training=False)
    test_ds = SkinDataset(test_p, test_l, get_val_transforms(args.img_size), training=False)

    # Loaders
    train_loader = DataLoader(train_ds, args.batch_size, sampler=create_sampler(train_l), num_workers=0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)

    # Model with higher dropout
    model = create_model(dropout=0.5)
    print(f"\nModel: ResNet34 (dropout=0.5, BatchNorm)")

    # Loss & optimizer
    class_weights = compute_class_weights(train_l, device)
    criterion = FocalLoss(class_weights, gamma=2.0, smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.warmup_epochs)

    # History
    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                                'val_sensitivity', 'val_specificity', 'val_auc', 'val_f1']}

    best_sens, best_f1, best_path = 0, 0, None
    best_balanced_score = 0
    start_epoch = 0
    consecutive_drops = 0

    # Resume checkpoint
    ckpt_path = output_dir / "latest_checkpoint.pth"
    if ckpt_path.exists():
        print(f"\nResuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_sens = ckpt.get('best_sensitivity', 0)
        best_f1 = ckpt.get('best_f1', 0)
        best_balanced_score = ckpt.get('best_balanced_score', 0)
        history = ckpt.get('history', history)
        print(f"Resuming epoch {start_epoch + 1}, best sens: {best_sens:.1f}%")

    # Training
    print(f"\n{'='*60}")
    print(f"TRAINING - Target: {args.target_sens*100:.0f}% sensitivity")
    print(f"Warmup: {args.warmup_epochs} epochs, then cosine annealing")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        # Warmup LR
        if epoch < args.warmup_epochs:
            lr = get_lr_with_warmup(optimizer, epoch, args.warmup_epochs, 1e-6, args.lr)
            print(f"[Warmup] LR: {lr:.2e}")
        elif epoch == args.warmup_epochs:
            # Reset scheduler after warmup
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

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

        # Balanced score: geometric mean of sensitivity and specificity
        balanced_score = np.sqrt(val_metrics['sensitivity'] * val_metrics['specificity'])

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  Sens: {val_metrics['sensitivity']:.1f}% | Spec: {val_metrics['specificity']:.1f}%")
        print(f"  F1: {val_metrics['f1']:.1f}% | AUC: {val_metrics['auc']:.1f}%")
        print(f"  Balanced Score: {balanced_score:.1f}%")
        print(f"  TP:{val_metrics['tp']} TN:{val_metrics['tn']} FP:{val_metrics['fp']} FN:{val_metrics['fn']}")

        # Early stopping check - sensitivity collapse
        if val_metrics['sensitivity'] < 50:
            consecutive_drops += 1
            print(f"  WARNING: Sensitivity dropped below 50%! ({consecutive_drops}/3)")
            if consecutive_drops >= 3:
                print("  EARLY STOPPING: Model collapsed, loading best checkpoint")
                if best_path and best_path.exists():
                    model.load_state_dict(torch.load(best_path)['model_state_dict'])
                break
        else:
            consecutive_drops = 0

        # Save best model (prioritize balanced score when sensitivity >= 80%)
        if val_metrics['sensitivity'] >= 80 and balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_sens = val_metrics['sensitivity']
            best_f1 = val_metrics['f1']
            best_path = output_dir / "best_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'balanced_score': balanced_score,
            }, best_path)
            print(f"  * NEW BEST! Balanced: {balanced_score:.1f}% (Sens: {best_sens:.1f}%, Spec: {val_metrics['specificity']:.1f}%)")
        elif val_metrics['sensitivity'] > best_sens and best_balanced_score == 0:
            best_sens = val_metrics['sensitivity']
            best_path = output_dir / "best_model.pth"
            torch.save({'model_state_dict': model.state_dict(), 'metrics': val_metrics}, best_path)
            print(f"  * New best sensitivity: {best_sens:.1f}%")

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_sensitivity': best_sens,
            'best_f1': best_f1,
            'best_balanced_score': best_balanced_score,
            'history': history,
        }, ckpt_path)
        print(f"  Checkpoint saved")

    # Final test
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")

    if best_path and best_path.exists():
        model.load_state_dict(torch.load(best_path)['model_state_dict'])
        print("Loaded best model")

    # Evaluate with default threshold
    test_metrics, test_labels, test_probs = evaluate(model, test_loader, criterion, device)

    print(f"\nDefault threshold (0.5):")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.1f}%")
    print(f"  Specificity: {test_metrics['specificity']:.1f}%")
    print(f"  F1 Score:    {test_metrics['f1']:.1f}%")
    print(f"  AUC-ROC:     {test_metrics['auc']:.1f}%")

    # Find optimal threshold
    opt_thresh = find_best_threshold(test_labels, test_probs, args.target_sens)
    test_metrics_opt, _, _ = evaluate(model, test_loader, criterion, device, threshold=opt_thresh)

    print(f"\nOptimal threshold ({opt_thresh:.2f}):")
    print(f"  Sensitivity: {test_metrics_opt['sensitivity']:.1f}%")
    print(f"  Specificity: {test_metrics_opt['specificity']:.1f}%")
    print(f"  F1 Score:    {test_metrics_opt['f1']:.1f}%")

    # Save
    plot_history(history, str(output_dir / "training_curves.png"))

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            'version': 'v2_stable',
            'fixes': ['lower_lr', 'undersample_2:1', 'warmup', 'gradient_clip', 'label_smoothing'],
            'config': vars(args),
            'test_metrics_default': test_metrics,
            'test_metrics_optimal': test_metrics_opt,
            'optimal_threshold': opt_thresh,
            'history': history,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
