"""
Combined ISIC 2019 + 2020 Skin Cancer Classifier - CPU Optimized

Combines ISIC 2019 and 2020 datasets for better malignant representation:
- ISIC 2020: 32,542 benign + 584 malignant (55:1 ratio)
- ISIC 2019: 15,991 benign + 9,340 malignant (1.7:1 ratio)
- Combined: 48,533 benign + 9,924 malignant (4.9:1 ratio - much better!)

CPU Optimizations:
- ResNet34 (lighter than ResNet50)
- 192x192 images (smaller than 224x224)
- Batch size 8 with gradient accumulation (effective batch 32)
- Checkpointing after every epoch

Usage:
    python train_combined_isic_cpu.py --epochs 20
"""

import os
import argparse
import json
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

# Use all CPU cores
torch.set_num_threads(os.cpu_count())


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()


def get_train_transforms(is_malignant: bool = False, img_size: int = 192):
    """Training transforms with augmentation."""
    base = [
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(20),
    ]

    if is_malignant:
        base.extend([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    else:
        base.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))

    base.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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


def create_model(num_classes=2, dropout=0.3):
    """ResNet34 - lighter for CPU."""
    model = models.resnet34(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes)
    )
    return model


def load_combined_isic(base_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load and combine ISIC 2019 + 2020 datasets.

    Expected structure:
        base_dir/
            isic/organized/benign/*.jpg      (ISIC 2020)
            isic/organized/malignant/*.jpg   (ISIC 2020)
            isic_2019/organized/benign/*.jpg (ISIC 2019)
            isic_2019/organized/malignant/*.jpg (ISIC 2019)
    """
    base = Path(base_dir)
    paths, labels = [], []

    # ISIC 2020
    isic_2020 = base / "isic" / "organized"
    # ISIC 2019
    isic_2019 = base / "isic_2019" / "organized"

    datasets = [
        (isic_2020, "ISIC 2020"),
        (isic_2019, "ISIC 2019"),
    ]

    print("\nLoading datasets:")
    print("-" * 50)

    total_benign, total_malignant = 0, 0

    for data_path, name in datasets:
        benign_dir = data_path / "benign"
        malignant_dir = data_path / "malignant"

        benign_count, malignant_count = 0, 0

        if benign_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img in benign_dir.glob(ext):
                    paths.append(str(img))
                    labels.append(0)
                    benign_count += 1

        if malignant_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img in malignant_dir.glob(ext):
                    paths.append(str(img))
                    labels.append(1)
                    malignant_count += 1

        print(f"  {name}: {benign_count:,} benign + {malignant_count:,} malignant")
        total_benign += benign_count
        total_malignant += malignant_count

    print("-" * 50)
    print(f"  TOTAL: {total_benign:,} benign + {total_malignant:,} malignant")
    print(f"  Combined ratio: {total_benign/max(total_malignant,1):.1f}:1")
    print(f"  Total images: {len(paths):,}")

    return paths, labels


def compute_class_weights(labels, device):
    """Compute weights with extra penalty for missing malignant."""
    counter = Counter(labels)
    total = len(labels)

    weights = []
    for i in range(len(counter)):
        w = (total / counter[i]) * (2.0 if i == 1 else 1.0)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)
    print(f"\nClass weights: Benign={weights[0]:.3f}, Malignant={weights[1]:.3f}")
    return weights.to(device)


def create_sampler(labels):
    """Weighted sampler for balanced batches."""
    counter = Counter(labels)
    weights = [1.0 / counter[l] for l in labels]
    return WeightedRandomSampler(weights, len(labels), replacement=True)


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
            optimizer.step()
            optimizer.zero_grad()

        loss_sum += loss.item() * accum_steps
        _, pred = out.max(1)
        total += lbls.size(0)
        correct += pred.eq(lbls).sum().item()

        pbar.set_postfix({'loss': f'{loss_sum/(i+1):.4f}', 'acc': f'{100*correct/total:.1f}%'})

    if (i + 1) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return loss_sum / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate with metrics."""
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
    parser.add_argument("--output_dir", default="./checkpoints/combined_isic")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--target_sens", type=float, default=0.90)
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"\n{'='*50}")
    print("COMBINED ISIC 2019 + 2020 TRAINING (CPU)")
    print(f"{'='*50}")
    print(f"CPU threads: {os.cpu_count()}")
    print(f"Effective batch size: {args.batch_size * args.accum_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load combined data
    paths, labels = load_combined_isic(args.data_dir)

    if len(paths) == 0:
        print("ERROR: No images found!")
        return

    # Split data
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

    # Model
    model = create_model()
    print(f"\nModel: ResNet34 (CPU optimized)")

    # Loss & optimizer
    class_weights = compute_class_weights(train_l, device)
    criterion = FocalLoss(class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # History
    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                                'val_sensitivity', 'val_specificity', 'val_auc', 'val_f1']}

    best_sens, best_f1, best_path = 0, 0, None
    start_epoch = 0

    # Resume checkpoint
    ckpt_path = output_dir / "latest_checkpoint.pth"
    if ckpt_path.exists():
        print(f"\nResuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_sens = ckpt.get('best_sensitivity', 0)
        best_f1 = ckpt.get('best_f1', 0)
        history = ckpt.get('history', history)
        print(f"Resuming epoch {start_epoch + 1}, best sens: {best_sens:.1f}%")

    # Training
    print(f"\n{'='*50}")
    print(f"TRAINING - Target: {args.target_sens*100:.0f}% sensitivity")
    print(f"{'='*50}\n")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.accum_steps
        )
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Record
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
        print(f"  Sens: {val_metrics['sensitivity']:.1f}% | Spec: {val_metrics['specificity']:.1f}%")
        print(f"  F1: {val_metrics['f1']:.1f}% | AUC: {val_metrics['auc']:.1f}%")
        print(f"  TP:{val_metrics['tp']} TN:{val_metrics['tn']} FP:{val_metrics['fp']} FN:{val_metrics['fn']}")

        # Save best
        if val_metrics['sensitivity'] >= args.target_sens * 100 and val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_sens = val_metrics['sensitivity']
            best_path = output_dir / "best_model.pth"
            torch.save({'model_state_dict': model.state_dict(), 'metrics': val_metrics}, best_path)
            print(f"  * NEW BEST! Sens: {best_sens:.1f}%, F1: {best_f1:.1f}%")
        elif val_metrics['sensitivity'] > best_sens:
            best_sens = val_metrics['sensitivity']
            best_path = output_dir / "best_model.pth"
            torch.save({'model_state_dict': model.state_dict(), 'metrics': val_metrics}, best_path)
            print(f"  * New best sensitivity: {best_sens:.1f}%")

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_sensitivity': best_sens,
            'best_f1': best_f1,
            'history': history,
        }, ckpt_path)
        print(f"  Checkpoint saved")

    # Final test
    print(f"\n{'='*50}")
    print("FINAL TEST RESULTS")
    print(f"{'='*50}")

    if best_path and best_path.exists():
        model.load_state_dict(torch.load(best_path)['model_state_dict'])

    test_metrics, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"Sensitivity: {test_metrics['sensitivity']:.1f}%")
    print(f"Specificity: {test_metrics['specificity']:.1f}%")
    print(f"Precision:   {test_metrics['precision']:.1f}%")
    print(f"F1 Score:    {test_metrics['f1']:.1f}%")
    print(f"AUC-ROC:     {test_metrics['auc']:.1f}%")
    print(f"TP:{test_metrics['tp']} TN:{test_metrics['tn']} FP:{test_metrics['fp']} FN:{test_metrics['fn']}")

    # Save
    plot_history(history, str(output_dir / "training_curves.png"))

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            'datasets': ['ISIC 2019', 'ISIC 2020'],
            'config': vars(args),
            'test_metrics': test_metrics,
            'history': history,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
