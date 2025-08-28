import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ham_path = "D:/Downloads/data/ham_compile"
coco_non_lesion_path = "D:/Downloads/data/coco_non_lesion"
output_dir = "../binary_classifier_model"
#output_dir = os.path.abspath("../binary_classifier_model")

os.makedirs(output_dir, exist_ok=True)

def list_images(root: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]
   
lesion_files = list_images(ham_path)
nonlesion_files = list_images(coco_non_lesion_path)

print(f"Found {len(lesion_files)} lesion images and {len(nonlesion_files)} non-lesion images.")

target = min(len(lesion_files), len(nonlesion_files))  # ~6800
random.seed(42)
lesion_files = random.sample(lesion_files, target)
nonlesion_files = random.sample(nonlesion_files, target)

print(f"Balanced to {len(lesion_files)} lesion and {len(nonlesion_files)} non-lesion.")

X = lesion_files + nonlesion_files
y = [1] * len(lesion_files) + [0] * len(nonlesion_files)

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class PathListDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], tfm):
        self.paths = paths
        self.labels = labels
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.tfm(img)
        label = self.labels[idx]
        return img, label

train_ds = PathListDataset(X_train, y_train, train_tfms)
val_ds   = PathListDataset(X_val, y_val, eval_tfms)
test_ds  = PathListDataset(X_test, y_test, eval_tfms)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)


# -----------------------------
# 6) MODEL: ResNet-18 (2 classes)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# -----------------------------
# 7) TRAIN / EVAL LOOPS
# -----------------------------
def run_epoch(loader: DataLoader, train: bool = True) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), torch.tensor(labels).to(device)

        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

if __name__ == "__main__":
    best_val_acc = 0.0
    EPOCHS = 10

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        scheduler.step()

        print(f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(output_dir, "best_resnet18_binary.pth")
            torch.save({"model": model.state_dict(),
                        "val_acc": val_acc}, ckpt_path)
            print(f"  ✅ Saved new best to {ckpt_path} (val_acc={val_acc:.4f})")
        else:
            print(f"  ⚠️ No improvement this epoch (best so far {best_val_acc:.4f})")


# -----------------------------
# 8) TEST SET EVAL (best model)
# -----------------------------

ckpt = torch.load(os.path.join(output_dir, "best_resnet18_binary.pth"), map_location=device)
model.load_state_dict(ckpt["model"])
test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"TEST  | loss {test_loss:.4f} acc {test_acc:.4f}")

# -----------------------------
# 9) INFERENCE HELPER
# -----------------------------
def predict_image(path: str):
    model.eval()
    img = Image.open(path).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    # probs[1] = P(lesion), probs[0] = P(non-lesion)
    return {"non_lesion": float(probs[0]), "lesion": float(probs[1])}

# Example:
# print(predict_image(r"D:\some\test\image.jpg"))