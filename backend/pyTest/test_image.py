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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same model architecture
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)

# Load checkpoint
ckpt_path = "../binary_classifier_model/best_resnet18_binary.pth"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
model = model.to(device)
model.eval()

print(f"Loaded model with val_acc={ckpt['val_acc']:.4f}")

eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(path: str):
    model.eval()
    img = Image.open(path).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    # probs[1] = P(lesion), probs[0] = P(non-lesion)
    return {"non_lesion": float(probs[0]), "lesion": float(probs[1])}

result = predict_image(r"D:/Downloads/pink.jpg")
print(result)