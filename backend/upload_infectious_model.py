"""
Upload the improved infectious disease model (86.5% AUC) to HuggingFace Hub.
"""
from huggingface_hub import HfApi, create_repo, upload_file
from pathlib import Path
import json

api = HfApi()
username = "PayamFard123"
repo_name = "dermaintel-infectious-disease"
repo_id = f"{username}/{repo_name}"

# Paths
model_dir = Path("checkpoints/infectious_improved")
model_path = model_dir / "best_model.pth"
results_path = model_dir / "results.json"

print("="*60)
print("UPLOADING IMPROVED INFECTIOUS DISEASE MODEL TO HUGGINGFACE")
print("="*60)
print(f"Repository: {repo_id}")
print(f"Model: {model_path}")

# Check model exists
if not model_path.exists():
    print(f"[ERROR] Model not found: {model_path}")
    exit(1)

# Create/verify repo
try:
    create_repo(repo_id, private=True, exist_ok=True)
    print(f"[OK] Repository ready: {repo_id}")
except Exception as e:
    print(f"[ERROR] Creating repo: {e}")
    exit(1)

# Upload model file
print("\nUploading model...")
try:
    upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="best_model.pth",
        repo_id=repo_id,
        commit_message="Upload improved infectious disease model (86.5% AUC, 48.9% F1)"
    )
    print(f"[OK] Uploaded: best_model.pth")
except Exception as e:
    print(f"[ERROR] Uploading model: {e}")

# Upload results JSON
if results_path.exists():
    print("Uploading results...")
    try:
        upload_file(
            path_or_fileobj=str(results_path),
            path_in_repo="results.json",
            repo_id=repo_id,
            commit_message="Upload training results"
        )
        print(f"[OK] Uploaded: results.json")
    except Exception as e:
        print(f"[ERROR] Uploading results: {e}")

# Create metadata JSON for model loading
metadata = {
    "architecture": "resnet34",
    "num_classes": 9,
    "class_names": [
        "candidiasis",
        "cellulitis",
        "folliculitis",
        "herpes_simplex",
        "impetigo",
        "molluscum_contagiosum",
        "scabies",
        "tinea_corporis",
        "warts"
    ],
    "metrics": {
        "auc": 86.5,
        "f1": 48.9,
        "accuracy": 46.0
    },
    "input_size": 224,
    "training_images": 9017
}

print("Uploading metadata...")
try:
    api.upload_file(
        path_or_fileobj=json.dumps(metadata, indent=2).encode(),
        path_in_repo="metadata.json",
        repo_id=repo_id,
        commit_message="Upload model metadata"
    )
    print(f"[OK] Uploaded: metadata.json")
except Exception as e:
    print(f"[ERROR] Uploading metadata: {e}")

# Create model card
model_card = """---
license: apache-2.0
tags:
- skin-disease
- infectious-disease
- dermatology
- medical-imaging
- pytorch
- resnet34
---

# DermaIntel Infectious Disease Classifier

Improved ResNet34 model for classifying 9 infectious skin conditions.

## Model Performance

| Metric | Value |
|--------|-------|
| **AUC** | 86.5% |
| **F1 Score** | 48.9% |
| **Accuracy** | 46.0% |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| tinea_corporis | 85% | 48% | 61% |
| warts | 70% | 32% | 44% |
| scabies | 22% | 90% | 35% |
| impetigo | 38% | 76% | 51% |
| herpes_simplex | 57% | 30% | 39% |
| molluscum_contagiosum | 28% | 62% | 38% |
| folliculitis | 25% | 56% | 34% |
| cellulitis | 41% | 43% | 42% |
| candidiasis | 30% | 56% | 39% |

## Training Details

- **Architecture**: ResNet34
- **Dataset**: Combined DermNet + Fitzpatrick17k + Kaggle (9,017 images)
- **Classes**: 9 infectious skin conditions
- **Training**: 12 epochs with focal loss, weighted sampling, cosine annealing

## Usage

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.resnet34(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.4),
    torch.nn.Linear(model.fc.in_features, 9)
)
checkpoint = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Class names
classes = [
    "candidiasis", "cellulitis", "folliculitis", "herpes_simplex",
    "impetigo", "molluscum_contagiosum", "scabies", "tinea_corporis", "warts"
]

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
image = Image.open("skin_condition.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred = probs.argmax(dim=1).item()

print(f"Prediction: {classes[pred]} ({probs[0][pred]*100:.1f}%)")
```

## Classes

0. candidiasis
1. cellulitis
2. folliculitis
3. herpes_simplex
4. impetigo
5. molluscum_contagiosum
6. scabies
7. tinea_corporis
8. warts

## License

Apache 2.0
"""

print("Uploading model card...")
try:
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Update model card with improved metrics"
    )
    print(f"[OK] Uploaded: README.md")
except Exception as e:
    print(f"[ERROR] Uploading model card: {e}")

print("\n" + "="*60)
print("UPLOAD COMPLETE!")
print("="*60)
print(f"Model available at: https://huggingface.co/{repo_id}")
