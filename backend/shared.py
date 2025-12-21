"""
Shared module containing loaded ML models, constants, and utility functions.
All routers import from this module to access models and common functionality.

Set TESTING=1 environment variable to skip model loading for faster tests.
"""

import os
import torch
from torch import nn
from torchvision import models, transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import cv2

# Check if running in test mode (skip model loading for faster tests)
TESTING_MODE = os.environ.get("TESTING", "0") == "1"

# Import configuration
from config import (
    UPLOAD_DIR,
    BINARY_CLASSIFIER_PATH,
    LESION_MODEL_PATH,
    ISIC_MODEL_PATH,
    ISIC_MODEL_FALLBACK_PATH,
    ISIC_2020_BINARY_PATH,
    INFECTIOUS_MODEL_DIR,
    INFLAMMATORY_MODEL_ID,
    MC_DROPOUT_SAMPLES,
    MODEL_INPUT_SIZE,
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure upload directory exists
UPLOAD_DIR.mkdir(exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sanitize_for_json(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def debug_log(message):
    """Helper function for timestamped debug logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] DEBUG: {message}")


# =============================================================================
# LABEL MAPPINGS
# =============================================================================

key_map = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratoses-Like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

binary_labels = {0: "non_lesion", 1: "lesion"}

isic_class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
isic_class_full_names = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC": "Squamous Cell Carcinoma"
}
isic_malignancy = {
    "MEL": "malignant",
    "NV": "benign",
    "BCC": "malignant",
    "AK": "pre-malignant",
    "BKL": "benign",
    "DF": "benign",
    "VASC": "benign",
    "SCC": "malignant"
}

isic_2020_binary_class_names = ["benign", "malignant"]

# =============================================================================
# CLINICAL CONSTANTS
# =============================================================================

INFLAMMATORY_CONDITIONS = {
    "Psoriasis",
    "Lichen Planus",
    "Lupus Erythematosus Chronicus Discoides",
    "Darier_s Disease",
    "Herpes Simplex"
}

ECZEMA_CONDITIONS = {
    "Atopic Dermatitis",
    "Eczema",
    "Contact Dermatitis",
    "Nummular Dermatitis",
    "Dyshidrotic Eczema"
}

URTICARIA_CONDITIONS = {
    "Urticaria",
    "Hives"
}

TRIAGE_CATEGORIES = {
    "neoplastic": {
        "description": "Pigmented lesions, moles, potential skin cancers",
        "conditions": ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses",
                       "Melanocytic Nevi", "Benign Keratoses-Like Lesions",
                       "Dermatofibroma", "Vascular Lesions"],
        "urgency_base": "medium",
        "requires_specialist": True
    },
    "inflammatory": {
        "description": "Inflammatory skin conditions like eczema, psoriasis",
        "conditions": ["Psoriasis", "Eczema", "Atopic Dermatitis", "Contact Dermatitis",
                       "Lichen Planus", "Lupus Erythematosus", "Urticaria", "Seborrheic Dermatitis"],
        "urgency_base": "low",
        "requires_specialist": False
    },
    "infectious": {
        "description": "Bacterial, viral, fungal, or parasitic skin infections",
        "conditions": ["Tinea_Corporis", "Tinea_Pedis", "Cellulitis", "Impetigo",
                       "Herpes Simplex", "Herpes Zoster", "Scabies", "Folliculitis"],
        "urgency_base": "medium",
        "requires_specialist": False
    },
    "traumatic": {
        "description": "Burns, wounds, physical injuries to skin",
        "conditions": ["First Degree Burn", "Second Degree Burn", "Third Degree Burn"],
        "urgency_base": "varies",
        "requires_specialist": True
    },
    "uncertain": {
        "description": "Unable to confidently categorize - requires clinical evaluation",
        "conditions": [],
        "urgency_base": "medium",
        "requires_specialist": True
    }
}

HIGH_RISK_CONDITIONS = {
    "Melanoma": {"category": "neoplastic", "min_confidence_override": 0.5},
    "Basal Cell Carcinoma": {"category": "neoplastic", "min_confidence_override": 0.6},
    "Actinic Keratoses": {"category": "neoplastic", "min_confidence_override": 0.6},
    "Third Degree Burn": {"category": "traumatic", "min_confidence_override": 0.5},
    "Cellulitis": {"category": "infectious", "min_confidence_override": 0.6},
}

# =============================================================================
# IMAGE TRANSFORMS
# =============================================================================

binary_transform = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.CenterCrop(MODEL_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

isic_transform = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# MODEL CREATION HELPERS
# =============================================================================

def create_isic_classifier(num_classes, feature_dim=2048):
    """Create ISIC classifier model architecture"""
    class ISICSkinClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.resnet50(weights=None)
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(0.5),
                nn.Linear(feature_dim, 512),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    return ISICSkinClassifier()


# =============================================================================
# MODEL LOADING
# =============================================================================

if TESTING_MODE:
    # Skip model loading in test mode for faster tests
    print("[TEST MODE] Skipping model loading - using mock models")

    # Create minimal mock models
    class MockModel(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.num_classes = num_classes
        def forward(self, x):
            batch = x.shape[0] if len(x.shape) > 0 else 1
            return torch.randn(batch, self.num_classes)
        def eval(self):
            return self

    class MockProcessor:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.randn(1, 3, 224, 224)}

    class MockOutput:
        def __init__(self, nc=7):
            self.logits = torch.randn(1, nc)

    class MockTransformer:
        def __init__(self, nc=7):
            self.nc = nc
            self.config = type('Config', (), {'id2label': {i: f'class_{i}' for i in range(nc)}})()
        def __call__(self, **kw):
            return MockOutput(self.nc)
        def eval(self):
            return self
        def to(self, d):
            return self

    binary_model = MockModel(2)
    isic_model = MockModel(8)
    isic_2020_binary_model = MockModel(2)
    lesion_model = MockTransformer(7)
    lesion_processor = MockProcessor()
    labels = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
    inflammatory_model = MockTransformer(26)
    inflammatory_processor = MockProcessor()
    inflammatory_labels = {i: f"condition_{i}" for i in range(26)}
    infectious_model = None
    infectious_processor = None
    infectious_labels = None
    infectious_class_metadata = {}
    isic_labels = {i: name for i, name in enumerate(isic_class_names)}
    print("[TEST MODE] Mock models initialized")

else:
    # Production mode - load real models
    # Binary lesion classifier
    print("Loading Binary Lesion Classifier...")
    binary_model = models.resnet18(weights=None)
    num_features = binary_model.fc.in_features
    binary_model.fc = torch.nn.Linear(num_features, 2)
    checkpoint = torch.load(str(BINARY_CLASSIFIER_PATH), map_location=device)
    binary_model.load_state_dict(checkpoint["model"])
    binary_model = binary_model.to(device)
    binary_model.eval()
    print(f"[OK] Binary lesion classifier loaded from {BINARY_CLASSIFIER_PATH}")

    # ISIC 8-class model
    print("Loading ISIC 8-class Skin Lesion Classification Model...")
    isic_model = None
    isic_labels = None
    try:
        isic_checkpoint_path = ISIC_MODEL_PATH
        if not isic_checkpoint_path.exists():
            isic_checkpoint_path = ISIC_MODEL_FALLBACK_PATH

        if isic_checkpoint_path.exists():
            isic_checkpoint = torch.load(str(isic_checkpoint_path), map_location=device, weights_only=False)
            model_config = isic_checkpoint.get('model_config', {})
            num_classes = model_config.get('num_classes', 8)

            if num_classes == 8:
                isic_model = create_isic_classifier(num_classes)
                isic_model.load_state_dict(isic_checkpoint['model_state_dict'])
                isic_model = isic_model.to(device)
                isic_model.eval()
                isic_labels = {i: name for i, name in enumerate(isic_class_names)}
                best_acc = isic_checkpoint.get('best_val_acc', 'N/A')
                print(f"[OK] ISIC 8-class model loaded from {isic_checkpoint_path}")
            else:
                print(f"[WARN] ISIC 8-class model not found")
        else:
            print(f"[WARN] ISIC checkpoint not found at {ISIC_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load ISIC 8-class model: {e}")

    # ISIC 2020 Binary model
    print("Loading ISIC 2020 Binary Classification Model...")
    isic_2020_binary_model = None
    try:
        if ISIC_2020_BINARY_PATH.exists():
            isic_2020_checkpoint = torch.load(str(ISIC_2020_BINARY_PATH), map_location=device, weights_only=False)
            model_config = isic_2020_checkpoint.get('model_config', {})
            num_classes = model_config.get('num_classes', 2)

            if num_classes == 2:
                isic_2020_binary_model = create_isic_classifier(num_classes)
                isic_2020_binary_model.load_state_dict(isic_2020_checkpoint['model_state_dict'])
                isic_2020_binary_model = isic_2020_binary_model.to(device)
                isic_2020_binary_model.eval()
                print(f"[OK] ISIC 2020 Binary model loaded from {ISIC_2020_BINARY_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load ISIC 2020 Binary model: {e}")

    # Lesion classification model (HuggingFace)
    print("Loading Lesion Classification Model...")
    lesion_model = AutoModelForImageClassification.from_pretrained(str(LESION_MODEL_PATH))
    lesion_processor = AutoImageProcessor.from_pretrained(str(LESION_MODEL_PATH))
    labels = lesion_model.config.id2label
    lesion_model.eval()
    print(f"[OK] Lesion classification model loaded from {LESION_MODEL_PATH}")

    # DinoV2 inflammatory model
    print("Loading DinoV2 skin disease classification model...")
    inflammatory_model = None
    inflammatory_processor = None
    inflammatory_labels = None
    try:
        inflammatory_model = AutoModelForImageClassification.from_pretrained(INFLAMMATORY_MODEL_ID)
        inflammatory_processor = AutoImageProcessor.from_pretrained(INFLAMMATORY_MODEL_ID)
        inflammatory_model.eval()
        inflammatory_labels = inflammatory_model.config.id2label
        print(f"[OK] DinoV2 skin disease model loaded: {INFLAMMATORY_MODEL_ID} ({len(inflammatory_labels)} conditions)")
    except Exception as e:
        print(f"[WARN] Could not load DinoV2 model: {e}")

    # Infectious disease model
    print("Loading Infectious Disease Classification Model...")
    infectious_model = None
    infectious_processor = None
    infectious_labels = None
    infectious_class_metadata = {}
    try:
        if INFECTIOUS_MODEL_DIR.exists():
            if (INFECTIOUS_MODEL_DIR / "model" / "config.json").exists():
                infectious_model = AutoModelForImageClassification.from_pretrained(str(INFECTIOUS_MODEL_DIR / "model"))
                infectious_processor = AutoImageProcessor.from_pretrained(str(INFECTIOUS_MODEL_DIR / "processor"))
                infectious_model.eval()
                infectious_labels = infectious_model.config.id2label
                print(f"[OK] Infectious disease model loaded from {INFECTIOUS_MODEL_DIR} (HuggingFace format)")
            elif (INFECTIOUS_MODEL_DIR / "model.pth").exists():
                with open(INFECTIOUS_MODEL_DIR / "metadata.json", 'r') as f:
                    infectious_metadata = json.load(f)

                infectious_labels = {i: name for i, name in enumerate(infectious_metadata['class_names'])}
                infectious_class_metadata = infectious_metadata.get('class_metadata', {})

                model_arch = infectious_metadata.get('config', {}).get('model_name', 'resnet50')
                num_classes = len(infectious_labels)

                # Load state dict first to detect actual architecture
                state_dict = torch.load(str(INFECTIOUS_MODEL_DIR / "model.pth"), map_location=device, weights_only=True)
                state_keys = list(state_dict.keys())

                # Detect actual architecture from state dict keys
                has_resnet_keys = any(k.startswith('layer1') for k in state_keys)
                has_efficientnet_keys = any(k.startswith('features.') for k in state_keys)
                has_sequential_fc = 'fc.1.weight' in state_keys or 'fc.1.bias' in state_keys

                if has_resnet_keys:
                    # ResNet architecture detected
                    if has_sequential_fc:
                        # ResNet with Sequential fc (Dropout + Linear)
                        infectious_model = models.resnet18(weights=None)
                        num_features = infectious_model.fc.in_features
                        infectious_model.fc = nn.Sequential(
                            nn.Dropout(p=0.3),
                            nn.Linear(num_features, num_classes)
                        )
                    else:
                        # ResNet with simple Linear fc
                        infectious_model = models.resnet18(weights=None)
                        num_features = infectious_model.fc.in_features
                        infectious_model.fc = nn.Linear(num_features, num_classes)
                elif has_efficientnet_keys:
                    # EfficientNet architecture
                    if model_arch == "efficientnet_b3":
                        infectious_model = models.efficientnet_b3(weights=None)
                    else:
                        infectious_model = models.efficientnet_b0(weights=None)
                    num_features = infectious_model.classifier[1].in_features
                    infectious_model.classifier = nn.Sequential(
                        nn.Dropout(p=0.3, inplace=True),
                        nn.Linear(num_features, num_classes)
                    )
                else:
                    # Fallback to ResNet50
                    infectious_model = models.resnet50(weights=None)
                    num_features = infectious_model.fc.in_features
                    infectious_model.fc = nn.Linear(num_features, num_classes)

                infectious_model.load_state_dict(state_dict)
                infectious_model = infectious_model.to(device)
                infectious_model.eval()

                infectious_processor = transforms.Compose([
                    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print(f"[OK] Infectious disease model loaded from {INFECTIOUS_MODEL_DIR} (PyTorch format, {num_classes} classes)")
    except Exception as e:
        print(f"[WARN] Could not load infectious disease model: {e}")

    print("\n" + "="*60)
    print("All models loaded successfully!")
    print("="*60 + "\n")
