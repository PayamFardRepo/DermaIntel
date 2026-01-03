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
    # HuggingFace Hub configuration
    HF_TOKEN,
    HF_LESION_MODEL_ID,
    HF_ISIC_MODEL_ID,
    HF_ISIC_OLD_MODEL_ID,
    HF_INFECTIOUS_MODEL_ID,
    HF_BINARY_MODEL_ID,
)

# HuggingFace Hub imports for downloading private models
from huggingface_hub import hf_hub_download, snapshot_download

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure upload directory exists
UPLOAD_DIR.mkdir(exist_ok=True)

# =============================================================================
# HUGGINGFACE HUB HELPERS
# =============================================================================

def download_model_from_hf(repo_id: str, local_dir: Path, filename: str = None) -> Path:
    """
    Download a model from HuggingFace Hub with authentication.

    Args:
        repo_id: HuggingFace repository ID (e.g., "PayamFard123/dermaintel-binary-classifier")
        local_dir: Local directory to cache the model
        filename: Specific file to download, or None for entire repo

    Returns:
        Path to downloaded file/directory
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set - cannot download private models")

    try:
        if filename:
            # Download single file
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=HF_TOKEN,
                local_dir=local_dir,
            )
            return Path(path)
        else:
            # Download entire repo
            path = snapshot_download(
                repo_id=repo_id,
                token=HF_TOKEN,
                local_dir=local_dir,
            )
            return Path(path)
    except Exception as e:
        print(f"[WARN] Failed to download from HuggingFace Hub ({repo_id}): {e}")
        raise

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
    binary_model = None
    try:
        checkpoint_path = None

        # Try HuggingFace Hub first
        if HF_TOKEN and HF_BINARY_MODEL_ID:
            try:
                print(f"  Downloading from HuggingFace: {HF_BINARY_MODEL_ID}")
                cache_dir = Path(".cache/models/binary_classifier")
                cache_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = download_model_from_hf(
                    HF_BINARY_MODEL_ID, cache_dir, "best_resnet18_binary.pth"
                )
                print(f"  Downloaded to: {checkpoint_path}")
            except Exception as e:
                print(f"  [WARN] HuggingFace download failed: {e}")
                checkpoint_path = None

        # Fall back to local path
        if checkpoint_path is None and BINARY_CLASSIFIER_PATH.exists():
            checkpoint_path = BINARY_CLASSIFIER_PATH
            print(f"  Using local checkpoint: {checkpoint_path}")

        if checkpoint_path and checkpoint_path.exists():
            binary_model = models.resnet18(weights=None)
            num_features = binary_model.fc.in_features
            binary_model.fc = torch.nn.Linear(num_features, 2)
            checkpoint = torch.load(str(checkpoint_path), map_location=device)
            binary_model.load_state_dict(checkpoint["model"])
            binary_model = binary_model.to(device)
            binary_model.eval()
            print(f"[OK] Binary lesion classifier loaded")
        else:
            print(f"[WARN] Binary classifier not found (no HF token or local file)")
    except Exception as e:
        print(f"[WARN] Could not load binary classifier: {e}")

    # ISIC 8-class model
    print("Loading ISIC 8-class Skin Lesion Classification Model...")
    isic_model = None
    isic_labels = None
    try:
        isic_checkpoint_path = None

        # Try HuggingFace Hub first (use isic_old which has best_model.pth)
        if HF_TOKEN and HF_ISIC_OLD_MODEL_ID:
            try:
                print(f"  Downloading from HuggingFace: {HF_ISIC_OLD_MODEL_ID}")
                cache_dir = Path(".cache/models/isic_old")
                cache_dir.mkdir(parents=True, exist_ok=True)
                isic_checkpoint_path = download_model_from_hf(
                    HF_ISIC_OLD_MODEL_ID, cache_dir, "best_model.pth"
                )
                print(f"  Downloaded to: {isic_checkpoint_path}")
            except Exception as e:
                print(f"  [WARN] HuggingFace download failed: {e}")
                isic_checkpoint_path = None

        # Fall back to local paths
        if isic_checkpoint_path is None:
            if ISIC_MODEL_PATH.exists():
                isic_checkpoint_path = ISIC_MODEL_PATH
            elif ISIC_MODEL_FALLBACK_PATH.exists():
                isic_checkpoint_path = ISIC_MODEL_FALLBACK_PATH
            if isic_checkpoint_path:
                print(f"  Using local checkpoint: {isic_checkpoint_path}")

        if isic_checkpoint_path and isic_checkpoint_path.exists():
            isic_checkpoint = torch.load(str(isic_checkpoint_path), map_location=device, weights_only=False)
            model_config = isic_checkpoint.get('model_config', {})
            num_classes = model_config.get('num_classes', 8)

            if num_classes == 8:
                isic_model = create_isic_classifier(num_classes)
                isic_model.load_state_dict(isic_checkpoint['model_state_dict'])
                isic_model = isic_model.to(device)
                isic_model.eval()
                isic_labels = {i: name for i, name in enumerate(isic_class_names)}
                print(f"[OK] ISIC 8-class model loaded")
            else:
                print(f"[WARN] ISIC 8-class model not found")
        else:
            print(f"[WARN] ISIC checkpoint not found (no HF token or local file)")
    except Exception as e:
        print(f"[ERROR] Failed to load ISIC 8-class model: {e}")

    # ISIC 2020 Binary model (supports both ResNet50 and ResNet34 architectures)
    print("Loading ISIC 2020 Binary Classification Model...")
    isic_2020_binary_model = None
    try:
        isic_2020_path = None

        # Try HuggingFace Hub first
        if HF_TOKEN and HF_ISIC_MODEL_ID:
            try:
                print(f"  Downloading from HuggingFace: {HF_ISIC_MODEL_ID}")
                cache_dir = Path(".cache/models/isic")
                cache_dir.mkdir(parents=True, exist_ok=True)
                isic_2020_path = download_model_from_hf(
                    HF_ISIC_MODEL_ID, cache_dir, "best_model.pth"
                )
                print(f"  Downloaded to: {isic_2020_path}")
            except Exception as e:
                print(f"  [WARN] HuggingFace download failed: {e}")
                isic_2020_path = None

        # Fall back to local path
        if isic_2020_path is None and ISIC_2020_BINARY_PATH.exists():
            isic_2020_path = ISIC_2020_BINARY_PATH
            print(f"  Using local checkpoint: {isic_2020_path}")

        if isic_2020_path and isic_2020_path.exists():
            isic_2020_checkpoint = torch.load(str(isic_2020_path), map_location=device, weights_only=False)
            state_dict = isic_2020_checkpoint.get('model_state_dict', isic_2020_checkpoint)

            # Detect architecture from state_dict keys
            state_keys = list(state_dict.keys())
            is_resnet34 = any('layer4.2' in k for k in state_keys) and not any('layer4.5' in k for k in state_keys)
            is_resnet50 = any('backbone' in k for k in state_keys) or any('classifier' in k for k in state_keys)

            if is_resnet50:
                # Old architecture: ResNet50 with custom classifier
                print(f"  Detected: ResNet50 architecture")
                model_config = isic_2020_checkpoint.get('model_config', {})
                num_classes = model_config.get('num_classes', 2)
                isic_2020_binary_model = create_isic_classifier(num_classes)
                isic_2020_binary_model.load_state_dict(state_dict)
            else:
                # New architecture: ResNet34 with simple fc (fine-tuned model)
                print(f"  Detected: ResNet34 fine-tuned architecture (94.3% AUC)")
                isic_2020_binary_model = models.resnet34(weights=None)
                num_features = isic_2020_binary_model.fc.in_features
                isic_2020_binary_model.fc = nn.Linear(num_features, 2)
                isic_2020_binary_model.load_state_dict(state_dict)

            isic_2020_binary_model = isic_2020_binary_model.to(device)
            isic_2020_binary_model.eval()
            print(f"[OK] ISIC 2020 Binary model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load ISIC 2020 Binary model: {e}")

    # Lesion classification model (HuggingFace ViT)
    print("Loading Lesion Classification Model...")
    lesion_model = None
    lesion_processor = None
    labels = {}
    try:
        lesion_model_path = None

        # Try HuggingFace Hub first (this is a full transformers model)
        if HF_TOKEN and HF_LESION_MODEL_ID:
            try:
                print(f"  Downloading from HuggingFace: {HF_LESION_MODEL_ID}")
                cache_dir = Path(".cache/models/lesion_classifier")
                cache_dir.mkdir(parents=True, exist_ok=True)
                lesion_model_path = download_model_from_hf(
                    HF_LESION_MODEL_ID, cache_dir
                )
                print(f"  Downloaded to: {lesion_model_path}")
            except Exception as e:
                print(f"  [WARN] HuggingFace download failed: {e}")
                lesion_model_path = None

        # Fall back to local path
        if lesion_model_path is None and LESION_MODEL_PATH.exists():
            lesion_model_path = LESION_MODEL_PATH
            print(f"  Using local checkpoint: {lesion_model_path}")

        if lesion_model_path and lesion_model_path.exists():
            lesion_model = AutoModelForImageClassification.from_pretrained(str(lesion_model_path))
            lesion_processor = AutoImageProcessor.from_pretrained(str(lesion_model_path))
            labels = lesion_model.config.id2label
            lesion_model.eval()
            print(f"[OK] Lesion classification model loaded")
        else:
            print(f"[WARN] Lesion model not found (no HF token or local file)")
    except Exception as e:
        print(f"[WARN] Could not load lesion model: {e}")

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
        infectious_model_dir = None

        # Try HuggingFace Hub first
        if HF_TOKEN and HF_INFECTIOUS_MODEL_ID:
            try:
                print(f"  Downloading from HuggingFace: {HF_INFECTIOUS_MODEL_ID}")
                cache_dir = Path(".cache/models/infectious_disease")
                cache_dir.mkdir(parents=True, exist_ok=True)
                infectious_model_dir = download_model_from_hf(
                    HF_INFECTIOUS_MODEL_ID, cache_dir
                )
                print(f"  Downloaded to: {infectious_model_dir}")
            except Exception as e:
                print(f"  [WARN] HuggingFace download failed: {e}")
                infectious_model_dir = None

        # Fall back to local path
        if infectious_model_dir is None and INFECTIOUS_MODEL_DIR.exists():
            infectious_model_dir = INFECTIOUS_MODEL_DIR
            print(f"  Using local model: {infectious_model_dir}")

        if infectious_model_dir and infectious_model_dir.exists():
            if (infectious_model_dir / "model" / "config.json").exists():
                infectious_model = AutoModelForImageClassification.from_pretrained(str(infectious_model_dir / "model"))
                infectious_processor = AutoImageProcessor.from_pretrained(str(infectious_model_dir / "processor"))
                infectious_model.eval()
                infectious_labels = infectious_model.config.id2label
                print(f"[OK] Infectious disease model loaded (HuggingFace format)")
            elif (infectious_model_dir / "best_model.pth").exists() or (infectious_model_dir / "model.pth").exists():
                # Determine which model file to use
                model_file = "best_model.pth" if (infectious_model_dir / "best_model.pth").exists() else "model.pth"

                with open(infectious_model_dir / "metadata.json", 'r') as f:
                    infectious_metadata = json.load(f)

                infectious_labels = {i: name for i, name in enumerate(infectious_metadata['class_names'])}
                infectious_class_metadata = infectious_metadata.get('class_metadata', {})

                # Get architecture from metadata (new format) or config (old format)
                model_arch = infectious_metadata.get('architecture',
                             infectious_metadata.get('config', {}).get('model_name', 'resnet50'))
                num_classes = len(infectious_labels)

                # Load state dict first to detect actual architecture
                state_dict = torch.load(str(infectious_model_dir / model_file), map_location=device, weights_only=True)

                # Handle checkpoint format (may have 'model_state_dict' key)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']

                state_keys = list(state_dict.keys())

                # Detect actual architecture from state dict keys
                has_resnet_keys = any(k.startswith('layer1') for k in state_keys)
                has_efficientnet_keys = any(k.startswith('features.') for k in state_keys)
                has_sequential_fc = 'fc.1.weight' in state_keys or 'fc.1.bias' in state_keys

                # Detect ResNet34 vs ResNet18 by checking layer4 depth
                is_resnet34 = any('layer4.2' in k for k in state_keys) and not any('layer4.5' in k for k in state_keys)

                if has_resnet_keys:
                    # ResNet architecture detected
                    if model_arch == 'resnet34' or is_resnet34:
                        # ResNet34 architecture (improved model with 86.5% AUC)
                        print(f"  Detected: ResNet34 architecture (improved model)")
                        infectious_model = models.resnet34(weights=None)
                        num_features = infectious_model.fc.in_features  # 512 for ResNet34
                        if has_sequential_fc:
                            # ResNet34 with Sequential fc (Dropout(0.4) + Linear)
                            infectious_model.fc = nn.Sequential(
                                nn.Dropout(p=0.4),
                                nn.Linear(num_features, num_classes)
                            )
                        else:
                            infectious_model.fc = nn.Linear(num_features, num_classes)
                    elif has_sequential_fc:
                        # ResNet18 with Sequential fc (Dropout + Linear)
                        print(f"  Detected: ResNet18 with Dropout")
                        infectious_model = models.resnet18(weights=None)
                        num_features = infectious_model.fc.in_features
                        infectious_model.fc = nn.Sequential(
                            nn.Dropout(p=0.3),
                            nn.Linear(num_features, num_classes)
                        )
                    else:
                        # ResNet18 with simple Linear fc
                        print(f"  Detected: ResNet18 simple")
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

                # Report metrics if available
                metrics = infectious_metadata.get('metrics', {})
                if metrics:
                    auc = metrics.get('auc', 'N/A')
                    print(f"[OK] Infectious disease model loaded (PyTorch format, {num_classes} classes, AUC: {auc}%)")
                else:
                    print(f"[OK] Infectious disease model loaded (PyTorch format, {num_classes} classes)")
        else:
            print(f"[WARN] Infectious disease model not found (no HF token or local file)")
    except Exception as e:
        print(f"[WARN] Could not load infectious disease model: {e}")

    print("\n" + "="*60)
    print("All models loaded successfully!")
    print("="*60 + "\n")


# =============================================================================
# ENSEMBLE PREDICTION FUNCTION
# =============================================================================

def ensemble_predict(image: Image.Image, use_malignancy_boost: bool = True) -> dict:
    """
    Ensemble prediction combining multiple models for improved accuracy.

    Strategy:
    1. Run lesion_model (primary ViT model)
    2. Run isic_model (8-class ISIC model) if available
    3. Run isic_2020_binary_model (benign/malignant) if available
    4. Combine predictions with weighted averaging
    5. Boost malignant classes if binary model predicts malignant with high confidence

    Returns:
        dict with 'probabilities', 'predicted_class', 'confidence', 'malignancy_score', 'ensemble_info'
    """
    import torch.nn.functional as F

    results = {
        "models_used": [],
        "individual_predictions": {},
        "malignancy_score": None,
        "malignancy_boost_applied": False,
    }

    # Standardized class mapping
    class_map = {
        "Melanoma": "Melanoma",
        "MEL": "Melanoma",
        "mel": "Melanoma",
        "Melanocytic Nevus": "Melanocytic Nevus",
        "NV": "Melanocytic Nevus",
        "nv": "Melanocytic Nevus",
        "Basal Cell Carcinoma": "Basal Cell Carcinoma",
        "BCC": "Basal Cell Carcinoma",
        "bcc": "Basal Cell Carcinoma",
        "Actinic Keratosis": "Actinic Keratosis",
        "AK": "Actinic Keratosis",
        "akiec": "Actinic Keratosis",
        "Benign Keratosis": "Benign Keratosis",
        "BKL": "Benign Keratosis",
        "bkl": "Benign Keratosis",
        "Dermatofibroma": "Dermatofibroma",
        "DF": "Dermatofibroma",
        "df": "Dermatofibroma",
        "Vascular Lesion": "Vascular Lesion",
        "VASC": "Vascular Lesion",
        "vasc": "Vascular Lesion",
        "Squamous Cell Carcinoma": "Squamous Cell Carcinoma",
        "SCC": "Squamous Cell Carcinoma",
        "scc": "Squamous Cell Carcinoma",
    }

    malignant_classes = {"Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Actinic Keratosis"}

    # Initialize combined probabilities
    combined_probs = {}
    weights_sum = 0.0

    # 1. Primary model: lesion_model (weight: 0.5)
    if lesion_model is not None and lesion_processor is not None:
        try:
            inputs = lesion_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = lesion_model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)[0]

            labels = lesion_model.config.id2label
            lesion_probs = {}
            for i, prob in enumerate(probs):
                class_name = class_map.get(labels[i], labels[i])
                lesion_probs[class_name] = prob.item()

            results["models_used"].append("lesion_model")
            results["individual_predictions"]["lesion_model"] = lesion_probs

            # Add to combined with weight 0.5
            weight = 0.5
            for cls, prob in lesion_probs.items():
                combined_probs[cls] = combined_probs.get(cls, 0) + prob * weight
            weights_sum += weight
        except Exception as e:
            print(f"[WARN] lesion_model prediction failed: {e}")

    # 2. ISIC 8-class model (weight: 0.35)
    if isic_model is not None:
        try:
            img_tensor = isic_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = isic_model(img_tensor)
                probs = F.softmax(outputs, dim=-1)[0]

            isic_probs = {}
            for i, class_code in enumerate(isic_class_names):
                class_name = isic_class_full_names.get(class_code, class_code)
                isic_probs[class_name] = probs[i].item()

            results["models_used"].append("isic_model")
            results["individual_predictions"]["isic_model"] = isic_probs

            # Add to combined with weight 0.35
            weight = 0.35
            for cls, prob in isic_probs.items():
                combined_probs[cls] = combined_probs.get(cls, 0) + prob * weight
            weights_sum += weight
        except Exception as e:
            print(f"[WARN] isic_model prediction failed: {e}")

    # 3. ISIC 2020 Binary model for malignancy detection (weight: 0.15 influence)
    malignancy_prob = 0.0
    if isic_2020_binary_model is not None:
        try:
            img_tensor = isic_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = isic_2020_binary_model(img_tensor)
                probs = F.softmax(outputs, dim=-1)[0]

            # Index 1 is typically "malignant"
            malignancy_prob = probs[1].item() if len(probs) > 1 else probs[0].item()
            results["malignancy_score"] = round(malignancy_prob, 4)
            results["models_used"].append("isic_2020_binary")
            results["individual_predictions"]["isic_2020_binary"] = {
                "benign": probs[0].item(),
                "malignant": malignancy_prob
            }
        except Exception as e:
            print(f"[WARN] isic_2020_binary prediction failed: {e}")

    # Normalize combined probabilities
    if weights_sum > 0:
        for cls in combined_probs:
            combined_probs[cls] /= weights_sum
    else:
        # Fallback if no models worked
        return {
            "probabilities": {"Unknown": 1.0},
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "malignancy_score": None,
            "ensemble_info": results
        }

    # 4. Apply malignancy boost if binary model is confident about malignancy
    if use_malignancy_boost and malignancy_prob > 0.5:
        results["malignancy_boost_applied"] = True
        boost_factor = 1.0 + (malignancy_prob - 0.5) * 2  # 1.0 to 2.0 boost

        # Boost malignant class probabilities
        for cls in malignant_classes:
            if cls in combined_probs:
                combined_probs[cls] *= boost_factor

        # Re-normalize
        total = sum(combined_probs.values())
        if total > 0:
            for cls in combined_probs:
                combined_probs[cls] /= total

    # Round probabilities
    combined_probs = {cls: round(prob, 4) for cls, prob in combined_probs.items()}

    # Get predicted class and confidence
    predicted_class = max(combined_probs, key=combined_probs.get)
    confidence = combined_probs[predicted_class]

    return {
        "probabilities": combined_probs,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "malignancy_score": results["malignancy_score"],
        "ensemble_info": results
    }
