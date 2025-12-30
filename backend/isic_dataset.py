"""
ISIC Archive Dataset Loader

Provides PyTorch Dataset classes for loading ISIC Archive images
with full support for the existing training pipeline.

Supports:
- ISIC 2016-2020 Challenge datasets
- HAM10000 (via ISIC)
- BCN20000
- Custom downloaded ISIC images

Classes:
- ISICDataset: General ISIC dataset loader
- ISICChallengeDataset: ISIC Challenge-specific loader
- CombinedSkinDataset: Combines multiple datasets for training
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Callable
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ISICDataset(Dataset):
    """
    General ISIC Archive Dataset loader.

    Supports loading images from:
    - Organized directory structure (diagnosis-based folders)
    - Flat directory with metadata CSV

    Diagnosis categories:
    - MEL: Melanoma (malignant)
    - NV: Melanocytic nevus (benign)
    - BCC: Basal cell carcinoma (malignant)
    - AK: Actinic keratosis (pre-malignant)
    - BKL: Benign keratosis (benign)
    - DF: Dermatofibroma (benign)
    - VASC: Vascular lesion (benign)
    - SCC: Squamous cell carcinoma (malignant)
    """

    # Standard ISIC class mapping (8 classes)
    CLASSES_8 = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    # HAM10000 / ISIC 2018 class mapping (7 classes)
    CLASSES_7 = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    # ISIC 2019 class mapping (9 classes)
    CLASSES_9 = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

    # Malignancy mapping
    MALIGNANCY_MAP = {
        'mel': 'malignant', 'MEL': 'malignant', 'melanoma': 'malignant',
        'bcc': 'malignant', 'BCC': 'malignant', 'basal cell carcinoma': 'malignant',
        'scc': 'malignant', 'SCC': 'malignant', 'squamous cell carcinoma': 'malignant',
        'akiec': 'pre-malignant', 'AK': 'pre-malignant', 'actinic keratosis': 'pre-malignant',
        'nv': 'benign', 'NV': 'benign', 'nevus': 'benign', 'melanocytic nevus': 'benign',
        'bkl': 'benign', 'BKL': 'benign', 'benign keratosis': 'benign', 'seborrheic keratosis': 'benign',
        'df': 'benign', 'DF': 'benign', 'dermatofibroma': 'benign',
        'vasc': 'benign', 'VASC': 'benign', 'vascular lesion': 'benign',
        'unk': 'unknown', 'UNK': 'unknown', 'unknown': 'unknown',
    }

    # Normalize diagnosis names
    DIAGNOSIS_NORMALIZE = {
        'melanoma': 'MEL', 'mel': 'MEL',
        'nevus': 'NV', 'nv': 'NV', 'melanocytic nevus': 'NV',
        'basal cell carcinoma': 'BCC', 'bcc': 'BCC',
        'actinic keratosis': 'AK', 'ak': 'AK', 'akiec': 'AK',
        'benign keratosis': 'BKL', 'bkl': 'BKL', 'seborrheic keratosis': 'BKL',
        'dermatofibroma': 'DF', 'df': 'DF',
        'vascular lesion': 'VASC', 'vasc': 'VASC',
        'squamous cell carcinoma': 'SCC', 'scc': 'SCC',
    }

    def __init__(
        self,
        root_dir: str,
        metadata_file: Optional[str] = None,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_classes: int = 8,
        binary_classification: bool = False,
        balance_classes: bool = False,
        include_metadata: bool = True,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
    ):
        """
        Initialize ISIC Dataset.

        Args:
            root_dir: Root directory containing images
            metadata_file: Path to metadata CSV file
            split: 'train', 'val', or 'test'
            transform: Image transforms
            target_transform: Label transforms
            num_classes: Number of classes (7, 8, or 9)
            binary_classification: If True, classify as malignant vs benign
            balance_classes: If True, apply class balancing
            include_metadata: If True, return patient metadata with samples
            image_extensions: List of valid image extensions
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes
        self.binary_classification = binary_classification
        self.balance_classes = balance_classes
        self.include_metadata = include_metadata
        self.image_extensions = image_extensions

        # Set class list based on num_classes
        if num_classes == 7:
            self.classes = self.CLASSES_7
        elif num_classes == 9:
            self.classes = self.CLASSES_9
        else:
            self.classes = self.CLASSES_8

        # Load metadata
        self.metadata_df = None
        if metadata_file and Path(metadata_file).exists():
            self.metadata_df = pd.read_csv(metadata_file)
            logger.info(f"Loaded metadata with {len(self.metadata_df)} records")

        # Load samples
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

        # Compute class weights for balancing
        if balance_classes:
            self.class_weights = self._compute_class_weights()

    def _load_samples(self) -> List[Dict]:
        """Load image samples with labels."""
        samples = []

        # Check for split-based directory structure (train/val/test with class subfolders)
        split_dir = self.root_dir / self.split
        # Only use split_dir if it has class subdirectories (not just flat images)
        has_class_subdirs = split_dir.exists() and any(
            d.is_dir() for d in split_dir.iterdir() if not d.name.startswith('.')
        )
        if has_class_subdirs:
            samples = self._load_from_split_dir(split_dir)
        # Check for organized directory structure (organized/category/diagnosis)
        elif (self.root_dir / "organized").exists():
            samples = self._load_from_organized(self.root_dir / "organized")
            # Apply split
            samples = self._apply_split(samples)
        elif self.metadata_df is not None:
            samples = self._load_from_metadata()
            # Apply split
            samples = self._apply_split(samples)
        else:
            # Try to load from flat directory or standard ISIC structure
            samples = self._load_from_directory()
            # Apply split
            samples = self._apply_split(samples)

        if not samples:
            logger.warning(f"No samples found in {self.root_dir}")

        return samples

    def _load_from_split_dir(self, split_dir: Path) -> List[Dict]:
        """Load samples from split directory structure (split/class/images)."""
        samples = []

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            diagnosis = class_dir.name.upper()
            normalized_diagnosis = self.DIAGNOSIS_NORMALIZE.get(diagnosis.lower(), diagnosis)

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    samples.append({
                        'image_path': img_path,
                        'image_id': img_path.stem,
                        'diagnosis': normalized_diagnosis,
                        'category': self.MALIGNANCY_MAP.get(normalized_diagnosis.lower(), 'unknown'),
                        'age': None,
                        'sex': None,
                        'anatomic_site': None,
                    })

        return samples

    def _apply_split(self, samples: List[Dict]) -> List[Dict]:
        """Apply train/val/test split to samples."""
        if not samples:
            return samples

        np.random.seed(42)
        indices = np.random.permutation(len(samples))

        n = len(samples)
        if self.split == 'train':
            indices = indices[:int(0.8 * n)]
        elif self.split == 'val':
            indices = indices[int(0.8 * n):int(0.9 * n)]
        else:  # test
            indices = indices[int(0.9 * n):]

        return [samples[i] for i in indices]

    def _load_from_organized(self, organized_dir: Path) -> List[Dict]:
        """Load samples from organized directory structure."""
        samples = []

        for category_dir in organized_dir.iterdir():
            if not category_dir.is_dir():
                continue

            for diagnosis_dir in category_dir.iterdir():
                if not diagnosis_dir.is_dir():
                    continue

                diagnosis = diagnosis_dir.name.upper()
                normalized_diagnosis = self.DIAGNOSIS_NORMALIZE.get(diagnosis.lower(), diagnosis)

                for img_path in diagnosis_dir.iterdir():
                    if img_path.suffix.lower() in self.image_extensions:
                        samples.append({
                            'image_path': img_path,
                            'image_id': img_path.stem,
                            'diagnosis': normalized_diagnosis,
                            'category': category_dir.name,
                            'age': None,
                            'sex': None,
                            'anatomic_site': None,
                        })

        return samples

    def _load_from_metadata(self) -> List[Dict]:
        """Load samples using metadata CSV file."""
        samples = []

        # Check multiple possible image directories
        possible_dirs = [
            self.root_dir / "images",
            self.root_dir / "ISIC_2019_Training_Input",
            self.root_dir / "ISIC_2020_Training_JPEG",
            self.root_dir / "ISIC2018_Task3_Training_Input",
            self.root_dir / "train",
            self.root_dir,
        ]
        images_dir = None
        for d in possible_dirs:
            if d.exists() and any(d.iterdir()):
                images_dir = d
                logger.info(f"Found images directory: {d}")
                break

        if images_dir is None:
            logger.warning(f"No images directory found in {self.root_dir}")
            return samples

        # Handle different metadata column names
        id_col = None
        for col in ['isic_id', 'image_id', 'image', 'ISIC_id', 'image_name']:
            if col in self.metadata_df.columns:
                id_col = col
                break

        diagnosis_col = None
        for col in ['diagnosis', 'dx', 'target', 'label', 'benign_malignant']:
            if col in self.metadata_df.columns:
                diagnosis_col = col
                break

        # Check for one-hot encoded labels (ISIC 2019 format)
        one_hot_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        is_one_hot = all(col in self.metadata_df.columns for col in one_hot_cols)
        if is_one_hot:
            logger.info("Detected one-hot encoded labels (ISIC 2019 format)")

        if id_col is None:
            logger.error("Could not find image ID column in metadata")
            return samples

        for _, row in self.metadata_df.iterrows():
            image_id = row[id_col]

            # Find image file
            img_path = None
            for ext in self.image_extensions:
                potential_path = images_dir / f"{image_id}{ext}"
                if potential_path.exists():
                    img_path = potential_path
                    break

            if img_path is None:
                # Try without extension
                for ext in self.image_extensions:
                    potential_path = images_dir / f"{image_id}{ext.upper()}"
                    if potential_path.exists():
                        img_path = potential_path
                        break

            if img_path is None:
                continue

            # Get diagnosis
            if is_one_hot:
                # Decode one-hot encoded labels
                for col in one_hot_cols:
                    if row.get(col, 0) == 1.0:
                        diagnosis = col
                        break
                else:
                    diagnosis = 'UNK'
            elif diagnosis_col:
                diagnosis = row.get(diagnosis_col, 'unknown')
                if isinstance(diagnosis, (int, float)):
                    # Handle numeric labels (ISIC 2020 format)
                    diagnosis = 'MEL' if diagnosis == 1 else 'NV'
            else:
                diagnosis = 'unknown'

            normalized_diagnosis = self.DIAGNOSIS_NORMALIZE.get(str(diagnosis).lower(), str(diagnosis).upper())

            samples.append({
                'image_path': img_path,
                'image_id': image_id,
                'diagnosis': normalized_diagnosis,
                'category': self.MALIGNANCY_MAP.get(diagnosis.lower() if isinstance(diagnosis, str) else 'unknown', 'unknown'),
                'age': row.get('age_approx') or row.get('age'),
                'sex': row.get('sex'),
                'anatomic_site': row.get('anatom_site_general') or row.get('localization'),
            })

        return samples

    def _load_from_directory(self) -> List[Dict]:
        """Load samples from flat directory structure."""
        samples = []

        # Check common ISIC directory structures
        for subdir in ['images', 'ISIC_2020_Training_JPEG', 'ISIC_2019_Training_Input',
                       'ISIC2018_Task3_Training_Input', 'train', 'training']:
            images_dir = self.root_dir / subdir
            if images_dir.exists():
                for img_path in images_dir.iterdir():
                    if img_path.suffix.lower() in self.image_extensions:
                        samples.append({
                            'image_path': img_path,
                            'image_id': img_path.stem,
                            'diagnosis': 'unknown',
                            'category': 'unknown',
                            'age': None,
                            'sex': None,
                            'anatomic_site': None,
                        })
                break

        return samples

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced sampling."""
        labels = [self._get_label(s['diagnosis']) for s in self.samples]
        class_counts = Counter(labels)
        total = len(labels)

        weights = []
        for i in range(len(self.classes) if not self.binary_classification else 2):
            count = class_counts.get(i, 1)
            weights.append(total / count)

        return torch.tensor(weights, dtype=torch.float32)

    def _get_label(self, diagnosis: str) -> int:
        """Convert diagnosis to label index."""
        if self.binary_classification:
            malignancy = self.MALIGNANCY_MAP.get(diagnosis.lower(), 'benign')
            return 1 if malignancy == 'malignant' else 0

        try:
            return self.classes.index(diagnosis)
        except ValueError:
            # Try normalized version
            normalized = self.DIAGNOSIS_NORMALIZE.get(diagnosis.lower(), diagnosis)
            try:
                return self.classes.index(normalized)
            except ValueError:
                return len(self.classes) - 1  # Unknown class

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Get label
        label = self._get_label(sample['diagnosis'])

        if self.target_transform:
            label = self.target_transform(label)

        result = {
            'image': image,
            'label': label,
            'image_id': sample['image_id'],
            'diagnosis': sample['diagnosis'],
        }

        if self.include_metadata:
            # Replace None values with defaults for DataLoader compatibility
            result['metadata'] = {
                'age': sample['age'] if sample['age'] is not None else -1,
                'sex': sample['sex'] if sample['sex'] is not None else 'unknown',
                'anatomic_site': sample['anatomic_site'] if sample['anatomic_site'] is not None else 'unknown',
                'category': sample['category'] if sample['category'] is not None else 'unknown',
            }

        return result

    def get_sampler(self) -> WeightedRandomSampler:
        """Get weighted sampler for balanced training."""
        labels = [self._get_label(s['diagnosis']) for s in self.samples]
        sample_weights = [self.class_weights[l].item() for l in labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights))


class ISICChallengeDataset(ISICDataset):
    """
    ISIC Challenge-specific dataset loader.

    Handles the specific format of ISIC Challenge datasets (2016-2020).
    """

    CHALLENGE_CONFIGS = {
        2016: {
            'classes': ['benign', 'malignant'],
            'num_classes': 2,
            'train_images': 'ISBI2016_ISIC_Part1_Training_Data',
            'train_labels': 'ISBI2016_ISIC_Part1_Training_GroundTruth.csv',
        },
        2017: {
            'classes': ['melanoma', 'nevus', 'seborrheic_keratosis'],
            'num_classes': 3,
            'train_images': 'ISIC-2017_Training_Data',
            'train_labels': 'ISIC-2017_Training_Part3_GroundTruth.csv',
        },
        2018: {
            'classes': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
            'num_classes': 7,
            'train_images': 'ISIC2018_Task3_Training_Input',
            'train_labels': 'ISIC2018_Task3_Training_GroundTruth.csv',
        },
        2019: {
            'classes': ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'],
            'num_classes': 9,
            'train_images': 'ISIC_2019_Training_Input',
            'train_labels': 'ISIC_2019_Training_GroundTruth.csv',
        },
        2020: {
            'classes': ['benign', 'malignant'],
            'num_classes': 2,
            'train_images': 'ISIC_2020_Training_JPEG',
            'train_labels': 'ISIC_2020_Training_GroundTruth.csv',
        },
    }

    def __init__(
        self,
        root_dir: str,
        year: int = 2020,
        split: str = 'train',
        transform: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize ISIC Challenge Dataset.

        Args:
            root_dir: Root directory containing challenge data
            year: Challenge year (2016-2020)
            split: 'train', 'val', or 'test'
            transform: Image transforms
        """
        self.year = year
        self.config = self.CHALLENGE_CONFIGS.get(year, self.CHALLENGE_CONFIGS[2020])

        # Set up paths based on challenge year
        root_path = Path(root_dir)
        images_dir = root_path / self.config['train_images']
        metadata_file = root_path / self.config['train_labels']

        # Override classes
        self.challenge_classes = self.config['classes']

        super().__init__(
            root_dir=root_dir,
            metadata_file=str(metadata_file) if metadata_file.exists() else None,
            split=split,
            transform=transform,
            num_classes=self.config['num_classes'],
            **kwargs
        )

        # Override class list
        self.classes = self.challenge_classes

    def _load_from_metadata(self) -> List[Dict]:
        """Load samples from challenge metadata format."""
        samples = []

        if self.metadata_df is None:
            return samples

        # Challenge datasets use different column formats
        if self.year == 2020:
            # ISIC 2020: binary classification (target: 0/1)
            id_col = 'image_name'
            label_col = 'target'
        elif self.year == 2019:
            # ISIC 2019: one-hot encoded labels
            id_col = 'image'
            label_col = None  # Multiple columns
        elif self.year == 2018:
            # ISIC 2018: one-hot encoded labels
            id_col = 'image'
            label_col = None
        else:
            id_col = 'image'
            label_col = 'label'

        images_dir = self.root_dir / self.config['train_images']

        for _, row in self.metadata_df.iterrows():
            image_id = row[id_col]

            # Find image
            img_path = None
            for ext in self.image_extensions:
                potential_path = images_dir / f"{image_id}{ext}"
                if potential_path.exists():
                    img_path = potential_path
                    break

            if img_path is None:
                continue

            # Get diagnosis based on year format
            if label_col:
                diagnosis = row[label_col]
                if isinstance(diagnosis, (int, float)):
                    diagnosis = self.classes[int(diagnosis)]
            else:
                # One-hot encoded - find the 1
                for cls in self.classes:
                    if cls in row and row[cls] == 1.0:
                        diagnosis = cls
                        break
                else:
                    diagnosis = 'unknown'

            samples.append({
                'image_path': img_path,
                'image_id': image_id,
                'diagnosis': diagnosis,
                'category': self.MALIGNANCY_MAP.get(diagnosis.lower(), 'unknown'),
                'age': row.get('age_approx'),
                'sex': row.get('sex'),
                'anatomic_site': row.get('anatom_site_general'),
            })

        return samples


class CombinedSkinDataset(Dataset):
    """
    Combined dataset that merges multiple skin image datasets.

    Supports combining:
    - ISIC Archive
    - HAM10000
    - DermNet
    - PH2
    - Custom datasets
    """

    def __init__(
        self,
        datasets: List[Dataset],
        unified_classes: Optional[List[str]] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize combined dataset.

        Args:
            datasets: List of datasets to combine
            unified_classes: Unified class list (if None, uses ISIC 8-class)
            transform: Optional transform to apply to all images
        """
        self.datasets = datasets
        self.unified_classes = unified_classes or ISICDataset.CLASSES_8
        self.transform = transform

        # Build combined index
        self.combined_samples = []
        for dataset_idx, dataset in enumerate(datasets):
            for sample_idx in range(len(dataset)):
                self.combined_samples.append((dataset_idx, sample_idx))

        logger.info(f"Combined dataset with {len(self.combined_samples)} total samples")

    def __len__(self):
        return len(self.combined_samples)

    def __getitem__(self, idx: int) -> Dict:
        dataset_idx, sample_idx = self.combined_samples[idx]
        sample = self.datasets[dataset_idx][sample_idx]

        # Apply additional transform if specified
        if self.transform and 'image' in sample:
            if isinstance(sample['image'], torch.Tensor):
                # Convert to PIL, transform, convert back
                to_pil = transforms.ToPILImage()
                image = to_pil(sample['image'])
                sample['image'] = self.transform(image)

        return sample


# Transform factories
def get_isic_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Get training transforms optimized for ISIC images."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_isic_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Get validation/test transforms for ISIC images."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def create_isic_dataloaders(
    root_dir: str,
    metadata_file: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    num_classes: int = 8,
    binary_classification: bool = False,
    balance_classes: bool = True,
    include_metadata: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for ISIC dataset.

    Args:
        root_dir: Root directory containing ISIC images
        metadata_file: Path to metadata CSV file
        batch_size: Batch size
        image_size: Image size for resizing
        num_workers: Number of data loading workers
        num_classes: Number of classes
        binary_classification: Whether to use binary classification
        balance_classes: Whether to balance classes during training
        include_metadata: Whether to include patient metadata in samples

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ISICDataset(
        root_dir=root_dir,
        metadata_file=metadata_file,
        split='train',
        transform=get_isic_train_transforms(image_size),
        num_classes=num_classes,
        binary_classification=binary_classification,
        balance_classes=balance_classes,
        include_metadata=include_metadata,
    )

    val_dataset = ISICDataset(
        root_dir=root_dir,
        metadata_file=metadata_file,
        split='val',
        transform=get_isic_val_transforms(image_size),
        num_classes=num_classes,
        binary_classification=binary_classification,
        include_metadata=include_metadata,
    )

    test_dataset = ISICDataset(
        root_dir=root_dir,
        metadata_file=metadata_file,
        split='test',
        transform=get_isic_val_transforms(image_size),
        num_classes=num_classes,
        binary_classification=binary_classification,
        include_metadata=include_metadata,
    )

    # Create dataloaders
    train_sampler = train_dataset.get_sampler() if balance_classes else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def print_dataset_stats(dataset: ISICDataset):
    """Print dataset statistics."""
    print(f"\nDataset Statistics ({dataset.split}):")
    print(f"  Total samples: {len(dataset)}")

    # Class distribution
    labels = [dataset._get_label(s['diagnosis']) for s in dataset.samples]
    class_counts = Counter(labels)

    print(f"\n  Class distribution:")
    for i, cls in enumerate(dataset.classes):
        count = class_counts.get(i, 0)
        pct = 100 * count / len(labels) if labels else 0
        print(f"    {cls:10}: {count:6} ({pct:5.1f}%)")


if __name__ == '__main__':
    # Test dataset loading
    import argparse

    parser = argparse.ArgumentParser(description="Test ISIC dataset loading")
    parser.add_argument("--root_dir", type=str, default="./data/isic",
                        help="Root directory containing ISIC images")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Path to metadata CSV file")
    args = parser.parse_args()

    print("=" * 80)
    print("ISIC Dataset Loader Test")
    print("=" * 80)

    # Test ISICDataset
    print("\nTesting ISICDataset...")
    dataset = ISICDataset(
        root_dir=args.root_dir,
        metadata_file=args.metadata,
        split='train',
        transform=get_isic_train_transforms(224),
        num_classes=8,
    )

    print_dataset_stats(dataset)

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n  Sample keys: {sample.keys()}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Label: {sample['label']} ({sample['diagnosis']})")

    # Test dataloader creation
    print("\nTesting dataloader creation...")
    train_loader, val_loader, test_loader = create_isic_dataloaders(
        root_dir=args.root_dir,
        metadata_file=args.metadata,
        batch_size=16,
        num_workers=0,  # For testing
    )

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
