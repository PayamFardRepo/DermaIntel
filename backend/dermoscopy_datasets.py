"""
Dataset loaders for dermoscopic image datasets (PH2 and HAM10000).

PH2 Dataset: 200 dermoscopic images from Pedro Hispano Hospital
HAM10000: Human Against Machine with 10000 training images
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from typing import Tuple, Dict, Optional, List
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm


class PH2Dataset(Dataset):
    """
    PH2 Dataset loader for dermoscopic images.

    The PH2 database contains 200 dermoscopic images of melanocytic lesions,
    including melanomas and nevi with expert annotations.

    Dataset structure:
    - Images: 8-bit RGB images
    - Masks: Binary segmentation masks
    - Clinical diagnosis labels
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        download: bool = False
    ):
        """
        Args:
            root_dir: Root directory where dataset is stored
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply to images
            download: Whether to download dataset if not present
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Create directories
        self.images_dir = self.root_dir / 'PH2Dataset' / 'images'
        self.masks_dir = self.root_dir / 'PH2Dataset' / 'masks'
        self.labels_file = self.root_dir / 'PH2Dataset' / 'labels.csv'

        if download and not self.images_dir.exists():
            self.download_ph2()

        # Load image paths and labels
        self.samples = self._load_samples()

    def download_ph2(self):
        """
        Download and prepare PH2 dataset.
        Note: PH2 requires academic access. This provides instructions.
        """
        print("=" * 80)
        print("PH2 DATASET DOWNLOAD INSTRUCTIONS")
        print("=" * 80)
        print("\nThe PH2 dataset requires academic registration.")
        print("\nSteps to download:")
        print("1. Visit: https://www.fc.up.pt/addi/ph2%20database.html")
        print("2. Fill out the request form with your academic information")
        print("3. Download the dataset ZIP file")
        print(f"4. Extract it to: {self.root_dir}")
        print("\nAlternatively, the dataset is available on:")
        print("- Kaggle: https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection")
        print("\nDirectory structure should be:")
        print(f"  {self.root_dir}/")
        print("    ├── PH2Dataset/")
        print("    │   ├── images/")
        print("    │   ├── masks/")
        print("    │   └── labels.csv")
        print("=" * 80)

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def _load_samples(self) -> List[Dict]:
        """Load image paths and corresponding labels"""
        samples = []

        if not self.images_dir.exists():
            print(f"Warning: Images directory not found at {self.images_dir}")
            return samples

        # Get all image files
        image_files = sorted(list(self.images_dir.glob('*.bmp')) +
                           list(self.images_dir.glob('*.jpg')) +
                           list(self.images_dir.glob('*.png')))

        # Split dataset (80% train, 10% val, 10% test)
        n = len(image_files)
        if self.split == 'train':
            image_files = image_files[:int(0.8 * n)]
        elif self.split == 'val':
            image_files = image_files[int(0.8 * n):int(0.9 * n)]
        else:  # test
            image_files = image_files[int(0.9 * n):]

        for img_path in image_files:
            # Look for corresponding mask
            mask_path = self.masks_dir / f"{img_path.stem}_mask.bmp"
            if not mask_path.exists():
                mask_path = self.masks_dir / f"{img_path.stem}_lesion.bmp"

            samples.append({
                'image': img_path,
                'mask': mask_path if mask_path.exists() else None,
                'id': img_path.stem
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image']).convert('RGB')

        # Load mask if available
        if sample['mask'] and sample['mask'].exists():
            mask = Image.open(sample['mask']).convert('L')
            mask = np.array(mask) > 127  # Binarize
        else:
            # Create dummy mask
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = torch.from_numpy(mask).long()

        return {
            'image': image,
            'mask': mask,
            'id': sample['id']
        }


class HAM10000Dataset(Dataset):
    """
    HAM10000 Dataset loader.

    Contains 10,015 dermatoscopic images with 7 diagnostic categories:
    - Actinic keratoses and intraepithelial carcinoma (akiec)
    - Basal cell carcinoma (bcc)
    - Benign keratosis-like lesions (bkl)
    - Dermatofibroma (df)
    - Melanoma (mel)
    - Melanocytic nevi (nv)
    - Vascular lesions (vasc)
    """

    CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    DERMOSCOPIC_FEATURES = [
        'pigment_network',
        'globules',
        'streaks',
        'blue_white_veil',
        'vascular_patterns',
        'regression'
    ]

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        download: bool = False,
        task: str = 'classification'  # 'classification' or 'segmentation'
    ):
        """
        Args:
            root_dir: Root directory where dataset is stored
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply to images
            download: Whether to download dataset
            task: 'classification' or 'segmentation'
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.task = task

        # Paths
        self.images_dir = self.root_dir / 'HAM10000' / 'images'
        self.metadata_file = self.root_dir / 'HAM10000' / 'HAM10000_metadata.csv'

        if download and not self.images_dir.exists():
            self.download_ham10000()

        # Load metadata
        self.samples = self._load_samples()

    def download_ham10000(self):
        """
        Download and prepare HAM10000 dataset.
        """
        print("=" * 80)
        print("HAM10000 DATASET DOWNLOAD INSTRUCTIONS")
        print("=" * 80)
        print("\nThe HAM10000 dataset is available on multiple platforms:")
        print("\n1. Kaggle (Recommended):")
        print("   https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("\n2. Harvard Dataverse:")
        print("   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("\n3. ISIC Archive:")
        print("   https://challenge.isic-archive.com/data/")
        print(f"\nExtract the dataset to: {self.root_dir}/HAM10000/")
        print("\nRequired files:")
        print("  - HAM10000_images_part_1/ (5000 images)")
        print("  - HAM10000_images_part_2/ (5015 images)")
        print("  - HAM10000_metadata.csv")
        print("  - HAM10000_segmentations_lesion_tschandl/ (optional masks)")
        print("=" * 80)

        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _load_samples(self) -> List[Dict]:
        """Load image paths and metadata"""
        samples = []

        if not self.metadata_file.exists():
            print(f"Warning: Metadata file not found at {self.metadata_file}")
            return samples

        # Load metadata CSV
        df = pd.read_csv(self.metadata_file)

        # Split by lesion_id to avoid data leakage
        unique_lesions = df['lesion_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_lesions)

        n = len(unique_lesions)
        if self.split == 'train':
            lesion_ids = unique_lesions[:int(0.8 * n)]
        elif self.split == 'val':
            lesion_ids = unique_lesions[int(0.8 * n):int(0.9 * n)]
        else:  # test
            lesion_ids = unique_lesions[int(0.9 * n):]

        # Filter dataframe
        df_split = df[df['lesion_id'].isin(lesion_ids)]

        for _, row in df_split.iterrows():
            image_id = row['image_id']

            # Find image file
            img_path = None
            for subdir in ['images', 'HAM10000_images_part_1', 'HAM10000_images_part_2']:
                potential_path = self.root_dir / 'HAM10000' / subdir / f"{image_id}.jpg"
                if potential_path.exists():
                    img_path = potential_path
                    break

            if img_path is None:
                continue

            samples.append({
                'image': img_path,
                'image_id': image_id,
                'lesion_id': row['lesion_id'],
                'dx': row['dx'],
                'dx_type': row.get('dx_type', 'unknown'),
                'age': row.get('age', -1),
                'sex': row.get('sex', 'unknown'),
                'localization': row.get('localization', 'unknown')
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image']).convert('RGB')

        # Get label
        label = self.CLASSES.index(sample['dx'])

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return {
            'image': image,
            'label': label,
            'dx': sample['dx'],
            'image_id': sample['image_id'],
            'metadata': {
                'age': sample['age'],
                'sex': sample['sex'],
                'localization': sample['localization']
            }
        }


def get_train_transforms(image_size=256):
    """Training data augmentation transforms"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size=256):
    """Validation transforms (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_dataloaders(
    dataset_name: str,
    root_dir: str,
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 4,
    download: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        dataset_name: 'PH2' or 'HAM10000'
        root_dir: Root directory for datasets
        batch_size: Batch size
        image_size: Image size for resizing
        num_workers: Number of workers for data loading
        download: Whether to download dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Choose dataset class
    if dataset_name.upper() == 'PH2':
        DatasetClass = PH2Dataset
    elif dataset_name.upper() == 'HAM10000':
        DatasetClass = HAM10000Dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create datasets
    train_dataset = DatasetClass(
        root_dir=root_dir,
        split='train',
        transform=get_train_transforms(image_size),
        download=download
    )

    val_dataset = DatasetClass(
        root_dir=root_dir,
        split='val',
        transform=get_val_transforms(image_size),
        download=False
    )

    test_dataset = DatasetClass(
        root_dir=root_dir,
        split='test',
        transform=get_val_transforms(image_size),
        download=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset loaders...\n")

    # Test PH2
    print("PH2 Dataset:")
    ph2_dataset = PH2Dataset(
        root_dir='./data',
        split='train',
        download=True
    )
    print(f"  Number of samples: {len(ph2_dataset)}")
    if len(ph2_dataset) > 0:
        sample = ph2_dataset[0]
        print(f"  Sample keys: {sample.keys()}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")

    # Test HAM10000
    print("\nHAM10000 Dataset:")
    ham_dataset = HAM10000Dataset(
        root_dir='./data',
        split='train',
        download=True
    )
    print(f"  Number of samples: {len(ham_dataset)}")
    if len(ham_dataset) > 0:
        sample = ham_dataset[0]
        print(f"  Sample keys: {sample.keys()}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Label: {sample['label']} ({sample['dx']})")
        print(f"  Classes: {HAM10000Dataset.CLASSES}")
