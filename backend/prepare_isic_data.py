"""
ISIC Data Preparation and Organization Script

Prepares the ISIC Archive data for training by:
1. Organizing images into class-based directories
2. Creating train/val/test splits
3. Generating unified metadata
4. Computing dataset statistics
5. Handling class imbalance analysis

Usage:
    python prepare_isic_data.py --input_dir ./data/isic --output_dir ./data/isic_prepared
    python prepare_isic_data.py --input_dir ./data/isic --analyze_only
    python prepare_isic_data.py --combine_datasets ./data/isic ./data/ham10000 --output_dir ./data/combined
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ISICDataPreparer:
    """
    Prepare and organize ISIC Archive data for training.
    """

    # Unified class mapping (8 classes)
    CLASS_MAPPING = {
        # Melanoma
        'melanoma': 'MEL',
        'mel': 'MEL',
        'MEL': 'MEL',

        # Melanocytic nevus
        'nevus': 'NV',
        'nv': 'NV',
        'NV': 'NV',
        'melanocytic nevus': 'NV',

        # Basal cell carcinoma
        'basal cell carcinoma': 'BCC',
        'bcc': 'BCC',
        'BCC': 'BCC',

        # Actinic keratosis
        'actinic keratosis': 'AK',
        'akiec': 'AK',
        'ak': 'AK',
        'AK': 'AK',
        'AKIEC': 'AK',

        # Benign keratosis
        'benign keratosis': 'BKL',
        'bkl': 'BKL',
        'BKL': 'BKL',
        'seborrheic keratosis': 'BKL',
        'solar lentigo': 'BKL',
        'lichenoid keratosis': 'BKL',

        # Dermatofibroma
        'dermatofibroma': 'DF',
        'df': 'DF',
        'DF': 'DF',

        # Vascular lesion
        'vascular lesion': 'VASC',
        'vasc': 'VASC',
        'VASC': 'VASC',
        'angioma': 'VASC',
        'angiokeratoma': 'VASC',
        'pyogenic granuloma': 'VASC',

        # Squamous cell carcinoma
        'squamous cell carcinoma': 'SCC',
        'scc': 'SCC',
        'SCC': 'SCC',
    }

    # Malignancy categories
    MALIGNANCY = {
        'MEL': 'malignant',
        'BCC': 'malignant',
        'SCC': 'malignant',
        'AK': 'pre-malignant',
        'NV': 'benign',
        'BKL': 'benign',
        'DF': 'benign',
        'VASC': 'benign',
    }

    # Standard class order
    CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
        copy_images: bool = True
    ):
        """
        Initialize the data preparer.

        Args:
            input_dir: Input directory containing ISIC images
            output_dir: Output directory for organized data
            image_extensions: Valid image file extensions
            copy_images: If True, copy images; if False, create symlinks (Unix only)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_extensions = [ext.lower() for ext in image_extensions]
        self.copy_images = copy_images

        # Metadata storage
        self.all_metadata = []
        self.class_counts = Counter()
        self.duplicate_hashes = set()

    def find_metadata_files(self) -> List[Path]:
        """Find all metadata CSV files in the input directory."""
        metadata_files = []

        patterns = [
            '**/*metadata*.csv',
            '**/*ground*truth*.csv',
            '**/*labels*.csv',
            '**/ISIC_*.csv',
        ]

        for pattern in patterns:
            metadata_files.extend(self.input_dir.glob(pattern))

        # Remove duplicates
        metadata_files = list(set(metadata_files))
        logger.info(f"Found {len(metadata_files)} metadata files")

        return metadata_files

    def find_image_directories(self) -> List[Path]:
        """Find all directories containing images."""
        image_dirs = set()

        for ext in self.image_extensions:
            for img_path in self.input_dir.rglob(f'*{ext}'):
                image_dirs.add(img_path.parent)

        logger.info(f"Found {len(image_dirs)} directories with images")
        return list(image_dirs)

    def load_metadata(self) -> pd.DataFrame:
        """Load and combine all metadata files."""
        metadata_files = self.find_metadata_files()

        if not metadata_files:
            logger.warning("No metadata files found")
            return pd.DataFrame()

        all_dfs = []

        for meta_file in metadata_files:
            try:
                df = pd.read_csv(meta_file)
                df['source_file'] = meta_file.name
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} records from {meta_file.name}")
            except Exception as e:
                logger.error(f"Error loading {meta_file}: {e}")

        if not all_dfs:
            return pd.DataFrame()

        # Combine all dataframes
        combined = pd.concat(all_dfs, ignore_index=True)

        # Standardize column names
        column_mapping = {
            'image_name': 'image_id',
            'image': 'image_id',
            'ISIC_id': 'image_id',
            'isic_id': 'image_id',
            'dx': 'diagnosis',
            'label': 'diagnosis',
            'age_approx': 'age',
            'anatom_site_general': 'anatomic_site',
            'anatom_site_general_challenge': 'anatomic_site',
            'localization': 'anatomic_site',
        }

        combined = combined.rename(columns={k: v for k, v in column_mapping.items() if k in combined.columns})

        # Handle ISIC 2019 format: one-hot encoded columns (MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK)
        one_hot_classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        if all(cls in combined.columns for cls in one_hot_classes):
            def get_diagnosis_from_onehot(row):
                for cls in one_hot_classes:
                    if row.get(cls, 0) == 1.0:
                        return cls
                return 'UNK'

            combined['diagnosis'] = combined.apply(get_diagnosis_from_onehot, axis=1)
            logger.info("Applied ISIC 2019 one-hot encoded diagnosis mapping")

        # Handle ISIC 2020 format: use 'target' and 'benign_malignant' columns
        # target=1 means melanoma (malignant), target=0 means benign
        elif 'target' in combined.columns or 'benign_malignant' in combined.columns:
            def get_diagnosis_from_target(row):
                # First check target column (most reliable for ISIC 2020)
                target = row.get('target', None)
                if target is not None and not pd.isna(target):
                    if int(target) == 1:
                        return 'MEL'  # Melanoma (malignant)
                    else:
                        return 'NV'  # Benign -> classify as nevus

                # Fallback to benign_malignant column
                bm = row.get('benign_malignant', '')
                if not pd.isna(bm):
                    if str(bm).lower() == 'malignant':
                        return 'MEL'
                    elif str(bm).lower() == 'benign':
                        return 'NV'

                # Last resort: check diagnosis column
                diagnosis = row.get('diagnosis', '')
                if not pd.isna(diagnosis) and str(diagnosis).lower() not in ['unknown', 'unk', '']:
                    return diagnosis

                return 'NV'  # Default to benign

            combined['diagnosis'] = combined.apply(get_diagnosis_from_target, axis=1)
            logger.info("Applied ISIC 2020 target-based diagnosis mapping")

        # Remove duplicates by image_id
        if 'image_id' in combined.columns:
            combined = combined.drop_duplicates(subset=['image_id'])

        logger.info(f"Combined metadata: {len(combined)} unique records")
        return combined

    def normalize_diagnosis(self, diagnosis) -> str:
        """Normalize diagnosis to standard class name."""
        # Handle various input types
        if diagnosis is None:
            return 'UNK'

        # Handle pandas Series or numpy arrays
        if hasattr(diagnosis, 'item'):
            diagnosis = diagnosis.item()
        elif hasattr(diagnosis, 'iloc'):
            diagnosis = diagnosis.iloc[0] if len(diagnosis) > 0 else 'UNK'

        # Check for NaN
        try:
            if pd.isna(diagnosis):
                return 'UNK'
        except (ValueError, TypeError):
            pass

        diagnosis_str = str(diagnosis).lower().strip()
        return self.CLASS_MAPPING.get(diagnosis_str, 'UNK')

    def compute_image_hash(self, image_path: Path) -> str:
        """Compute MD5 hash of an image for duplicate detection."""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def find_all_images(self) -> Dict[str, Path]:
        """Find all images and create a mapping from ID to path."""
        image_map = {}

        for ext in self.image_extensions:
            for img_path in self.input_dir.rglob(f'*{ext}'):
                # Use filename (without extension) as ID
                image_id = img_path.stem
                image_map[image_id] = img_path

            # Also check uppercase extensions
            for img_path in self.input_dir.rglob(f'*{ext.upper()}'):
                image_id = img_path.stem
                image_map[image_id] = img_path

        logger.info(f"Found {len(image_map)} unique images")
        return image_map

    def create_directory_structure(self):
        """Create output directory structure."""
        # Main directories
        for split in ['train', 'val', 'test']:
            for cls in self.CLASSES + ['UNK']:
                (self.output_dir / split / cls).mkdir(parents=True, exist_ok=True)

        # Metadata directory
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)

        logger.info(f"Created directory structure at {self.output_dir}")

    def prepare_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        deduplicate: bool = True,
        balance_classes: bool = False,
        max_per_class: Optional[int] = None,
        seed: int = 42
    ):
        """
        Prepare and organize the dataset.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            deduplicate: Remove duplicate images
            balance_classes: Undersample majority classes
            max_per_class: Maximum samples per class (None for no limit)
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        # Create output directories
        self.create_directory_structure()

        # Load metadata
        metadata_df = self.load_metadata()

        # Find all images
        image_map = self.find_all_images()

        if metadata_df.empty:
            logger.warning("No metadata available. Organizing by directory structure.")
            self._organize_by_directory(image_map)
            return

        # Process each image
        processed_records = []
        seen_hashes = set()

        logger.info("Processing images...")
        for image_id, image_path in tqdm(image_map.items(), desc="Processing"):
            # Check for duplicates
            if deduplicate:
                try:
                    img_hash = self.compute_image_hash(image_path)
                    if img_hash in seen_hashes:
                        continue
                    seen_hashes.add(img_hash)
                except Exception:
                    pass

            # Get metadata for this image
            record = {'image_id': image_id, 'image_path': str(image_path)}

            if 'image_id' in metadata_df.columns:
                meta_row = metadata_df[metadata_df['image_id'] == image_id]
                if not meta_row.empty:
                    meta_row = meta_row.iloc[0]
                    # Safely extract values from pandas Series
                    def safe_get(row, key, default=None):
                        try:
                            if key not in row.index:
                                return default
                            val = row[key]
                            # Handle NaN
                            if pd.isna(val):
                                return default
                            # Convert numpy types to Python native types
                            if hasattr(val, 'item') and np.ndim(val) == 0:
                                return val.item()
                            return val
                        except Exception:
                            return default

                    record.update({
                        'diagnosis': safe_get(meta_row, 'diagnosis', 'UNK'),
                        'age': safe_get(meta_row, 'age'),
                        'sex': safe_get(meta_row, 'sex'),
                        'anatomic_site': safe_get(meta_row, 'anatomic_site'),
                    })
                else:
                    record['diagnosis'] = 'UNK'
            else:
                record['diagnosis'] = 'UNK'

            # Normalize diagnosis
            record['normalized_class'] = self.normalize_diagnosis(record.get('diagnosis', 'UNK'))
            record['malignancy'] = self.MALIGNANCY.get(record['normalized_class'], 'unknown')

            processed_records.append(record)

        logger.info(f"Processed {len(processed_records)} images")

        # Create DataFrame
        df = pd.DataFrame(processed_records)

        # Balance classes if requested
        if balance_classes or max_per_class:
            df = self._balance_classes(df, max_per_class)

        # Split data stratified by class
        df = self._stratified_split(df, train_ratio, val_ratio, test_ratio, seed)

        # Copy/link images to output directories
        self._organize_images(df)

        # Save metadata
        self._save_metadata(df)

        # Print statistics
        self._print_statistics(df)

    def _organize_by_directory(self, image_map: Dict[str, Path]):
        """Organize images when no metadata is available."""
        logger.info("Organizing by directory structure...")

        for image_id, image_path in tqdm(image_map.items(), desc="Organizing"):
            # Try to infer class from directory name
            parent_dir = image_path.parent.name.lower()
            normalized_class = self.CLASS_MAPPING.get(parent_dir, 'UNK')

            # Default to train split
            output_path = self.output_dir / 'train' / normalized_class / image_path.name

            if self.copy_images:
                shutil.copy2(image_path, output_path)
            else:
                output_path.symlink_to(image_path.absolute())

    def _balance_classes(
        self,
        df: pd.DataFrame,
        max_per_class: Optional[int] = None
    ) -> pd.DataFrame:
        """Balance classes by undersampling."""
        class_counts = df['normalized_class'].value_counts()

        if max_per_class is None:
            # Use median count as target
            max_per_class = int(class_counts.median())

        logger.info(f"Balancing classes to max {max_per_class} per class")

        balanced_dfs = []
        for cls in df['normalized_class'].unique():
            cls_df = df[df['normalized_class'] == cls]
            if len(cls_df) > max_per_class:
                cls_df = cls_df.sample(n=max_per_class, random_state=42)
            balanced_dfs.append(cls_df)

        return pd.concat(balanced_dfs, ignore_index=True)

    def _stratified_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int
    ) -> pd.DataFrame:
        """Create stratified train/val/test splits."""
        np.random.seed(seed)

        df['split'] = 'train'

        for cls in df['normalized_class'].unique():
            cls_indices = df[df['normalized_class'] == cls].index.tolist()
            np.random.shuffle(cls_indices)

            n = len(cls_indices)
            train_end = int(train_ratio * n)
            val_end = train_end + int(val_ratio * n)

            train_idx = cls_indices[:train_end]
            val_idx = cls_indices[train_end:val_end]
            test_idx = cls_indices[val_end:]

            df.loc[val_idx, 'split'] = 'val'
            df.loc[test_idx, 'split'] = 'test'

        return df

    def _organize_images(self, df: pd.DataFrame):
        """Copy or link images to organized directories."""
        logger.info("Organizing images into directories...")

        def process_row(row):
            src_path = Path(row['image_path'])
            if not src_path.exists():
                return False

            dst_dir = self.output_dir / row['split'] / row['normalized_class']
            dst_path = dst_dir / src_path.name

            if dst_path.exists():
                return True

            try:
                if self.copy_images:
                    shutil.copy2(src_path, dst_path)
                else:
                    dst_path.symlink_to(src_path.absolute())
                return True
            except Exception as e:
                logger.error(f"Error processing {src_path}: {e}")
                return False

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying images"):
                pass

    def _save_metadata(self, df: pd.DataFrame):
        """Save metadata files."""
        # Full metadata
        df.to_csv(self.output_dir / 'metadata' / 'full_metadata.csv', index=False)

        # Per-split metadata
        for split in ['train', 'val', 'test']:
            split_df = df[df['split'] == split]
            split_df.to_csv(self.output_dir / 'metadata' / f'{split}_metadata.csv', index=False)

        # Class mapping
        class_info = {
            'classes': self.CLASSES,
            'class_to_idx': {cls: i for i, cls in enumerate(self.CLASSES)},
            'malignancy_mapping': self.MALIGNANCY,
        }
        with open(self.output_dir / 'metadata' / 'class_info.json', 'w') as f:
            json.dump(class_info, f, indent=2)

        logger.info(f"Metadata saved to {self.output_dir / 'metadata'}")

    def _print_statistics(self, df: pd.DataFrame):
        """Print dataset statistics."""
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)

        print(f"\nTotal images: {len(df)}")

        # Per-split statistics
        print("\nSplit distribution:")
        for split in ['train', 'val', 'test']:
            count = len(df[df['split'] == split])
            pct = 100 * count / len(df)
            print(f"  {split:5}: {count:6} ({pct:5.1f}%)")

        # Per-class statistics
        print("\nClass distribution:")
        class_counts = df['normalized_class'].value_counts()
        for cls in self.CLASSES + ['UNK']:
            if cls in class_counts:
                count = class_counts[cls]
                pct = 100 * count / len(df)
                malignancy = self.MALIGNANCY.get(cls, 'unknown')
                print(f"  {cls:4} ({malignancy:12}): {count:6} ({pct:5.1f}%)")

        # Malignancy distribution
        print("\nMalignancy distribution:")
        mal_counts = df['malignancy'].value_counts()
        for mal in ['malignant', 'pre-malignant', 'benign', 'unknown']:
            if mal in mal_counts:
                count = mal_counts[mal]
                pct = 100 * count / len(df)
                print(f"  {mal:12}: {count:6} ({pct:5.1f}%)")

        # Per-split per-class distribution
        print("\nDetailed split distribution:")
        for split in ['train', 'val', 'test']:
            split_df = df[df['split'] == split]
            print(f"\n  {split.upper()}:")
            split_counts = split_df['normalized_class'].value_counts()
            for cls in self.CLASSES:
                if cls in split_counts:
                    count = split_counts[cls]
                    print(f"    {cls}: {count}")

        print("\n" + "=" * 80)

    def analyze_dataset(self):
        """Analyze dataset without modifying files."""
        logger.info("Analyzing dataset...")

        metadata_df = self.load_metadata()
        image_map = self.find_all_images()

        if metadata_df.empty:
            print("\nNo metadata found. Image statistics only:")
            print(f"Total images: {len(image_map)}")

            # Analyze by directory
            dir_counts = Counter()
            for img_path in image_map.values():
                dir_counts[img_path.parent.name] += 1

            print("\nImages per directory:")
            for dir_name, count in dir_counts.most_common():
                print(f"  {dir_name}: {count}")
            return

        # Create analysis DataFrame
        records = []
        for image_id, image_path in image_map.items():
            record = {'image_id': image_id}

            if 'image_id' in metadata_df.columns:
                meta_row = metadata_df[metadata_df['image_id'] == image_id]
                if not meta_row.empty:
                    meta_row = meta_row.iloc[0]
                    record.update({
                        'diagnosis': meta_row.get('diagnosis', 'UNK'),
                        'age': meta_row.get('age'),
                        'sex': meta_row.get('sex'),
                    })

            record['normalized_class'] = self.normalize_diagnosis(record.get('diagnosis', 'UNK'))
            records.append(record)

        df = pd.DataFrame(records)
        self._print_statistics(df)

        # Plot class distribution
        self._plot_class_distribution(df)

    def _plot_class_distribution(self, df: pd.DataFrame):
        """Plot class distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Class distribution
        class_counts = df['normalized_class'].value_counts()
        colors = ['red' if self.MALIGNANCY.get(c) == 'malignant'
                  else 'orange' if self.MALIGNANCY.get(c) == 'pre-malignant'
                  else 'green' for c in class_counts.index]

        axes[0].bar(class_counts.index, class_counts.values, color=colors)
        axes[0].set_title('Class Distribution')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)

        # Malignancy distribution
        mal_counts = df['normalized_class'].map(lambda x: self.MALIGNANCY.get(x, 'unknown')).value_counts()
        mal_colors = {'malignant': 'red', 'pre-malignant': 'orange', 'benign': 'green', 'unknown': 'gray'}
        axes[1].pie(mal_counts.values, labels=mal_counts.index, autopct='%1.1f%%',
                    colors=[mal_colors.get(m, 'gray') for m in mal_counts.index])
        axes[1].set_title('Malignancy Distribution')

        plt.tight_layout()
        output_path = self.output_dir / 'class_distribution.png' if self.output_dir.exists() else Path('class_distribution.png')
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Class distribution plot saved to {output_path}")


def combine_datasets(
    dataset_dirs: List[str],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """
    Combine multiple skin lesion datasets into a unified format.

    Args:
        dataset_dirs: List of dataset directories to combine
        output_dir: Output directory for combined dataset
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
    """
    logger.info(f"Combining {len(dataset_dirs)} datasets...")

    # Create output preparer
    preparer = ISICDataPreparer(
        input_dir=dataset_dirs[0],  # Use first as base
        output_dir=output_dir
    )
    preparer.create_directory_structure()

    all_records = []

    for dataset_dir in dataset_dirs:
        logger.info(f"Processing {dataset_dir}...")

        temp_preparer = ISICDataPreparer(
            input_dir=dataset_dir,
            output_dir=output_dir
        )

        metadata_df = temp_preparer.load_metadata()
        image_map = temp_preparer.find_all_images()

        for image_id, image_path in image_map.items():
            record = {
                'image_id': image_id,
                'image_path': str(image_path),
                'source': Path(dataset_dir).name
            }

            if not metadata_df.empty and 'image_id' in metadata_df.columns:
                meta_row = metadata_df[metadata_df['image_id'] == image_id]
                if not meta_row.empty:
                    meta_row = meta_row.iloc[0]
                    record.update({
                        'diagnosis': meta_row.get('diagnosis', 'UNK'),
                        'age': meta_row.get('age'),
                        'sex': meta_row.get('sex'),
                    })

            record['normalized_class'] = preparer.normalize_diagnosis(record.get('diagnosis', 'UNK'))
            record['malignancy'] = preparer.MALIGNANCY.get(record['normalized_class'], 'unknown')
            all_records.append(record)

    # Create combined DataFrame and process
    df = pd.DataFrame(all_records)
    df = preparer._stratified_split(df, train_ratio, val_ratio, 1 - train_ratio - val_ratio, 42)
    preparer._organize_images(df)
    preparer._save_metadata(df)
    preparer._print_statistics(df)

    logger.info(f"Combined dataset saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ISIC Archive data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input_dir', type=str, default='./data/isic',
                        help='Input directory containing ISIC images')
    parser.add_argument('--output_dir', type=str, default='./data/isic_prepared',
                        help='Output directory for organized data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--deduplicate', action='store_true', default=True,
                        help='Remove duplicate images')
    parser.add_argument('--balance', action='store_true',
                        help='Balance classes by undersampling')
    parser.add_argument('--max_per_class', type=int, default=None,
                        help='Maximum samples per class')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Only analyze dataset without organizing')
    parser.add_argument('--combine_datasets', nargs='+',
                        help='Combine multiple datasets')
    parser.add_argument('--symlink', action='store_true',
                        help='Use symlinks instead of copying (Unix only)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    if args.combine_datasets:
        combine_datasets(
            dataset_dirs=args.combine_datasets,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        return

    preparer = ISICDataPreparer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copy_images=not args.symlink
    )

    if args.analyze_only:
        preparer.analyze_dataset()
    else:
        preparer.prepare_data(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1 - args.train_ratio - args.val_ratio,
            deduplicate=args.deduplicate,
            balance_classes=args.balance,
            max_per_class=args.max_per_class,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
