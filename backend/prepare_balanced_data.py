"""
Prepare ISIC 2020 data for balanced training.

This script:
1. Reads the metadata CSV
2. Organizes images into benign/malignant folders
3. Creates symlinks (or copies) to avoid duplicating 24GB of data

Usage:
    python prepare_balanced_data.py
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
DATA_DIR = Path("data/isic")
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = DATA_DIR / "organized"
METADATA_FILE = DATA_DIR / "ISIC_2020_metadata.csv"

def main():
    print("=" * 60)
    print("PREPARING ISIC 2020 DATA FOR BALANCED TRAINING")
    print("=" * 60)

    # Read metadata
    df = pd.read_csv(METADATA_FILE)
    print(f"\nLoaded {len(df)} records from metadata")
    print(f"\nClass distribution:")
    print(df['benign_malignant'].value_counts())
    print(f"\nImbalance ratio: {len(df[df.benign_malignant=='benign']) / len(df[df.benign_malignant=='malignant']):.1f}:1")

    # Create output directories
    benign_dir = OUTPUT_DIR / "benign"
    malignant_dir = OUTPUT_DIR / "malignant"
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOrganizing images into {OUTPUT_DIR}")

    # Count existing files
    existing_benign = len(list(benign_dir.glob("*.jpg")))
    existing_malignant = len(list(malignant_dir.glob("*.jpg")))

    if existing_benign > 0 or existing_malignant > 0:
        print(f"Found existing organized data:")
        print(f"  Benign: {existing_benign}")
        print(f"  Malignant: {existing_malignant}")

        response = input("\nReorganize? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping reorganization.")
            return

    # Process each image
    benign_count = 0
    malignant_count = 0
    missing_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
        image_name = row['image_name']
        label = row['benign_malignant']

        # Find source image
        src_path = TRAIN_DIR / f"{image_name}.jpg"
        if not src_path.exists():
            # Try without extension in name
            src_path = TRAIN_DIR / image_name
            if not src_path.exists():
                missing_count += 1
                continue

        # Determine destination
        if label == 'malignant':
            dst_path = malignant_dir / f"{image_name}.jpg"
            malignant_count += 1
        else:
            dst_path = benign_dir / f"{image_name}.jpg"
            benign_count += 1

        # Create symlink (saves disk space) or copy
        if not dst_path.exists():
            try:
                # Try symlink first (Windows requires admin or developer mode)
                os.symlink(src_path.absolute(), dst_path)
            except OSError:
                # Fall back to copy
                shutil.copy2(src_path, dst_path)

    print(f"\n✓ Organization complete!")
    print(f"  Benign: {benign_count} images -> {benign_dir}")
    print(f"  Malignant: {malignant_count} images -> {malignant_dir}")
    if missing_count > 0:
        print(f"  Missing: {missing_count} images not found")

    print(f"\n" + "=" * 60)
    print("READY FOR TRAINING")
    print("=" * 60)
    print(f"\nRun this command to train:")
    print(f"  python train_balanced_classifier.py --data_dir {OUTPUT_DIR} --epochs 20 --use_focal_loss")
    print()

if __name__ == "__main__":
    main()
