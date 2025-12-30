"""
Organize ISIC 2019 images into benign/malignant folders.

Creates:
  data/isic_2019/organized/
    benign/      (~15,991 images)
    malignant/   (~9,340 images including AK)
"""

import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
ISIC_2019_DIR = Path("data/isic_2019")
IMAGES_DIR = ISIC_2019_DIR / "ISIC_2019_Training_Input"
OUTPUT_DIR = ISIC_2019_DIR / "organized"

def main():
    print("=" * 60)
    print("ORGANIZING ISIC 2019 INTO BENIGN/MALIGNANT")
    print("=" * 60)

    # Create output directories
    benign_dir = OUTPUT_DIR / "benign"
    malignant_dir = OUTPUT_DIR / "malignant"
    benign_dir.mkdir(parents=True, exist_ok=True)
    malignant_dir.mkdir(parents=True, exist_ok=True)

    # Read metadata
    df = pd.read_csv(ISIC_2019_DIR / "ISIC_2019_metadata.csv")
    print(f"\nLoaded {len(df)} records from metadata")

    benign_count = 0
    malignant_count = 0
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing"):
        image_name = row['image']
        src_path = IMAGES_DIR / f"{image_name}.jpg"

        if not src_path.exists():
            missing += 1
            continue

        # Malignant: MEL, BCC, SCC, AK (pre-malignant but high-risk)
        is_malignant = (row['MEL'] == 1.0 or row['BCC'] == 1.0 or
                        row['SCC'] == 1.0 or row['AK'] == 1.0)

        if is_malignant:
            dst = malignant_dir / f"{image_name}.jpg"
            if not dst.exists():
                shutil.copy2(src_path, dst)
            malignant_count += 1
        else:
            dst = benign_dir / f"{image_name}.jpg"
            if not dst.exists():
                shutil.copy2(src_path, dst)
            benign_count += 1

    print(f"\nDone!")
    print(f"  Benign:    {benign_count:,} -> {benign_dir}")
    print(f"  Malignant: {malignant_count:,} -> {malignant_dir}")
    if missing > 0:
        print(f"  Missing:   {missing}")

    print(f"\nYou can now upload these folders to Google Drive")
    print(f"and add them to the ISIC 2020 benign/malignant folders.")

if __name__ == "__main__":
    main()
