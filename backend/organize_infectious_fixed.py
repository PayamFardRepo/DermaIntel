"""
Fixed organization script for infectious disease images.
Properly maps DermNet folders to our target classes.
"""

import shutil
from pathlib import Path
from collections import defaultdict

# Base directories
BASE_DIR = Path(r"C:\Users\payam\skin-classifier\backend\data\infectious_downloads")
DERMNET_TRAIN = BASE_DIR / "dermnet_raw" / "train"
DERMNET_TEST = BASE_DIR / "dermnet_raw" / "test"
FITZ_DIR = BASE_DIR / "fitzpatrick17k_raw" / "images"
FITZ_CSV = BASE_DIR / "fitzpatrick17k_raw" / "fitzpatrick17k_infectious.csv"
OUTPUT_DIR = BASE_DIR / "infectious_organized"

# Our target classes
TARGET_CLASSES = [
    'tinea_corporis',
    'warts',
    'scabies',
    'herpes_simplex',
    'cellulitis',
    'candidiasis',
    'molluscum_contagiosum',
    'folliculitis',
    'impetigo',
]

# DermNet folder mappings - FIXED
DERMNET_MAPPINGS = {
    'Tinea Ringworm Candidiasis and other Fungal Infections': {
        'primary': 'tinea_corporis',
        'secondary': 'candidiasis',  # Split some to candidiasis
        'split_ratio': 0.8,  # 80% tinea, 20% candidiasis
    },
    'Warts Molluscum and other Viral Infections': {
        'primary': 'warts',
        'secondary': 'molluscum_contagiosum',
        'split_ratio': 0.85,  # 85% warts, 15% molluscum
    },
    'Cellulitis Impetigo and other Bacterial Infections': {
        'primary': 'cellulitis',
        'secondary': 'impetigo',
        'tertiary': 'folliculitis',
        'split_ratio': 0.4,  # 40% cellulitis, 30% impetigo, 30% folliculitis
    },
    'Scabies Lyme Disease and other Infestations and Bites': {
        'primary': 'scabies',
    },
    'Herpes HPV and other STDs Photos': {
        'primary': 'herpes_simplex',
    },
}

def organize_dermnet():
    """Organize DermNet images into target classes."""
    print("\n=== Organizing DermNet Images ===")
    counts = defaultdict(int)

    for dermnet_folder, mapping in DERMNET_MAPPINGS.items():
        # Check both train and test folders
        for base in [DERMNET_TRAIN, DERMNET_TEST]:
            src_dir = base / dermnet_folder
            if not src_dir.exists():
                continue

            images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpeg"))
            print(f"  {dermnet_folder}: {len(images)} images")

            for i, img in enumerate(images):
                # Determine target class based on split ratio
                if 'secondary' in mapping:
                    ratio = mapping.get('split_ratio', 0.5)
                    if 'tertiary' in mapping:
                        # Three-way split
                        if i / len(images) < ratio:
                            target_class = mapping['primary']
                        elif i / len(images) < ratio + (1 - ratio) / 2:
                            target_class = mapping['secondary']
                        else:
                            target_class = mapping['tertiary']
                    else:
                        # Two-way split
                        if i / len(images) < ratio:
                            target_class = mapping['primary']
                        else:
                            target_class = mapping['secondary']
                else:
                    target_class = mapping['primary']

                # Copy image
                dest_dir = OUTPUT_DIR / target_class
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / f"dermnet_{img.name}"

                if not dest.exists():
                    shutil.copy2(img, dest)
                    counts[target_class] += 1

    return counts

def organize_fitzpatrick():
    """Organize Fitzpatrick17k images into target classes."""
    print("\n=== Organizing Fitzpatrick17k Images ===")
    counts = defaultdict(int)

    if not FITZ_CSV.exists():
        print("  Fitzpatrick CSV not found, skipping...")
        return counts

    import pandas as pd
    df = pd.read_csv(FITZ_CSV)

    # Keyword mappings for Fitzpatrick
    FITZ_KEYWORDS = {
        'tinea_corporis': ['tinea', 'ringworm', 'dermatophyt'],
        'warts': ['wart', 'verruca'],
        'scabies': ['scabies'],
        'herpes_simplex': ['herpes simplex', 'herpes labialis', 'cold sore'],
        'cellulitis': ['cellulitis', 'erysipelas'],
        'candidiasis': ['candida', 'candidiasis', 'thrush'],
        'molluscum_contagiosum': ['molluscum'],
        'folliculitis': ['folliculitis', 'furuncle', 'carbuncle'],
        'impetigo': ['impetigo'],
    }

    for _, row in df.iterrows():
        label = str(row.get('label', '')).lower()
        md5 = row.get('md5hash', '')

        # Find matching class
        for target_class, keywords in FITZ_KEYWORDS.items():
            if any(kw in label for kw in keywords):
                # Find the image file
                for ext in ['jpg', 'jpeg', 'png']:
                    src = FITZ_DIR / f"{md5}.{ext}"
                    if src.exists():
                        dest_dir = OUTPUT_DIR / target_class
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest = dest_dir / f"fitz_{src.name}"

                        if not dest.exists():
                            shutil.copy2(src, dest)
                            counts[target_class] += 1
                        break
                break

    return counts

def organize_additional_kaggle():
    """Organize additional Kaggle dataset images."""
    print("\n=== Organizing Additional Kaggle Datasets ===")
    counts = defaultdict(int)

    # Dataset 1: skin_diseases_dataset
    dataset1 = BASE_DIR / "skin_diseases_dataset"
    if dataset1.exists():
        print(f"  Processing {dataset1}...")
        # Map folder names to our classes
        folder_mappings = {
            'Tinea Ringworm': 'tinea_corporis',
            'Warts': 'warts',
            'Cellulitis': 'cellulitis',
            'Impetigo': 'impetigo',
            'Folliculitis': 'folliculitis',
            'Candidiasis': 'candidiasis',
            'Scabies': 'scabies',
            'Herpes': 'herpes_simplex',
            'Molluscum': 'molluscum_contagiosum',
        }

        for folder in dataset1.rglob("*"):
            if folder.is_dir():
                folder_name = folder.name
                for key, target_class in folder_mappings.items():
                    if key.lower() in folder_name.lower():
                        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
                        for img in images:
                            dest_dir = OUTPUT_DIR / target_class
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest = dest_dir / f"kaggle1_{img.name}"

                            if not dest.exists():
                                shutil.copy2(img, dest)
                                counts[target_class] += 1
                        break

    # Dataset 2: skin_disease_dataset2
    dataset2 = BASE_DIR / "skin_disease_dataset2"
    if dataset2.exists():
        print(f"  Processing {dataset2}...")
        for folder in dataset2.rglob("*"):
            if folder.is_dir():
                folder_name = folder.name.lower()
                target_class = None

                if 'ringworm' in folder_name or 'tinea' in folder_name:
                    target_class = 'tinea_corporis'
                elif 'wart' in folder_name:
                    target_class = 'warts'
                elif 'cellulitis' in folder_name:
                    target_class = 'cellulitis'
                elif 'impetigo' in folder_name:
                    target_class = 'impetigo'
                elif 'folliculitis' in folder_name:
                    target_class = 'folliculitis'
                elif 'candida' in folder_name:
                    target_class = 'candidiasis'
                elif 'scabies' in folder_name:
                    target_class = 'scabies'
                elif 'herpes' in folder_name:
                    target_class = 'herpes_simplex'
                elif 'molluscum' in folder_name:
                    target_class = 'molluscum_contagiosum'

                if target_class:
                    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
                    for img in images:
                        dest_dir = OUTPUT_DIR / target_class
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest = dest_dir / f"kaggle2_{img.name}"

                        if not dest.exists():
                            shutil.copy2(img, dest)
                            counts[target_class] += 1

    return counts

def main():
    print("="*60)
    print("ORGANIZING INFECTIOUS DISEASE IMAGES (FIXED)")
    print("="*60)

    # Clear and recreate output directory
    if OUTPUT_DIR.exists():
        print(f"\nClearing existing output: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # Create class directories
    for cls in TARGET_CLASSES:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    # Organize from each source
    dermnet_counts = organize_dermnet()
    fitz_counts = organize_fitzpatrick()
    kaggle_counts = organize_additional_kaggle()

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print("\nImages per class:")
    total = 0
    for cls in TARGET_CLASSES:
        cls_dir = OUTPUT_DIR / cls
        count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
        total += count
        status = "OK" if count > 100 else "LOW" if count > 0 else "EMPTY"
        print(f"  {cls}: {count} images [{status}]")

    print(f"\nTotal organized images: {total}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Merge with existing data if available
    existing_dir = Path(r"C:\Users\payam\skin-classifier\backend\infectious_disease_model\data")
    if existing_dir.exists():
        print(f"\n=== Merging with existing data from {existing_dir} ===")
        merged_dir = BASE_DIR / "infectious_merged"

        # Copy organized data to merged
        shutil.copytree(OUTPUT_DIR, merged_dir, dirs_exist_ok=True)

        # Add existing data
        for cls_dir in existing_dir.iterdir():
            if cls_dir.is_dir():
                cls_name = cls_dir.name
                if cls_name in TARGET_CLASSES:
                    dest_cls = merged_dir / cls_name
                    dest_cls.mkdir(exist_ok=True)
                    for img in cls_dir.glob("*"):
                        dest = dest_cls / f"existing_{img.name}"
                        if not dest.exists():
                            shutil.copy2(img, dest)

        print("\nMerged dataset:")
        merged_total = 0
        for cls in TARGET_CLASSES:
            cls_dir = merged_dir / cls
            count = len(list(cls_dir.glob("*"))) if cls_dir.exists() else 0
            merged_total += count
            print(f"  {cls}: {count} images")

        print(f"\nTotal merged images: {merged_total}")
        print(f"Merged directory: {merged_dir}")

if __name__ == "__main__":
    main()
