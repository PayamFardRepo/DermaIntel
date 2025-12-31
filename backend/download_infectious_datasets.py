"""
Download and Organize Infectious Disease Datasets
==================================================
Downloads images from multiple sources to improve infectious disease model accuracy.

Sources:
1. DermNet (Kaggle) - 21,844 images, includes infectious conditions
2. Fitzpatrick17k (GitHub) - 16,577 images, 114 skin conditions
3. SCIN (Google) - 10,000+ real-world images

Usage:
    python download_infectious_datasets.py --output ./data/infectious_combined

Requirements:
    pip install kaggle requests pandas tqdm gdown google-cloud-storage

    For Kaggle: Set up ~/.kaggle/kaggle.json with your API credentials
    https://www.kaggle.com/docs/api
"""

import os
import sys
import json
import shutil
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import subprocess

# Infectious disease class mappings
INFECTIOUS_CLASSES = {
    # Target classes for our model
    'tinea_corporis': ['tinea', 'ringworm', 'dermatophytosis', 'tinea corporis', 'tinea cruris', 'tinea pedis'],
    'warts': ['wart', 'verruca', 'hpv', 'plantar wart', 'common wart', 'genital wart'],
    'scabies': ['scabies', 'sarcoptes'],
    'herpes_simplex': ['herpes simplex', 'hsv', 'cold sore', 'herpes labialis', 'genital herpes'],
    'cellulitis': ['cellulitis', 'erysipelas'],
    'candidiasis': ['candida', 'candidiasis', 'thrush', 'yeast infection', 'intertrigo candida'],
    'molluscum_contagiosum': ['molluscum', 'molluscum contagiosum'],
    'folliculitis': ['folliculitis', 'furuncle', 'carbuncle', 'boil'],
    'impetigo': ['impetigo', 'ecthyma'],
}

# DermNet class mappings to our classes
DERMNET_MAPPINGS = {
    'Tinea Ringworm Candidiasis and other Fungal Infections': ['tinea_corporis', 'candidiasis'],
    'Cellulitis Impetigo and other Bacterial Infections': ['cellulitis', 'impetigo'],
    'Warts Molluscum and other Viral Infections': ['warts', 'molluscum_contagiosum'],
    'Scabies Lyme Disease and other Infestations and Bites': ['scabies'],
    'Herpes HPV and other STDs': ['herpes_simplex', 'warts'],
}


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))


def download_dermnet(output_dir: Path):
    """
    Download DermNet dataset from Kaggle.
    Requires kaggle API credentials in ~/.kaggle/kaggle.json
    """
    print("\n" + "="*60)
    print("DOWNLOADING DERMNET DATASET")
    print("="*60)

    dermnet_dir = output_dir / "dermnet_raw"
    dermnet_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Check if kaggle is installed
        import kaggle

        print("Downloading from Kaggle...")
        kaggle.api.dataset_download_files(
            'shubhamgoel27/dermnet',
            path=str(dermnet_dir),
            unzip=True
        )
        print(f"[OK] DermNet downloaded to {dermnet_dir}")
        return dermnet_dir

    except ImportError:
        print("[WARN] Kaggle package not installed. Install with: pip install kaggle")
        print("       Then set up API credentials: https://www.kaggle.com/docs/api")
    except Exception as e:
        print(f"[ERROR] Failed to download DermNet: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/shubhamgoel27/dermnet")
        print("2. Click 'Download' button")
        print(f"3. Extract to: {dermnet_dir}")

    return None


def download_fitzpatrick17k(output_dir: Path):
    """
    Download Fitzpatrick17k dataset from GitHub.
    """
    print("\n" + "="*60)
    print("DOWNLOADING FITZPATRICK17K DATASET")
    print("="*60)

    fitz_dir = output_dir / "fitzpatrick17k_raw"
    fitz_dir.mkdir(parents=True, exist_ok=True)

    # Download the CSV with image URLs and labels
    csv_url = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
    csv_path = fitz_dir / "fitzpatrick17k.csv"

    try:
        print("Downloading Fitzpatrick17k metadata...")
        response = requests.get(csv_url)
        response.raise_for_status()

        with open(csv_path, 'wb') as f:
            f.write(response.content)

        print(f"[OK] Metadata downloaded: {csv_path}")

        # Read CSV and filter for infectious conditions
        df = pd.read_csv(csv_path)
        print(f"Total images in Fitzpatrick17k: {len(df)}")

        # Find infectious disease related conditions
        infectious_keywords = []
        for keywords in INFECTIOUS_CLASSES.values():
            infectious_keywords.extend(keywords)

        # Filter for infectious conditions
        mask = df['label'].str.lower().apply(
            lambda x: any(kw in x.lower() for kw in infectious_keywords)
        )
        infectious_df = df[mask]
        print(f"Infectious disease images found: {len(infectious_df)}")

        # Save filtered CSV
        infectious_csv = fitz_dir / "fitzpatrick17k_infectious.csv"
        infectious_df.to_csv(infectious_csv, index=False)

        # Download images
        images_dir = fitz_dir / "images"
        images_dir.mkdir(exist_ok=True)

        print(f"\nDownloading {len(infectious_df)} infectious disease images...")

        def download_image(row):
            try:
                url = row['url']
                md5 = row['md5hash']
                ext = url.split('.')[-1].split('?')[0][:4]
                filename = f"{md5}.{ext}"
                filepath = images_dir / filename

                if filepath.exists():
                    return True

                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    return True
            except Exception as e:
                pass
            return False

        successful = 0
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(download_image, row) for _, row in infectious_df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                if future.result():
                    successful += 1

        print(f"[OK] Downloaded {successful}/{len(infectious_df)} images")
        return fitz_dir

    except Exception as e:
        print(f"[ERROR] Failed to download Fitzpatrick17k: {e}")

    return None


def download_scin(output_dir: Path):
    """
    Download SCIN dataset from Google Cloud Storage.
    """
    print("\n" + "="*60)
    print("DOWNLOADING SCIN DATASET (Google/Stanford)")
    print("="*60)

    scin_dir = output_dir / "scin_raw"
    scin_dir.mkdir(parents=True, exist_ok=True)

    # SCIN is hosted on Google Cloud Storage
    # We can access it without authentication for public data

    try:
        # Try using gsutil if available
        print("Attempting to download SCIN dataset...")
        print("Note: SCIN requires Google Cloud SDK or manual download")

        # Download the metadata CSV first
        metadata_url = "https://raw.githubusercontent.com/google-research-datasets/scin/main/scin_metadata.csv"

        try:
            response = requests.get(metadata_url, timeout=30)
            if response.status_code == 200:
                csv_path = scin_dir / "scin_metadata.csv"
                with open(csv_path, 'wb') as f:
                    f.write(response.content)
                print(f"[OK] SCIN metadata downloaded: {csv_path}")
        except:
            pass

        print("\nManual download instructions for SCIN:")
        print("1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        print("2. Run: gsutil -m cp -r gs://dx-scin-public-data/* " + str(scin_dir))
        print("\nOr use the demo notebook:")
        print("https://github.com/google-research-datasets/scin/blob/main/scin_demo.ipynb")

        return scin_dir

    except Exception as e:
        print(f"[ERROR] Failed to download SCIN: {e}")

    return None


def download_kaggle_skin_diseases(output_dir: Path):
    """
    Download additional skin disease datasets from Kaggle.
    """
    print("\n" + "="*60)
    print("DOWNLOADING ADDITIONAL KAGGLE DATASETS")
    print("="*60)

    datasets = [
        ('ismailpromus/skin-diseases-image-dataset', 'skin_diseases_dataset'),
        ('subirbiswas19/skin-disease-dataset', 'skin_disease_dataset2'),
    ]

    downloaded = []

    try:
        import kaggle

        for dataset_name, folder_name in datasets:
            dataset_dir = output_dir / folder_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            try:
                print(f"\nDownloading {dataset_name}...")
                kaggle.api.dataset_download_files(
                    dataset_name,
                    path=str(dataset_dir),
                    unzip=True
                )
                print(f"[OK] Downloaded to {dataset_dir}")
                downloaded.append(dataset_dir)
            except Exception as e:
                print(f"[WARN] Failed to download {dataset_name}: {e}")

    except ImportError:
        print("[WARN] Kaggle package not installed")
        print("Manual download links:")
        print("1. https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset")
        print("2. https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset")

    return downloaded


def organize_infectious_images(output_dir: Path, source_dirs: list):
    """
    Organize all downloaded images into a unified structure for training.
    """
    print("\n" + "="*60)
    print("ORGANIZING INFECTIOUS DISEASE IMAGES")
    print("="*60)

    organized_dir = output_dir / "infectious_organized"

    # Create class directories
    for class_name in INFECTIOUS_CLASSES.keys():
        (organized_dir / class_name).mkdir(parents=True, exist_ok=True)

    total_copied = {cls: 0 for cls in INFECTIOUS_CLASSES.keys()}

    # Process DermNet
    dermnet_dir = output_dir / "dermnet_raw"
    if dermnet_dir.exists():
        print("\nProcessing DermNet images...")
        for dermnet_class, our_classes in DERMNET_MAPPINGS.items():
            src_dir = dermnet_dir / "train" / dermnet_class
            if not src_dir.exists():
                src_dir = dermnet_dir / "test" / dermnet_class

            if src_dir.exists():
                images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
                for img in images:
                    # Copy to first matching class
                    target_class = our_classes[0]
                    dest = organized_dir / target_class / f"dermnet_{img.name}"
                    if not dest.exists():
                        shutil.copy2(img, dest)
                        total_copied[target_class] += 1

    # Process Fitzpatrick17k
    fitz_dir = output_dir / "fitzpatrick17k_raw"
    fitz_csv = fitz_dir / "fitzpatrick17k_infectious.csv"
    if fitz_csv.exists():
        print("\nProcessing Fitzpatrick17k images...")
        df = pd.read_csv(fitz_csv)
        images_dir = fitz_dir / "images"

        for _, row in df.iterrows():
            label = row['label'].lower()
            md5 = row['md5hash']

            # Find matching class
            for our_class, keywords in INFECTIOUS_CLASSES.items():
                if any(kw in label for kw in keywords):
                    # Find the image file
                    for ext in ['jpg', 'jpeg', 'png']:
                        src = images_dir / f"{md5}.{ext}"
                        if src.exists():
                            dest = organized_dir / our_class / f"fitz_{src.name}"
                            if not dest.exists():
                                shutil.copy2(src, dest)
                                total_copied[our_class] += 1
                            break
                    break

    # Print summary
    print("\n" + "="*60)
    print("ORGANIZATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {organized_dir}")
    print("\nImages per class:")

    grand_total = 0
    for class_name, count in total_copied.items():
        existing = len(list((organized_dir / class_name).glob("*")))
        print(f"  {class_name}: {existing} images")
        grand_total += existing

    print(f"\nTotal organized images: {grand_total}")

    return organized_dir


def merge_with_existing(organized_dir: Path, existing_dir: Path):
    """
    Merge newly downloaded images with existing infectious disease data.
    """
    print("\n" + "="*60)
    print("MERGING WITH EXISTING DATA")
    print("="*60)

    if not existing_dir.exists():
        print(f"[WARN] Existing directory not found: {existing_dir}")
        return

    merged_dir = organized_dir.parent / "infectious_merged"

    # Copy existing data first
    if existing_dir.exists():
        print(f"Copying existing data from {existing_dir}...")
        shutil.copytree(existing_dir, merged_dir, dirs_exist_ok=True)

    # Add new data
    print(f"Adding new data from {organized_dir}...")
    for class_dir in organized_dir.iterdir():
        if class_dir.is_dir():
            dest_class_dir = merged_dir / class_dir.name
            dest_class_dir.mkdir(parents=True, exist_ok=True)

            for img in class_dir.glob("*"):
                dest = dest_class_dir / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)

    # Print final counts
    print("\nFinal merged dataset:")
    total = 0
    for class_dir in sorted(merged_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            print(f"  {class_dir.name}: {count} images")
            total += count

    print(f"\nTotal images: {total}")
    return merged_dir


def main():
    parser = argparse.ArgumentParser(description="Download infectious disease datasets")
    parser.add_argument("--output", type=str, default="./data/infectious_downloads",
                        help="Output directory for downloaded data")
    parser.add_argument("--existing", type=str, default="./infectious_disease_model/data",
                        help="Path to existing infectious disease data")
    parser.add_argument("--skip-dermnet", action="store_true", help="Skip DermNet download")
    parser.add_argument("--skip-fitzpatrick", action="store_true", help="Skip Fitzpatrick17k download")
    parser.add_argument("--skip-scin", action="store_true", help="Skip SCIN download")
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip additional Kaggle datasets")
    parser.add_argument("--organize-only", action="store_true", help="Only organize existing downloads")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("INFECTIOUS DISEASE DATASET DOWNLOADER")
    print("="*60)
    print(f"Output directory: {output_dir}")

    source_dirs = []

    if not args.organize_only:
        # Download datasets
        if not args.skip_dermnet:
            dermnet_dir = download_dermnet(output_dir)
            if dermnet_dir:
                source_dirs.append(dermnet_dir)

        if not args.skip_fitzpatrick:
            fitz_dir = download_fitzpatrick17k(output_dir)
            if fitz_dir:
                source_dirs.append(fitz_dir)

        if not args.skip_scin:
            scin_dir = download_scin(output_dir)
            if scin_dir:
                source_dirs.append(scin_dir)

        if not args.skip_kaggle:
            kaggle_dirs = download_kaggle_skin_diseases(output_dir)
            source_dirs.extend(kaggle_dirs)

    # Organize images
    organized_dir = organize_infectious_images(output_dir, source_dirs)

    # Merge with existing data
    existing_dir = Path(args.existing)
    if existing_dir.exists():
        merged_dir = merge_with_existing(organized_dir, existing_dir)
        print(f"\n[OK] Merged dataset ready at: {merged_dir}")
    else:
        print(f"\n[OK] Organized dataset ready at: {organized_dir}")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review the organized images for quality")
    print("2. Remove any mislabeled or low-quality images")
    print("3. Run the improved training script:")
    print("   python train_infectious_improved.py --data ./data/infectious_merged")
    print("="*60)


if __name__ == "__main__":
    main()
