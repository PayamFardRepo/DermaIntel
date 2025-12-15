"""
Data Augmentation API Router

Provides endpoints for synthetic data augmentation to generate
training data for rare skin conditions.
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import json
import base64
from datetime import datetime
from PIL import Image
import io

from database import get_db, User
from auth import get_current_active_user, get_current_professional_user
from synthetic_data_augmentation import (
    SyntheticDataAugmentor,
    AugmentationType,
    AugmentationConfig
)

router = APIRouter()

# Initialize augmentor with default output directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
AUGMENTED_DIR = os.path.join(DATA_DIR, 'augmented')
os.makedirs(AUGMENTED_DIR, exist_ok=True)

augmentor = SyntheticDataAugmentor(output_dir=AUGMENTED_DIR)


@router.get("/augmentation/types")
async def get_augmentation_types(
    current_user: User = Depends(get_current_active_user)
):
    """Get available augmentation types and their descriptions"""
    return {
        "augmentation_types": [
            {
                "id": "geometric",
                "name": "Geometric Transformations",
                "description": "Rotation, scaling, flipping, and shear transformations",
                "techniques": ["rotation", "scale", "flip", "shear", "crop"]
            },
            {
                "id": "color",
                "name": "Color Augmentations",
                "description": "Brightness, contrast, saturation, and hue adjustments",
                "techniques": ["brightness", "contrast", "saturation", "hue_shift"]
            },
            {
                "id": "noise",
                "name": "Noise & Blur",
                "description": "Gaussian noise, blur, and sharpening effects",
                "techniques": ["gaussian_noise", "blur", "sharpen"]
            },
            {
                "id": "advanced",
                "name": "Advanced Techniques",
                "description": "Elastic deformation, grid distortion, and random erasing",
                "techniques": ["elastic_deformation", "grid_distortion", "random_erasing"]
            },
            {
                "id": "dermatology",
                "name": "Dermatology-Specific",
                "description": "Flash simulation, skin tone variation, lesion enhancement",
                "techniques": ["flash_simulation", "skin_tone_variation", "border_enhancement", "camera_quality"]
            },
            {
                "id": "mixup",
                "name": "Mixup/CutMix",
                "description": "Blend or combine regions from different images",
                "techniques": ["mixup", "cutmix"]
            }
        ]
    }


@router.get("/augmentation/rare-conditions")
async def get_rare_conditions(
    current_user: User = Depends(get_current_active_user)
):
    """Get list of rare conditions that typically need augmentation"""
    return {
        "rare_conditions": augmentor.RARE_CONDITIONS,
        "description": "These conditions typically have fewer training samples and benefit most from data augmentation"
    }


@router.post("/augmentation/augment-image")
async def augment_single_image(
    file: UploadFile = File(...),
    augmentation_types: str = Form(default="geometric,color,dermatology"),
    num_augmentations: int = Form(default=5),
    current_user: User = Depends(get_current_active_user)
):
    """
    Augment a single image and return augmented versions

    Args:
        file: Image file to augment
        augmentation_types: Comma-separated list of augmentation types
        num_augmentations: Number of augmented versions to generate (1-20)
    """
    # Validate number of augmentations
    if num_augmentations < 1 or num_augmentations > 20:
        raise HTTPException(status_code=400, detail="num_augmentations must be between 1 and 20")

    # Read image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Parse augmentation types
    aug_type_list = []
    for aug_type in augmentation_types.split(','):
        aug_type = aug_type.strip().lower()
        try:
            aug_type_list.append(AugmentationType(aug_type))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid augmentation type: {aug_type}. Valid types: geometric, color, noise, advanced, dermatology, mixup"
            )

    # Generate augmentations
    try:
        results = augmentor.augment_image(
            image,
            augmentation_types=aug_type_list,
            num_augmentations=num_augmentations
        )

        # Convert to base64 for response
        augmented_images = []
        for idx, (aug_image, params) in enumerate(results):
            buffer = io.BytesIO()
            aug_image.save(buffer, format='JPEG', quality=90)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            augmented_images.append({
                "index": idx,
                "image_base64": img_base64,
                "parameters": params
            })

        return {
            "success": True,
            "original_filename": file.filename,
            "augmentation_types": augmentation_types.split(','),
            "num_augmentations": len(augmented_images),
            "augmented_images": augmented_images
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Augmentation failed: {str(e)}")


@router.post("/augmentation/preview")
async def preview_augmentation(
    file: UploadFile = File(...),
    augmentation_type: str = Form(default="geometric"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Preview a single augmentation type on an image

    Returns one augmented version to preview the effect
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    try:
        aug_type = AugmentationType(augmentation_type.strip().lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid augmentation type: {augmentation_type}"
        )

    # Generate single augmentation
    results = augmentor.augment_image(
        image,
        augmentation_types=[aug_type],
        num_augmentations=1
    )

    aug_image, params = results[0]

    # Convert to base64
    buffer = io.BytesIO()
    aug_image.save(buffer, format='JPEG', quality=90)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # Also return original for comparison
    orig_buffer = io.BytesIO()
    image.save(orig_buffer, format='JPEG', quality=90)
    orig_base64 = base64.b64encode(orig_buffer.getvalue()).decode('utf-8')

    return {
        "augmentation_type": augmentation_type,
        "original_image": orig_base64,
        "augmented_image": img_base64,
        "parameters_applied": params
    }


@router.get("/augmentation/dataset-statistics")
async def get_dataset_statistics(
    data_dir: str = None,
    current_user: User = Depends(get_current_professional_user)
):
    """
    Get statistics about class distribution in a dataset

    Requires professional user access.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # Security check - ensure path is within allowed directories
    allowed_dirs = [DATA_DIR, AUGMENTED_DIR]
    if not any(os.path.commonpath([data_dir, allowed]) == allowed for allowed in allowed_dirs):
        raise HTTPException(status_code=403, detail="Access to this directory is not allowed")

    if not os.path.exists(data_dir):
        raise HTTPException(status_code=404, detail="Data directory not found")

    stats = augmentor.get_dataset_statistics(data_dir)

    if "error" in stats:
        raise HTTPException(status_code=404, detail=stats["error"])

    return stats


@router.get("/augmentation/recommendations")
async def get_augmentation_recommendations(
    data_dir: str = None,
    target_balance: float = 0.8,
    current_user: User = Depends(get_current_professional_user)
):
    """
    Get recommended augmentation strategy to balance a dataset

    Args:
        data_dir: Root directory of the dataset
        target_balance: Target ratio of min/max class (0.1-1.0)
    """
    if target_balance < 0.1 or target_balance > 1.0:
        raise HTTPException(status_code=400, detail="target_balance must be between 0.1 and 1.0")

    if data_dir is None:
        data_dir = DATA_DIR

    if not os.path.exists(data_dir):
        raise HTTPException(status_code=404, detail="Data directory not found")

    recommendations = augmentor.recommend_augmentation(data_dir, target_balance)

    if "error" in recommendations:
        raise HTTPException(status_code=404, detail=recommendations["error"])

    return recommendations


@router.post("/augmentation/augment-dataset")
async def augment_dataset(
    background_tasks: BackgroundTasks,
    condition: str = Form(...),
    target_count: int = Form(default=1000),
    augmentation_types: str = Form(default="geometric,color,dermatology"),
    source_dir: str = Form(default=None),
    current_user: User = Depends(get_current_professional_user)
):
    """
    Augment all images for a specific condition to reach target count

    This operation runs in the background for large datasets.

    Requires professional user access.
    """
    if target_count < 10 or target_count > 10000:
        raise HTTPException(status_code=400, detail="target_count must be between 10 and 10000")

    if source_dir is None:
        source_dir = DATA_DIR

    if not os.path.exists(source_dir):
        raise HTTPException(status_code=404, detail="Source directory not found")

    # Parse augmentation types
    aug_type_list = []
    for aug_type in augmentation_types.split(','):
        aug_type = aug_type.strip().lower()
        try:
            aug_type_list.append(AugmentationType(aug_type))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid augmentation type: {aug_type}"
            )

    # Create a job ID for tracking
    job_id = f"aug_{condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run augmentation in background
    def run_augmentation():
        result = augmentor.augment_dataset(
            source_dir=source_dir,
            condition=condition,
            target_count=target_count,
            augmentation_types=aug_type_list
        )
        # Save result to a job file
        job_file = os.path.join(AUGMENTED_DIR, f"{job_id}_result.json")
        with open(job_file, 'w') as f:
            json.dump(result, f, indent=2)

    background_tasks.add_task(run_augmentation)

    return {
        "message": "Augmentation job started",
        "job_id": job_id,
        "condition": condition,
        "target_count": target_count,
        "augmentation_types": augmentation_types.split(','),
        "status": "processing"
    }


@router.get("/augmentation/job-status/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_professional_user)
):
    """Get status of an augmentation job"""
    job_file = os.path.join(AUGMENTED_DIR, f"{job_id}_result.json")

    if os.path.exists(job_file):
        with open(job_file, 'r') as f:
            result = json.load(f)
        return {
            "job_id": job_id,
            "status": "completed",
            "result": result
        }
    else:
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Job is still running or not found"
        }


@router.post("/augmentation/mixup")
async def mixup_images(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    alpha: float = Form(default=0.4),
    current_user: User = Depends(get_current_active_user)
):
    """
    Apply mixup augmentation between two images

    Args:
        file1: First image
        file2: Second image
        alpha: Beta distribution parameter (0.1-1.0)
    """
    if alpha < 0.1 or alpha > 1.0:
        raise HTTPException(status_code=400, detail="alpha must be between 0.1 and 1.0")

    try:
        contents1 = await file1.read()
        contents2 = await file2.read()
        image1 = Image.open(io.BytesIO(contents1)).convert('RGB')
        image2 = Image.open(io.BytesIO(contents2)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Apply mixup
    mixed_image, lam = augmentor.mixup(image1, image2, alpha)

    # Convert to base64
    buffer = io.BytesIO()
    mixed_image.save(buffer, format='JPEG', quality=90)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        "mixed_image": img_base64,
        "lambda": lam,
        "alpha": alpha,
        "description": f"Image 1 contributes {lam*100:.1f}%, Image 2 contributes {(1-lam)*100:.1f}%"
    }


@router.post("/augmentation/cutmix")
async def cutmix_images(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    alpha: float = Form(default=1.0),
    current_user: User = Depends(get_current_active_user)
):
    """
    Apply CutMix augmentation between two images

    Args:
        file1: Base image
        file2: Image to cut from
        alpha: Beta distribution parameter (0.1-2.0)
    """
    if alpha < 0.1 or alpha > 2.0:
        raise HTTPException(status_code=400, detail="alpha must be between 0.1 and 2.0")

    try:
        contents1 = await file1.read()
        contents2 = await file2.read()
        image1 = Image.open(io.BytesIO(contents1)).convert('RGB')
        image2 = Image.open(io.BytesIO(contents2)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Apply cutmix
    mixed_image, lam, bbox = augmentor.cutmix(image1, image2, alpha)

    # Convert to base64
    buffer = io.BytesIO()
    mixed_image.save(buffer, format='JPEG', quality=90)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        "mixed_image": img_base64,
        "lambda": lam,
        "alpha": alpha,
        "bounding_box": {
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3]
        },
        "description": f"Image 1 area: {lam*100:.1f}%, Cut region from Image 2: {(1-lam)*100:.1f}%"
    }


@router.get("/augmentation/config")
async def get_augmentation_config(
    current_user: User = Depends(get_current_active_user)
):
    """Get current augmentation configuration parameters"""
    config = augmentor.config

    return {
        "geometric": {
            "rotation_range": config.rotation_range,
            "scale_range": config.scale_range,
            "horizontal_flip": config.horizontal_flip,
            "vertical_flip": config.vertical_flip,
            "shear_range": config.shear_range
        },
        "color": {
            "brightness_range": config.brightness_range,
            "contrast_range": config.contrast_range,
            "saturation_range": config.saturation_range,
            "hue_shift_range": config.hue_shift_range
        },
        "noise": {
            "gaussian_noise_std": config.gaussian_noise_std,
            "blur_radius_range": config.blur_radius_range
        },
        "advanced": {
            "elastic_alpha": config.elastic_alpha,
            "elastic_sigma": config.elastic_sigma,
            "grid_distortion_steps": config.grid_distortion_steps
        },
        "dermatology": {
            "simulate_flash": config.simulate_flash,
            "skin_tone_variation": config.skin_tone_variation,
            "lesion_border_enhance": config.lesion_border_enhance
        }
    }


@router.post("/augmentation/update-config")
async def update_augmentation_config(
    rotation_range_min: int = Form(default=-30),
    rotation_range_max: int = Form(default=30),
    scale_range_min: float = Form(default=0.8),
    scale_range_max: float = Form(default=1.2),
    horizontal_flip: bool = Form(default=True),
    vertical_flip: bool = Form(default=False),
    brightness_range_min: float = Form(default=0.7),
    brightness_range_max: float = Form(default=1.3),
    contrast_range_min: float = Form(default=0.7),
    contrast_range_max: float = Form(default=1.3),
    current_user: User = Depends(get_current_professional_user)
):
    """Update augmentation configuration parameters"""
    global augmentor

    new_config = AugmentationConfig(
        rotation_range=(rotation_range_min, rotation_range_max),
        scale_range=(scale_range_min, scale_range_max),
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        brightness_range=(brightness_range_min, brightness_range_max),
        contrast_range=(contrast_range_min, contrast_range_max)
    )

    augmentor = SyntheticDataAugmentor(output_dir=AUGMENTED_DIR, config=new_config)

    return {
        "message": "Configuration updated successfully",
        "new_config": {
            "rotation_range": new_config.rotation_range,
            "scale_range": new_config.scale_range,
            "horizontal_flip": new_config.horizontal_flip,
            "vertical_flip": new_config.vertical_flip,
            "brightness_range": new_config.brightness_range,
            "contrast_range": new_config.contrast_range
        }
    }


@router.get("/augmentation/log")
async def get_augmentation_log(
    limit: int = 100,
    current_user: User = Depends(get_current_professional_user)
):
    """Get recent augmentation log entries"""
    log_entries = augmentor.augmentation_log[-limit:]

    return {
        "total_entries": len(augmentor.augmentation_log),
        "returned_entries": len(log_entries),
        "log": [
            {
                "original_path": entry.original_path,
                "augmented_path": entry.augmented_path,
                "condition": entry.condition,
                "augmentation_types": entry.augmentation_types,
                "created_at": entry.created_at
            }
            for entry in log_entries
        ]
    }
