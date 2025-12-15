"""
Synthetic Data Augmentation Module for Rare Skin Conditions

This module provides comprehensive data augmentation techniques to generate
synthetic training data for rare skin conditions, helping to balance datasets
and improve model performance on underrepresented classes.

Techniques included:
1. Geometric transformations (rotation, flip, scale, crop, affine)
2. Color/intensity augmentations (brightness, contrast, saturation, hue)
3. Noise and blur augmentations (Gaussian, motion blur, noise injection)
4. Advanced techniques (elastic deformation, grid distortion)
5. Dermatology-specific augmentations (lighting simulation, skin tone variation)
6. Mixup and CutMix for hybrid sample generation
"""

import os
import json
import random
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import io
import base64
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AugmentationType(str, Enum):
    """Types of augmentation techniques available"""
    GEOMETRIC = "geometric"
    COLOR = "color"
    NOISE = "noise"
    ADVANCED = "advanced"
    DERMATOLOGY = "dermatology"
    MIXUP = "mixup"


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters"""
    # Geometric
    rotation_range: Tuple[int, int] = (-30, 30)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    horizontal_flip: bool = True
    vertical_flip: bool = False
    shear_range: Tuple[float, float] = (-0.1, 0.1)

    # Color
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_shift_range: Tuple[int, int] = (-20, 20)

    # Noise
    gaussian_noise_std: float = 0.05
    blur_radius_range: Tuple[float, float] = (0.5, 2.0)

    # Advanced
    elastic_alpha: float = 50.0
    elastic_sigma: float = 5.0
    grid_distortion_steps: int = 5

    # Dermatology-specific
    simulate_flash: bool = True
    skin_tone_variation: bool = True
    lesion_border_enhance: bool = True


@dataclass
class AugmentationResult:
    """Result of an augmentation operation"""
    original_path: str
    augmented_path: str
    condition: str
    augmentation_types: List[str]
    parameters: Dict[str, Any]
    created_at: str


class SyntheticDataAugmentor:
    """
    Synthetic Data Augmentation for Rare Skin Conditions

    Generates augmented training data using various techniques to help
    balance datasets and improve model performance on rare conditions.
    """

    # Rare conditions that typically need augmentation
    RARE_CONDITIONS = [
        "melanoma",
        "merkel_cell_carcinoma",
        "dermatofibrosarcoma",
        "kaposi_sarcoma",
        "cutaneous_lymphoma",
        "paget_disease",
        "angiosarcoma",
        "atypical_fibroxanthoma",
        "microcystic_adnexal_carcinoma",
        "sebaceous_carcinoma",
        "eccrine_porocarcinoma",
        "lentigo_maligna",
        "acral_melanoma",
        "amelanotic_melanoma",
        "desmoplastic_melanoma"
    ]

    def __init__(self,
                 output_dir: str = "augmented_data",
                 config: Optional[AugmentationConfig] = None):
        """
        Initialize the augmentor

        Args:
            output_dir: Directory to save augmented images
            config: Augmentation configuration parameters
        """
        self.output_dir = output_dir
        self.config = config or AugmentationConfig()
        self.augmentation_log: List[AugmentationResult] = []

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"SyntheticDataAugmentor initialized with output dir: {output_dir}")

    def augment_image(self,
                      image: Image.Image,
                      augmentation_types: List[AugmentationType] = None,
                      num_augmentations: int = 1) -> List[Tuple[Image.Image, Dict]]:
        """
        Apply augmentations to a single image

        Args:
            image: PIL Image to augment
            augmentation_types: List of augmentation types to apply
            num_augmentations: Number of augmented versions to generate

        Returns:
            List of (augmented_image, parameters_dict) tuples
        """
        if augmentation_types is None:
            augmentation_types = [
                AugmentationType.GEOMETRIC,
                AugmentationType.COLOR,
                AugmentationType.NOISE
            ]

        results = []

        for _ in range(num_augmentations):
            aug_image = image.copy()
            applied_params = {}

            for aug_type in augmentation_types:
                # Skip mixup - it requires two images and is handled separately
                if aug_type == AugmentationType.MIXUP:
                    continue

                if aug_type == AugmentationType.GEOMETRIC:
                    aug_image, params = self._apply_geometric(aug_image)
                elif aug_type == AugmentationType.COLOR:
                    aug_image, params = self._apply_color(aug_image)
                elif aug_type == AugmentationType.NOISE:
                    aug_image, params = self._apply_noise(aug_image)
                elif aug_type == AugmentationType.ADVANCED:
                    aug_image, params = self._apply_advanced(aug_image)
                elif aug_type == AugmentationType.DERMATOLOGY:
                    aug_image, params = self._apply_dermatology_specific(aug_image)
                else:
                    continue  # Skip unknown types

                applied_params[aug_type.value] = params

            results.append((aug_image, applied_params))

        return results

    def _apply_geometric(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply geometric transformations"""
        params = {}

        # Random rotation
        angle = random.uniform(*self.config.rotation_range)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
        params['rotation'] = angle

        # Random scale
        scale = random.uniform(*self.config.scale_range)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.BICUBIC)
        # Crop or pad to original size
        image = self._resize_to_original(image, (image.width, image.height))
        params['scale'] = scale

        # Horizontal flip
        if self.config.horizontal_flip and random.random() > 0.5:
            image = ImageOps.mirror(image)
            params['horizontal_flip'] = True
        else:
            params['horizontal_flip'] = False

        # Vertical flip
        if self.config.vertical_flip and random.random() > 0.5:
            image = ImageOps.flip(image)
            params['vertical_flip'] = True
        else:
            params['vertical_flip'] = False

        # Shear transformation
        shear = random.uniform(*self.config.shear_range)
        if abs(shear) > 0.01:
            image = self._apply_shear(image, shear)
            params['shear'] = shear

        return image, params

    def _apply_color(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply color/intensity augmentations"""
        params = {}

        # Brightness adjustment
        brightness = random.uniform(*self.config.brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        params['brightness'] = brightness

        # Contrast adjustment
        contrast = random.uniform(*self.config.contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        params['contrast'] = contrast

        # Saturation adjustment
        saturation = random.uniform(*self.config.saturation_range)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
        params['saturation'] = saturation

        # Hue shift (convert to HSV, shift, convert back)
        hue_shift = random.randint(*self.config.hue_shift_range)
        if abs(hue_shift) > 0:
            image = self._shift_hue(image, hue_shift)
            params['hue_shift'] = hue_shift

        return image, params

    def _apply_noise(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply noise and blur augmentations"""
        params = {}

        # Gaussian noise
        if random.random() > 0.5:
            image = self._add_gaussian_noise(image, self.config.gaussian_noise_std)
            params['gaussian_noise'] = True

        # Random blur
        if random.random() > 0.7:
            blur_radius = random.uniform(*self.config.blur_radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            params['blur_radius'] = blur_radius

        # Sharpen occasionally
        if random.random() > 0.8:
            image = image.filter(ImageFilter.SHARPEN)
            params['sharpen'] = True

        return image, params

    def _apply_advanced(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply advanced augmentation techniques"""
        params = {}

        # Elastic deformation
        if random.random() > 0.5:
            image = self._elastic_transform(image)
            params['elastic_deformation'] = True

        # Grid distortion
        if random.random() > 0.7:
            image = self._grid_distortion(image)
            params['grid_distortion'] = True

        # Random erasing (cutout)
        if random.random() > 0.8:
            image = self._random_erasing(image)
            params['random_erasing'] = True

        return image, params

    def _apply_dermatology_specific(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply dermatology-specific augmentations"""
        params = {}

        # Simulate different lighting conditions (flash, ambient)
        if self.config.simulate_flash and random.random() > 0.5:
            image = self._simulate_flash(image)
            params['flash_simulation'] = True

        # Skin tone variation
        if self.config.skin_tone_variation and random.random() > 0.6:
            image = self._vary_skin_tone(image)
            params['skin_tone_variation'] = True

        # Lesion border enhancement
        if self.config.lesion_border_enhance and random.random() > 0.7:
            image = self._enhance_lesion_border(image)
            params['border_enhancement'] = True

        # Simulate different camera qualities
        if random.random() > 0.6:
            image = self._simulate_camera_quality(image)
            params['camera_quality_variation'] = True

        return image, params

    def _apply_shear(self, image: Image.Image, shear: float) -> Image.Image:
        """Apply shear transformation"""
        width, height = image.size
        xshift = abs(shear) * width
        new_width = width + int(round(xshift))

        coeffs = (1, shear, -xshift if shear > 0 else 0,
                  0, 1, 0)

        image = image.transform((new_width, height), Image.AFFINE, coeffs, Image.BICUBIC)
        image = image.crop((0, 0, width, height))
        return image

    def _resize_to_original(self, image: Image.Image, original_size: Tuple[int, int]) -> Image.Image:
        """Resize/crop image to match original size"""
        return image.resize(original_size, Image.BICUBIC)

    def _shift_hue(self, image: Image.Image, shift: int) -> Image.Image:
        """Shift hue of image"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy for HSV manipulation
        img_array = np.array(image)

        # Simple hue shift using RGB manipulation
        # This is a simplified version - for production, use proper HSV conversion
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

        # Rotate color channels slightly based on shift
        shift_factor = shift / 360.0
        if shift > 0:
            new_r = np.clip(r * (1 - shift_factor) + g * shift_factor, 0, 255)
            new_g = np.clip(g * (1 - shift_factor) + b * shift_factor, 0, 255)
            new_b = np.clip(b * (1 - shift_factor) + r * shift_factor, 0, 255)
        else:
            shift_factor = abs(shift_factor)
            new_r = np.clip(r * (1 - shift_factor) + b * shift_factor, 0, 255)
            new_g = np.clip(g * (1 - shift_factor) + r * shift_factor, 0, 255)
            new_b = np.clip(b * (1 - shift_factor) + g * shift_factor, 0, 255)

        img_array = np.stack([new_r, new_g, new_b], axis=2).astype(np.uint8)
        return Image.fromarray(img_array)

    def _add_gaussian_noise(self, image: Image.Image, std: float) -> Image.Image:
        """Add Gaussian noise to image"""
        img_array = np.array(image).astype(float)
        noise = np.random.normal(0, std * 255, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    def _elastic_transform(self, image: Image.Image) -> Image.Image:
        """Apply elastic deformation"""
        img_array = np.array(image)
        shape = img_array.shape[:2]

        # Generate random displacement fields
        dx = np.random.randn(*shape) * self.config.elastic_alpha
        dy = np.random.randn(*shape) * self.config.elastic_alpha

        # Smooth the displacement fields
        from scipy.ndimage import gaussian_filter
        dx = gaussian_filter(dx, self.config.elastic_sigma)
        dy = gaussian_filter(dy, self.config.elastic_sigma)

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        # Apply displacement
        indices_x = np.clip(x + dx, 0, shape[1] - 1).astype(int)
        indices_y = np.clip(y + dy, 0, shape[0] - 1).astype(int)

        # Apply to each channel
        if len(img_array.shape) == 3:
            result = np.zeros_like(img_array)
            for c in range(img_array.shape[2]):
                result[:, :, c] = img_array[indices_y, indices_x, c]
        else:
            result = img_array[indices_y, indices_x]

        return Image.fromarray(result.astype(np.uint8))

    def _grid_distortion(self, image: Image.Image) -> Image.Image:
        """Apply grid-based distortion"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Create distortion grid
        steps = self.config.grid_distortion_steps
        grid_x = np.linspace(0, width, steps)
        grid_y = np.linspace(0, height, steps)

        # Add random perturbations to grid points
        perturb_x = np.random.uniform(-width * 0.05, width * 0.05, (steps, steps))
        perturb_y = np.random.uniform(-height * 0.05, height * 0.05, (steps, steps))

        # Simplified grid distortion using PIL transform
        # For a more accurate implementation, use cv2.remap
        quad = (
            random.randint(-10, 10), random.randint(-10, 10),
            width + random.randint(-10, 10), random.randint(-10, 10),
            width + random.randint(-10, 10), height + random.randint(-10, 10),
            random.randint(-10, 10), height + random.randint(-10, 10)
        )

        coeffs = self._find_perspective_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [(quad[0], quad[1]), (quad[2], quad[3]), (quad[4], quad[5]), (quad[6], quad[7])]
        )

        return image.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    def _find_perspective_coeffs(self, source_coords, target_coords):
        """Calculate perspective transform coefficients"""
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])

        A = np.matrix(matrix, dtype=float)
        B = np.array([s for pair in source_coords for s in pair]).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    def _random_erasing(self, image: Image.Image,
                        area_ratio: Tuple[float, float] = (0.02, 0.1)) -> Image.Image:
        """Apply random erasing (cutout) augmentation"""
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Calculate erasing area
        area = height * width
        target_area = random.uniform(*area_ratio) * area

        aspect_ratio = random.uniform(0.5, 2.0)
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))

        if h < height and w < width:
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)

            # Fill with mean color of the image
            mean_color = img_array.mean(axis=(0, 1)).astype(np.uint8)
            img_array[y:y+h, x:x+w] = mean_color

        return Image.fromarray(img_array)

    def _simulate_flash(self, image: Image.Image) -> Image.Image:
        """Simulate flash photography lighting"""
        img_array = np.array(image).astype(float)
        height, width = img_array.shape[:2]

        # Create a radial gradient for flash effect
        center_x, center_y = width // 2, height // 2
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)

        # Flash intensity decreases with distance from center
        flash_intensity = 1 - (dist_from_center / max_dist) * 0.3
        flash_intensity = flash_intensity[:, :, np.newaxis]

        # Apply flash effect
        flashed = img_array * flash_intensity * random.uniform(1.1, 1.3)
        flashed = np.clip(flashed, 0, 255).astype(np.uint8)

        return Image.fromarray(flashed)

    def _vary_skin_tone(self, image: Image.Image) -> Image.Image:
        """Vary skin tone to simulate different skin types"""
        img_array = np.array(image).astype(float)

        # Adjust color balance to simulate different skin tones
        # This shifts towards warmer (more red/yellow) or cooler (more blue) tones
        tone_shift = random.uniform(-0.15, 0.15)

        if tone_shift > 0:
            # Warmer skin tone
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * (1 + tone_shift), 0, 255)  # Red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * (1 + tone_shift * 0.5), 0, 255)  # Green
        else:
            # Cooler skin tone
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * (1 + abs(tone_shift)), 0, 255)  # Blue

        return Image.fromarray(img_array.astype(np.uint8))

    def _enhance_lesion_border(self, image: Image.Image) -> Image.Image:
        """Enhance lesion borders for better visibility"""
        # Apply edge enhancement
        enhanced = image.filter(ImageFilter.EDGE_ENHANCE)

        # Blend with original
        return Image.blend(image, enhanced, alpha=0.3)

    def _simulate_camera_quality(self, image: Image.Image) -> Image.Image:
        """Simulate different camera qualities"""
        quality_factor = random.choice(['high', 'medium', 'low'])

        if quality_factor == 'low':
            # Reduce resolution and add compression artifacts
            small_size = (image.width // 2, image.height // 2)
            image = image.resize(small_size, Image.BILINEAR)
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

            # Add slight blur
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

        elif quality_factor == 'medium':
            # Slight quality reduction
            small_size = (int(image.width * 0.75), int(image.height * 0.75))
            image = image.resize(small_size, Image.BILINEAR)
            image = image.resize((int(image.width / 0.75), int(image.height / 0.75)), Image.BILINEAR)

        return image

    def mixup(self,
              image1: Image.Image,
              image2: Image.Image,
              alpha: float = 0.4) -> Tuple[Image.Image, float]:
        """
        Apply mixup augmentation between two images

        Args:
            image1: First image
            image2: Second image
            alpha: Beta distribution parameter

        Returns:
            Mixed image and the lambda value used
        """
        # Ensure same size
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.BICUBIC)

        # Sample lambda from beta distribution
        lam = np.random.beta(alpha, alpha)

        # Mix images
        img1_array = np.array(image1).astype(float)
        img2_array = np.array(image2).astype(float)

        mixed = lam * img1_array + (1 - lam) * img2_array
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)

        return Image.fromarray(mixed), lam

    def cutmix(self,
               image1: Image.Image,
               image2: Image.Image,
               alpha: float = 1.0) -> Tuple[Image.Image, float, Tuple[int, int, int, int]]:
        """
        Apply CutMix augmentation between two images

        Args:
            image1: First image (base)
            image2: Second image (to cut from)
            alpha: Beta distribution parameter

        Returns:
            Mixed image, lambda value, and bounding box coordinates
        """
        # Ensure same size
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.BICUBIC)

        width, height = image1.size

        # Sample lambda from beta distribution
        lam = np.random.beta(alpha, alpha)

        # Calculate cut size
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        # Random position for the cut
        cx = random.randint(0, width)
        cy = random.randint(0, height)

        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(width, cx + cut_w // 2)
        bby2 = min(height, cy + cut_h // 2)

        # Create mixed image
        img1_array = np.array(image1)
        img2_array = np.array(image2)

        img1_array[bby1:bby2, bbx1:bbx2] = img2_array[bby1:bby2, bbx1:bbx2]

        # Recalculate lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))

        return Image.fromarray(img1_array), lam, (bbx1, bby1, bbx2, bby2)

    def augment_dataset(self,
                        source_dir: str,
                        condition: str,
                        target_count: int = 1000,
                        augmentation_types: List[AugmentationType] = None) -> Dict[str, Any]:
        """
        Augment all images for a specific condition to reach target count

        Args:
            source_dir: Directory containing source images
            condition: Condition name (subdirectory name)
            target_count: Target number of images after augmentation
            augmentation_types: Types of augmentation to apply

        Returns:
            Statistics about the augmentation process
        """
        condition_dir = os.path.join(source_dir, condition)
        output_condition_dir = os.path.join(self.output_dir, condition)
        os.makedirs(output_condition_dir, exist_ok=True)

        if not os.path.exists(condition_dir):
            logger.warning(f"Source directory not found: {condition_dir}")
            return {"error": f"Source directory not found: {condition_dir}"}

        # Get existing images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        source_images = [
            f for f in os.listdir(condition_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

        current_count = len(source_images)

        if current_count == 0:
            return {"error": f"No images found in {condition_dir}"}

        # Calculate augmentations needed per image
        augmentations_needed = target_count - current_count

        if augmentations_needed <= 0:
            return {
                "condition": condition,
                "original_count": current_count,
                "target_count": target_count,
                "augmented_count": 0,
                "message": "Target count already reached"
            }

        augs_per_image = max(1, augmentations_needed // current_count)
        extra_augs = augmentations_needed % current_count

        logger.info(f"Augmenting {condition}: {current_count} -> {target_count} images")

        augmented_count = 0

        for idx, image_file in enumerate(source_images):
            image_path = os.path.join(condition_dir, image_file)

            try:
                image = Image.open(image_path).convert('RGB')

                # Determine number of augmentations for this image
                num_augs = augs_per_image + (1 if idx < extra_augs else 0)

                # Generate augmented versions
                augmented_images = self.augment_image(
                    image,
                    augmentation_types=augmentation_types,
                    num_augmentations=num_augs
                )

                # Save augmented images
                for aug_idx, (aug_image, params) in enumerate(augmented_images):
                    base_name = os.path.splitext(image_file)[0]
                    aug_filename = f"{base_name}_aug_{augmented_count:04d}.jpg"
                    aug_path = os.path.join(output_condition_dir, aug_filename)

                    aug_image.save(aug_path, 'JPEG', quality=95)
                    augmented_count += 1

                    # Log the augmentation
                    self.augmentation_log.append(AugmentationResult(
                        original_path=image_path,
                        augmented_path=aug_path,
                        condition=condition,
                        augmentation_types=[t.value for t in (augmentation_types or [])],
                        parameters=params,
                        created_at=datetime.now().isoformat()
                    ))

            except Exception as e:
                logger.error(f"Error augmenting {image_path}: {e}")
                continue

        # Also copy original images to output directory
        for image_file in source_images:
            src_path = os.path.join(condition_dir, image_file)
            dst_path = os.path.join(output_condition_dir, image_file)
            if not os.path.exists(dst_path):
                Image.open(src_path).save(dst_path)

        return {
            "condition": condition,
            "original_count": current_count,
            "target_count": target_count,
            "augmented_count": augmented_count,
            "total_count": current_count + augmented_count,
            "output_directory": output_condition_dir
        }

    def get_dataset_statistics(self, data_dir: str) -> Dict[str, Any]:
        """
        Get statistics about class distribution in a dataset

        Args:
            data_dir: Root directory of the dataset

        Returns:
            Dictionary with class counts and imbalance metrics
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        class_counts = {}

        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                count = len([
                    f for f in os.listdir(class_dir)
                    if os.path.splitext(f)[1].lower() in image_extensions
                ])
                class_counts[class_name] = count

        if not class_counts:
            return {"error": "No classes found in directory"}

        total = sum(class_counts.values())
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())

        # Calculate imbalance ratio
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        # Identify rare classes (less than 10% of max class)
        rare_threshold = max_count * 0.1
        rare_classes = [
            cls for cls, count in class_counts.items()
            if count < rare_threshold
        ]

        return {
            "total_images": total,
            "num_classes": len(class_counts),
            "class_counts": class_counts,
            "max_class": max(class_counts, key=class_counts.get),
            "max_count": max_count,
            "min_class": min(class_counts, key=class_counts.get),
            "min_count": min_count,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "rare_classes": rare_classes,
            "rare_threshold": rare_threshold
        }

    def recommend_augmentation(self, data_dir: str, target_balance: float = 0.8) -> Dict[str, Any]:
        """
        Recommend augmentation strategy to balance dataset

        Args:
            data_dir: Root directory of the dataset
            target_balance: Target ratio of min/max class (0-1)

        Returns:
            Recommended augmentation counts per class
        """
        stats = self.get_dataset_statistics(data_dir)

        if "error" in stats:
            return stats

        max_count = stats["max_count"]
        target_count = int(max_count * target_balance)

        recommendations = {}

        for class_name, count in stats["class_counts"].items():
            if count < target_count:
                needed = target_count - count
                recommendations[class_name] = {
                    "current_count": count,
                    "target_count": target_count,
                    "augmentations_needed": needed,
                    "augmentation_factor": round(needed / count, 2) if count > 0 else float('inf'),
                    "priority": "high" if class_name in stats["rare_classes"] else "medium"
                }
            else:
                recommendations[class_name] = {
                    "current_count": count,
                    "target_count": target_count,
                    "augmentations_needed": 0,
                    "augmentation_factor": 0,
                    "priority": "none"
                }

        return {
            "target_count": target_count,
            "target_balance": target_balance,
            "recommendations": recommendations,
            "total_augmentations_needed": sum(
                r["augmentations_needed"] for r in recommendations.values()
            )
        }

    def augment_single_image_base64(self,
                                     image_base64: str,
                                     augmentation_types: List[str] = None,
                                     num_augmentations: int = 5) -> List[Dict[str, Any]]:
        """
        Augment a single image provided as base64

        Args:
            image_base64: Base64 encoded image
            augmentation_types: List of augmentation type names
            num_augmentations: Number of augmented versions to generate

        Returns:
            List of augmented images as base64 with parameters
        """
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Convert string types to enum
        aug_types = None
        if augmentation_types:
            aug_types = [AugmentationType(t) for t in augmentation_types]

        # Generate augmentations
        results = self.augment_image(image, aug_types, num_augmentations)

        # Convert back to base64
        output = []
        for aug_image, params in results:
            buffer = io.BytesIO()
            aug_image.save(buffer, format='JPEG', quality=95)
            aug_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            output.append({
                "image_base64": aug_base64,
                "parameters": params
            })

        return output

    def save_augmentation_log(self, filepath: str = None):
        """Save augmentation log to JSON file"""
        if filepath is None:
            filepath = os.path.join(self.output_dir, "augmentation_log.json")

        with open(filepath, 'w') as f:
            json.dump([asdict(r) for r in self.augmentation_log], f, indent=2)

        logger.info(f"Augmentation log saved to {filepath}")


# Convenience function for quick augmentation
def quick_augment(image_path: str,
                  num_augmentations: int = 5,
                  output_dir: str = None) -> List[str]:
    """
    Quick function to augment a single image

    Args:
        image_path: Path to the image file
        num_augmentations: Number of augmented versions
        output_dir: Output directory (default: same as input)

    Returns:
        List of paths to augmented images
    """
    if output_dir is None:
        output_dir = os.path.dirname(image_path)

    augmentor = SyntheticDataAugmentor(output_dir=output_dir)
    image = Image.open(image_path).convert('RGB')

    results = augmentor.augment_image(
        image,
        augmentation_types=[
            AugmentationType.GEOMETRIC,
            AugmentationType.COLOR,
            AugmentationType.DERMATOLOGY
        ],
        num_augmentations=num_augmentations
    )

    output_paths = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for idx, (aug_image, params) in enumerate(results):
        output_path = os.path.join(output_dir, f"{base_name}_aug_{idx:03d}.jpg")
        aug_image.save(output_path, 'JPEG', quality=95)
        output_paths.append(output_path)

    return output_paths


if __name__ == "__main__":
    # Example usage
    print("Synthetic Data Augmentation Module")
    print("=" * 50)

    # Create augmentor
    augmentor = SyntheticDataAugmentor(output_dir="augmented_data")

    # Show available augmentation types
    print("\nAvailable Augmentation Types:")
    for aug_type in AugmentationType:
        print(f"  - {aug_type.value}")

    # Show rare conditions
    print(f"\nRare conditions that benefit from augmentation:")
    for condition in augmentor.RARE_CONDITIONS[:5]:
        print(f"  - {condition}")
    print(f"  ... and {len(augmentor.RARE_CONDITIONS) - 5} more")

    print("\nModule loaded successfully!")
