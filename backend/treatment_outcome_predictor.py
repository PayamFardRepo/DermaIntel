"""
Treatment Outcome Predictor
Generates AI-enhanced "after" treatment images using image processing techniques
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from typing import Tuple, Dict

# Validate imports on load
print("[TreatmentPredictor] Module loading...")
print(f"[TreatmentPredictor] OpenCV version: {cv2.__version__}")
print(f"[TreatmentPredictor] NumPy version: {np.__version__}")


class TreatmentOutcomePredictor:
    """
    Simulates treatment outcomes using image processing techniques
    """

    def __init__(self):
        self.treatment_configs = {
            'topical-steroid': {
                'redness_reduction': 0.65,
                'smoothing': 0.7,
                'brightness_increase': 1.08,
                'saturation_decrease': 0.75
            },
            'laser-therapy': {
                'redness_reduction': 0.80,
                'smoothing': 0.85,
                'brightness_increase': 1.12,
                'saturation_decrease': 0.65
            },
            'cryotherapy': {
                'redness_reduction': 0.85,
                'smoothing': 0.90,
                'brightness_increase': 1.15,
                'saturation_decrease': 0.60
            },
            'prescription-cream': {
                'redness_reduction': 0.70,
                'smoothing': 0.75,
                'brightness_increase': 1.10,
                'saturation_decrease': 0.70
            }
        }

        # Condition-specific modifiers
        self.condition_profiles = {
            # Inflammatory conditions - respond well to treatment
            'eczema': {
                'response_rate': 1.2,  # 20% better response
                'redness_multiplier': 1.3,
                'smoothing_multiplier': 1.2,
                'notes': 'Inflammatory condition that responds well to topical treatments'
            },
            'atopic dermatitis': {
                'response_rate': 1.2,
                'redness_multiplier': 1.3,
                'smoothing_multiplier': 1.2,
                'notes': 'Similar to eczema, responds well to steroids'
            },
            'psoriasis': {
                'response_rate': 0.9,  # Slower response
                'redness_multiplier': 1.1,
                'smoothing_multiplier': 0.8,  # Scaling persists
                'notes': 'Chronic condition, slower response, may have residual scaling'
            },
            'acne': {
                'response_rate': 0.85,
                'redness_multiplier': 1.0,
                'smoothing_multiplier': 1.4,  # Texture improvement important
                'notes': 'May worsen initially, texture improvement over time'
            },
            'rosacea': {
                'response_rate': 0.95,
                'redness_multiplier': 1.4,  # Redness is main feature
                'smoothing_multiplier': 0.9,
                'notes': 'Chronic condition focused on redness reduction'
            },

            # Infectious conditions - variable response
            'cellulitis': {
                'response_rate': 1.3,  # Rapid response to antibiotics
                'redness_multiplier': 1.5,
                'smoothing_multiplier': 1.1,
                'notes': 'Bacterial infection, responds rapidly to antibiotics'
            },
            'impetigo': {
                'response_rate': 1.4,  # Fast response
                'redness_multiplier': 1.4,
                'smoothing_multiplier': 1.3,
                'notes': 'Bacterial infection, quick healing with treatment'
            },
            'fungal infection': {
                'response_rate': 0.8,  # Slower
                'redness_multiplier': 1.1,
                'smoothing_multiplier': 1.0,
                'notes': 'Requires longer treatment duration'
            },

            # Pigmented lesions - minimal/no treatment response
            'melanoma': {
                'response_rate': 0.0,  # No improvement with topical treatments
                'redness_multiplier': 0.0,
                'smoothing_multiplier': 0.0,
                'notes': 'Requires surgical intervention, not responsive to topical treatments'
            },
            'nevus': {
                'response_rate': 0.1,  # Minimal change
                'redness_multiplier': 0.2,
                'smoothing_multiplier': 0.1,
                'notes': 'Benign mole, minimal response to topical treatments'
            },
            'seborrheic keratosis': {
                'response_rate': 0.2,
                'redness_multiplier': 0.3,
                'smoothing_multiplier': 0.2,
                'notes': 'Benign growth, best treated with removal procedures'
            },

            # Vascular conditions
            'hemangioma': {
                'response_rate': 0.7,
                'redness_multiplier': 1.2,
                'smoothing_multiplier': 0.8,
                'notes': 'Vascular lesion, laser therapy most effective'
            },

            # Precancerous/cancerous lesions
            'basal cell carcinoma': {
                'response_rate': 0.1,
                'redness_multiplier': 0.2,
                'smoothing_multiplier': 0.1,
                'notes': 'Requires surgical removal or specialized treatment'
            },
            'squamous cell carcinoma': {
                'response_rate': 0.1,
                'redness_multiplier': 0.2,
                'smoothing_multiplier': 0.1,
                'notes': 'Requires surgical removal or specialized treatment'
            },
            'actinic keratosis': {
                'response_rate': 0.6,
                'redness_multiplier': 0.8,
                'smoothing_multiplier': 0.9,
                'notes': 'Precancerous lesion, responds to prescription creams and cryotherapy'
            },

            # Urticaria/allergic
            'urticaria': {
                'response_rate': 1.5,  # Fast response to antihistamines
                'redness_multiplier': 1.6,
                'smoothing_multiplier': 1.3,
                'notes': 'Allergic reaction, responds quickly to antihistamines'
            }
        }

        # Treatment-specific overrides: some conditions respond differently to surgical vs topical
        self.treatment_overrides = {
            'melanoma': {
                'surgical-excision': {
                    'response_rate': 1.8,
                    'redness_multiplier': 1.5,
                    'smoothing_multiplier': 1.3,
                    'notes': 'Surgery is curative - complete removal'
                },
                'mohs-surgery': {
                    'response_rate': 1.9,
                    'redness_multiplier': 1.5,
                    'smoothing_multiplier': 1.3,
                    'notes': 'Highest cure rate'
                },
                'immunotherapy': {
                    'response_rate': 1.2,
                    'redness_multiplier': 0.8,
                    'smoothing_multiplier': 0.7,
                    'notes': 'For advanced/metastatic'
                },
                # Topicals remain at 0.0 (default from condition_profiles)
            },
            'basal cell carcinoma': {
                'mohs-bcc': {'response_rate': 1.9, 'notes': 'Gold standard with 99% cure'},
                'excision-bcc': {'response_rate': 1.8, 'notes': 'Surgical removal'},
                'edc-bcc': {'response_rate': 1.7, 'notes': 'Electrodesiccation & curettage'},
                'imiquimod': {'response_rate': 1.5, 'notes': 'Topical for superficial BCC'},
            },
            'squamous cell carcinoma': {
                'mohs-bcc': {'response_rate': 1.9, 'notes': 'Best for SCC too'},
                'excision-bcc': {'response_rate': 1.8, 'notes': 'Surgical removal'},
                'edc-bcc': {'response_rate': 1.6, 'notes': 'For small low-risk'},
            },
            'seborrheic keratosis': {
                'cryotherapy-sk': {'response_rate': 1.7, 'notes': 'Freezing removes growth'},
                'curettage': {'response_rate': 1.8, 'notes': 'Scraping procedure'},
                'electrodesiccation': {'response_rate': 1.75, 'notes': 'Electrical removal'},
            },
            'wart': {
                'cryotherapy-wart': {'response_rate': 1.6, 'notes': 'Liquid nitrogen'},
                'salicylic-acid': {'response_rate': 1.2, 'notes': 'OTC slow method'},
                'cantharidin': {'response_rate': 1.5, 'notes': 'Blistering agent'},
                'immunotherapy-wart': {'response_rate': 1.4, 'notes': 'Candida injection'},
            },
            'acne': {
                'oral-isotretinoin': {'response_rate': 1.7, 'notes': 'Accutane - most effective'},
                'topical-retinoid': {'response_rate': 1.4, 'notes': 'Tretinoin'},
                'benzoyl-peroxide': {'response_rate': 1.2, 'notes': 'OTC antibacterial'},
                'laser-acne': {'response_rate': 1.3, 'notes': 'Laser therapy'},
            }
        }

    def predict_outcome(
        self,
        image_bytes: bytes,
        treatment_type: str,
        timeframe: str,
        improvement_percentage: int,
        diagnosis: str = None
    ) -> bytes:
        """
        Generate predicted treatment outcome image with condition-specific processing

        Args:
            image_bytes: Original image bytes
            treatment_type: Type of treatment (e.g., 'laser-therapy')
            timeframe: '6months', '1year', or '2years'
            improvement_percentage: Expected improvement (0-100)
            diagnosis: The diagnosed condition (e.g., 'eczema', 'melanoma')

        Returns:
            Processed image bytes showing predicted outcome
        """
        # Load image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get treatment configuration
        config = self.treatment_configs.get(
            treatment_type,
            self.treatment_configs['topical-steroid']
        )

        # Get condition-specific profile
        condition_profile = self._get_condition_profile(diagnosis)

        # Adjust intensity based on timeframe
        timeframe_multiplier = self._get_timeframe_multiplier(timeframe)

        # Adjust based on improvement percentage and condition response rate
        improvement_factor = (improvement_percentage / 100.0) * condition_profile['response_rate']

        # Apply condition-specific modifiers to treatment config
        redness_intensity = config['redness_reduction'] * timeframe_multiplier * improvement_factor * condition_profile['redness_multiplier']
        smoothing_intensity = config['smoothing'] * timeframe_multiplier * improvement_factor * condition_profile['smoothing_multiplier']

        # Apply treatments with condition-specific adjustments
        processed_image = self._reduce_redness(
            image,
            redness_intensity
        )
        processed_image = self._smooth_skin(
            processed_image,
            smoothing_intensity
        )
        processed_image = self._adjust_brightness(
            processed_image,
            1 + (config['brightness_increase'] - 1) * timeframe_multiplier * improvement_factor
        )
        processed_image = self._reduce_saturation(
            processed_image,
            config['saturation_decrease'] * timeframe_multiplier * improvement_factor
        )

        # Add subtle enhancement
        processed_image = self._enhance_image(processed_image, improvement_factor)

        # Convert back to bytes
        output_bytes = io.BytesIO()
        processed_image.save(output_bytes, format='JPEG', quality=95)
        output_bytes.seek(0)

        return output_bytes.getvalue()

    def _get_timeframe_multiplier(self, timeframe: str) -> float:
        """Get intensity multiplier based on timeframe"""
        multipliers = {
            '6months': 0.7,
            '1year': 0.9,
            '2years': 1.0
        }
        return multipliers.get(timeframe, 0.9)

    def _get_condition_profile(self, diagnosis: str) -> Dict:
        """
        Get condition-specific profile with fuzzy matching

        Args:
            diagnosis: The diagnosed condition name

        Returns:
            Dictionary with response_rate, redness_multiplier, smoothing_multiplier
        """
        # Default profile for unknown conditions
        default_profile = {
            'response_rate': 1.0,
            'redness_multiplier': 1.0,
            'smoothing_multiplier': 1.0,
            'notes': 'Standard treatment response'
        }

        if not diagnosis:
            return default_profile

        # Normalize diagnosis for matching
        diagnosis_lower = diagnosis.lower().strip()

        # Direct match
        if diagnosis_lower in self.condition_profiles:
            return self.condition_profiles[diagnosis_lower]

        # Fuzzy matching - check if diagnosis contains any condition keywords
        for condition_key, profile in self.condition_profiles.items():
            if condition_key in diagnosis_lower or diagnosis_lower in condition_key:
                return profile

        # Check for common aliases
        aliases = {
            'bcc': 'basal cell carcinoma',
            'scc': 'squamous cell carcinoma',
            'ak': 'actinic keratosis',
            'mel': 'melanoma',
            'nv': 'nevus',
            'dermatitis': 'eczema',
            'hives': 'urticaria',
            'tinea': 'fungal infection',
            'ringworm': 'fungal infection'
        }

        for alias, condition in aliases.items():
            if alias in diagnosis_lower:
                if condition in self.condition_profiles:
                    return self.condition_profiles[condition]

        return default_profile

    def _reduce_redness(self, image: Image.Image, intensity: float) -> Image.Image:
        """Reduce red tones in the image using HSV color space"""
        # Convert to numpy array
        img_array = np.array(image)

        # Convert RGB to HSV
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

        # HSV: Hue (0-180), Saturation (0-255), Value (0-255)
        # Red hues are in ranges: 0-10 and 170-180 in OpenCV
        hue = img_hsv[:, :, 0]
        sat = img_hsv[:, :, 1]

        # Create mask for red pixels
        red_mask1 = (hue <= 10)
        red_mask2 = (hue >= 170)
        red_mask = (red_mask1 | red_mask2) & (sat > 30)  # Only affect saturated reds

        # Reduce saturation of red pixels to make them less red
        reduction_factor = intensity  # intensity is like 0.65, so we reduce by 35%
        img_hsv[:, :, 1][red_mask] = img_hsv[:, :, 1][red_mask] * reduction_factor

        # Convert back to RGB
        img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(img_rgb)

    def _smooth_skin(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply skin smoothing effect"""
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Calculate kernel size based on intensity
        kernel_size = int(5 + intensity * 10)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Must be odd

        # Apply bilateral filter (preserves edges while smoothing)
        smoothed = cv2.bilateralFilter(img_cv, kernel_size, 75, 75)

        # Blend original and smoothed based on intensity
        blended = cv2.addWeighted(img_cv, 1 - intensity, smoothed, intensity, 0)

        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

    def _adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """Adjust image brightness"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _reduce_saturation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Reduce color saturation (for inflammation reduction)"""
        enhancer = ImageEnhance.Color(image)
        # intensity close to 1 = more saturation reduction
        saturation_factor = 1 - (1 - intensity) * 0.5
        return enhancer.enhance(saturation_factor)

    def _enhance_image(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply subtle overall enhancement"""
        # Slight contrast boost
        contrast_enhancer = ImageEnhance.Contrast(image)
        enhanced = contrast_enhancer.enhance(1 + intensity * 0.1)

        # Slight sharpness for clarity
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpness_enhancer.enhance(1 + intensity * 0.15)

        return enhanced

    # ==================== ENHANCEMENT METHODS ====================

    def _detect_severity(self, image: Image.Image) -> str:
        """
        Detect severity level based on image analysis
        Returns: 'mild', 'moderate', or 'severe'
        """
        img_array = np.array(image)
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Analyze redness (hue 0-10 and 170-180)
        hue = img_hsv[:, :, 0]
        sat = img_hsv[:, :, 1]
        val = img_hsv[:, :, 2]

        # Count red pixels
        red_mask1 = (hue <= 10) & (sat > 30)
        red_mask2 = (hue >= 170) & (sat > 30)
        red_pixels = np.sum(red_mask1 | red_mask2)
        total_pixels = hue.size
        red_percentage = (red_pixels / total_pixels) * 100

        # Analyze average saturation in red areas
        red_areas = img_hsv[red_mask1 | red_mask2]
        avg_saturation = np.mean(red_areas[:, 1]) if len(red_areas) > 0 else 0

        # Determine severity
        if red_percentage > 30 or avg_saturation > 150:
            return 'severe'
        elif red_percentage > 15 or avg_saturation > 100:
            return 'moderate'
        else:
            return 'mild'

    def _apply_severity_modifier(self, base_improvement: float, severity: str) -> float:
        """Modify improvement based on severity"""
        severity_modifiers = {
            'mild': 1.3,      # 30% better response
            'moderate': 1.0,  # Standard response
            'severe': 0.7     # 30% slower response
        }
        return base_improvement * severity_modifiers.get(severity, 1.0)

    def _add_scarring(self, image: Image.Image, diagnosis: str, severity: str) -> Image.Image:
        """
        Add realistic scarring/hyperpigmentation for relevant conditions
        """
        # Conditions that cause scarring
        scarring_conditions = ['acne', 'cellulitis', 'impetigo', 'severe eczema']

        should_scar = any(cond in diagnosis.lower() for cond in scarring_conditions)
        if not should_scar and severity != 'severe':
            return image

        img_array = np.array(image)
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Find areas that were likely inflamed (slightly red/dark)
        hue = img_hsv[:, :, 0]
        sat = img_hsv[:, :, 1]
        val = img_hsv[:, :, 2]

        # Create mask for potential scar areas
        scar_mask = ((hue <= 20) | (hue >= 160)) & (sat > 20) & (sat < 100)

        # Add hyperpigmentation (darken slightly)
        darkening_factor = 0.85 if severity == 'severe' else 0.92
        img_hsv[:, :, 2][scar_mask] = (img_hsv[:, :, 2][scar_mask] * darkening_factor).astype(np.uint8)

        # Reduce saturation in scar areas
        img_hsv[:, :, 1][scar_mask] = (img_hsv[:, :, 1][scar_mask] * 0.7).astype(np.uint8)

        # Convert back
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(img_rgb)

    def _apply_side_effects(self, image: Image.Image, treatment_type: str, weeks_elapsed: int) -> Image.Image:
        """
        Apply realistic side effects based on treatment type
        """
        if treatment_type == 'topical-steroid' and weeks_elapsed < 8:
            # Early steroid use: slight skin lightening
            img_array = np.array(image)
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

            # Slight brightening
            img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.05, 0, 255).astype(np.uint8)

            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            return Image.fromarray(img_rgb)

        # Other treatments have minimal visible side effects
        return image

    def _simulate_surgical_removal(self, image: Image.Image, healing_progress: float = 1.0) -> Image.Image:
        """
        Simulate surgical removal of a lesion.

        This method:
        1. Detects the lesion (darker/colored area, typically in center)
        2. Replaces it with surrounding skin texture
        3. Adds a subtle surgical scar

        Args:
            image: Original image with lesion
            healing_progress: 0.0 to 1.0, where 1.0 is fully healed

        Returns:
            Image with lesion removed and scar added
        """
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Convert to different color spaces for analysis
        img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)

        # Create a mask for the lesion (typically darker, more saturated, or different hue)
        # Focus on center region where lesion is likely to be
        center_x, center_y = w // 2, h // 2

        # Create distance-from-center weight (lesions typically centered)
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        center_weight = 1 - np.clip(dist_from_center / (min(h, w) * 0.4), 0, 1)

        # Detect lesion based on:
        # 1. Lower brightness (darker)
        # 2. Different color from surrounding skin
        brightness = img_lab[:, :, 0].astype(float)

        # Get surrounding skin brightness (edges of image)
        edge_mask = np.zeros((h, w), dtype=bool)
        edge_width = max(10, min(h, w) // 8)
        edge_mask[:edge_width, :] = True
        edge_mask[-edge_width:, :] = True
        edge_mask[:, :edge_width] = True
        edge_mask[:, -edge_width:] = True

        surrounding_brightness = np.median(brightness[edge_mask])
        surrounding_color = np.median(img_array[edge_mask], axis=0)

        # Create lesion mask based on brightness difference and center weighting
        brightness_diff = surrounding_brightness - brightness
        lesion_score = (brightness_diff / 255.0) * center_weight

        # Also consider saturation (melanoma often has higher saturation)
        saturation = img_hsv[:, :, 1].astype(float)
        surrounding_sat = np.median(saturation[edge_mask])
        sat_diff = saturation - surrounding_sat
        lesion_score += np.clip(sat_diff / 255.0, 0, 1) * center_weight * 0.5

        # Threshold to create binary mask
        lesion_mask = lesion_score > 0.15

        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        lesion_mask = cv2.morphologyEx(lesion_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)

        # Dilate slightly to ensure full coverage
        lesion_mask = cv2.dilate(lesion_mask, kernel, iterations=2)

        # Create smooth transition mask
        lesion_mask_float = cv2.GaussianBlur(lesion_mask.astype(float), (21, 21), 0)
        lesion_mask_float = np.clip(lesion_mask_float * healing_progress, 0, 1)

        # Create the "healed" skin by using surrounding skin color with slight variation
        healed_skin = np.zeros_like(img_array)
        for c in range(3):
            # Add subtle texture variation
            noise = np.random.normal(0, 5, (h, w))
            healed_skin[:, :, c] = np.clip(surrounding_color[c] + noise, 0, 255)

        # Apply Gaussian blur to healed skin for smooth texture
        healed_skin = cv2.GaussianBlur(healed_skin, (15, 15), 0)

        # Blend original and healed based on mask
        result = img_array.copy().astype(float)
        for c in range(3):
            result[:, :, c] = (
                img_array[:, :, c] * (1 - lesion_mask_float) +
                healed_skin[:, :, c] * lesion_mask_float
            )

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Add subtle scar if healing is complete
        if healing_progress > 0.8:
            # Find the center of the lesion area for scar placement
            if np.any(lesion_mask):
                lesion_coords = np.where(lesion_mask > 0)
                scar_center_y = int(np.mean(lesion_coords[0]))
                scar_center_x = int(np.mean(lesion_coords[1]))

                # Create a thin linear scar
                scar_length = max(20, int(np.std(lesion_coords[1]) * 1.5))
                scar_half = scar_length // 2

                # Draw scar line (slightly lighter than surrounding skin)
                for i in range(-scar_half, scar_half):
                    x = scar_center_x + i
                    y = scar_center_y + int(i * 0.1)  # Slight angle
                    if 0 <= x < w and 0 <= y < h:
                        # Scar is slightly pinker/lighter
                        scar_color = np.clip(surrounding_color * 1.05 + np.array([10, 5, 5]), 0, 255)
                        # Blend with existing pixel
                        scar_weight = 0.6 * np.exp(-(abs(i) / scar_half) ** 2)
                        result[y, x] = (
                            result[y, x] * (1 - scar_weight) +
                            scar_color * scar_weight
                        ).astype(np.uint8)

        return Image.fromarray(result)

    def _is_surgical_cancer_treatment(self, diagnosis: str, treatment_type: str) -> bool:
        """Check if this is a surgical treatment for cancer/melanoma"""
        if not diagnosis:
            return False

        diagnosis_lower = diagnosis.lower()
        treatment_lower = treatment_type.lower() if treatment_type else ''

        # Cancer/malignant conditions
        cancer_conditions = ['melanoma', 'carcinoma', 'basal cell', 'squamous cell', 'bcc', 'scc']
        is_cancer = any(cond in diagnosis_lower for cond in cancer_conditions)

        # Surgical treatments
        surgical_treatments = ['surgical', 'excision', 'mohs', 'surgery', 'curettage', 'electrodesiccation']
        is_surgical = any(surg in treatment_lower for surg in surgical_treatments)

        return is_cancer and is_surgical

    def _generate_progressive_timeline(
        self,
        base_image: Image.Image,
        config: Dict,
        condition_profile: Dict,
        timeframe: str,
        projected_improvement: int,
        diagnosis: str,
        severity: str,
        treatment_type: str
    ) -> list:
        """
        Generate multiple images showing progression over time
        Returns list of (weeks, improvement_percentage, image_bytes)
        """
        timeframe_weeks = {'6months': 24, '1year': 52, '2years': 104}
        total_weeks = timeframe_weeks.get(timeframe, 52)

        # Define checkpoints
        if timeframe == '6months':
            checkpoints = [4, 8, 16, 24]
        elif timeframe == '1year':
            checkpoints = [8, 16, 32, 52]
        else:  # 2years
            checkpoints = [12, 26, 52, 104]

        # Check for treatment-specific overrides (same logic as predict_outcome_enhanced)
        response_rate = condition_profile['response_rate']
        redness_multiplier = condition_profile['redness_multiplier']
        smoothing_multiplier = condition_profile['smoothing_multiplier']

        normalized_diagnosis = diagnosis.lower().strip() if diagnosis else ''
        if normalized_diagnosis in self.treatment_overrides:
            if treatment_type in self.treatment_overrides[normalized_diagnosis]:
                override = self.treatment_overrides[normalized_diagnosis][treatment_type]
                response_rate = override['response_rate']
                if 'redness_multiplier' in override:
                    redness_multiplier = override['redness_multiplier']
                if 'smoothing_multiplier' in override:
                    smoothing_multiplier = override['smoothing_multiplier']
                print(f"[TIMELINE OVERRIDE] {normalized_diagnosis} + {treatment_type}: response={response_rate}")

        timeline_images = []

        for weeks in checkpoints:
            # Calculate progress (non-linear - faster at start)
            progress_ratio = np.sqrt(weeks / total_weeks)  # Square root for non-linear
            current_improvement = projected_improvement * progress_ratio

            # Apply severity modifier
            current_improvement = self._apply_severity_modifier(current_improvement, severity)

            # Calculate intensities using the (possibly overridden) response_rate
            improvement_factor = (current_improvement / 100.0) * response_rate

            # Check if this is surgical cancer treatment
            is_surgical_cancer = self._is_surgical_cancer_treatment(diagnosis, treatment_type)

            if is_surgical_cancer:
                # Use surgical removal with progressive healing
                # Scale healing progress to ensure final image matches the main outcome
                # Use same formula as predict_outcome_enhanced for consistency
                healing_progress = progress_ratio * improvement_factor
                healing_progress = min(healing_progress * 1.5, 1.0)  # Same scaling as final image
                processed = self._simulate_surgical_removal(base_image.copy(), healing_progress=healing_progress)
            else:
                # Use the (possibly overridden) multipliers
                redness_intensity = config['redness_reduction'] * improvement_factor * redness_multiplier
                smoothing_intensity = config['smoothing'] * improvement_factor * smoothing_multiplier

                # Process image
                processed = self._reduce_redness(base_image.copy(), redness_intensity)
                processed = self._smooth_skin(processed, smoothing_intensity)
                processed = self._adjust_brightness(
                    processed,
                    1 + (config['brightness_increase'] - 1) * improvement_factor
                )
                processed = self._reduce_saturation(
                    processed,
                    config['saturation_decrease'] * improvement_factor
                )

                # Apply side effects for early weeks
                if weeks < 16:
                    processed = self._apply_side_effects(processed, treatment_type, weeks)

                # Add scarring for final image
                if weeks == checkpoints[-1]:
                    processed = self._add_scarring(processed, diagnosis, severity)

            # Convert to bytes
            output_bytes = io.BytesIO()
            processed.save(output_bytes, format='JPEG', quality=95)
            output_bytes.seek(0)

            timeline_images.append({
                'weeks': weeks,
                'improvement': int(current_improvement),
                'image_bytes': output_bytes.getvalue()
            })

        return timeline_images

    def _get_treatment_combination_multiplier(self, combination: str) -> float:
        """
        Get multiplier for treatment combinations
        Format: 'treatment1+treatment2'
        """
        combination_multipliers = {
            'topical-steroid+moisturizer': 1.2,
            'laser-therapy+topical': 1.3,
            'prescription-cream+moisturizer': 1.15,
            'cryotherapy+topical': 1.25
        }

        # Check if it's a combination
        if '+' in combination:
            return combination_multipliers.get(combination, 1.1)  # Default 10% bonus
        return 1.0

    def _apply_compliance_factor(self, improvement: float, compliance: str) -> float:
        """
        Modify improvement based on patient compliance
        compliance: 'perfect', 'good', 'typical', 'poor'
        """
        compliance_factors = {
            'perfect': 1.0,    # 100% compliance
            'good': 0.85,      # 85% compliance
            'typical': 0.7,    # 70% compliance
            'poor': 0.4        # 40% compliance
        }
        return improvement * compliance_factors.get(compliance, 0.7)

    def _apply_age_factor(self, improvement: float, age: int) -> float:
        """
        Modify improvement based on patient age
        Younger skin heals faster
        """
        if age < 20:
            return improvement * 1.2   # 20% better
        elif age < 40:
            return improvement * 1.0   # Standard
        elif age < 60:
            return improvement * 0.9   # 10% slower
        else:
            return improvement * 0.75  # 25% slower

    def _apply_skin_type_factor(self, improvement: float, skin_type: str) -> float:
        """
        Modify improvement based on Fitzpatrick skin type
        Higher types have more hyperpigmentation risk
        """
        # Types I-VI
        type_factors = {
            'I': 1.0,    # Very fair
            'II': 1.0,   # Fair
            'III': 0.95, # Medium
            'IV': 0.9,   # Olive
            'V': 0.85,   # Brown
            'VI': 0.8    # Dark brown/black
        }
        return improvement * type_factors.get(skin_type, 1.0)

    def _apply_location_factor(self, improvement: float, body_location: str) -> float:
        """
        Modify improvement based on body location
        """
        location_factors = {
            'face': 1.2,       # Heals faster
            'scalp': 1.1,
            'chest': 1.0,
            'back': 0.95,
            'arms': 0.95,
            'legs': 0.9,
            'hands': 0.85,     # Heals slower
            'feet': 0.8        # Heals slowest
        }
        return improvement * location_factors.get(body_location.lower(), 1.0)

    def generate_confidence_intervals(
        self,
        base_improvement: float,
        severity: str
    ) -> Dict:
        """
        Generate best-case, typical, and worst-case scenarios
        """
        # Base on severity and natural variation
        variation = {
            'mild': 0.15,      # ±15% variation
            'moderate': 0.20,  # ±20% variation
            'severe': 0.30     # ±30% variation
        }

        var = variation.get(severity, 0.20)

        return {
            'best_case': round(min(base_improvement * (1 + var), 95), 1),  # Cap at 95%, round to 1 decimal
            'typical': round(base_improvement, 1),
            'worst_case': round(max(base_improvement * (1 - var), 10), 1)  # Floor at 10%, round to 1 decimal
        }

    def predict_outcome_enhanced(
        self,
        image_bytes: bytes,
        treatment_type: str,
        timeframe: str,
        improvement_percentage: int,
        diagnosis: str = None,
        # Optional parameters for enhanced features
        compliance: str = 'typical',
        age: int = None,
        skin_type: str = None,
        body_location: str = None,
        generate_timeline: bool = False,
        generate_confidence: bool = False
    ) -> Dict:
        """
        Enhanced prediction with all new features

        Returns dict with:
        - after_image: Final outcome image bytes
        - severity: Detected severity level
        - confidence_intervals: Dict with best/typical/worst cases (if requested)
        - timeline_images: List of progressive images (if requested)
        - metadata: Dict with all factors applied
        """
        # Load and prepare image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Detect severity automatically
        severity = self._detect_severity(image)

        # Get treatment and condition profiles
        config = self.treatment_configs.get(treatment_type, self.treatment_configs['topical-steroid'])
        condition_profile = self._get_condition_profile(diagnosis)

        # Base improvement calculation
        base_improvement = improvement_percentage

        # Apply all modifying factors
        improvement = base_improvement

        # 1. Treatment combination
        combo_mult = self._get_treatment_combination_multiplier(treatment_type)
        improvement *= combo_mult

        # 2. Compliance
        improvement = self._apply_compliance_factor(improvement, compliance)

        # 3. Age (if provided)
        if age:
            improvement = self._apply_age_factor(improvement, age)

        # 4. Skin type (if provided)
        if skin_type:
            improvement = self._apply_skin_type_factor(improvement, skin_type)

        # 5. Body location (if provided)
        if body_location:
            improvement = self._apply_location_factor(improvement, body_location)

        # 6. Severity
        improvement = self._apply_severity_modifier(improvement, severity)

        # 7. Condition-specific response (check for treatment-specific override first)
        response_rate = condition_profile['response_rate']
        redness_multiplier = condition_profile['redness_multiplier']
        smoothing_multiplier = condition_profile['smoothing_multiplier']

        # Normalize diagnosis for override matching
        normalized_diagnosis = diagnosis.lower().strip() if diagnosis else ''

        # Check if there's a treatment-specific override for this condition+treatment combo
        if normalized_diagnosis in self.treatment_overrides:
            if treatment_type in self.treatment_overrides[normalized_diagnosis]:
                override = self.treatment_overrides[normalized_diagnosis][treatment_type]
                response_rate = override['response_rate']
                # Override visual multipliers if provided
                if 'redness_multiplier' in override:
                    redness_multiplier = override['redness_multiplier']
                if 'smoothing_multiplier' in override:
                    smoothing_multiplier = override['smoothing_multiplier']
                print(f"[TREATMENT OVERRIDE] {normalized_diagnosis} + {treatment_type}: response={response_rate}, redness={redness_multiplier}, smoothing={smoothing_multiplier}")

        improvement *= response_rate

        # Cap improvement
        improvement = min(improvement, 95)

        # Generate final outcome image
        timeframe_multiplier = self._get_timeframe_multiplier(timeframe)
        improvement_factor = improvement / 100.0

        # Check if this is a surgical treatment for cancer/melanoma
        is_surgical_cancer = self._is_surgical_cancer_treatment(diagnosis, treatment_type)

        if is_surgical_cancer:
            # Use surgical removal simulation for cancer surgeries
            print(f"[SURGICAL REMOVAL] Applying surgical removal simulation for {diagnosis} + {treatment_type}")
            healing_progress = timeframe_multiplier * improvement_factor
            processed_image = self._simulate_surgical_removal(image, healing_progress=min(healing_progress * 1.5, 1.0))
        else:
            # Use standard image processing for non-surgical treatments
            processed_image = self._reduce_redness(
                image,
                config['redness_reduction'] * timeframe_multiplier * improvement_factor * redness_multiplier
            )
            processed_image = self._smooth_skin(
                processed_image,
                config['smoothing'] * timeframe_multiplier * improvement_factor * smoothing_multiplier
            )
            processed_image = self._adjust_brightness(
                processed_image,
                1 + (config['brightness_increase'] - 1) * timeframe_multiplier * improvement_factor
            )
            processed_image = self._reduce_saturation(
                processed_image,
                config['saturation_decrease'] * timeframe_multiplier * improvement_factor
            )
            processed_image = self._enhance_image(processed_image, improvement_factor)

            # Add scarring if applicable (for non-surgical inflammatory conditions)
            processed_image = self._add_scarring(processed_image, diagnosis or '', severity)

        # Convert to bytes
        output_bytes = io.BytesIO()
        processed_image.save(output_bytes, format='JPEG', quality=95)
        output_bytes.seek(0)

        result = {
            'after_image': output_bytes.getvalue(),
            'severity': severity,
            'final_improvement': int(improvement),
            'metadata': {
                'base_improvement': base_improvement,
                'combination_multiplier': combo_mult,
                'compliance': compliance,
                'age_applied': age is not None,
                'skin_type_applied': skin_type is not None,
                'location_applied': body_location is not None,
                'severity_detected': severity,
                'condition_response_rate': condition_profile['response_rate']
            }
        }

        # Generate confidence intervals if requested
        if generate_confidence:
            result['confidence_intervals'] = self.generate_confidence_intervals(improvement, severity)

        # Generate timeline if requested
        if generate_timeline:
            timeline = self._generate_progressive_timeline(
                image,
                config,
                condition_profile,
                timeframe,
                int(improvement),
                diagnosis or '',
                severity,
                treatment_type
            )
            result['timeline_images'] = timeline

        return result


# Singleton instance
treatment_predictor = TreatmentOutcomePredictor()
print("[TreatmentPredictor] Module loaded successfully - predictor ready")
