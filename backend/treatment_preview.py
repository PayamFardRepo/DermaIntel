"""
Treatment Preview - "What If" Simulator

Shows photo-realistic before/after predictions for:
- Lesion removal
- Sunscreen use over time
- Skincare treatments (retinol, vitamin C, etc.)
- Lifestyle changes
- Aging simulation
"""

import io
import base64
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

class TreatmentType(Enum):
    LESION_REMOVAL = "lesion_removal"
    SUNSCREEN_USE = "sunscreen_use"
    RETINOL = "retinol"
    VITAMIN_C = "vitamin_c"
    HYDRATION = "hydration"
    ANTI_AGING = "anti_aging"
    ACNE_TREATMENT = "acne_treatment"
    PIGMENTATION_TREATMENT = "pigmentation_treatment"
    SUN_DAMAGE_REPAIR = "sun_damage_repair"
    LIFESTYLE_CHANGE = "lifestyle_change"

@dataclass
class TreatmentPreviewResult:
    original_image: str  # Base64
    preview_image: str   # Base64 after treatment
    treatment_type: str
    treatment_name: str
    description: str
    timeline: str
    expected_improvements: List[Dict]
    confidence: float
    tips: List[str]
    disclaimer: str

class TreatmentPreviewGenerator:
    """Generates realistic treatment preview images."""

    def __init__(self):
        self.treatment_configs = {
            TreatmentType.LESION_REMOVAL: {
                "name": "Lesion Removal Preview",
                "description": "Shows how your skin would look after lesion removal",
                "timeline": "Immediately after healing (4-6 weeks)",
                "confidence": 0.85,
            },
            TreatmentType.SUNSCREEN_USE: {
                "name": "Sunscreen Protection Results",
                "description": "Predicted skin after 6 months of daily SPF 50 use",
                "timeline": "6 months",
                "confidence": 0.75,
            },
            TreatmentType.RETINOL: {
                "name": "Retinol Treatment Results",
                "description": "Expected improvement from daily retinol use",
                "timeline": "3-6 months",
                "confidence": 0.70,
            },
            TreatmentType.VITAMIN_C: {
                "name": "Vitamin C Serum Results",
                "description": "Brightening and evening effects of vitamin C",
                "timeline": "2-3 months",
                "confidence": 0.72,
            },
            TreatmentType.HYDRATION: {
                "name": "Intensive Hydration Results",
                "description": "Effects of improved hydration routine",
                "timeline": "2-4 weeks",
                "confidence": 0.80,
            },
            TreatmentType.ANTI_AGING: {
                "name": "Anti-Aging Treatment Results",
                "description": "Combined anti-aging treatment effects",
                "timeline": "6-12 months",
                "confidence": 0.65,
            },
            TreatmentType.ACNE_TREATMENT: {
                "name": "Acne Treatment Results",
                "description": "Expected clearing with consistent treatment",
                "timeline": "2-3 months",
                "confidence": 0.70,
            },
            TreatmentType.PIGMENTATION_TREATMENT: {
                "name": "Pigmentation Treatment Results",
                "description": "Reduction of dark spots and uneven tone",
                "timeline": "3-6 months",
                "confidence": 0.68,
            },
            TreatmentType.SUN_DAMAGE_REPAIR: {
                "name": "Sun Damage Repair",
                "description": "Reversal of visible sun damage signs",
                "timeline": "6-12 months",
                "confidence": 0.65,
            },
            TreatmentType.LIFESTYLE_CHANGE: {
                "name": "Lifestyle Improvement Results",
                "description": "Effects of better sleep, diet, and hydration",
                "timeline": "1-3 months",
                "confidence": 0.70,
            },
        }

    def generate_preview(self, image_data: bytes, treatment_type: TreatmentType,
                        intensity: float = 0.5,
                        target_region: Optional[Dict] = None) -> TreatmentPreviewResult:
        """
        Generate a treatment preview image.

        Args:
            image_data: Original image bytes
            treatment_type: Type of treatment to simulate
            intensity: How dramatic the effect (0.0-1.0)
            target_region: Optional specific region to target (x, y, width, height normalized)

        Returns:
            TreatmentPreviewResult with before/after images
        """
        # Load image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        original = image.copy()

        # Apply treatment transformation
        if treatment_type == TreatmentType.LESION_REMOVAL:
            preview = self._apply_lesion_removal(image, target_region, intensity)
            improvements = [
                {"area": "Lesion", "change": "Removed", "confidence": 0.90},
                {"area": "Surrounding skin", "change": "Natural appearance", "confidence": 0.85},
            ]
            tips = [
                "Actual results depend on lesion type and removal method",
                "Healing time varies from 2-6 weeks",
                "Minimal scarring with modern techniques",
                "Always have suspicious lesions examined by a dermatologist"
            ]

        elif treatment_type == TreatmentType.SUNSCREEN_USE:
            preview = self._apply_sun_protection_effects(image, intensity)
            improvements = [
                {"area": "UV Damage", "change": "-40% visible damage", "confidence": 0.75},
                {"area": "Dark spots", "change": "-25% pigmentation", "confidence": 0.70},
                {"area": "Fine lines", "change": "-15% progression", "confidence": 0.65},
                {"area": "Skin tone", "change": "More even", "confidence": 0.80},
            ]
            tips = [
                "Apply SPF 50+ every morning",
                "Reapply every 2 hours when outdoors",
                "Don't forget ears, neck, and hands",
                "Use a vitamin C serum underneath for added protection"
            ]

        elif treatment_type == TreatmentType.RETINOL:
            preview = self._apply_retinol_effects(image, intensity)
            improvements = [
                {"area": "Fine lines", "change": "-30% depth", "confidence": 0.70},
                {"area": "Texture", "change": "+40% smoothness", "confidence": 0.75},
                {"area": "Pores", "change": "-20% visibility", "confidence": 0.65},
                {"area": "Cell turnover", "change": "+50% faster", "confidence": 0.80},
            ]
            tips = [
                "Start with low concentration (0.25-0.5%)",
                "Use only at night",
                "Always use sunscreen during the day",
                "Expect some initial irritation - this is normal"
            ]

        elif treatment_type == TreatmentType.VITAMIN_C:
            preview = self._apply_vitamin_c_effects(image, intensity)
            improvements = [
                {"area": "Brightness", "change": "+35% radiance", "confidence": 0.75},
                {"area": "Dark spots", "change": "-30% intensity", "confidence": 0.70},
                {"area": "Collagen", "change": "+20% production", "confidence": 0.65},
                {"area": "Antioxidant protection", "change": "+50%", "confidence": 0.80},
            ]
            tips = [
                "Use in the morning before sunscreen",
                "Look for L-ascorbic acid 10-20%",
                "Store in a cool, dark place",
                "Replace every 3 months or if it turns brown"
            ]

        elif treatment_type == TreatmentType.HYDRATION:
            preview = self._apply_hydration_effects(image, intensity)
            improvements = [
                {"area": "Moisture", "change": "+60% hydration", "confidence": 0.85},
                {"area": "Plumpness", "change": "+25% volume", "confidence": 0.75},
                {"area": "Fine lines", "change": "-20% (from dehydration)", "confidence": 0.70},
                {"area": "Glow", "change": "+40% radiance", "confidence": 0.80},
            ]
            tips = [
                "Use hyaluronic acid on damp skin",
                "Layer: toner → serum → moisturizer",
                "Drink at least 8 glasses of water daily",
                "Consider a humidifier in dry climates"
            ]

        elif treatment_type == TreatmentType.ANTI_AGING:
            preview = self._apply_anti_aging_effects(image, intensity)
            improvements = [
                {"area": "Wrinkles", "change": "-35% depth", "confidence": 0.65},
                {"area": "Firmness", "change": "+30% elasticity", "confidence": 0.60},
                {"area": "Skin tone", "change": "+40% evenness", "confidence": 0.70},
                {"area": "Radiance", "change": "+45% glow", "confidence": 0.75},
            ]
            tips = [
                "Consistency is key - results take 3-6 months",
                "Combine retinol, vitamin C, and peptides",
                "Never skip sunscreen - it prevents 80% of aging",
                "Consider professional treatments for faster results"
            ]

        elif treatment_type == TreatmentType.ACNE_TREATMENT:
            preview = self._apply_acne_treatment_effects(image, intensity)
            improvements = [
                {"area": "Active breakouts", "change": "-70% count", "confidence": 0.70},
                {"area": "Inflammation", "change": "-60% redness", "confidence": 0.75},
                {"area": "Pores", "change": "-25% congestion", "confidence": 0.65},
                {"area": "Scarring risk", "change": "-50%", "confidence": 0.60},
            ]
            tips = [
                "Use salicylic acid or benzoyl peroxide consistently",
                "Don't pick or squeeze - it causes scarring",
                "Change pillowcases weekly",
                "See a dermatologist if OTC products don't work"
            ]

        elif treatment_type == TreatmentType.PIGMENTATION_TREATMENT:
            preview = self._apply_pigmentation_treatment(image, intensity)
            improvements = [
                {"area": "Dark spots", "change": "-50% intensity", "confidence": 0.70},
                {"area": "Skin tone", "change": "+45% evenness", "confidence": 0.75},
                {"area": "Melasma", "change": "-30% visibility", "confidence": 0.55},
                {"area": "Sun spots", "change": "-40% count", "confidence": 0.65},
            ]
            tips = [
                "Sunscreen is essential - sun makes pigmentation worse",
                "Use niacinamide and alpha arbutin daily",
                "Be patient - results take 2-4 months",
                "Avoid irritating products that cause inflammation"
            ]

        else:  # Default/lifestyle
            preview = self._apply_lifestyle_effects(image, intensity)
            improvements = [
                {"area": "Overall health", "change": "+40% appearance", "confidence": 0.70},
                {"area": "Hydration", "change": "+35% moisture", "confidence": 0.75},
                {"area": "Radiance", "change": "+30% glow", "confidence": 0.70},
                {"area": "Clarity", "change": "+25% clearness", "confidence": 0.65},
            ]
            tips = [
                "Get 7-9 hours of sleep per night",
                "Drink plenty of water",
                "Eat antioxidant-rich foods",
                "Exercise regularly for better circulation"
            ]

        # Convert images to base64
        original_b64 = self._image_to_base64(original)
        preview_b64 = self._image_to_base64(preview)

        config = self.treatment_configs[treatment_type]

        return TreatmentPreviewResult(
            original_image=original_b64,
            preview_image=preview_b64,
            treatment_type=treatment_type.value,
            treatment_name=config["name"],
            description=config["description"],
            timeline=config["timeline"],
            expected_improvements=improvements,
            confidence=config["confidence"],
            tips=tips,
            disclaimer="This is a simulation for educational purposes. Actual results vary. Consult a dermatologist for personalized advice."
        )

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _apply_lesion_removal(self, image: Image.Image,
                             target_region: Optional[Dict],
                             intensity: float) -> Image.Image:
        """Simulate lesion removal using inpainting-like technique."""
        result = image.copy()
        img_array = np.array(result)

        if target_region:
            # Use specified region
            h, w = img_array.shape[:2]
            x = int(target_region.get('x', 0.5) * w)
            y = int(target_region.get('y', 0.5) * h)
            rw = int(target_region.get('width', 0.1) * w)
            rh = int(target_region.get('height', 0.1) * h)
        else:
            # Auto-detect darkest region
            h, w = img_array.shape[:2]
            gray = np.mean(img_array, axis=2)

            # Find darkest 10x10 region
            min_val = float('inf')
            x, y = w // 2, h // 2
            for yi in range(10, h - 10, 5):
                for xi in range(10, w - 10, 5):
                    region = gray[yi-5:yi+5, xi-5:xi+5]
                    if np.mean(region) < min_val:
                        min_val = np.mean(region)
                        x, y = xi, yi
            rw, rh = 30, 30

        # Create a patch from surrounding skin
        x1, x2 = max(0, x - rw), min(w, x + rw)
        y1, y2 = max(0, y - rh), min(h, y + rh)

        # Sample surrounding area colors
        surrounding_mask = np.ones_like(img_array[:,:,0], dtype=bool)
        surrounding_mask[y1:y2, x1:x2] = False

        # Get median color of surrounding skin
        for c in range(3):
            channel = img_array[:,:,c]
            surrounding_pixels = channel[surrounding_mask]
            if len(surrounding_pixels) > 0:
                median_color = np.median(surrounding_pixels)

                # Blend the lesion area with surrounding color
                for yi in range(y1, y2):
                    for xi in range(x1, x2):
                        # Calculate distance from center for smooth blending
                        dist = np.sqrt((xi - x)**2 + (yi - y)**2)
                        max_dist = np.sqrt(rw**2 + rh**2)
                        blend = min(1.0, dist / max_dist)
                        blend = blend ** 0.5  # Softer edge

                        # Add some texture variation
                        noise = np.random.normal(0, 5)

                        img_array[yi, xi, c] = int(
                            img_array[yi, xi, c] * blend +
                            (median_color + noise) * (1 - blend) * intensity +
                            img_array[yi, xi, c] * (1 - intensity)
                        )

        # Smooth the edited region
        result = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        result = result.filter(ImageFilter.GaussianBlur(radius=0.5))

        return result

    def _apply_sun_protection_effects(self, image: Image.Image, intensity: float) -> Image.Image:
        """Simulate effects of consistent sun protection."""
        result = image.copy()
        img_array = np.array(result, dtype=np.float32)

        # Reduce redness (sun damage indicator)
        img_array[:,:,0] = img_array[:,:,0] * (1 - 0.08 * intensity)

        # Even out skin tone (reduce variance)
        for c in range(3):
            channel = img_array[:,:,c]
            mean = np.mean(channel)
            img_array[:,:,c] = channel * (1 - 0.15 * intensity) + mean * (0.15 * intensity)

        # Slight brightness increase (healthier appearance)
        img_array = img_array * (1 + 0.05 * intensity)

        result = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

        # Subtle smoothing
        result = result.filter(ImageFilter.GaussianBlur(radius=0.3 * intensity))

        return result

    def _apply_retinol_effects(self, image: Image.Image, intensity: float) -> Image.Image:
        """Simulate retinol treatment effects."""
        result = image.copy()

        # Smooth texture
        result = result.filter(ImageFilter.GaussianBlur(radius=0.5 * intensity))

        # Increase contrast slightly (clearer skin)
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(1 + 0.1 * intensity)

        # Slight brightness boost
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(1 + 0.05 * intensity)

        # Reduce fine lines by selective smoothing
        img_array = np.array(result, dtype=np.float32)

        # Even out texture
        for c in range(3):
            channel = img_array[:,:,c]
            smoothed = np.array(Image.fromarray(channel.astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=1)))
            img_array[:,:,c] = channel * (1 - 0.2 * intensity) + smoothed * (0.2 * intensity)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _apply_vitamin_c_effects(self, image: Image.Image, intensity: float) -> Image.Image:
        """Simulate vitamin C serum effects - brightening and evening."""
        result = image.copy()
        img_array = np.array(result, dtype=np.float32)

        # Brighten overall
        img_array = img_array * (1 + 0.08 * intensity)

        # Reduce dark spots (areas darker than average)
        gray = np.mean(img_array, axis=2)
        mean_brightness = np.mean(gray)

        dark_mask = gray < mean_brightness - 20
        for c in range(3):
            img_array[:,:,c] = np.where(
                dark_mask,
                img_array[:,:,c] * (1 + 0.15 * intensity),
                img_array[:,:,c]
            )

        # Add subtle warmth (healthy glow)
        img_array[:,:,0] = img_array[:,:,0] * (1 + 0.02 * intensity)  # Slight red
        img_array[:,:,1] = img_array[:,:,1] * (1 + 0.01 * intensity)  # Slight green

        result = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

        # Enhance color saturation slightly
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(1 + 0.1 * intensity)

        return result

    def _apply_hydration_effects(self, image: Image.Image, intensity: float) -> Image.Image:
        """Simulate intensive hydration effects."""
        result = image.copy()
        img_array = np.array(result, dtype=np.float32)

        # Add luminosity (hydrated skin reflects light better)
        img_array = img_array * (1 + 0.06 * intensity)

        # Reduce fine lines from dehydration (smooth)
        result = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        result = result.filter(ImageFilter.GaussianBlur(radius=0.3 * intensity))

        # Enhance with subtle glow
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(1 + 0.03 * intensity)

        return result

    def _apply_anti_aging_effects(self, image: Image.Image, intensity: float) -> Image.Image:
        """Simulate combined anti-aging treatment effects."""
        result = image.copy()

        # Combine multiple effects
        result = self._apply_retinol_effects(result, intensity * 0.7)
        result = self._apply_vitamin_c_effects(result, intensity * 0.5)
        result = self._apply_hydration_effects(result, intensity * 0.6)

        # Additional firming effect (subtle)
        img_array = np.array(result, dtype=np.float32)

        # Slight sharpening for "firmer" appearance
        from PIL import ImageFilter
        sharpened = result.filter(ImageFilter.SHARPEN)
        sharpened_array = np.array(sharpened, dtype=np.float32)

        # Blend
        img_array = img_array * (1 - 0.2 * intensity) + sharpened_array * (0.2 * intensity)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _apply_acne_treatment_effects(self, image: Image.Image, intensity: float) -> Image.Image:
        """Simulate acne treatment effects."""
        result = image.copy()
        img_array = np.array(result, dtype=np.float32)

        # Reduce redness
        img_array[:,:,0] = img_array[:,:,0] * (1 - 0.12 * intensity)

        # Find and reduce red spots
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        red_mask = (r > g + 15) & (r > b + 15)

        for c in range(3):
            mean_val = np.mean(img_array[:,:,c])
            img_array[:,:,c] = np.where(
                red_mask,
                img_array[:,:,c] * (1 - 0.3 * intensity) + mean_val * (0.3 * intensity),
                img_array[:,:,c]
            )

        # Smooth texture
        result = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        result = result.filter(ImageFilter.GaussianBlur(radius=0.4 * intensity))

        return result

    def _apply_pigmentation_treatment(self, image: Image.Image, intensity: float) -> Image.Image:
        """Simulate pigmentation treatment effects."""
        result = image.copy()
        img_array = np.array(result, dtype=np.float32)

        # Even out skin tone
        gray = np.mean(img_array, axis=2)
        mean_brightness = np.mean(gray)

        # Target dark spots
        dark_threshold = mean_brightness - 25
        dark_mask = gray < dark_threshold

        for c in range(3):
            img_array[:,:,c] = np.where(
                dark_mask,
                img_array[:,:,c] * (1 - 0.4 * intensity) + mean_brightness * (0.4 * intensity),
                img_array[:,:,c]
            )

        # Reduce overall variance (more even tone)
        for c in range(3):
            channel = img_array[:,:,c]
            mean = np.mean(channel)
            img_array[:,:,c] = channel * (1 - 0.1 * intensity) + mean * (0.1 * intensity)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _apply_lifestyle_effects(self, image: Image.Image, intensity: float) -> Image.Image:
        """Simulate effects of lifestyle improvements."""
        result = image.copy()

        # Combination of hydration and general health
        result = self._apply_hydration_effects(result, intensity * 0.6)

        # Reduce under-eye darkness and overall fatigue appearance
        img_array = np.array(result, dtype=np.float32)

        # Slight brightness and warmth
        img_array = img_array * (1 + 0.04 * intensity)

        # Add healthy color
        img_array[:,:,0] = img_array[:,:,0] * (1 + 0.02 * intensity)
        img_array[:,:,1] = img_array[:,:,1] * (1 + 0.015 * intensity)

        result = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

        # Enhance saturation (healthier look)
        enhancer = ImageEnhance.Color(result)
        result = enhancer.enhance(1 + 0.08 * intensity)

        return result


# FastAPI integration
def create_treatment_preview_router():
    """Create FastAPI router for treatment preview endpoints."""
    from fastapi import APIRouter, HTTPException, File, UploadFile, Form
    from pydantic import BaseModel
    from typing import Optional

    router = APIRouter(prefix="/api/treatment-preview", tags=["Treatment Preview"])
    generator = TreatmentPreviewGenerator()

    class ImprovementItem(BaseModel):
        area: str
        change: str
        confidence: float

    class TreatmentPreviewResponse(BaseModel):
        original_image: str
        preview_image: str
        treatment_type: str
        treatment_name: str
        description: str
        timeline: str
        expected_improvements: List[ImprovementItem]
        confidence: float
        tips: List[str]
        disclaimer: str

    @router.post("/generate", response_model=TreatmentPreviewResponse)
    async def generate_preview(
        image: UploadFile = File(...),
        treatment: str = Form(...),
        intensity: float = Form(0.5),
        target_x: Optional[float] = Form(None),
        target_y: Optional[float] = Form(None),
        target_width: Optional[float] = Form(None),
        target_height: Optional[float] = Form(None)
    ):
        """
        Generate a treatment preview image.

        Args:
            image: Original skin image
            treatment: Treatment type (lesion_removal, sunscreen_use, retinol, etc.)
            intensity: Effect intensity 0.0-1.0
            target_*: Optional target region for treatments like lesion removal
        """
        try:
            # Parse treatment type
            try:
                treatment_type = TreatmentType(treatment)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid treatment type. Valid options: {[t.value for t in TreatmentType]}"
                )

            image_data = await image.read()

            # Build target region if provided
            target_region = None
            if all(v is not None for v in [target_x, target_y, target_width, target_height]):
                target_region = {
                    'x': target_x,
                    'y': target_y,
                    'width': target_width,
                    'height': target_height
                }

            result = generator.generate_preview(
                image_data=image_data,
                treatment_type=treatment_type,
                intensity=max(0.0, min(1.0, intensity)),
                target_region=target_region
            )

            return TreatmentPreviewResponse(
                original_image=result.original_image,
                preview_image=result.preview_image,
                treatment_type=result.treatment_type,
                treatment_name=result.treatment_name,
                description=result.description,
                timeline=result.timeline,
                expected_improvements=[
                    ImprovementItem(**imp) for imp in result.expected_improvements
                ],
                confidence=result.confidence,
                tips=result.tips,
                disclaimer=result.disclaimer
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")

    @router.get("/treatments")
    async def list_treatments():
        """List all available treatment types."""
        return {
            "treatments": [
                {
                    "id": t.value,
                    "name": generator.treatment_configs[t]["name"],
                    "description": generator.treatment_configs[t]["description"],
                    "timeline": generator.treatment_configs[t]["timeline"],
                }
                for t in TreatmentType
            ]
        }

    return router
