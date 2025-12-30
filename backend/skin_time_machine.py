"""
Skin Time Machine - Age Progression Simulator

Shows users how their skin will look in 10, 20, 30 years:
- Without care: Aging based on current lifestyle
- With skincare: Slowed aging with proper routine
- Highly shareable results

Now powered by Fast-AgingGAN for realistic AI-based aging!
"""

import io
import base64
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

logger = logging.getLogger(__name__)

# Try to import Fast-AgingGAN
try:
    from fast_aging_gan import get_aging_model, is_model_available, download_model
    FAST_AGING_GAN_AVAILABLE = True
    logger.info("Fast-AgingGAN module loaded successfully")
except ImportError:
    FAST_AGING_GAN_AVAILABLE = False
    logger.warning("Fast-AgingGAN module not available, using filter-based aging")


class AgingScenario(Enum):
    NO_CARE = "no_care"  # Continue current habits (bad)
    BASIC_CARE = "basic_care"  # Basic skincare routine
    OPTIMAL_CARE = "optimal_care"  # Full anti-aging regimen


@dataclass
class AgingFactors:
    """Lifestyle factors that affect skin aging."""
    smoking: bool = False
    sun_exposure: str = "moderate"  # low, moderate, high
    sleep_hours: float = 7
    water_intake: str = "moderate"  # low, moderate, high
    uses_sunscreen: bool = False
    uses_retinol: bool = False
    stress_level: str = "moderate"  # low, moderate, high
    diet_quality: str = "moderate"  # poor, moderate, good


@dataclass
class TimeMachineResult:
    """Result of age progression simulation."""
    original_image: str  # Base64
    projections: List[Dict]  # List of {years, scenario, image, description}
    current_age: int
    skin_age: int
    aging_rate: str  # "accelerated", "normal", "slowed"
    key_factors: List[Dict]
    recommendations: List[str]
    share_text: str
    ai_powered: bool = False  # Whether real AI was used


class SkinTimeMachine:
    """Generates realistic age progression images using AI."""

    def __init__(self):
        # Aging acceleration factors by lifestyle
        self.lifestyle_factors = {
            "smoking": 0.15,  # 15% faster aging
            "high_sun": 0.20,  # 20% faster
            "low_sleep": 0.10,  # 10% faster
            "dehydration": 0.08,
            "high_stress": 0.12,
            "poor_diet": 0.08,
        }

        # Protection factors
        self.protection_factors = {
            "sunscreen": -0.15,  # 15% slower aging
            "retinol": -0.20,
            "good_sleep": -0.05,
            "hydration": -0.05,
            "antioxidants": -0.08,
        }

        # AI model (lazy loaded)
        self._ai_model = None
        self._ai_model_loaded = False
        self._ai_aged_cache = {}  # Cache the AI-aged image per session

    def _get_ai_model(self):
        """Lazy load the AI aging model."""
        if not FAST_AGING_GAN_AVAILABLE:
            return None

        if not self._ai_model_loaded:
            try:
                self._ai_model = get_aging_model()
                if self._ai_model.load_model():
                    self._ai_model_loaded = True
                    logger.info("Fast-AgingGAN model loaded successfully")
                else:
                    logger.warning("Failed to load Fast-AgingGAN model")
                    self._ai_model = None
            except Exception as e:
                logger.error(f"Error loading Fast-AgingGAN: {e}")
                self._ai_model = None

        return self._ai_model

    def _age_with_ai(self, image: Image.Image) -> Optional[Image.Image]:
        """Age a face using the AI model."""
        model = self._get_ai_model()
        if model is None:
            return None

        try:
            aged = model.age_face(image)
            return aged
        except Exception as e:
            logger.error(f"AI aging failed: {e}")
            return None

    def generate_progression(
        self,
        image_data: bytes,
        current_age: int,
        factors: AgingFactors,
        years_ahead: List[int] = [10, 20, 30]
    ) -> TimeMachineResult:
        """Generate age progression images."""

        # Load image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        original = image.copy()

        # Calculate aging rate based on lifestyle
        aging_rate = self._calculate_aging_rate(factors)

        # Estimate current skin age
        skin_age = self._estimate_skin_age(current_age, factors)

        # Try to get AI-aged version (maximum aging)
        ai_aged_image = self._age_with_ai(image.copy())
        ai_powered = ai_aged_image is not None

        if ai_powered:
            logger.info("Using Fast-AgingGAN for realistic aging")
        else:
            logger.info("Using filter-based aging (AI model not available)")

        # Generate projections for each scenario and timeframe
        projections = []

        for years in years_ahead:
            for scenario in AgingScenario:
                if ai_powered:
                    # Use AI-based aging with blending
                    aged_image = self._apply_ai_aging(
                        original.copy(),
                        ai_aged_image,
                        years,
                        scenario,
                        factors,
                        current_age
                    )
                else:
                    # Fall back to filter-based aging
                    aged_image = self._apply_filter_aging(
                        image.copy(),
                        years,
                        scenario,
                        factors,
                        current_age
                    )

                projection = {
                    "years": years,
                    "future_age": current_age + years,
                    "scenario": scenario.value,
                    "scenario_name": self._get_scenario_name(scenario),
                    "image": self._image_to_base64(aged_image),
                    "description": self._get_scenario_description(scenario, years),
                    "skin_age_at_time": self._project_skin_age(
                        skin_age, years, scenario, factors
                    )
                }
                projections.append(projection)

        # Identify key factors
        key_factors = self._analyze_key_factors(factors)

        # Generate recommendations
        recommendations = self._generate_recommendations(factors, aging_rate)

        # Determine aging rate category
        if aging_rate > 1.15:
            rate_category = "accelerated"
        elif aging_rate < 0.90:
            rate_category = "slowed"
        else:
            rate_category = "normal"

        # Generate share text
        share_text = self._generate_share_text(
            current_age, skin_age, rate_category, years_ahead[-1]
        )

        return TimeMachineResult(
            original_image=self._image_to_base64(original),
            projections=projections,
            current_age=current_age,
            skin_age=skin_age,
            aging_rate=rate_category,
            key_factors=key_factors,
            recommendations=recommendations,
            share_text=share_text,
            ai_powered=ai_powered
        )

    def _apply_ai_aging(
        self,
        original: Image.Image,
        ai_aged: Image.Image,
        years: int,
        scenario: AgingScenario,
        factors: AgingFactors,
        current_age: int
    ) -> Image.Image:
        """
        Apply AI-based aging by blending original with fully-aged image,
        then applying additional filter effects for more visible results.

        The AI model produces a "fully aged" version.
        We amplify and blend based on years/scenario, then add filter effects.
        """
        # Ensure same size
        if original.size != ai_aged.size:
            ai_aged = ai_aged.resize(original.size, Image.Resampling.LANCZOS)

        # Convert to numpy
        orig_arr = np.array(original, dtype=np.float32)
        aged_arr = np.array(ai_aged, dtype=np.float32)

        # Calculate the difference (aging effect)
        diff = aged_arr - orig_arr

        # Amplify the AI differences (the model is too subtle)
        # This makes wrinkles and age effects more visible
        amplification = 3.0  # Amplify AI aging effect 3x

        # Calculate blend factor based on years and scenario
        # More aggressive blending for visible results
        if scenario == AgingScenario.NO_CARE:
            # +10y = 50%, +20y = 80%, +30y = 100%
            base_blend = 0.3 + (years / 30.0) * 0.7
            lifestyle_mult = self._calculate_aging_rate(factors)
            blend = base_blend * min(1.3, lifestyle_mult)
        elif scenario == AgingScenario.BASIC_CARE:
            # +10y = 30%, +20y = 50%, +30y = 70%
            blend = 0.2 + (years / 30.0) * 0.5
        else:  # OPTIMAL_CARE
            # +10y = 20%, +20y = 35%, +30y = 50%
            blend = 0.1 + (years / 30.0) * 0.4

        # Clamp blend factor
        blend = max(0.0, min(1.0, blend))

        # Apply amplified aging
        blended = orig_arr + (diff * amplification * blend)

        # Now apply additional filter-based effects for more drama
        # These stack on top of the AI aging
        filter_intensity = blend * 0.6  # Reduce filter intensity since we have AI

        # Add subtle wrinkle texture
        h, w = blended.shape[:2]
        np.random.seed(42)
        wrinkle_noise = np.random.randn(h, w) * 5 * filter_intensity
        for c in range(3):
            blended[:, :, c] = blended[:, :, c] + wrinkle_noise

        # Reduce luminosity (aging skin is less radiant)
        blended = blended * (1 - 0.08 * filter_intensity)

        # Add slight yellow/sallow tint for no_care scenario
        if scenario == AgingScenario.NO_CARE:
            blended[:, :, 1] = blended[:, :, 1] * (1 + 0.03 * filter_intensity)  # Yellow
            blended[:, :, 2] = blended[:, :, 2] * (1 - 0.04 * filter_intensity)  # Less blue
        elif scenario == AgingScenario.OPTIMAL_CARE:
            # Healthy glow for optimal care
            blended = blended * (1 + 0.02 * filter_intensity)
            blended[:, :, 0] = blended[:, :, 0] * 1.01  # Slight warmth

        # Add age spots for bad scenarios
        if scenario == AgingScenario.NO_CARE and years >= 20:
            blended = self._add_age_spots_array(blended, filter_intensity * 0.5, factors)

        result = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

        return result

    def _add_age_spots_array(
        self,
        img: np.ndarray,
        intensity: float,
        factors: AgingFactors
    ) -> np.ndarray:
        """Add age spots to numpy array."""
        h, w = img.shape[:2]

        spot_multiplier = 1.0
        if factors.sun_exposure == "high":
            spot_multiplier = 1.5

        num_spots = int(3 * intensity * spot_multiplier)

        np.random.seed(123)
        for _ in range(num_spots):
            x = np.random.randint(int(w * 0.2), int(w * 0.8))
            y = np.random.randint(int(h * 0.1), int(h * 0.7))
            radius = np.random.randint(3, 8)

            for yi in range(max(0, y - radius), min(h, y + radius)):
                for xi in range(max(0, x - radius), min(w, x + radius)):
                    dist = np.sqrt((xi - x) ** 2 + (yi - y) ** 2)
                    if dist < radius:
                        spot_blend = (1 - (dist / radius)) * 0.25 * intensity
                        img[yi, xi, 0] = img[yi, xi, 0] * (1 - spot_blend) + 139 * spot_blend
                        img[yi, xi, 1] = img[yi, xi, 1] * (1 - spot_blend) + 90 * spot_blend
                        img[yi, xi, 2] = img[yi, xi, 2] * (1 - spot_blend) + 43 * spot_blend

        return img

    def _apply_filter_aging(
        self,
        image: Image.Image,
        years: int,
        scenario: AgingScenario,
        factors: AgingFactors,
        current_age: int
    ) -> Image.Image:
        """Apply filter-based aging effects (fallback when AI not available)."""

        # Determine intensity based on scenario
        if scenario == AgingScenario.NO_CARE:
            intensity = 1.0 + (self._calculate_aging_rate(factors) - 1.0) * 0.5
        elif scenario == AgingScenario.BASIC_CARE:
            intensity = 0.7
        else:  # OPTIMAL_CARE
            intensity = 0.4

        # Scale by years
        year_factor = years / 30.0  # Normalize to 30 years
        intensity *= year_factor

        img_array = np.array(image, dtype=np.float32)

        # 1. Add wrinkles (texture)
        img_array = self._add_wrinkles(img_array, intensity, current_age + years)

        # 2. Reduce skin elasticity (slight sag simulation)
        img_array = self._reduce_elasticity(img_array, intensity)

        # 3. Add age spots
        img_array = self._add_age_spots(img_array, intensity, scenario, factors)

        # 4. Reduce skin luminosity
        img_array = self._reduce_luminosity(img_array, intensity)

        # 5. Add skin thinning effect
        img_array = self._thin_skin(img_array, intensity)

        # 6. Adjust color (more yellow/sallow with age)
        img_array = self._adjust_skin_color(img_array, intensity, scenario)

        result = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

        return result

    def _calculate_aging_rate(self, factors: AgingFactors) -> float:
        """Calculate overall aging rate multiplier."""
        rate = 1.0

        # Negative factors (accelerate aging)
        if factors.smoking:
            rate += self.lifestyle_factors["smoking"]

        if factors.sun_exposure == "high":
            rate += self.lifestyle_factors["high_sun"]
        elif factors.sun_exposure == "low":
            rate -= 0.05

        if factors.sleep_hours < 6:
            rate += self.lifestyle_factors["low_sleep"]
        elif factors.sleep_hours >= 8:
            rate -= 0.03

        if factors.water_intake == "low":
            rate += self.lifestyle_factors["dehydration"]
        elif factors.water_intake == "high":
            rate += self.protection_factors["hydration"]

        if factors.stress_level == "high":
            rate += self.lifestyle_factors["high_stress"]

        if factors.diet_quality == "poor":
            rate += self.lifestyle_factors["poor_diet"]
        elif factors.diet_quality == "good":
            rate += self.protection_factors["antioxidants"]

        # Positive factors (slow aging)
        if factors.uses_sunscreen:
            rate += self.protection_factors["sunscreen"]

        if factors.uses_retinol:
            rate += self.protection_factors["retinol"]

        return max(0.5, min(1.5, rate))

    def _estimate_skin_age(self, current_age: int, factors: AgingFactors) -> int:
        """Estimate current biological skin age."""
        aging_rate = self._calculate_aging_rate(factors)

        # Skin age deviation based on lifestyle
        deviation = (aging_rate - 1.0) * current_age * 0.3

        skin_age = current_age + int(deviation)
        return max(18, min(current_age + 15, skin_age))

    def _project_skin_age(
        self,
        current_skin_age: int,
        years: int,
        scenario: AgingScenario,
        factors: AgingFactors
    ) -> int:
        """Project skin age for a future date based on scenario."""

        if scenario == AgingScenario.NO_CARE:
            # Continue current bad habits
            rate = self._calculate_aging_rate(factors)
            rate = max(rate, 1.1)  # At least 10% accelerated
        elif scenario == AgingScenario.BASIC_CARE:
            rate = 0.95  # 5% slower than chronological
        else:  # OPTIMAL_CARE
            rate = 0.80  # 20% slower

        projected = current_skin_age + int(years * rate)
        return projected

    # Filter-based aging effects (used as fallback)
    def _add_wrinkles(self, img: np.ndarray, intensity: float, target_age: int) -> np.ndarray:
        """Add wrinkle-like texture to skin."""
        h, w = img.shape[:2]

        wrinkle_intensity = intensity * min(1.0, (target_age - 30) / 40)
        if wrinkle_intensity <= 0:
            return img

        np.random.seed(42)
        noise = np.random.randn(h, w) * 8 * wrinkle_intensity

        for c in range(3):
            img[:, :, c] = img[:, :, c] + noise

        return img

    def _reduce_elasticity(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """Simulate reduced skin elasticity."""
        h, w = img.shape[:2]

        shift = int(h * 0.01 * intensity)
        if shift > 0:
            lower_portion = int(h * 0.6)
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=0.3 * intensity))
            img = np.array(img_pil, dtype=np.float32)

        return img

    def _add_age_spots(
        self,
        img: np.ndarray,
        intensity: float,
        scenario: AgingScenario,
        factors: AgingFactors
    ) -> np.ndarray:
        """Add age/sun spots."""
        h, w = img.shape[:2]

        spot_multiplier = 1.0
        if factors.sun_exposure == "high":
            spot_multiplier = 1.5
        if scenario == AgingScenario.NO_CARE:
            spot_multiplier *= 1.3
        elif scenario == AgingScenario.OPTIMAL_CARE:
            spot_multiplier *= 0.5

        num_spots = int(5 * intensity * spot_multiplier)

        np.random.seed(123)
        for _ in range(num_spots):
            x = np.random.randint(int(w * 0.2), int(w * 0.8))
            y = np.random.randint(int(h * 0.1), int(h * 0.7))
            radius = np.random.randint(2, 6)

            for yi in range(max(0, y - radius), min(h, y + radius)):
                for xi in range(max(0, x - radius), min(w, x + radius)):
                    dist = np.sqrt((xi - x) ** 2 + (yi - y) ** 2)
                    if dist < radius:
                        blend = 1 - (dist / radius)
                        blend *= 0.3 * intensity
                        img[yi, xi, 0] = img[yi, xi, 0] * (1 - blend) + 139 * blend
                        img[yi, xi, 1] = img[yi, xi, 1] * (1 - blend) + 90 * blend
                        img[yi, xi, 2] = img[yi, xi, 2] * (1 - blend) + 43 * blend

        return img

    def _reduce_luminosity(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """Reduce skin luminosity/glow."""
        reduction = 1 - (0.08 * intensity)
        img = img * reduction
        return img

    def _thin_skin(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """Simulate thinner skin."""
        mean = np.mean(img)
        img = (img - mean) * (1 + 0.05 * intensity) + mean
        return img

    def _adjust_skin_color(
        self,
        img: np.ndarray,
        intensity: float,
        scenario: AgingScenario
    ) -> np.ndarray:
        """Adjust skin color for age."""

        if scenario == AgingScenario.OPTIMAL_CARE:
            img[:, :, 0] = img[:, :, 0] * (1 + 0.02 * intensity)
        else:
            img[:, :, 1] = img[:, :, 1] * (1 + 0.03 * intensity)
            img[:, :, 2] = img[:, :, 2] * (1 - 0.05 * intensity)

            if scenario == AgingScenario.NO_CARE:
                img[:, :, 0] = img[:, :, 0] * (1 - 0.03 * intensity)

        return img

    def _get_scenario_name(self, scenario: AgingScenario) -> str:
        """Get human-readable scenario name."""
        names = {
            AgingScenario.NO_CARE: "Without Skincare",
            AgingScenario.BASIC_CARE: "With Basic Care",
            AgingScenario.OPTIMAL_CARE: "With Optimal Care",
        }
        return names[scenario]

    def _get_scenario_description(self, scenario: AgingScenario, years: int) -> str:
        """Get description for scenario."""
        if scenario == AgingScenario.NO_CARE:
            return f"Your skin in {years} years if you continue current habits without improvement"
        elif scenario == AgingScenario.BASIC_CARE:
            return f"Your skin in {years} years with basic skincare (cleanser, moisturizer, SPF)"
        else:
            return f"Your skin in {years} years with optimal anti-aging routine"

    def _analyze_key_factors(self, factors: AgingFactors) -> List[Dict]:
        """Analyze which factors are most impacting aging."""
        key_factors = []

        if factors.smoking:
            key_factors.append({
                "factor": "Smoking",
                "impact": "negative",
                "severity": "high",
                "description": "Accelerates skin aging by 15%+ through reduced blood flow and collagen breakdown"
            })

        if factors.sun_exposure == "high" and not factors.uses_sunscreen:
            key_factors.append({
                "factor": "Unprotected Sun Exposure",
                "impact": "negative",
                "severity": "high",
                "description": "Causes up to 80% of visible skin aging (photoaging)"
            })

        if factors.uses_sunscreen:
            key_factors.append({
                "factor": "Daily Sunscreen",
                "impact": "positive",
                "severity": "high",
                "description": "Reduces visible aging by up to 24% and prevents sun damage"
            })

        if factors.uses_retinol:
            key_factors.append({
                "factor": "Retinol Use",
                "impact": "positive",
                "severity": "high",
                "description": "Proven to reduce wrinkles and stimulate collagen by 20%+"
            })

        if factors.sleep_hours < 6:
            key_factors.append({
                "factor": "Sleep Deprivation",
                "impact": "negative",
                "severity": "medium",
                "description": "Poor sleep accelerates aging and reduces skin repair"
            })

        if factors.water_intake == "low":
            key_factors.append({
                "factor": "Dehydration",
                "impact": "negative",
                "severity": "medium",
                "description": "Leads to dull, dry skin that shows age faster"
            })

        if factors.stress_level == "high":
            key_factors.append({
                "factor": "High Stress",
                "impact": "negative",
                "severity": "medium",
                "description": "Cortisol breaks down collagen and causes inflammation"
            })

        return key_factors

    def _generate_recommendations(
        self,
        factors: AgingFactors,
        aging_rate: float
    ) -> List[str]:
        """Generate personalized recommendations."""
        recs = []

        if not factors.uses_sunscreen:
            recs.append("Start daily SPF 50 sunscreen - this alone can prevent 80% of visible aging")

        if not factors.uses_retinol:
            recs.append("Add retinol to your night routine - the #1 proven anti-aging ingredient")

        if factors.smoking:
            recs.append("Quit smoking - your skin will show improvement within weeks")

        if factors.sun_exposure == "high":
            recs.append("Reduce sun exposure and always wear protective clothing outdoors")

        if factors.sleep_hours < 7:
            recs.append("Aim for 7-9 hours of sleep for optimal skin repair")

        if factors.water_intake == "low":
            recs.append("Drink 8+ glasses of water daily for hydrated, plump skin")

        if factors.stress_level == "high":
            recs.append("Practice stress management - meditation, exercise, or hobbies")

        if factors.diet_quality != "good":
            recs.append("Eat antioxidant-rich foods (berries, leafy greens, fatty fish)")

        recs.append("Use vitamin C serum in the morning for brightening and protection")

        return recs[:5]

    def _generate_share_text(
        self,
        current_age: int,
        skin_age: int,
        rate_category: str,
        max_years: int
    ) -> str:
        """Generate shareable text."""
        diff = current_age - skin_age

        if diff > 0:
            age_msg = f"My skin age is {skin_age} - that's {diff} years younger than me!"
            emoji = "🎉"
        elif diff < 0:
            age_msg = f"Time to step up my skincare game!"
            emoji = "💪"
        else:
            age_msg = f"My skin age matches my real age"
            emoji = "✨"

        return (
            f"{emoji} Just used the AI Skin Time Machine!\n\n"
            f"{age_msg}\n\n"
            f"Saw my face at age {current_age + max_years}... "
            f"{'Motivation to take care of my skin!' if diff < 3 else 'Looking good!'}\n\n"
            f"#SkinTimeMachine #DermAIPro #AntiAging #Skincare"
        )

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


# FastAPI Router
def create_time_machine_router():
    """Create FastAPI router for time machine endpoints."""
    from fastapi import APIRouter, HTTPException, File, UploadFile, Form
    from pydantic import BaseModel
    from typing import Optional

    router = APIRouter(prefix="/api/time-machine", tags=["Skin Time Machine"])
    time_machine = SkinTimeMachine()

    class KeyFactor(BaseModel):
        factor: str
        impact: str
        severity: str
        description: str

    class Projection(BaseModel):
        years: int
        future_age: int
        scenario: str
        scenario_name: str
        image: str
        description: str
        skin_age_at_time: int

    class TimeMachineResponse(BaseModel):
        original_image: str
        projections: List[Projection]
        current_age: int
        skin_age: int
        aging_rate: str
        key_factors: List[KeyFactor]
        recommendations: List[str]
        share_text: str
        ai_powered: bool = False

    @router.post("/generate", response_model=TimeMachineResponse)
    async def generate_time_machine(
        image: UploadFile = File(...),
        age: int = Form(...),
        smoking: bool = Form(False),
        sun_exposure: str = Form("moderate"),
        sleep_hours: float = Form(7),
        water_intake: str = Form("moderate"),
        uses_sunscreen: bool = Form(False),
        uses_retinol: bool = Form(False),
        stress_level: str = Form("moderate"),
        diet_quality: str = Form("moderate"),
        years: str = Form("10,20,30")
    ):
        """
        Generate age progression simulation using AI.

        Shows how skin will look in the future based on lifestyle and skincare choices.
        Powered by Fast-AgingGAN for realistic results.
        """
        try:
            image_data = await image.read()

            factors = AgingFactors(
                smoking=smoking,
                sun_exposure=sun_exposure,
                sleep_hours=sleep_hours,
                water_intake=water_intake,
                uses_sunscreen=uses_sunscreen,
                uses_retinol=uses_retinol,
                stress_level=stress_level,
                diet_quality=diet_quality
            )

            years_list = [int(y.strip()) for y in years.split(",")]

            result = time_machine.generate_progression(
                image_data=image_data,
                current_age=age,
                factors=factors,
                years_ahead=years_list
            )

            return TimeMachineResponse(
                original_image=result.original_image,
                projections=[
                    Projection(**p) for p in result.projections
                ],
                current_age=result.current_age,
                skin_age=result.skin_age,
                aging_rate=result.aging_rate,
                key_factors=[KeyFactor(**f) for f in result.key_factors],
                recommendations=result.recommendations,
                share_text=result.share_text,
                ai_powered=result.ai_powered
            )

        except Exception as e:
            logger.error(f"Time machine generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    @router.get("/scenarios")
    async def get_scenarios():
        """Get available aging scenarios."""
        return {
            "scenarios": [
                {
                    "id": "no_care",
                    "name": "Without Skincare",
                    "description": "Continue current habits without skincare routine"
                },
                {
                    "id": "basic_care",
                    "name": "Basic Care",
                    "description": "Cleanser, moisturizer, and daily SPF"
                },
                {
                    "id": "optimal_care",
                    "name": "Optimal Care",
                    "description": "Full anti-aging routine with retinol, vitamin C, and SPF"
                }
            ]
        }

    @router.get("/model-status")
    async def get_model_status():
        """Check if AI model is available."""
        model_available = False
        model_loaded = False

        if FAST_AGING_GAN_AVAILABLE:
            from fast_aging_gan import is_model_available as check_model
            model_available = check_model()

            if model_available:
                try:
                    model = time_machine._get_ai_model()
                    model_loaded = model is not None and model.is_loaded
                except:
                    pass

        return {
            "fast_aging_gan_available": FAST_AGING_GAN_AVAILABLE,
            "model_downloaded": model_available,
            "model_loaded": model_loaded,
            "ai_powered": model_loaded
        }

    @router.post("/download-model")
    async def download_ai_model():
        """Download the AI aging model."""
        if not FAST_AGING_GAN_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="Fast-AgingGAN module not available"
            )

        try:
            from fast_aging_gan import download_model
            success = download_model()
            if success:
                return {"status": "success", "message": "Model downloaded successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to download model")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

    return router
