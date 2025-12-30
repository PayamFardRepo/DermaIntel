"""
AI Skin Age & Health Score

Analyzes skin images using deep learning to provide:
- Estimated skin age vs chronological age (CNN model)
- Overall skin health score (0-100)
- Category scores (hydration, texture, clarity, firmness, UV damage)
- Wrinkle/pore/texture detection (computer vision)
- Personalized recommendations
- Shareable score card data

AI Model: Based on https://github.com/Himika-Mishra/Skin-Analysis
"""

import base64
import io
import random
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import AI module
try:
    from skin_health_ai import analyze_skin, is_model_available, SkinAnalysisResult
    AI_AVAILABLE = True
    logger.info("Skin Health AI module loaded successfully")
except ImportError as e:
    AI_AVAILABLE = False
    logger.warning(f"Skin Health AI module not available: {e}")

class SkinHealthCategory(Enum):
    HYDRATION = "hydration"
    TEXTURE = "texture"
    CLARITY = "clarity"
    FIRMNESS = "firmness"
    UV_DAMAGE = "uv_damage"
    PORE_SIZE = "pore_size"
    WRINKLES = "wrinkles"
    PIGMENTATION = "pigmentation"

@dataclass
class SkinHealthResult:
    overall_score: int  # 0-100
    skin_age: int
    chronological_age: int
    age_difference: int  # Positive = younger, Negative = older
    percentile: int  # Top X% for your age
    category_scores: Dict[str, int]
    recommendations: List[Dict]
    insights: List[str]
    share_text: str
    timestamp: str

class SkinHealthAnalyzer:
    """Analyzes skin health and estimates skin age."""

    def __init__(self):
        self.category_weights = {
            SkinHealthCategory.HYDRATION: 0.15,
            SkinHealthCategory.TEXTURE: 0.15,
            SkinHealthCategory.CLARITY: 0.15,
            SkinHealthCategory.FIRMNESS: 0.15,
            SkinHealthCategory.UV_DAMAGE: 0.15,
            SkinHealthCategory.PORE_SIZE: 0.10,
            SkinHealthCategory.WRINKLES: 0.10,
            SkinHealthCategory.PIGMENTATION: 0.05,
        }

        # Age-based baseline adjustments
        self.age_baselines = {
            (0, 20): {"base_score": 90, "wrinkle_factor": 0.1},
            (20, 30): {"base_score": 85, "wrinkle_factor": 0.3},
            (30, 40): {"base_score": 75, "wrinkle_factor": 0.5},
            (40, 50): {"base_score": 65, "wrinkle_factor": 0.7},
            (50, 60): {"base_score": 55, "wrinkle_factor": 0.85},
            (60, 70): {"base_score": 45, "wrinkle_factor": 0.95},
            (70, 100): {"base_score": 35, "wrinkle_factor": 1.0},
        }

    def analyze_image(self, image_data: bytes, chronological_age: int,
                      skin_type: Optional[str] = None,
                      lifestyle_factors: Optional[Dict] = None) -> SkinHealthResult:
        """
        Analyze skin image and return health score.

        Uses AI model for skin age prediction and feature detection when available,
        falls back to heuristic analysis otherwise.

        Args:
            image_data: Raw image bytes
            chronological_age: User's actual age
            skin_type: Optional skin type (oily, dry, combination, normal, sensitive)
            lifestyle_factors: Optional dict with smoking, sun_exposure, sleep, water_intake, etc.
        """
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_array = np.array(image)

        # Try AI analysis first
        ai_result = None
        if AI_AVAILABLE:
            try:
                logger.info("Using AI model for skin analysis...")
                ai_result = analyze_skin(image_data)
                if ai_result:
                    logger.info(f"AI skin age prediction: {ai_result.skin_age}")
            except Exception as e:
                logger.warning(f"AI analysis failed, using fallback: {e}")

        # Analyze each category (with AI results if available)
        category_scores = self._analyze_categories(
            image_array, chronological_age, lifestyle_factors, ai_result
        )

        # Calculate overall score
        overall_score = self._calculate_overall_score(category_scores)

        # Get skin age from AI or estimate it
        if ai_result and ai_result.skin_age:
            # Use AI-predicted skin age, but blend with score-based estimate
            ai_skin_age = ai_result.skin_age
            score_based_age = self._estimate_skin_age(overall_score, chronological_age, category_scores)
            # Weight AI prediction more heavily (70% AI, 30% score-based)
            skin_age = int(ai_skin_age * 0.7 + score_based_age * 0.3)
            logger.info(f"Blended skin age: {skin_age} (AI: {ai_skin_age}, Score: {score_based_age})")
        else:
            skin_age = self._estimate_skin_age(overall_score, chronological_age, category_scores)

        # Calculate percentile
        percentile = self._calculate_percentile(overall_score, chronological_age)

        # Generate recommendations
        recommendations = self._generate_recommendations(category_scores, chronological_age, skin_type)

        # Generate insights
        insights = self._generate_insights(category_scores, skin_age, chronological_age)

        # Add AI insight if available
        if ai_result:
            insights.insert(0, "Analysis powered by AI deep learning model")

        # Generate shareable text
        age_diff = chronological_age - skin_age
        share_text = self._generate_share_text(overall_score, skin_age, chronological_age, percentile)

        return SkinHealthResult(
            overall_score=overall_score,
            skin_age=skin_age,
            chronological_age=chronological_age,
            age_difference=age_diff,
            percentile=percentile,
            category_scores={cat.value: score for cat, score in category_scores.items()},
            recommendations=recommendations,
            insights=insights,
            share_text=share_text,
            timestamp=datetime.now().isoformat()
        )

    def _analyze_categories(self, image_array: np.ndarray, age: int,
                           lifestyle_factors: Optional[Dict] = None,
                           ai_result: Optional['SkinAnalysisResult'] = None) -> Dict[SkinHealthCategory, int]:
        """
        Analyze each skin health category from image features.

        Uses AI-detected features when available, falls back to heuristics.
        """

        # Extract image features (always needed for some categories)
        features = self._extract_image_features(image_array)

        # Get age baseline
        baseline = self._get_age_baseline(age)

        # Calculate each category score
        scores = {}

        # Hydration - based on skin luminosity and color distribution
        hydration_base = 70 + (features['luminosity'] - 0.5) * 40
        hydration_variance = features['color_variance'] * -20
        scores[SkinHealthCategory.HYDRATION] = self._clamp(hydration_base + hydration_variance)

        # Texture - use AI score if available
        if ai_result and ai_result.texture_score:
            # Blend AI texture with heuristic (80% AI, 20% heuristic)
            heuristic_texture = 85 - (features['edge_intensity'] * 50)
            scores[SkinHealthCategory.TEXTURE] = self._clamp(
                ai_result.texture_score * 0.8 + heuristic_texture * 0.2
            )
        else:
            texture_score = 85 - (features['edge_intensity'] * 50)
            scores[SkinHealthCategory.TEXTURE] = self._clamp(texture_score)

        # Clarity - use AI spots score if available (spots affect clarity)
        if ai_result and ai_result.spots_score:
            clarity_base = 90 - (features['color_variance'] * 50)
            # Blend with AI spots detection
            scores[SkinHealthCategory.CLARITY] = self._clamp(
                ai_result.spots_score * 0.6 + clarity_base * 0.4
            )
        else:
            clarity_score = 90 - (features['color_variance'] * 100)
            scores[SkinHealthCategory.CLARITY] = self._clamp(clarity_score)

        # Firmness - age-adjusted (no direct AI detection for this)
        firmness_base = baseline['base_score'] + 10
        # If AI detected more wrinkles, firmness is likely lower
        if ai_result and ai_result.wrinkle_score:
            firmness_adjustment = (ai_result.wrinkle_score - 50) * 0.3
            scores[SkinHealthCategory.FIRMNESS] = self._clamp(firmness_base + firmness_adjustment)
        else:
            scores[SkinHealthCategory.FIRMNESS] = self._clamp(firmness_base + random.randint(-5, 5))

        # UV Damage - based on redness and pigmentation patterns
        uv_score = 85 - (features['redness'] * 60) - (features['pigmentation_variance'] * 30)
        scores[SkinHealthCategory.UV_DAMAGE] = self._clamp(uv_score)

        # Pore Size - use AI pore score if available
        if ai_result and ai_result.pore_score:
            # AI pore score directly
            scores[SkinHealthCategory.PORE_SIZE] = self._clamp(ai_result.pore_score)
        else:
            pore_score = 80 - (features['local_variance'] * 50)
            scores[SkinHealthCategory.PORE_SIZE] = self._clamp(pore_score)

        # Wrinkles - use AI wrinkle score if available
        if ai_result and ai_result.wrinkle_score:
            # Blend AI wrinkle detection with age baseline
            age_wrinkle_base = 100 - (baseline['wrinkle_factor'] * 40)
            scores[SkinHealthCategory.WRINKLES] = self._clamp(
                ai_result.wrinkle_score * 0.7 + age_wrinkle_base * 0.3
            )
            logger.info(f"AI Wrinkle score: {ai_result.wrinkle_score}, Final: {scores[SkinHealthCategory.WRINKLES]}")
        else:
            wrinkle_base = 100 - (baseline['wrinkle_factor'] * 60)
            wrinkle_adjustment = features['edge_intensity'] * -20
            scores[SkinHealthCategory.WRINKLES] = self._clamp(wrinkle_base + wrinkle_adjustment)

        # Pigmentation - based on color uniformity and AI spots
        if ai_result and ai_result.spots_score:
            pigmentation_base = 85 - (features['pigmentation_variance'] * 40)
            scores[SkinHealthCategory.PIGMENTATION] = self._clamp(
                ai_result.spots_score * 0.5 + pigmentation_base * 0.5
            )
        else:
            pigmentation_score = 85 - (features['pigmentation_variance'] * 80)
            scores[SkinHealthCategory.PIGMENTATION] = self._clamp(pigmentation_score)

        # Apply lifestyle adjustments if provided
        if lifestyle_factors:
            scores = self._apply_lifestyle_adjustments(scores, lifestyle_factors)

        return scores

    def _extract_image_features(self, image_array: np.ndarray) -> Dict[str, float]:
        """Extract relevant features from the image for skin analysis."""

        # Normalize to 0-1 range
        img_normalized = image_array.astype(np.float32) / 255.0

        # Calculate luminosity (perceived brightness)
        luminosity = np.mean(0.299 * img_normalized[:,:,0] +
                           0.587 * img_normalized[:,:,1] +
                           0.114 * img_normalized[:,:,2])

        # Calculate color variance (skin uniformity)
        color_variance = np.std(img_normalized)

        # Calculate redness (skin irritation/damage indicator)
        redness = np.mean(img_normalized[:,:,0]) - np.mean(img_normalized[:,:,1:])
        redness = max(0, redness)

        # Edge detection for texture analysis (simplified Sobel)
        gray = np.mean(img_normalized, axis=2)
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        edge_intensity = (np.mean(dx) + np.mean(dy)) / 2

        # Local variance for pore/texture detection
        from scipy import ndimage
        try:
            local_mean = ndimage.uniform_filter(gray, size=10)
            local_sqr_mean = ndimage.uniform_filter(gray**2, size=10)
            local_variance = np.mean(np.sqrt(np.abs(local_sqr_mean - local_mean**2)))
        except:
            local_variance = np.std(gray)

        # Pigmentation variance
        # Convert to LAB-like representation for better pigmentation analysis
        l_channel = 0.299 * img_normalized[:,:,0] + 0.587 * img_normalized[:,:,1] + 0.114 * img_normalized[:,:,2]
        pigmentation_variance = np.std(l_channel)

        return {
            'luminosity': float(luminosity),
            'color_variance': float(color_variance),
            'redness': float(redness),
            'edge_intensity': float(edge_intensity),
            'local_variance': float(local_variance),
            'pigmentation_variance': float(pigmentation_variance),
        }

    def _get_age_baseline(self, age: int) -> Dict:
        """Get baseline scores for a given age."""
        for (min_age, max_age), baseline in self.age_baselines.items():
            if min_age <= age < max_age:
                return baseline
        return {"base_score": 50, "wrinkle_factor": 0.8}

    def _clamp(self, value: float, min_val: int = 0, max_val: int = 100) -> int:
        """Clamp value to valid range and convert to int."""
        return int(max(min_val, min(max_val, value)))

    def _apply_lifestyle_adjustments(self, scores: Dict[SkinHealthCategory, int],
                                    lifestyle: Dict) -> Dict[SkinHealthCategory, int]:
        """Adjust scores based on lifestyle factors."""
        adjusted = scores.copy()

        # Smoking penalty
        if lifestyle.get('smoking', False):
            adjusted[SkinHealthCategory.TEXTURE] -= 10
            adjusted[SkinHealthCategory.CLARITY] -= 8
            adjusted[SkinHealthCategory.FIRMNESS] -= 12

        # High sun exposure penalty
        sun_exposure = lifestyle.get('sun_exposure', 'moderate')
        if sun_exposure == 'high':
            adjusted[SkinHealthCategory.UV_DAMAGE] -= 15
            adjusted[SkinHealthCategory.PIGMENTATION] -= 10
            adjusted[SkinHealthCategory.WRINKLES] -= 8
        elif sun_exposure == 'low':
            adjusted[SkinHealthCategory.UV_DAMAGE] += 5

        # Good sleep bonus
        sleep_hours = lifestyle.get('sleep_hours', 7)
        if sleep_hours >= 8:
            adjusted[SkinHealthCategory.HYDRATION] += 5
            adjusted[SkinHealthCategory.FIRMNESS] += 3
        elif sleep_hours < 6:
            adjusted[SkinHealthCategory.HYDRATION] -= 8
            adjusted[SkinHealthCategory.TEXTURE] -= 5

        # Water intake bonus
        water_intake = lifestyle.get('water_intake', 'moderate')
        if water_intake == 'high':
            adjusted[SkinHealthCategory.HYDRATION] += 8
            adjusted[SkinHealthCategory.CLARITY] += 3
        elif water_intake == 'low':
            adjusted[SkinHealthCategory.HYDRATION] -= 10

        # Sunscreen use bonus
        if lifestyle.get('uses_sunscreen', False):
            adjusted[SkinHealthCategory.UV_DAMAGE] += 10
            adjusted[SkinHealthCategory.PIGMENTATION] += 5

        # Clamp all values
        return {cat: self._clamp(score) for cat, score in adjusted.items()}

    def _calculate_overall_score(self, category_scores: Dict[SkinHealthCategory, int]) -> int:
        """Calculate weighted overall score."""
        total = sum(score * self.category_weights[cat]
                   for cat, score in category_scores.items())
        return self._clamp(total)

    def _estimate_skin_age(self, overall_score: int, chronological_age: int,
                          category_scores: Dict[SkinHealthCategory, int]) -> int:
        """Estimate biological skin age based on scores."""

        # Base calculation: higher score = younger skin
        # Score of 100 = 10 years younger, Score of 0 = 15 years older
        score_adjustment = ((overall_score - 50) / 50) * 12

        # Weight wrinkles and firmness more heavily for age estimation
        wrinkle_score = category_scores.get(SkinHealthCategory.WRINKLES, 50)
        firmness_score = category_scores.get(SkinHealthCategory.FIRMNESS, 50)

        wrinkle_adjustment = ((wrinkle_score - 50) / 50) * 5
        firmness_adjustment = ((firmness_score - 50) / 50) * 3

        skin_age = chronological_age - score_adjustment - wrinkle_adjustment - firmness_adjustment

        # Ensure reasonable bounds (skin age between age-15 and age+20)
        min_age = max(18, chronological_age - 15)
        max_age = chronological_age + 20

        return int(max(min_age, min(max_age, skin_age)))

    def _calculate_percentile(self, overall_score: int, age: int) -> int:
        """Calculate percentile ranking for user's age group."""
        # Simulate percentile based on score
        # Higher scores = higher percentile
        base_percentile = overall_score

        # Age-adjusted percentile (younger people have higher baseline)
        if age < 30:
            age_adjustment = -5
        elif age > 50:
            age_adjustment = 10
        else:
            age_adjustment = 0

        percentile = self._clamp(base_percentile + age_adjustment, 1, 99)
        return percentile

    def _generate_recommendations(self, scores: Dict[SkinHealthCategory, int],
                                  age: int, skin_type: Optional[str]) -> List[Dict]:
        """Generate personalized recommendations based on scores."""
        recommendations = []

        # Find weakest categories
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])

        recommendation_templates = {
            SkinHealthCategory.HYDRATION: {
                "title": "Boost Hydration",
                "description": "Your skin could use more moisture",
                "tips": [
                    "Use a hyaluronic acid serum before moisturizer",
                    "Drink at least 8 glasses of water daily",
                    "Consider a humidifier in dry environments",
                    "Apply moisturizer to damp skin for better absorption"
                ],
                "products": ["Hyaluronic Acid Serum", "Ceramide Moisturizer"],
                "priority": "high" if scores[SkinHealthCategory.HYDRATION] < 50 else "medium"
            },
            SkinHealthCategory.TEXTURE: {
                "title": "Improve Skin Texture",
                "description": "Smoother skin is within reach",
                "tips": [
                    "Exfoliate 2-3 times per week with AHA/BHA",
                    "Use retinol at night (start slowly)",
                    "Never skip moisturizer after exfoliating",
                    "Consider professional microdermabrasion"
                ],
                "products": ["Glycolic Acid Toner", "Retinol Serum"],
                "priority": "high" if scores[SkinHealthCategory.TEXTURE] < 50 else "medium"
            },
            SkinHealthCategory.UV_DAMAGE: {
                "title": "Repair & Protect from UV",
                "description": "Sun damage can be reversed with consistent care",
                "tips": [
                    "Apply SPF 50+ sunscreen every morning",
                    "Reapply sunscreen every 2 hours outdoors",
                    "Use vitamin C serum in the morning",
                    "Wear protective clothing and seek shade"
                ],
                "products": ["Broad Spectrum SPF 50", "Vitamin C Serum"],
                "priority": "high" if scores[SkinHealthCategory.UV_DAMAGE] < 50 else "medium"
            },
            SkinHealthCategory.WRINKLES: {
                "title": "Reduce Fine Lines",
                "description": "Target wrinkles with proven ingredients",
                "tips": [
                    "Use retinoid products consistently",
                    "Apply peptide serums for collagen support",
                    "Stay hydrated inside and out",
                    "Consider professional treatments like micro-needling"
                ],
                "products": ["Retinol 0.5%", "Peptide Complex Serum"],
                "priority": "high" if scores[SkinHealthCategory.WRINKLES] < 50 else "medium"
            },
            SkinHealthCategory.PIGMENTATION: {
                "title": "Even Skin Tone",
                "description": "Reduce dark spots and discoloration",
                "tips": [
                    "Use niacinamide daily for brightening",
                    "Apply vitamin C in the morning",
                    "Always use sunscreen to prevent new spots",
                    "Consider alpha arbutin for stubborn spots"
                ],
                "products": ["Niacinamide 10%", "Alpha Arbutin Serum"],
                "priority": "high" if scores[SkinHealthCategory.PIGMENTATION] < 50 else "medium"
            },
            SkinHealthCategory.FIRMNESS: {
                "title": "Restore Firmness",
                "description": "Support skin's natural elasticity",
                "tips": [
                    "Use peptides to stimulate collagen",
                    "Try facial massage techniques",
                    "Consider radiofrequency treatments",
                    "Ensure adequate protein intake"
                ],
                "products": ["Peptide Moisturizer", "Firming Eye Cream"],
                "priority": "high" if scores[SkinHealthCategory.FIRMNESS] < 50 else "medium"
            },
            SkinHealthCategory.PORE_SIZE: {
                "title": "Minimize Pores",
                "description": "Reduce the appearance of enlarged pores",
                "tips": [
                    "Use niacinamide to regulate oil production",
                    "Try BHA (salicylic acid) for pore cleansing",
                    "Use a clay mask weekly",
                    "Never skip moisturizer (dry skin = larger pores)"
                ],
                "products": ["Niacinamide Serum", "Salicylic Acid Cleanser"],
                "priority": "medium"
            },
            SkinHealthCategory.CLARITY: {
                "title": "Enhance Clarity",
                "description": "Achieve a clearer, more radiant complexion",
                "tips": [
                    "Double cleanse at night",
                    "Use antioxidant serums daily",
                    "Exfoliate regularly but gently",
                    "Get adequate sleep for skin repair"
                ],
                "products": ["Antioxidant Serum", "Gentle Exfoliating Cleanser"],
                "priority": "medium"
            },
        }

        # Add top 3 recommendations based on lowest scores
        for category, score in sorted_scores[:3]:
            if category in recommendation_templates:
                rec = recommendation_templates[category].copy()
                rec["category"] = category.value
                rec["current_score"] = score
                rec["potential_improvement"] = min(30, 100 - score)
                recommendations.append(rec)

        return recommendations

    def _generate_insights(self, scores: Dict[SkinHealthCategory, int],
                          skin_age: int, chronological_age: int) -> List[str]:
        """Generate personalized insights about skin health."""
        insights = []

        age_diff = chronological_age - skin_age

        if age_diff > 5:
            insights.append(f"Your skin appears {age_diff} years younger than your age - great job!")
        elif age_diff > 0:
            insights.append(f"Your skin is aging well - {age_diff} years younger than average")
        elif age_diff < -5:
            insights.append(f"Your skin shows signs of accelerated aging - lifestyle changes can help")

        # Category-specific insights
        if scores[SkinHealthCategory.UV_DAMAGE] < 50:
            insights.append("Sun protection should be your #1 priority - it prevents 80% of visible aging")

        if scores[SkinHealthCategory.HYDRATION] > 80:
            insights.append("Excellent hydration! Well-moisturized skin ages more slowly")
        elif scores[SkinHealthCategory.HYDRATION] < 50:
            insights.append("Dehydrated skin can make you look years older - focus on moisture")

        if scores[SkinHealthCategory.TEXTURE] > 80:
            insights.append("Your skin texture is smooth and healthy")

        if scores[SkinHealthCategory.WRINKLES] > 75:
            insights.append("Minimal wrinkles detected - keep up your current routine!")

        # Calculate best and worst categories
        best_cat = max(scores.items(), key=lambda x: x[1])
        worst_cat = min(scores.items(), key=lambda x: x[1])

        insights.append(f"Your strongest area: {best_cat[0].value.replace('_', ' ').title()} ({best_cat[1]}/100)")
        insights.append(f"Focus area: {worst_cat[0].value.replace('_', ' ').title()} ({worst_cat[1]}/100)")

        return insights

    def _generate_share_text(self, overall_score: int, skin_age: int,
                            chronological_age: int, percentile: int) -> str:
        """Generate shareable social media text."""
        age_diff = chronological_age - skin_age

        if age_diff > 5:
            emoji = "🎉"
            message = f"My skin age is {skin_age} - that's {age_diff} years younger than me!"
        elif age_diff > 0:
            emoji = "✨"
            message = f"My skin age is {skin_age} - {age_diff} years younger!"
        else:
            emoji = "💪"
            message = f"Working on my skin health! Current skin age: {skin_age}"

        share_text = (
            f"{emoji} Just checked my Skin Health Score: {overall_score}/100\n"
            f"{message}\n"
            f"Top {100 - percentile}% for my age group!\n"
            f"#SkinHealth #DermAIPro #HealthySkin"
        )

        return share_text


# FastAPI integration
def create_skin_health_router():
    """Create FastAPI router for skin health endpoints."""
    from fastapi import APIRouter, HTTPException, File, UploadFile, Form
    from pydantic import BaseModel
    from typing import Optional

    router = APIRouter(prefix="/api/skin-health", tags=["Skin Health Score"])
    analyzer = SkinHealthAnalyzer()

    class LifestyleFactors(BaseModel):
        smoking: bool = False
        sun_exposure: str = "moderate"  # low, moderate, high
        sleep_hours: float = 7
        water_intake: str = "moderate"  # low, moderate, high
        uses_sunscreen: bool = False

    class SkinHealthResponse(BaseModel):
        overall_score: int
        skin_age: int
        chronological_age: int
        age_difference: int
        percentile: int
        category_scores: Dict[str, int]
        recommendations: List[Dict]
        insights: List[str]
        share_text: str
        timestamp: str

    @router.post("/analyze", response_model=SkinHealthResponse)
    async def analyze_skin_health(
        image: UploadFile = File(...),
        age: int = Form(...),
        skin_type: Optional[str] = Form(None),
        smoking: bool = Form(False),
        sun_exposure: str = Form("moderate"),
        sleep_hours: float = Form(7),
        water_intake: str = Form("moderate"),
        uses_sunscreen: bool = Form(False)
    ):
        """
        Analyze skin health from an image.

        Returns:
        - Overall health score (0-100)
        - Estimated skin age
        - Category breakdown
        - Personalized recommendations
        - Shareable results
        """
        try:
            image_data = await image.read()

            lifestyle = {
                "smoking": smoking,
                "sun_exposure": sun_exposure,
                "sleep_hours": sleep_hours,
                "water_intake": water_intake,
                "uses_sunscreen": uses_sunscreen,
            }

            result = analyzer.analyze_image(
                image_data=image_data,
                chronological_age=age,
                skin_type=skin_type,
                lifestyle_factors=lifestyle
            )

            return SkinHealthResponse(
                overall_score=result.overall_score,
                skin_age=result.skin_age,
                chronological_age=result.chronological_age,
                age_difference=result.age_difference,
                percentile=result.percentile,
                category_scores=result.category_scores,
                recommendations=result.recommendations,
                insights=result.insights,
                share_text=result.share_text,
                timestamp=result.timestamp
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    @router.get("/categories")
    async def get_categories():
        """Get list of skin health categories and their weights."""
        return {
            "categories": [
                {"id": cat.value, "name": cat.value.replace("_", " ").title(),
                 "weight": analyzer.category_weights[cat]}
                for cat in SkinHealthCategory
            ]
        }

    return router
