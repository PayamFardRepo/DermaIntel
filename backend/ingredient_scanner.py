"""
Ingredient Scanner - Product Safety Analyzer

Scan skincare product barcodes or ingredient lists to:
- Check compatibility with user's skin type & conditions
- Flag allergens, irritants, comedogenic ingredients
- Rate effectiveness for specific goals
- Suggest better alternatives
"""

import re
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class SkinType(Enum):
    OILY = "oily"
    DRY = "dry"
    COMBINATION = "combination"
    NORMAL = "normal"
    SENSITIVE = "sensitive"


class SkinConcern(Enum):
    ACNE = "acne"
    AGING = "aging"
    HYPERPIGMENTATION = "hyperpigmentation"
    REDNESS = "redness"
    DRYNESS = "dryness"
    OILINESS = "oiliness"
    SENSITIVITY = "sensitivity"
    DULLNESS = "dullness"
    DARK_SPOTS = "dark_spots"
    FINE_LINES = "fine_lines"
    LARGE_PORES = "large_pores"
    UNEVEN_TEXTURE = "uneven_texture"


class IngredientRating(Enum):
    EXCELLENT = "excellent"  # Great for your skin
    GOOD = "good"  # Beneficial
    NEUTRAL = "neutral"  # Neither good nor bad
    CAUTION = "caution"  # May cause issues
    AVOID = "avoid"  # Should avoid for your skin


@dataclass
class IngredientAnalysis:
    name: str
    category: str
    rating: str
    rating_reason: str
    comedogenic_rating: int  # 0-5 scale
    irritation_potential: str  # low, medium, high
    benefits: List[str]
    concerns: List[str]
    good_for: List[str]  # Skin types/concerns
    avoid_if: List[str]  # Conditions to avoid


@dataclass
class ProductAnalysis:
    product_name: Optional[str]
    overall_rating: str  # A, B, C, D, F
    overall_score: int  # 0-100
    compatibility_score: int  # 0-100 for user's skin
    ingredient_count: int
    ingredients: List[IngredientAnalysis]
    summary: str
    pros: List[str]
    cons: List[str]
    warnings: List[str]
    alternatives: List[Dict]
    effectiveness_for_goals: Dict[str, int]  # goal -> score
    timestamp: str


class IngredientDatabase:
    """Database of skincare ingredients and their properties."""

    def __init__(self):
        # Comprehensive ingredient database
        self.ingredients = {
            # RETINOIDS
            "retinol": {
                "category": "Retinoid",
                "comedogenic": 0,
                "irritation": "medium",
                "benefits": ["Anti-aging", "Reduces wrinkles", "Increases cell turnover", "Fades dark spots"],
                "concerns": ["Can cause irritation", "Sun sensitivity", "Not for pregnant women"],
                "good_for": [SkinConcern.AGING, SkinConcern.FINE_LINES, SkinConcern.DARK_SPOTS, SkinConcern.UNEVEN_TEXTURE],
                "avoid_if": ["pregnancy", "sensitive skin beginners", "rosacea flare"],
            },
            "tretinoin": {
                "category": "Retinoid (Rx)",
                "comedogenic": 0,
                "irritation": "high",
                "benefits": ["Strongest anti-aging", "Treats acne", "Increases collagen"],
                "concerns": ["Prescription only", "High irritation potential", "Peeling and redness"],
                "good_for": [SkinConcern.AGING, SkinConcern.ACNE, SkinConcern.FINE_LINES],
                "avoid_if": ["pregnancy", "sensitive skin", "eczema"],
            },
            "retinaldehyde": {
                "category": "Retinoid",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Gentler than retinol", "Anti-aging", "Less irritating"],
                "concerns": ["Less potent than tretinoin"],
                "good_for": [SkinConcern.AGING, SkinConcern.SENSITIVITY],
                "avoid_if": ["pregnancy"],
            },

            # VITAMIN C
            "ascorbic acid": {
                "category": "Vitamin C",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Brightening", "Antioxidant", "Boosts collagen", "Fades dark spots"],
                "concerns": ["Can oxidize quickly", "May tingle on sensitive skin"],
                "good_for": [SkinConcern.DULLNESS, SkinConcern.DARK_SPOTS, SkinConcern.AGING],
                "avoid_if": [],
            },
            "sodium ascorbyl phosphate": {
                "category": "Vitamin C Derivative",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Stable vitamin C", "Brightening", "Gentle"],
                "concerns": ["Less potent than L-ascorbic acid"],
                "good_for": [SkinConcern.DULLNESS, SkinConcern.SENSITIVITY],
                "avoid_if": [],
            },

            # HYDROXY ACIDS
            "salicylic acid": {
                "category": "BHA",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Unclogs pores", "Treats acne", "Exfoliates", "Oil-soluble"],
                "concerns": ["Can be drying", "Not for very dry skin"],
                "good_for": [SkinConcern.ACNE, SkinConcern.LARGE_PORES, SkinConcern.OILINESS],
                "avoid_if": ["aspirin allergy", "very dry skin"],
            },
            "glycolic acid": {
                "category": "AHA",
                "comedogenic": 0,
                "irritation": "medium",
                "benefits": ["Exfoliates", "Brightens", "Smooths texture", "Stimulates collagen"],
                "concerns": ["Sun sensitivity", "Can irritate sensitive skin"],
                "good_for": [SkinConcern.DULLNESS, SkinConcern.UNEVEN_TEXTURE, SkinConcern.AGING],
                "avoid_if": ["sensitive skin", "active rosacea"],
            },
            "lactic acid": {
                "category": "AHA",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Gentle exfoliation", "Hydrating", "Brightening"],
                "concerns": ["Mild sun sensitivity"],
                "good_for": [SkinConcern.DRYNESS, SkinConcern.DULLNESS, SkinConcern.SENSITIVITY],
                "avoid_if": [],
            },
            "mandelic acid": {
                "category": "AHA",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Very gentle AHA", "Good for dark skin", "Antibacterial"],
                "concerns": [],
                "good_for": [SkinConcern.ACNE, SkinConcern.HYPERPIGMENTATION, SkinConcern.SENSITIVITY],
                "avoid_if": [],
            },

            # HYDRATING
            "hyaluronic acid": {
                "category": "Humectant",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Intense hydration", "Plumps skin", "Reduces fine lines"],
                "concerns": ["Can draw moisture from skin in dry climates if not sealed"],
                "good_for": [SkinConcern.DRYNESS, SkinConcern.FINE_LINES],
                "avoid_if": [],
            },
            "glycerin": {
                "category": "Humectant",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Hydrating", "Skin barrier support", "Non-irritating"],
                "concerns": [],
                "good_for": [SkinConcern.DRYNESS],
                "avoid_if": [],
            },
            "squalane": {
                "category": "Emollient",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Moisturizing", "Non-greasy", "Antioxidant", "Mimics skin's natural oils"],
                "concerns": [],
                "good_for": [SkinConcern.DRYNESS, SkinConcern.SENSITIVITY],
                "avoid_if": [],
            },
            "ceramides": {
                "category": "Lipid",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Repairs skin barrier", "Locks in moisture", "Reduces sensitivity"],
                "concerns": [],
                "good_for": [SkinConcern.DRYNESS, SkinConcern.SENSITIVITY, SkinConcern.REDNESS],
                "avoid_if": [],
            },

            # NIACINAMIDE
            "niacinamide": {
                "category": "Vitamin B3",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Minimizes pores", "Brightens", "Reduces oil", "Strengthens barrier"],
                "concerns": ["High concentrations may cause flushing"],
                "good_for": [SkinConcern.LARGE_PORES, SkinConcern.OILINESS, SkinConcern.DULLNESS, SkinConcern.ACNE],
                "avoid_if": [],
            },

            # PROBLEMATIC INGREDIENTS
            "fragrance": {
                "category": "Fragrance",
                "comedogenic": 0,
                "irritation": "high",
                "benefits": ["Pleasant scent"],
                "concerns": ["Common allergen", "Can cause irritation", "Photosensitivity"],
                "good_for": [],
                "avoid_if": ["sensitive skin", "eczema", "rosacea", "allergies"],
            },
            "parfum": {
                "category": "Fragrance",
                "comedogenic": 0,
                "irritation": "high",
                "benefits": ["Pleasant scent"],
                "concerns": ["Common allergen", "Can cause irritation"],
                "good_for": [],
                "avoid_if": ["sensitive skin", "eczema", "rosacea"],
            },
            "alcohol denat": {
                "category": "Alcohol",
                "comedogenic": 0,
                "irritation": "high",
                "benefits": ["Quick absorption", "Mattifying"],
                "concerns": ["Very drying", "Damages skin barrier", "Increases oil long-term"],
                "good_for": [],
                "avoid_if": ["dry skin", "sensitive skin", "aging concerns"],
            },
            "isopropyl alcohol": {
                "category": "Alcohol",
                "comedogenic": 0,
                "irritation": "high",
                "benefits": [],
                "concerns": ["Extremely drying", "Irritating"],
                "good_for": [],
                "avoid_if": ["all skin types for leave-on products"],
            },

            # COMEDOGENIC INGREDIENTS
            "coconut oil": {
                "category": "Oil",
                "comedogenic": 4,
                "irritation": "low",
                "benefits": ["Very moisturizing", "Antibacterial"],
                "concerns": ["Highly comedogenic", "Can clog pores"],
                "good_for": [SkinConcern.DRYNESS],
                "avoid_if": ["acne-prone skin", "oily skin"],
            },
            "isopropyl myristate": {
                "category": "Emollient",
                "comedogenic": 5,
                "irritation": "low",
                "benefits": ["Smooth texture"],
                "concerns": ["Highly comedogenic"],
                "good_for": [],
                "avoid_if": ["acne-prone skin"],
            },
            "isopropyl palmitate": {
                "category": "Emollient",
                "comedogenic": 4,
                "irritation": "low",
                "benefits": ["Emollient"],
                "concerns": ["Comedogenic"],
                "good_for": [],
                "avoid_if": ["acne-prone skin"],
            },

            # SUNSCREEN FILTERS
            "zinc oxide": {
                "category": "Mineral Sunscreen",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Broad spectrum UV protection", "Gentle", "Anti-inflammatory"],
                "concerns": ["Can leave white cast"],
                "good_for": [SkinConcern.SENSITIVITY, SkinConcern.REDNESS],
                "avoid_if": [],
            },
            "titanium dioxide": {
                "category": "Mineral Sunscreen",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["UV protection", "Gentle"],
                "concerns": ["White cast", "Less broad spectrum than zinc"],
                "good_for": [SkinConcern.SENSITIVITY],
                "avoid_if": [],
            },
            "avobenzone": {
                "category": "Chemical Sunscreen",
                "comedogenic": 0,
                "irritation": "medium",
                "benefits": ["Excellent UVA protection"],
                "concerns": ["Can degrade in sun", "May irritate sensitive skin"],
                "good_for": [],
                "avoid_if": ["very sensitive skin"],
            },

            # PEPTIDES
            "peptides": {
                "category": "Peptide",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Stimulates collagen", "Firms skin", "Reduces wrinkles"],
                "concerns": [],
                "good_for": [SkinConcern.AGING, SkinConcern.FINE_LINES],
                "avoid_if": [],
            },
            "matrixyl": {
                "category": "Peptide",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Anti-wrinkle", "Collagen boosting"],
                "concerns": [],
                "good_for": [SkinConcern.AGING, SkinConcern.FINE_LINES],
                "avoid_if": [],
            },

            # BRIGHTENING
            "alpha arbutin": {
                "category": "Brightening Agent",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Fades dark spots", "Gentle", "Evens skin tone"],
                "concerns": [],
                "good_for": [SkinConcern.DARK_SPOTS, SkinConcern.HYPERPIGMENTATION],
                "avoid_if": [],
            },
            "kojic acid": {
                "category": "Brightening Agent",
                "comedogenic": 0,
                "irritation": "medium",
                "benefits": ["Fades hyperpigmentation", "Antioxidant"],
                "concerns": ["Can cause sensitivity", "May irritate"],
                "good_for": [SkinConcern.DARK_SPOTS, SkinConcern.HYPERPIGMENTATION],
                "avoid_if": ["sensitive skin"],
            },
            "azelaic acid": {
                "category": "Brightening/Anti-acne",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Brightens", "Anti-acne", "Anti-inflammatory", "Safe for pregnancy"],
                "concerns": [],
                "good_for": [SkinConcern.ACNE, SkinConcern.HYPERPIGMENTATION, SkinConcern.REDNESS],
                "avoid_if": [],
            },

            # SOOTHING
            "centella asiatica": {
                "category": "Soothing",
                "comedogenic": 1,
                "irritation": "low",
                "benefits": ["Calming", "Healing", "Anti-inflammatory", "Strengthens skin"],
                "concerns": [],
                "good_for": [SkinConcern.SENSITIVITY, SkinConcern.REDNESS, SkinConcern.ACNE],
                "avoid_if": [],
            },
            "aloe vera": {
                "category": "Soothing",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Soothing", "Hydrating", "Healing"],
                "concerns": [],
                "good_for": [SkinConcern.SENSITIVITY, SkinConcern.REDNESS],
                "avoid_if": [],
            },
            "allantoin": {
                "category": "Soothing",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Healing", "Soothing", "Moisturizing"],
                "concerns": [],
                "good_for": [SkinConcern.SENSITIVITY],
                "avoid_if": [],
            },

            # ANTIOXIDANTS
            "vitamin e": {
                "category": "Antioxidant",
                "comedogenic": 2,
                "irritation": "low",
                "benefits": ["Antioxidant", "Moisturizing", "Healing"],
                "concerns": ["Mildly comedogenic for some"],
                "good_for": [SkinConcern.DRYNESS, SkinConcern.AGING],
                "avoid_if": ["very acne-prone skin"],
            },
            "tocopherol": {
                "category": "Vitamin E",
                "comedogenic": 2,
                "irritation": "low",
                "benefits": ["Antioxidant", "Stabilizes other ingredients"],
                "concerns": [],
                "good_for": [SkinConcern.AGING],
                "avoid_if": [],
            },
            "green tea extract": {
                "category": "Antioxidant",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Antioxidant", "Anti-inflammatory", "Soothing"],
                "concerns": [],
                "good_for": [SkinConcern.AGING, SkinConcern.REDNESS],
                "avoid_if": [],
            },

            # PRESERVATIVES
            "phenoxyethanol": {
                "category": "Preservative",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Safe preservative"],
                "concerns": ["Rare sensitivity"],
                "good_for": [],
                "avoid_if": [],
            },
            "parabens": {
                "category": "Preservative",
                "comedogenic": 0,
                "irritation": "low",
                "benefits": ["Effective preservative"],
                "concerns": ["Controversial but deemed safe by FDA", "Some prefer to avoid"],
                "good_for": [],
                "avoid_if": ["personal preference"],
            },
        }

        # Common ingredient name variations
        self.aliases = {
            "vitamin c": "ascorbic acid",
            "l-ascorbic acid": "ascorbic acid",
            "ha": "hyaluronic acid",
            "sodium hyaluronate": "hyaluronic acid",
            "vitamin b3": "niacinamide",
            "nicotinamide": "niacinamide",
            "retinoid": "retinol",
            "bha": "salicylic acid",
            "cica": "centella asiatica",
            "madecassoside": "centella asiatica",
            "tocopheryl acetate": "vitamin e",
            "parfum": "fragrance",
            "ethanol": "alcohol denat",
        }

    def lookup(self, ingredient_name: str) -> Optional[Dict]:
        """Look up an ingredient by name."""
        name = ingredient_name.lower().strip()

        # Check direct match
        if name in self.ingredients:
            return self.ingredients[name]

        # Check aliases
        if name in self.aliases:
            return self.ingredients.get(self.aliases[name])

        # Partial match
        for key in self.ingredients:
            if key in name or name in key:
                return self.ingredients[key]

        return None


class IngredientScanner:
    """Scans and analyzes product ingredients."""

    def __init__(self):
        self.db = IngredientDatabase()

    def analyze_ingredients(
        self,
        ingredients_text: str,
        skin_type: SkinType,
        concerns: List[SkinConcern],
        allergies: List[str] = None,
        product_name: str = None
    ) -> ProductAnalysis:
        """Analyze a list of ingredients."""

        # Parse ingredients
        ingredient_list = self._parse_ingredients(ingredients_text)

        # Analyze each ingredient
        analyzed = []
        for ing_name in ingredient_list:
            analysis = self._analyze_ingredient(ing_name, skin_type, concerns, allergies or [])
            if analysis:
                analyzed.append(analysis)

        # Calculate scores
        overall_score = self._calculate_overall_score(analyzed)
        compatibility_score = self._calculate_compatibility(analyzed, skin_type, concerns)
        overall_rating = self._score_to_rating(overall_score)

        # Generate summary
        summary = self._generate_summary(analyzed, skin_type, concerns)

        # Extract pros and cons
        pros = self._extract_pros(analyzed)
        cons = self._extract_cons(analyzed)
        warnings = self._extract_warnings(analyzed, skin_type, concerns, allergies or [])

        # Calculate effectiveness for goals
        effectiveness = self._calculate_effectiveness(analyzed, concerns)

        # Suggest alternatives if needed
        alternatives = self._suggest_alternatives(analyzed, skin_type, concerns, overall_score)

        return ProductAnalysis(
            product_name=product_name,
            overall_rating=overall_rating,
            overall_score=overall_score,
            compatibility_score=compatibility_score,
            ingredient_count=len(analyzed),
            ingredients=analyzed,
            summary=summary,
            pros=pros,
            cons=cons,
            warnings=warnings,
            alternatives=alternatives,
            effectiveness_for_goals=effectiveness,
            timestamp=datetime.now().isoformat()
        )

    def _parse_ingredients(self, text: str) -> List[str]:
        """Parse ingredient list text into individual ingredients."""
        # Remove common prefixes
        text = re.sub(r'^ingredients?:?\s*', '', text, flags=re.IGNORECASE)

        # Split by comma, handling parentheses
        ingredients = []
        current = ""
        paren_depth = 0

        for char in text:
            if char == '(':
                paren_depth += 1
                current += char
            elif char == ')':
                paren_depth -= 1
                current += char
            elif char == ',' and paren_depth == 0:
                if current.strip():
                    ingredients.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            ingredients.append(current.strip())

        return ingredients

    def _analyze_ingredient(
        self,
        name: str,
        skin_type: SkinType,
        concerns: List[SkinConcern],
        allergies: List[str]
    ) -> Optional[IngredientAnalysis]:
        """Analyze a single ingredient."""

        info = self.db.lookup(name)

        if not info:
            # Unknown ingredient
            return IngredientAnalysis(
                name=name,
                category="Unknown",
                rating=IngredientRating.NEUTRAL.value,
                rating_reason="Not in database - likely a filler or less common ingredient",
                comedogenic_rating=0,
                irritation_potential="unknown",
                benefits=[],
                concerns=["Unknown ingredient - research recommended"],
                good_for=[],
                avoid_if=[]
            )

        # Determine rating based on skin type and concerns
        rating, reason = self._rate_for_user(info, skin_type, concerns, allergies)

        return IngredientAnalysis(
            name=name,
            category=info["category"],
            rating=rating.value,
            rating_reason=reason,
            comedogenic_rating=info["comedogenic"],
            irritation_potential=info["irritation"],
            benefits=info["benefits"],
            concerns=info["concerns"],
            good_for=[c.value if isinstance(c, SkinConcern) else c for c in info["good_for"]],
            avoid_if=info["avoid_if"]
        )

    def _rate_for_user(
        self,
        info: Dict,
        skin_type: SkinType,
        concerns: List[SkinConcern],
        allergies: List[str]
    ) -> tuple:
        """Rate an ingredient for specific user."""

        rating = IngredientRating.NEUTRAL
        reasons = []

        # Check if good for user's concerns
        concern_match = False
        for concern in concerns:
            if concern in info["good_for"]:
                concern_match = True
                reasons.append(f"Good for {concern.value}")

        if concern_match:
            rating = IngredientRating.GOOD

        # Check comedogenic for acne-prone
        if SkinConcern.ACNE in concerns or skin_type == SkinType.OILY:
            if info["comedogenic"] >= 3:
                rating = IngredientRating.CAUTION
                reasons.append(f"Comedogenic rating: {info['comedogenic']}/5")
            elif info["comedogenic"] >= 4:
                rating = IngredientRating.AVOID
                reasons.append(f"Highly comedogenic: {info['comedogenic']}/5")

        # Check irritation for sensitive skin
        if skin_type == SkinType.SENSITIVE or SkinConcern.SENSITIVITY in concerns:
            if info["irritation"] == "high":
                rating = IngredientRating.AVOID
                reasons.append("High irritation potential for sensitive skin")
            elif info["irritation"] == "medium":
                rating = IngredientRating.CAUTION
                reasons.append("May irritate sensitive skin")

        # Check avoid_if conditions
        for condition in info["avoid_if"]:
            if skin_type.value in condition.lower() or any(c.value in condition.lower() for c in concerns):
                rating = IngredientRating.CAUTION
                reasons.append(f"May not be suitable: {condition}")

        # Check for allergies
        for allergy in allergies:
            if allergy.lower() in info.get("category", "").lower():
                rating = IngredientRating.AVOID
                reasons.append(f"Matches allergy: {allergy}")

        # Upgrade to excellent if very beneficial
        if rating == IngredientRating.GOOD and len(info["benefits"]) >= 3:
            rating = IngredientRating.EXCELLENT
            reasons.insert(0, "Highly beneficial ingredient")

        if not reasons:
            reasons = ["Standard ingredient"]

        return rating, "; ".join(reasons)

    def _calculate_overall_score(self, ingredients: List[IngredientAnalysis]) -> int:
        """Calculate overall product score."""
        if not ingredients:
            return 50

        scores = []
        weights = []

        for i, ing in enumerate(ingredients):
            # Earlier ingredients are typically in higher concentrations
            weight = max(0.5, 1.0 - (i * 0.05))
            weights.append(weight)

            if ing.rating == IngredientRating.EXCELLENT.value:
                scores.append(95)
            elif ing.rating == IngredientRating.GOOD.value:
                scores.append(80)
            elif ing.rating == IngredientRating.NEUTRAL.value:
                scores.append(60)
            elif ing.rating == IngredientRating.CAUTION.value:
                scores.append(40)
            else:  # AVOID
                scores.append(20)

        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        return int(weighted_sum / total_weight)

    def _calculate_compatibility(
        self,
        ingredients: List[IngredientAnalysis],
        skin_type: SkinType,
        concerns: List[SkinConcern]
    ) -> int:
        """Calculate compatibility with user's skin."""
        if not ingredients:
            return 50

        avoid_count = sum(1 for i in ingredients if i.rating == IngredientRating.AVOID.value)
        caution_count = sum(1 for i in ingredients if i.rating == IngredientRating.CAUTION.value)
        good_count = sum(1 for i in ingredients if i.rating in [IngredientRating.GOOD.value, IngredientRating.EXCELLENT.value])

        # Start at 70
        score = 70

        # Deduct for problematic ingredients
        score -= avoid_count * 15
        score -= caution_count * 5

        # Add for beneficial ingredients
        score += good_count * 3

        return max(0, min(100, score))

    def _score_to_rating(self, score: int) -> str:
        """Convert score to letter rating."""
        if score >= 85:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 55:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"

    def _generate_summary(
        self,
        ingredients: List[IngredientAnalysis],
        skin_type: SkinType,
        concerns: List[SkinConcern]
    ) -> str:
        """Generate product summary."""
        excellent = [i.name for i in ingredients if i.rating == IngredientRating.EXCELLENT.value]
        avoid = [i.name for i in ingredients if i.rating == IngredientRating.AVOID.value]

        parts = []

        if excellent:
            parts.append(f"Contains excellent ingredients: {', '.join(excellent[:3])}")

        if avoid:
            parts.append(f"Contains ingredients to avoid for your skin: {', '.join(avoid[:2])}")
        elif not avoid and len(excellent) > 0:
            parts.append(f"Good compatibility with {skin_type.value} skin")

        if not parts:
            parts.append("Average product with no standout ingredients for your profile")

        return ". ".join(parts) + "."

    def _extract_pros(self, ingredients: List[IngredientAnalysis]) -> List[str]:
        """Extract product pros."""
        pros = set()
        for ing in ingredients:
            if ing.rating in [IngredientRating.EXCELLENT.value, IngredientRating.GOOD.value]:
                for benefit in ing.benefits[:2]:
                    pros.add(benefit)
        return list(pros)[:5]

    def _extract_cons(self, ingredients: List[IngredientAnalysis]) -> List[str]:
        """Extract product cons."""
        cons = set()
        for ing in ingredients:
            if ing.rating in [IngredientRating.CAUTION.value, IngredientRating.AVOID.value]:
                for concern in ing.concerns[:1]:
                    cons.add(f"{ing.name}: {concern}")
        return list(cons)[:5]

    def _extract_warnings(
        self,
        ingredients: List[IngredientAnalysis],
        skin_type: SkinType,
        concerns: List[SkinConcern],
        allergies: List[str]
    ) -> List[str]:
        """Extract warnings."""
        warnings = []

        avoid_ings = [i for i in ingredients if i.rating == IngredientRating.AVOID.value]
        if avoid_ings:
            warnings.append(f"⚠️ Contains {len(avoid_ings)} ingredient(s) you should avoid")

        high_comedogenic = [i for i in ingredients if i.comedogenic_rating >= 4]
        if high_comedogenic and (SkinConcern.ACNE in concerns or skin_type == SkinType.OILY):
            warnings.append(f"⚠️ Contains pore-clogging ingredients: {', '.join([i.name for i in high_comedogenic])}")

        irritants = [i for i in ingredients if i.irritation_potential == "high"]
        if irritants and skin_type == SkinType.SENSITIVE:
            warnings.append(f"⚠️ May irritate sensitive skin: {', '.join([i.name for i in irritants])}")

        return warnings

    def _calculate_effectiveness(
        self,
        ingredients: List[IngredientAnalysis],
        concerns: List[SkinConcern]
    ) -> Dict[str, int]:
        """Calculate effectiveness for each concern."""
        effectiveness = {}

        for concern in concerns:
            score = 50  # Base
            for ing in ingredients:
                if concern.value in ing.good_for:
                    score += 10
            effectiveness[concern.value] = min(100, score)

        return effectiveness

    def _suggest_alternatives(
        self,
        ingredients: List[IngredientAnalysis],
        skin_type: SkinType,
        concerns: List[SkinConcern],
        overall_score: int
    ) -> List[Dict]:
        """Suggest alternative products/ingredients."""
        alternatives = []

        if overall_score < 60:
            if SkinConcern.ACNE in concerns:
                alternatives.append({
                    "type": "ingredient",
                    "suggestion": "Look for products with Salicylic Acid, Niacinamide, or Azelaic Acid",
                    "reason": "Better suited for acne-prone skin"
                })

            if skin_type == SkinType.SENSITIVE:
                alternatives.append({
                    "type": "ingredient",
                    "suggestion": "Choose fragrance-free products with Centella Asiatica or Ceramides",
                    "reason": "Gentler on sensitive skin"
                })

            if SkinConcern.AGING in concerns:
                alternatives.append({
                    "type": "ingredient",
                    "suggestion": "Look for Retinol, Vitamin C, and Peptides",
                    "reason": "Proven anti-aging ingredients"
                })

        return alternatives


# FastAPI Router
def create_ingredient_scanner_router():
    """Create FastAPI router for ingredient scanner."""
    from fastapi import APIRouter, HTTPException, Form
    from pydantic import BaseModel
    from typing import Optional

    router = APIRouter(prefix="/api/ingredient-scanner", tags=["Ingredient Scanner"])
    scanner = IngredientScanner()

    class IngredientResult(BaseModel):
        name: str
        category: str
        rating: str
        rating_reason: str
        comedogenic_rating: int
        irritation_potential: str
        benefits: List[str]
        concerns: List[str]
        good_for: List[str]
        avoid_if: List[str]

    class ScanResponse(BaseModel):
        product_name: Optional[str]
        overall_rating: str
        overall_score: int
        compatibility_score: int
        ingredient_count: int
        ingredients: List[IngredientResult]
        summary: str
        pros: List[str]
        cons: List[str]
        warnings: List[str]
        alternatives: List[Dict]
        effectiveness_for_goals: Dict[str, int]
        timestamp: str

    @router.post("/scan", response_model=ScanResponse)
    async def scan_ingredients(
        ingredients: str = Form(...),
        skin_type: str = Form("normal"),
        concerns: str = Form(""),
        allergies: str = Form(""),
        product_name: Optional[str] = Form(None)
    ):
        """
        Scan and analyze product ingredients.

        Args:
            ingredients: Comma-separated ingredient list
            skin_type: oily, dry, combination, normal, sensitive
            concerns: Comma-separated concerns (acne, aging, etc.)
            allergies: Comma-separated allergies
            product_name: Optional product name
        """
        try:
            # Parse skin type
            try:
                skin_type_enum = SkinType(skin_type.lower())
            except ValueError:
                skin_type_enum = SkinType.NORMAL

            # Parse concerns
            concern_list = []
            if concerns:
                for c in concerns.split(","):
                    try:
                        concern_list.append(SkinConcern(c.strip().lower()))
                    except ValueError:
                        pass

            # Parse allergies
            allergy_list = [a.strip() for a in allergies.split(",") if a.strip()]

            result = scanner.analyze_ingredients(
                ingredients_text=ingredients,
                skin_type=skin_type_enum,
                concerns=concern_list,
                allergies=allergy_list,
                product_name=product_name
            )

            return ScanResponse(
                product_name=result.product_name,
                overall_rating=result.overall_rating,
                overall_score=result.overall_score,
                compatibility_score=result.compatibility_score,
                ingredient_count=result.ingredient_count,
                ingredients=[IngredientResult(
                    name=i.name,
                    category=i.category,
                    rating=i.rating,
                    rating_reason=i.rating_reason,
                    comedogenic_rating=i.comedogenic_rating,
                    irritation_potential=i.irritation_potential,
                    benefits=i.benefits,
                    concerns=i.concerns,
                    good_for=i.good_for,
                    avoid_if=i.avoid_if
                ) for i in result.ingredients],
                summary=result.summary,
                pros=result.pros,
                cons=result.cons,
                warnings=result.warnings,
                alternatives=result.alternatives,
                effectiveness_for_goals=result.effectiveness_for_goals,
                timestamp=result.timestamp
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

    @router.get("/concerns")
    async def list_concerns():
        """List available skin concerns."""
        return {
            "concerns": [
                {"id": c.value, "name": c.value.replace("_", " ").title()}
                for c in SkinConcern
            ]
        }

    @router.get("/skin-types")
    async def list_skin_types():
        """List available skin types."""
        return {
            "skin_types": [
                {"id": t.value, "name": t.value.title()}
                for t in SkinType
            ]
        }

    @router.get("/ingredient/{name}")
    async def lookup_ingredient(name: str):
        """Look up a single ingredient."""
        db = IngredientDatabase()
        info = db.lookup(name)

        if not info:
            raise HTTPException(status_code=404, detail="Ingredient not found")

        return {
            "name": name,
            "info": info
        }

    return router
