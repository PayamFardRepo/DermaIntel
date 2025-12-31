"""
Environmental Shield - Location-Based Skin Protection

Real-time location-based skin protection alerts using OpenWeatherMap API:
- UV index and sunscreen reminders
- Pollution levels affecting skin
- Humidity adjustments
- Pollen/allergy alerts
- Temperature impacts
"""

import math
import httpx
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import random
import logging

logger = logging.getLogger(__name__)

# OpenWeatherMap API Key - from environment or fallback
import os
# API Key must be set in environment variables (never commit API keys!)
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")


class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class AlertType(Enum):
    UV = "uv"
    POLLUTION = "pollution"
    HUMIDITY = "humidity"
    POLLEN = "pollen"
    TEMPERATURE = "temperature"
    WIND = "wind"


@dataclass
class EnvironmentalAlert:
    type: str
    priority: str
    title: str
    message: str
    icon: str
    action: str
    expires_at: str


@dataclass
class SkinProtectionPlan:
    morning_routine: List[Dict]
    midday_actions: List[Dict]
    evening_routine: List[Dict]
    products_needed: List[Dict]
    lifestyle_tips: List[str]


@dataclass
class EnvironmentalData:
    uv_index: float
    uv_level: str
    pollution_aqi: int
    pollution_level: str
    humidity: int
    humidity_impact: str
    pollen_count: int
    pollen_level: str
    temperature: float
    temperature_impact: str
    wind_speed: float
    overall_skin_risk: str
    alerts: List[EnvironmentalAlert]
    protection_plan: SkinProtectionPlan
    next_sunscreen_reminder: str
    timestamp: str
    location_name: Optional[str] = None
    weather_description: Optional[str] = None


class EnvironmentalShield:
    """Analyzes environmental factors and provides skin protection guidance using real weather data."""

    def __init__(self):
        self.api_key = OPENWEATHERMAP_API_KEY

        # UV Index thresholds
        self.uv_levels = {
            (0, 2): ("Low", "low"),
            (3, 5): ("Moderate", "medium"),
            (6, 7): ("High", "high"),
            (8, 10): ("Very High", "urgent"),
            (11, float('inf')): ("Extreme", "urgent"),
        }

        # AQI thresholds (OpenWeatherMap uses 1-5 scale, we convert to US EPA scale)
        self.aqi_levels = {
            (0, 50): ("Good", "low"),
            (51, 100): ("Moderate", "low"),
            (101, 150): ("Unhealthy for Sensitive", "medium"),
            (151, 200): ("Unhealthy", "high"),
            (201, 300): ("Very Unhealthy", "urgent"),
            (301, float('inf')): ("Hazardous", "urgent"),
        }

        # Pollen thresholds
        self.pollen_levels = {
            (0, 2.4): ("Low", "low"),
            (2.5, 4.8): ("Moderate", "medium"),
            (4.9, 7.2): ("High", "high"),
            (7.3, float('inf')): ("Very High", "urgent"),
        }

    async def _fetch_weather_data(self, lat: float, lon: float) -> Dict:
        """Fetch current weather data from OpenWeatherMap API."""
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "imperial"  # Fahrenheit
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def _fetch_uv_data(self, lat: float, lon: float) -> Dict:
        """Fetch UV index data from OpenWeatherMap One Call API."""
        # Use One Call API 3.0 for UV data
        url = f"https://api.openweathermap.org/data/2.5/uvi"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"UV API failed, using One Call API: {e}")
            # Fallback: try One Call API 2.5
            url = f"https://api.openweathermap.org/data/2.5/onecall"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "exclude": "minutely,hourly,daily,alerts"
            }
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    return {"value": data.get("current", {}).get("uvi", 0)}
            except:
                return None

    async def _fetch_air_pollution(self, lat: float, lon: float) -> Dict:
        """Fetch air pollution data from OpenWeatherMap API."""
        url = f"https://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def _convert_aqi_to_us_scale(self, owm_aqi: int, pm25: float = None) -> int:
        """Convert OpenWeatherMap AQI (1-5) to US EPA scale (0-500)."""
        # OpenWeatherMap AQI: 1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor
        # US EPA AQI: 0-50 Good, 51-100 Moderate, 101-150 USG, 151-200 Unhealthy, 201-300 Very Unhealthy, 301+ Hazardous

        if pm25 is not None:
            # Calculate based on PM2.5 if available (more accurate)
            if pm25 <= 12:
                return int(pm25 * 50 / 12)
            elif pm25 <= 35.4:
                return int(50 + (pm25 - 12) * 50 / 23.4)
            elif pm25 <= 55.4:
                return int(100 + (pm25 - 35.4) * 50 / 20)
            elif pm25 <= 150.4:
                return int(150 + (pm25 - 55.4) * 50 / 95)
            elif pm25 <= 250.4:
                return int(200 + (pm25 - 150.4) * 100 / 100)
            else:
                return int(300 + (pm25 - 250.4) * 100 / 150)

        # Fallback to simple conversion
        aqi_mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 275}
        return aqi_mapping.get(owm_aqi, 50)

    async def get_environmental_data(
        self,
        latitude: float,
        longitude: float,
        skin_type: str = "normal",
        conditions: List[str] = None
    ) -> EnvironmentalData:
        """
        Get real environmental data from OpenWeatherMap and skin protection recommendations.
        """
        try:
            # Fetch all data concurrently
            weather_task = self._fetch_weather_data(latitude, longitude)
            uv_task = self._fetch_uv_data(latitude, longitude)
            pollution_task = self._fetch_air_pollution(latitude, longitude)

            weather_data, uv_data, pollution_data = await asyncio.gather(
                weather_task, uv_task, pollution_task,
                return_exceptions=True
            )

            # Process weather data
            if isinstance(weather_data, Exception):
                logger.error(f"Weather API error: {weather_data}")
                weather_data = {}

            # Process UV data
            if isinstance(uv_data, Exception) or uv_data is None:
                logger.warning(f"UV API error, estimating from time of day")
                uv_data = {}

            # Process pollution data
            if isinstance(pollution_data, Exception):
                logger.error(f"Pollution API error: {pollution_data}")
                pollution_data = {}

            # Build environmental data from API responses
            env_data = self._process_api_data(weather_data, uv_data, pollution_data)

        except Exception as e:
            logger.error(f"Failed to fetch environmental data: {e}")
            # Fallback to simulated data if APIs fail
            env_data = self._simulate_environmental_data(latitude, longitude)

        # Analyze and generate alerts
        alerts = self._generate_alerts(env_data, skin_type, conditions or [])

        # Generate protection plan
        protection_plan = self._create_protection_plan(env_data, skin_type, conditions or [])

        # Calculate next sunscreen reminder
        next_reminder = self._calculate_sunscreen_reminder(env_data["uv_index"])

        # Determine overall risk
        overall_risk = self._calculate_overall_risk(env_data)

        return EnvironmentalData(
            uv_index=env_data["uv_index"],
            uv_level=env_data["uv_level"],
            pollution_aqi=env_data["aqi"],
            pollution_level=env_data["aqi_level"],
            humidity=env_data["humidity"],
            humidity_impact=env_data["humidity_impact"],
            pollen_count=env_data["pollen"],
            pollen_level=env_data["pollen_level"],
            temperature=env_data["temperature"],
            temperature_impact=env_data["temp_impact"],
            wind_speed=env_data["wind"],
            overall_skin_risk=overall_risk,
            alerts=alerts,
            protection_plan=protection_plan,
            next_sunscreen_reminder=next_reminder,
            timestamp=datetime.now().isoformat(),
            location_name=env_data.get("location_name"),
            weather_description=env_data.get("weather_description")
        )

    def _process_api_data(self, weather: Dict, uv: Dict, pollution: Dict) -> Dict:
        """Process API responses into unified environmental data."""

        # Extract temperature (already in Fahrenheit from API)
        temperature = weather.get("main", {}).get("temp", 70)

        # Extract humidity
        humidity = weather.get("main", {}).get("humidity", 50)

        # Extract wind speed (mph since we used imperial units)
        wind = weather.get("wind", {}).get("speed", 5)

        # Extract UV index
        uv_index = 0
        if uv and isinstance(uv, dict):
            uv_index = uv.get("value", 0)

        # If UV is 0 or missing, estimate based on time of day
        if uv_index == 0:
            hour = datetime.now().hour
            if 6 <= hour <= 18:
                # Simple estimation based on time
                base_uv = 5 + (3 * math.sin((hour - 6) * math.pi / 12))
                uv_index = round(max(0, base_uv), 1)

        # Determine UV level
        uv_level = "Low"
        for (low, high), (level, _) in self.uv_levels.items():
            if low <= uv_index <= high:
                uv_level = level
                break

        # Extract AQI from pollution data
        aqi = 50  # Default
        if pollution and "list" in pollution and len(pollution["list"]) > 0:
            owm_aqi = pollution["list"][0].get("main", {}).get("aqi", 1)
            pm25 = pollution["list"][0].get("components", {}).get("pm2_5")
            aqi = self._convert_aqi_to_us_scale(owm_aqi, pm25)

        # Determine AQI level
        aqi_level = "Good"
        for (low, high), (level, _) in self.aqi_levels.items():
            if low <= aqi <= high:
                aqi_level = level
                break

        # Humidity impact
        if humidity < 30:
            humidity_impact = "Very dry - extra moisturizer needed"
        elif humidity < 50:
            humidity_impact = "Comfortable - normal routine"
        elif humidity < 70:
            humidity_impact = "Humid - lighter products recommended"
        else:
            humidity_impact = "Very humid - skip heavy creams"

        # Temperature impact
        if temperature < 50:
            temp_impact = "Cold - protect skin barrier, use richer moisturizer"
        elif temperature < 70:
            temp_impact = "Mild - ideal for skin"
        elif temperature < 85:
            temp_impact = "Warm - stay hydrated, reapply sunscreen more often"
        else:
            temp_impact = "Hot - minimize sun exposure, cool compresses help"

        # Pollen estimation (OpenWeatherMap doesn't provide pollen, so we estimate)
        # Higher in spring/summer, during daytime, with moderate wind
        month = datetime.now().month
        hour = datetime.now().hour
        pollen_base = 2.0
        if 3 <= month <= 6:  # Spring
            pollen_base = 5.0
        elif 7 <= month <= 9:  # Summer
            pollen_base = 4.0

        if 8 <= hour <= 18:  # Daytime
            pollen_base *= 1.3

        if 5 <= wind <= 15:  # Moderate wind spreads pollen
            pollen_base *= 1.2

        pollen = round(min(10, pollen_base + random.uniform(-1, 1)), 1)

        # Determine pollen level
        pollen_level = "Low"
        for (low, high), (level, _) in self.pollen_levels.items():
            if low <= pollen <= high:
                pollen_level = level
                break

        # Location info
        location_name = weather.get("name", "Unknown")
        weather_desc = ""
        if "weather" in weather and len(weather["weather"]) > 0:
            weather_desc = weather["weather"][0].get("description", "").title()

        return {
            "uv_index": round(uv_index, 1),
            "uv_level": uv_level,
            "aqi": aqi,
            "aqi_level": aqi_level,
            "humidity": humidity,
            "humidity_impact": humidity_impact,
            "pollen": pollen,
            "pollen_level": pollen_level,
            "temperature": round(temperature, 1),
            "temp_impact": temp_impact,
            "wind": round(wind, 1),
            "location_name": location_name,
            "weather_description": weather_desc
        }

    def _simulate_environmental_data(self, lat: float, lon: float) -> Dict:
        """Fallback: Simulate environmental data if API fails."""
        hour = datetime.now().hour

        # Base UV on time of day and latitude
        if 6 <= hour <= 18:
            base_uv = 5 + (3 * math.sin((hour - 6) * math.pi / 12))
            lat_factor = 1 - (abs(lat) / 90) * 0.5
            uv_index = base_uv * lat_factor
        else:
            uv_index = 0

        uv_index = max(0, uv_index + random.uniform(-1, 1))
        uv_index = round(uv_index, 1)

        uv_level = "Low"
        for (low, high), (level, _) in self.uv_levels.items():
            if low <= uv_index <= high:
                uv_level = level
                break

        base_aqi = 50 + random.randint(-20, 50)
        aqi = max(0, min(300, base_aqi))

        aqi_level = "Good"
        for (low, high), (level, _) in self.aqi_levels.items():
            if low <= aqi <= high:
                aqi_level = level
                break

        humidity = random.randint(30, 80)
        if humidity < 30:
            humidity_impact = "Very dry - extra moisturizer needed"
        elif humidity < 50:
            humidity_impact = "Comfortable - normal routine"
        elif humidity < 70:
            humidity_impact = "Humid - lighter products recommended"
        else:
            humidity_impact = "Very humid - skip heavy creams"

        pollen = round(random.uniform(0, 10), 1)
        pollen_level = "Low"
        for (low, high), (level, _) in self.pollen_levels.items():
            if low <= pollen <= high:
                pollen_level = level
                break

        temperature = random.randint(50, 95)
        if temperature < 50:
            temp_impact = "Cold - protect skin barrier, use richer moisturizer"
        elif temperature < 70:
            temp_impact = "Mild - ideal for skin"
        elif temperature < 85:
            temp_impact = "Warm - stay hydrated, reapply sunscreen more often"
        else:
            temp_impact = "Hot - minimize sun exposure, cool compresses help"

        wind = random.randint(0, 25)

        return {
            "uv_index": uv_index,
            "uv_level": uv_level,
            "aqi": aqi,
            "aqi_level": aqi_level,
            "humidity": humidity,
            "humidity_impact": humidity_impact,
            "pollen": pollen,
            "pollen_level": pollen_level,
            "temperature": temperature,
            "temp_impact": temp_impact,
            "wind": wind,
            "location_name": "Unknown (API unavailable)",
            "weather_description": "Data unavailable"
        }

    def _generate_alerts(
        self,
        env_data: Dict,
        skin_type: str,
        conditions: List[str]
    ) -> List[EnvironmentalAlert]:
        """Generate alerts based on environmental conditions."""
        alerts = []

        # UV Alert
        if env_data["uv_index"] >= 3:
            priority = "medium"
            if env_data["uv_index"] >= 6:
                priority = "high"
            if env_data["uv_index"] >= 8:
                priority = "urgent"

            alerts.append(EnvironmentalAlert(
                type=AlertType.UV.value,
                priority=priority,
                title=f"UV Index: {env_data['uv_index']} ({env_data['uv_level']})",
                message=self._get_uv_message(env_data["uv_index"]),
                icon="☀️",
                action="Apply SPF 50+ now" if env_data["uv_index"] >= 3 else "SPF recommended",
                expires_at=(datetime.now() + timedelta(hours=2)).isoformat()
            ))

        # Pollution Alert
        if env_data["aqi"] > 100:
            priority = "medium" if env_data["aqi"] < 150 else "high"

            alerts.append(EnvironmentalAlert(
                type=AlertType.POLLUTION.value,
                priority=priority,
                title=f"Air Quality: {env_data['aqi_level']} (AQI: {env_data['aqi']})",
                message="High pollution can damage skin barrier and cause premature aging",
                icon="🏭",
                action="Double cleanse tonight and use antioxidant serum",
                expires_at=(datetime.now() + timedelta(hours=6)).isoformat()
            ))

        # Humidity Alert
        if env_data["humidity"] < 30 or env_data["humidity"] > 70:
            alerts.append(EnvironmentalAlert(
                type=AlertType.HUMIDITY.value,
                priority="low",
                title=f"Humidity: {env_data['humidity']}%",
                message=env_data["humidity_impact"],
                icon="💧" if env_data["humidity"] < 30 else "💦",
                action="Adjust moisturizer accordingly",
                expires_at=(datetime.now() + timedelta(hours=12)).isoformat()
            ))

        # Pollen Alert (especially for sensitive skin)
        if env_data["pollen"] > 4.8 and ("sensitive" in skin_type or "eczema" in conditions or "rosacea" in conditions):
            alerts.append(EnvironmentalAlert(
                type=AlertType.POLLEN.value,
                priority="medium",
                title=f"High Pollen: {env_data['pollen_level']}",
                message="Pollen can trigger skin inflammation and sensitivity",
                icon="🌸",
                action="Wash face after outdoor exposure, use barrier cream",
                expires_at=(datetime.now() + timedelta(hours=8)).isoformat()
            ))

        # Temperature Alert
        if env_data["temperature"] > 85 or env_data["temperature"] < 40:
            alerts.append(EnvironmentalAlert(
                type=AlertType.TEMPERATURE.value,
                priority="low",
                title=f"Temperature: {env_data['temperature']}°F",
                message=env_data["temp_impact"],
                icon="🌡️",
                action="Adjust skincare routine for temperature",
                expires_at=(datetime.now() + timedelta(hours=6)).isoformat()
            ))

        # Wind Alert
        if env_data["wind"] > 15:
            alerts.append(EnvironmentalAlert(
                type=AlertType.WIND.value,
                priority="low",
                title=f"High Wind: {env_data['wind']} mph",
                message="Wind can dry out skin and spread allergens",
                icon="💨",
                action="Use a barrier cream and stay hydrated",
                expires_at=(datetime.now() + timedelta(hours=4)).isoformat()
            ))

        # Sort by priority
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(key=lambda a: priority_order.get(a.priority, 4))

        return alerts

    def _get_uv_message(self, uv_index: float) -> str:
        """Get appropriate UV message."""
        if uv_index < 3:
            return "Low UV - minimal protection needed for short exposure"
        elif uv_index < 6:
            return "Moderate UV - wear sunscreen and sunglasses outdoors"
        elif uv_index < 8:
            return "High UV - reduce sun exposure during midday hours"
        elif uv_index < 11:
            return "Very High UV - extra protection required, seek shade"
        else:
            return "Extreme UV - avoid sun exposure, stay indoors if possible"

    def _create_protection_plan(
        self,
        env_data: Dict,
        skin_type: str,
        conditions: List[str]
    ) -> SkinProtectionPlan:
        """Create a personalized protection plan."""

        morning = []
        midday = []
        evening = []
        products = []
        tips = []

        # Morning routine based on conditions
        morning.append({
            "step": 1,
            "action": "Gentle cleanser",
            "reason": "Start with clean skin"
        })

        if env_data["aqi"] > 100:
            morning.append({
                "step": 2,
                "action": "Antioxidant serum (Vitamin C)",
                "reason": "Protects against pollution damage"
            })
            products.append({
                "type": "Vitamin C Serum",
                "priority": "high",
                "reason": "High pollution protection"
            })

        if env_data["humidity"] < 40:
            morning.append({
                "step": 3,
                "action": "Hyaluronic acid serum",
                "reason": "Boost hydration in dry air"
            })

        morning.append({
            "step": 4 if len(morning) > 2 else 3,
            "action": "Moisturizer",
            "reason": "Lock in hydration and protect barrier"
        })

        if env_data["uv_index"] >= 1:
            spf_level = "SPF 30" if env_data["uv_index"] < 6 else "SPF 50+"
            morning.append({
                "step": len(morning) + 1,
                "action": f"Apply {spf_level} sunscreen",
                "reason": f"UV Index is {env_data['uv_index']}"
            })
            products.append({
                "type": f"Sunscreen {spf_level}",
                "priority": "essential",
                "reason": "UV protection"
            })

        # Midday actions
        if env_data["uv_index"] >= 3:
            midday.append({
                "time": "Every 2 hours outdoors",
                "action": "Reapply sunscreen",
                "reason": "Maintain protection"
            })

        if env_data["uv_index"] >= 6:
            midday.append({
                "time": "10 AM - 4 PM",
                "action": "Seek shade when possible",
                "reason": "Peak UV hours"
            })

        if env_data["temperature"] > 85:
            midday.append({
                "time": "As needed",
                "action": "Use facial mist to cool skin",
                "reason": "Prevent heat damage"
            })
            products.append({
                "type": "Facial Mist",
                "priority": "medium",
                "reason": "Cool and hydrate throughout day"
            })

        # Evening routine
        evening.append({
            "step": 1,
            "action": "Double cleanse" if env_data["aqi"] > 100 else "Gentle cleanser",
            "reason": "Remove pollution and sunscreen"
        })

        if env_data["aqi"] > 100:
            evening.append({
                "step": 2,
                "action": "Antioxidant treatment",
                "reason": "Repair pollution damage"
            })

        evening.append({
            "step": len(evening) + 1,
            "action": "Treatment serum (retinol if not sensitive)",
            "reason": "Repair and rejuvenate overnight"
        })

        evening.append({
            "step": len(evening) + 1,
            "action": "Rich night moisturizer",
            "reason": "Deep overnight hydration"
        })

        # Lifestyle tips
        tips.append("Wear UV-protective sunglasses to prevent crow's feet")

        if env_data["uv_index"] >= 6:
            tips.append("Wear a wide-brimmed hat outdoors")

        if env_data["aqi"] > 100:
            tips.append("Consider an air purifier indoors")

        if env_data["humidity"] < 40:
            tips.append("Use a humidifier at home")
            tips.append("Drink extra water to hydrate from within")

        if env_data["temperature"] > 85:
            tips.append("Take cool showers - hot water strips skin oils")

        if env_data["pollen"] > 4.8:
            tips.append("Change clothes after being outdoors")
            tips.append("Wash hair before bed to remove pollen")

        return SkinProtectionPlan(
            morning_routine=morning,
            midday_actions=midday,
            evening_routine=evening,
            products_needed=products,
            lifestyle_tips=tips[:5]
        )

    def _calculate_sunscreen_reminder(self, uv_index: float) -> str:
        """Calculate when to remind about sunscreen reapplication."""
        if uv_index < 3:
            return "No reminder needed - low UV"

        now = datetime.now()

        if uv_index >= 8:
            next_time = now + timedelta(minutes=90)
        elif uv_index >= 6:
            next_time = now + timedelta(hours=2)
        else:
            next_time = now + timedelta(hours=2, minutes=30)

        return next_time.strftime("%I:%M %p")

    def _calculate_overall_risk(self, env_data: Dict) -> str:
        """Calculate overall skin risk level."""
        risk_score = 0

        if env_data["uv_index"] >= 8:
            risk_score += 3
        elif env_data["uv_index"] >= 6:
            risk_score += 2
        elif env_data["uv_index"] >= 3:
            risk_score += 1

        if env_data["aqi"] >= 150:
            risk_score += 2
        elif env_data["aqi"] >= 100:
            risk_score += 1

        if env_data["humidity"] < 30 or env_data["humidity"] > 80:
            risk_score += 1

        if env_data["pollen"] >= 7.2:
            risk_score += 1

        if env_data["temperature"] > 90 or env_data["temperature"] < 35:
            risk_score += 1

        if risk_score >= 5:
            return "High"
        elif risk_score >= 3:
            return "Moderate"
        elif risk_score >= 1:
            return "Low"
        else:
            return "Minimal"


# FastAPI Router
def create_environmental_shield_router():
    """Create FastAPI router for environmental shield."""
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel
    from typing import Optional

    router = APIRouter(prefix="/api/environmental-shield", tags=["Environmental Shield"])
    shield = EnvironmentalShield()

    class RoutineStep(BaseModel):
        step: Optional[int] = None
        time: Optional[str] = None
        action: str
        reason: str

    class ProductNeeded(BaseModel):
        type: str
        priority: str
        reason: str

    class ProtectionPlanResponse(BaseModel):
        morning_routine: List[RoutineStep]
        midday_actions: List[RoutineStep]
        evening_routine: List[RoutineStep]
        products_needed: List[ProductNeeded]
        lifestyle_tips: List[str]

    class AlertResponse(BaseModel):
        type: str
        priority: str
        title: str
        message: str
        icon: str
        action: str
        expires_at: str

    class EnvironmentalResponse(BaseModel):
        uv_index: float
        uv_level: str
        pollution_aqi: int
        pollution_level: str
        humidity: int
        humidity_impact: str
        pollen_count: float
        pollen_level: str
        temperature: float
        temperature_impact: str
        wind_speed: float
        overall_skin_risk: str
        alerts: List[AlertResponse]
        protection_plan: ProtectionPlanResponse
        next_sunscreen_reminder: str
        timestamp: str
        location_name: Optional[str] = None
        weather_description: Optional[str] = None

    @router.get("/status", response_model=EnvironmentalResponse)
    async def get_environmental_status(
        latitude: float = Query(..., description="Latitude"),
        longitude: float = Query(..., description="Longitude"),
        skin_type: str = Query("normal", description="Skin type"),
        conditions: str = Query("", description="Comma-separated skin conditions")
    ):
        """
        Get current environmental conditions and skin protection recommendations.

        Uses OpenWeatherMap API for real-time weather, UV, and air quality data.
        Returns personalized skin protection alerts and recommendations.
        """
        try:
            condition_list = [c.strip() for c in conditions.split(",") if c.strip()]

            result = await shield.get_environmental_data(
                latitude=latitude,
                longitude=longitude,
                skin_type=skin_type,
                conditions=condition_list
            )

            return EnvironmentalResponse(
                uv_index=result.uv_index,
                uv_level=result.uv_level,
                pollution_aqi=result.pollution_aqi,
                pollution_level=result.pollution_level,
                humidity=result.humidity,
                humidity_impact=result.humidity_impact,
                pollen_count=result.pollen_count,
                pollen_level=result.pollen_level,
                temperature=result.temperature,
                temperature_impact=result.temperature_impact,
                wind_speed=result.wind_speed,
                overall_skin_risk=result.overall_skin_risk,
                alerts=[AlertResponse(
                    type=a.type,
                    priority=a.priority,
                    title=a.title,
                    message=a.message,
                    icon=a.icon,
                    action=a.action,
                    expires_at=a.expires_at
                ) for a in result.alerts],
                protection_plan=ProtectionPlanResponse(
                    morning_routine=[RoutineStep(**s) for s in result.protection_plan.morning_routine],
                    midday_actions=[RoutineStep(**s) for s in result.protection_plan.midday_actions],
                    evening_routine=[RoutineStep(**s) for s in result.protection_plan.evening_routine],
                    products_needed=[ProductNeeded(**p) for p in result.protection_plan.products_needed],
                    lifestyle_tips=result.protection_plan.lifestyle_tips
                ),
                next_sunscreen_reminder=result.next_sunscreen_reminder,
                timestamp=result.timestamp,
                location_name=result.location_name,
                weather_description=result.weather_description
            )

        except Exception as e:
            logger.error(f"Environmental shield error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get environmental data: {str(e)}")

    @router.get("/uv-info")
    async def get_uv_info():
        """Get UV index scale information."""
        return {
            "scale": [
                {"range": "0-2", "level": "Low", "risk": "Minimal", "protection": "Optional"},
                {"range": "3-5", "level": "Moderate", "risk": "Moderate", "protection": "Recommended"},
                {"range": "6-7", "level": "High", "risk": "High", "protection": "Required"},
                {"range": "8-10", "level": "Very High", "risk": "Very High", "protection": "Extra protection"},
                {"range": "11+", "level": "Extreme", "risk": "Extreme", "protection": "Avoid sun exposure"},
            ]
        }

    @router.get("/aqi-info")
    async def get_aqi_info():
        """Get Air Quality Index scale information."""
        return {
            "scale": [
                {"range": "0-50", "level": "Good", "skin_impact": "No impact"},
                {"range": "51-100", "level": "Moderate", "skin_impact": "Minimal"},
                {"range": "101-150", "level": "Unhealthy for Sensitive", "skin_impact": "Use antioxidants"},
                {"range": "151-200", "level": "Unhealthy", "skin_impact": "Double cleanse, barrier cream"},
                {"range": "201-300", "level": "Very Unhealthy", "skin_impact": "Minimize outdoor exposure"},
                {"range": "301+", "level": "Hazardous", "skin_impact": "Stay indoors"},
            ]
        }

    return router
