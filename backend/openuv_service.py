"""
OpenUV API Service

Provides real-time and historical UV index data based on GPS coordinates.
This service integrates with the OpenUV API to get accurate UV readings
for workout locations from Apple Watch/HealthKit.

API Documentation: https://www.openuv.io/uvindex
Free tier: 50 requests/day (sufficient for most personal use)

Setup:
1. Sign up at https://www.openuv.io/
2. Get your API key from the dashboard
3. Add to .env: OPENUV_API_KEY=your_api_key_here
"""

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system env vars
import httpx
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

# API Configuration
OPENUV_API_URL = "https://api.openuv.io/api/v1"
OPENUV_API_KEY = os.getenv("OPENUV_API_KEY", "")

# Cache for UV data to minimize API calls
_uv_cache: Dict[str, Tuple[datetime, Any]] = {}
CACHE_DURATION_MINUTES = 30


@dataclass
class UVReading:
    """UV reading data from OpenUV API"""
    uv_index: float
    uv_index_max: float
    uv_time: datetime
    ozone: float
    ozone_time: datetime
    safe_exposure_times: Dict[str, Optional[int]]  # Minutes for each skin type
    sun_info: Dict[str, datetime]  # sunrise, sunset, solar_noon, etc.
    latitude: float
    longitude: float


@dataclass
class UVForecast:
    """Hourly UV forecast"""
    uv_index: float
    uv_time: datetime


@dataclass
class UVProtectionAdvice:
    """Sun protection recommendations based on UV index and skin type"""
    uv_index: float
    skin_type: int  # Fitzpatrick scale 1-6
    safe_exposure_minutes: Optional[int]
    protection_required: bool
    recommendations: List[str]
    risk_level: str  # low, moderate, high, very_high, extreme


class OpenUVService:
    """Service for interacting with OpenUV API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENUV_API_KEY
        self._client: Optional[httpx.AsyncClient] = None
        self._request_count = 0
        self._last_reset = datetime.now().date()

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=OPENUV_API_URL,
                headers={
                    "x-access-token": self.api_key,
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _check_rate_limit(self):
        """Check and update rate limit counter"""
        today = datetime.now().date()
        if today != self._last_reset:
            self._request_count = 0
            self._last_reset = today

        if self._request_count >= 50:
            raise Exception("OpenUV daily rate limit reached (50 requests/day)")

        self._request_count += 1

    def _get_cache_key(self, lat: float, lon: float, dt: Optional[datetime] = None) -> str:
        """Generate cache key for UV data"""
        # Round coordinates to 2 decimal places (about 1km precision)
        lat_rounded = round(lat, 2)
        lon_rounded = round(lon, 2)
        date_str = (dt or datetime.now()).strftime("%Y-%m-%d-%H")
        return f"{lat_rounded}_{lon_rounded}_{date_str}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get cached UV data if still valid"""
        if key in _uv_cache:
            cached_time, data = _uv_cache[key]
            if datetime.now() - cached_time < timedelta(minutes=CACHE_DURATION_MINUTES):
                return data
            else:
                del _uv_cache[key]
        return None

    def _set_cache(self, key: str, data: Any):
        """Cache UV data"""
        _uv_cache[key] = (datetime.now(), data)

        # Clean old cache entries
        now = datetime.now()
        expired_keys = [
            k for k, (t, _) in _uv_cache.items()
            if now - t > timedelta(minutes=CACHE_DURATION_MINUTES * 2)
        ]
        for k in expired_keys:
            del _uv_cache[k]

    async def get_uv_index(
        self,
        latitude: float,
        longitude: float,
        altitude: Optional[float] = None,
        ozone: Optional[float] = None,
        dt: Optional[datetime] = None
    ) -> Optional[UVReading]:
        """
        Get current UV index for a location.

        Args:
            latitude: GPS latitude (-90 to 90)
            longitude: GPS longitude (-180 to 180)
            altitude: Altitude in meters (optional, defaults to sea level)
            ozone: Ozone level in Dobson Units (optional, API will fetch if not provided)
            dt: Datetime for the reading (optional, for historical reference)

        Returns:
            UVReading object with UV data, or None if request fails
        """
        if not self.is_configured:
            logger.warning("OpenUV API key not configured")
            return None

        # Check cache first
        cache_key = self._get_cache_key(latitude, longitude, dt)
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.debug(f"Using cached UV data for {cache_key}")
            return cached

        try:
            self._check_rate_limit()

            client = await self._get_client()

            params = {
                "lat": latitude,
                "lng": longitude
            }
            if altitude:
                params["alt"] = altitude
            if ozone:
                params["ozone"] = ozone
            if dt:
                params["dt"] = dt.isoformat()

            response = await client.get("/uv", params=params)
            response.raise_for_status()

            data = response.json()
            result = data.get("result", {})

            # Parse safe exposure times
            safe_times = {}
            for key, value in result.get("safe_exposure_time", {}).items():
                safe_times[key] = int(value) if value else None

            # Parse sun info
            sun_info = {}
            sun_info_data = result.get("sun_info", {}).get("sun_times", {})
            for key, value in sun_info_data.items():
                if value:
                    try:
                        sun_info[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except:
                        pass

            reading = UVReading(
                uv_index=result.get("uv", 0),
                uv_index_max=result.get("uv_max", 0),
                uv_time=datetime.fromisoformat(result.get("uv_time", "").replace("Z", "+00:00")) if result.get("uv_time") else datetime.now(),
                ozone=result.get("ozone", 0),
                ozone_time=datetime.fromisoformat(result.get("ozone_time", "").replace("Z", "+00:00")) if result.get("ozone_time") else datetime.now(),
                safe_exposure_times=safe_times,
                sun_info=sun_info,
                latitude=latitude,
                longitude=longitude
            )

            # Cache the result
            self._set_cache(cache_key, reading)

            logger.info(f"UV index at ({latitude}, {longitude}): {reading.uv_index}")
            return reading

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.error("OpenUV API key invalid or expired")
            else:
                logger.error(f"OpenUV API error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Failed to get UV index: {str(e)}")
            return None

    async def get_uv_forecast(
        self,
        latitude: float,
        longitude: float,
        altitude: Optional[float] = None
    ) -> List[UVForecast]:
        """
        Get hourly UV forecast for a location (next 48 hours).

        Args:
            latitude: GPS latitude
            longitude: GPS longitude
            altitude: Altitude in meters (optional)

        Returns:
            List of hourly UV forecasts
        """
        if not self.is_configured:
            return []

        try:
            self._check_rate_limit()

            client = await self._get_client()

            params = {
                "lat": latitude,
                "lng": longitude
            }
            if altitude:
                params["alt"] = altitude

            response = await client.get("/forecast", params=params)
            response.raise_for_status()

            data = response.json()
            forecasts = []

            for item in data.get("result", []):
                forecasts.append(UVForecast(
                    uv_index=item.get("uv", 0),
                    uv_time=datetime.fromisoformat(item.get("uv_time", "").replace("Z", "+00:00"))
                ))

            return forecasts

        except Exception as e:
            logger.error(f"Failed to get UV forecast: {str(e)}")
            return []

    async def get_uv_for_workout(
        self,
        latitude: float,
        longitude: float,
        workout_start: datetime,
        workout_duration_seconds: int
    ) -> Tuple[float, float, str]:
        """
        Get UV exposure for a workout session.

        Args:
            latitude: Workout location latitude
            longitude: Workout location longitude
            workout_start: Start time of workout
            workout_duration_seconds: Duration of workout in seconds

        Returns:
            Tuple of (average_uv_index, uv_dose, risk_level)
        """
        # For historical workouts, estimate UV based on time of day
        # OpenUV only provides current and forecast data

        reading = await self.get_uv_index(latitude, longitude)

        if reading:
            avg_uv = reading.uv_index
        else:
            # Estimate based on time of day if API fails
            hour = workout_start.hour
            if 10 <= hour <= 14:
                avg_uv = 8.0
            elif 8 <= hour <= 16:
                avg_uv = 5.0
            elif 6 <= hour <= 18:
                avg_uv = 2.0
            else:
                avg_uv = 0.0

        # Calculate UV dose (UV index * hours)
        duration_hours = workout_duration_seconds / 3600
        uv_dose = avg_uv * duration_hours

        # Determine risk level
        if avg_uv < 3:
            risk_level = "low"
        elif avg_uv < 6:
            risk_level = "moderate"
        elif avg_uv < 8:
            risk_level = "high"
        elif avg_uv < 11:
            risk_level = "very_high"
        else:
            risk_level = "extreme"

        return avg_uv, uv_dose, risk_level

    async def get_uv_for_locations(
        self,
        locations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get UV readings for multiple workout locations.
        Batches requests to respect rate limits.

        Args:
            locations: List of dicts with latitude, longitude, timestamp

        Returns:
            List of UV readings with original location data
        """
        results = []

        for loc in locations:
            lat = loc.get("latitude")
            lon = loc.get("longitude")
            timestamp_str = loc.get("timestamp")

            if lat is None or lon is None:
                continue

            timestamp = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except:
                    timestamp = datetime.now()

            # Get UV data
            reading = await self.get_uv_index(lat, lon, dt=timestamp)

            result = {
                "latitude": lat,
                "longitude": lon,
                "timestamp": timestamp_str,
                "uv_index": reading.uv_index if reading else self._estimate_uv(timestamp),
                "uv_max": reading.uv_index_max if reading else None,
                "source": "openuv" if reading else "estimated",
            }

            if reading:
                result["safe_exposure_times"] = reading.safe_exposure_times
                result["ozone"] = reading.ozone

            results.append(result)

            # Small delay to be nice to the API
            await asyncio.sleep(0.1)

        return results

    def _estimate_uv(self, dt: Optional[datetime] = None) -> float:
        """Estimate UV index based on time of day"""
        if dt is None:
            dt = datetime.now()

        hour = dt.hour

        # Summer vs winter adjustment (northern hemisphere)
        month = dt.month
        seasonal_factor = 1.0
        if month in [6, 7, 8]:  # Summer
            seasonal_factor = 1.3
        elif month in [12, 1, 2]:  # Winter
            seasonal_factor = 0.6

        # Time of day UV curve
        if 10 <= hour <= 14:
            base_uv = 8.0
        elif 8 <= hour <= 16:
            base_uv = 5.0
        elif 6 <= hour <= 18:
            base_uv = 2.0
        else:
            base_uv = 0.0

        return round(base_uv * seasonal_factor, 1)

    def get_protection_advice(
        self,
        uv_index: float,
        skin_type: int = 2,  # Default to Type II (fair, burns easily)
        outdoor_duration_minutes: int = 30
    ) -> UVProtectionAdvice:
        """
        Get sun protection recommendations based on UV and skin type.

        Args:
            uv_index: Current UV index
            skin_type: Fitzpatrick skin type (1-6)
            outdoor_duration_minutes: Planned outdoor duration

        Returns:
            Protection advice with recommendations
        """
        # Base safe exposure times by skin type (minutes to burn at UV=10)
        base_times = {
            1: 10,   # Very fair, always burns
            2: 15,   # Fair, burns easily
            3: 20,   # Medium, sometimes burns
            4: 30,   # Olive, rarely burns
            5: 45,   # Brown, very rarely burns
            6: 60,   # Dark brown/black, never burns
        }

        base_time = base_times.get(skin_type, 15)

        # Calculate safe exposure time
        if uv_index > 0:
            safe_minutes = int((base_time * 10) / uv_index)
        else:
            safe_minutes = None  # Unlimited

        # Determine risk level
        if uv_index < 3:
            risk_level = "low"
        elif uv_index < 6:
            risk_level = "moderate"
        elif uv_index < 8:
            risk_level = "high"
        elif uv_index < 11:
            risk_level = "very_high"
        else:
            risk_level = "extreme"

        # Protection required?
        protection_required = uv_index >= 3 or (safe_minutes and outdoor_duration_minutes > safe_minutes)

        # Generate recommendations
        recommendations = []

        if uv_index < 3:
            recommendations.append("Low UV levels - minimal sun protection needed")
            recommendations.append("Wear sunglasses on bright days")
        elif uv_index < 6:
            recommendations.append("Apply SPF 30+ sunscreen")
            recommendations.append("Wear sunglasses and a hat")
            recommendations.append("Seek shade during midday hours")
        elif uv_index < 8:
            recommendations.append("Apply SPF 50+ sunscreen every 2 hours")
            recommendations.append("Wear protective clothing, hat, and sunglasses")
            recommendations.append("Reduce sun exposure between 10am-4pm")
            recommendations.append("Seek shade whenever possible")
        elif uv_index < 11:
            recommendations.append("Minimize sun exposure between 10am-4pm")
            recommendations.append("Apply SPF 50+ sunscreen liberally and reapply often")
            recommendations.append("Wear UPF clothing, wide-brim hat, and UV-blocking sunglasses")
            recommendations.append("Stay in shade - unprotected skin burns quickly")
        else:
            recommendations.append("AVOID sun exposure if possible")
            recommendations.append("If outdoors, stay in full shade")
            recommendations.append("Use maximum protection: SPF 50+, full coverage clothing")
            recommendations.append("Unprotected skin can burn in minutes")

        if safe_minutes and outdoor_duration_minutes > safe_minutes:
            recommendations.append(
                f"Your planned {outdoor_duration_minutes} min outdoors exceeds safe exposure "
                f"({safe_minutes} min) for your skin type. Take extra precautions!"
            )

        return UVProtectionAdvice(
            uv_index=uv_index,
            skin_type=skin_type,
            safe_exposure_minutes=safe_minutes,
            protection_required=protection_required,
            recommendations=recommendations,
            risk_level=risk_level
        )


# Global service instance
_openuv_service: Optional[OpenUVService] = None


def get_openuv_service() -> OpenUVService:
    """Get or create the global OpenUV service instance"""
    global _openuv_service
    if _openuv_service is None:
        _openuv_service = OpenUVService()
    return _openuv_service


async def cleanup_openuv_service():
    """Cleanup the OpenUV service on shutdown"""
    global _openuv_service
    if _openuv_service:
        await _openuv_service.close()
        _openuv_service = None
