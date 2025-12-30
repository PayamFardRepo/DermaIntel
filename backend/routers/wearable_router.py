"""
Wearable Integration Router

API endpoints for connecting wearable devices (Apple Watch, Fitbit, Garmin)
and syncing UV exposure data for correlation with lesion changes.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import httpx
import os

from database import (
    SessionLocal, User, WearableDevice, WearableUVReading,
    WearableDailyUVSummary, WearableLesionCorrelation, LesionGroup,
    AnalysisHistory
)
from auth import get_current_active_user as get_current_user

router = APIRouter(prefix="/wearables", tags=["Wearable Integration"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class DeviceConnectRequest(BaseModel):
    device_type: str = Field(..., description="apple_watch, fitbit, garmin, samsung, withings")
    device_model: Optional[str] = None
    device_name: Optional[str] = None
    auth_code: Optional[str] = Field(None, description="OAuth authorization code")
    access_token: Optional[str] = Field(None, description="Direct access token if available")
    refresh_token: Optional[str] = None


class DeviceSettingsUpdate(BaseModel):
    device_name: Optional[str] = None
    auto_sync_enabled: Optional[bool] = None
    sync_frequency_minutes: Optional[int] = None
    uv_alert_threshold: Optional[float] = None
    outdoor_detection_enabled: Optional[bool] = None


class UVReadingCreate(BaseModel):
    reading_timestamp: datetime
    duration_seconds: int = 60
    uv_index: float
    uv_source: str = "device_sensor"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    activity_type: Optional[str] = None
    is_outdoor: Optional[bool] = True
    weather_condition: Optional[str] = None


class BulkUVReadingsRequest(BaseModel):
    device_id: int
    readings: List[UVReadingCreate]


class CorrelationAnalysisRequest(BaseModel):
    lesion_group_id: int
    analysis_days: int = Field(90, ge=7, le=365)


class HealthKitWorkout(BaseModel):
    id: str
    type: str
    startDate: str
    endDate: str
    duration: float
    isOutdoor: bool


class HealthKitUVReading(BaseModel):
    startDate: str
    endDate: str
    value: float
    source: str


class WorkoutLocation(BaseModel):
    latitude: float
    longitude: float
    timestamp: str


class HealthKitSyncRequest(BaseModel):
    startDate: str
    endDate: str
    outdoorMinutes: int
    estimatedUVExposure: float
    workouts: List[HealthKitWorkout] = []
    uvReadings: List[HealthKitUVReading] = []
    workoutLocations: List[WorkoutLocation] = []


# =============================================================================
# DEVICE CONNECTION ENDPOINTS
# =============================================================================

@router.get("/supported-devices")
async def get_supported_devices():
    """Get list of supported wearable devices and their capabilities."""
    return {
        "devices": [
            {
                "type": "apple_watch",
                "name": "Apple Watch",
                "logo": "apple_watch_logo.png",
                "capabilities": {
                    "uv_sensor": False,
                    "location": True,
                    "activity": True,
                    "heart_rate": True
                },
                "oauth_required": True,
                "setup_url": "https://developer.apple.com/healthkit/",
                "notes": "UV data derived from location + weather APIs"
            },
            {
                "type": "fitbit",
                "name": "Fitbit",
                "logo": "fitbit_logo.png",
                "capabilities": {
                    "uv_sensor": False,
                    "location": True,
                    "activity": True,
                    "heart_rate": True
                },
                "oauth_required": True,
                "setup_url": "https://dev.fitbit.com/",
                "notes": "Outdoor activity detection + location-based UV"
            },
            {
                "type": "garmin",
                "name": "Garmin",
                "logo": "garmin_logo.png",
                "capabilities": {
                    "uv_sensor": True,
                    "location": True,
                    "activity": True,
                    "heart_rate": True
                },
                "oauth_required": True,
                "setup_url": "https://developer.garmin.com/",
                "notes": "Some models have built-in UV sensors"
            },
            {
                "type": "samsung",
                "name": "Samsung Galaxy Watch",
                "logo": "samsung_logo.png",
                "capabilities": {
                    "uv_sensor": False,
                    "location": True,
                    "activity": True,
                    "heart_rate": True
                },
                "oauth_required": True,
                "setup_url": "https://developer.samsung.com/health",
                "notes": "Samsung Health integration"
            },
            {
                "type": "withings",
                "name": "Withings",
                "logo": "withings_logo.png",
                "capabilities": {
                    "uv_sensor": False,
                    "location": False,
                    "activity": True,
                    "heart_rate": True
                },
                "oauth_required": True,
                "setup_url": "https://developer.withings.com/",
                "notes": "Activity tracking only, UV from manual entry"
            }
        ]
    }


@router.get("/oauth/{device_type}/url")
async def get_oauth_url(
    device_type: str,
    redirect_uri: str = Query(..., description="Callback URL after OAuth"),
    current_user: User = Depends(get_current_user)
):
    """Get OAuth authorization URL for a device type."""

    oauth_configs = {
        "fitbit": {
            "auth_url": "https://www.fitbit.com/oauth2/authorize",
            "client_id": os.getenv("FITBIT_CLIENT_ID", ""),
            "scope": "activity heartrate location profile",
            "response_type": "code"
        },
        "garmin": {
            "auth_url": "https://connect.garmin.com/oauthConfirm",
            "client_id": os.getenv("GARMIN_CLIENT_ID", ""),
            "scope": "activity_read",
            "response_type": "code"
        },
        "withings": {
            "auth_url": "https://account.withings.com/oauth2_user/authorize2",
            "client_id": os.getenv("WITHINGS_CLIENT_ID", ""),
            "scope": "user.activity",
            "response_type": "code"
        }
    }

    if device_type not in oauth_configs:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported device type: {device_type}. Use apple_watch for HealthKit."
        )

    config = oauth_configs[device_type]

    if not config["client_id"]:
        raise HTTPException(
            status_code=503,
            detail=f"{device_type} integration not configured. Please set API credentials."
        )

    # Build OAuth URL
    oauth_url = (
        f"{config['auth_url']}?"
        f"client_id={config['client_id']}&"
        f"response_type={config['response_type']}&"
        f"scope={config['scope']}&"
        f"redirect_uri={redirect_uri}&"
        f"state={current_user.id}"
    )

    return {"oauth_url": oauth_url, "device_type": device_type}


@router.post("/connect")
async def connect_device(
    request: DeviceConnectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Connect a new wearable device."""

    # Check if device already connected
    existing = db.query(WearableDevice).filter(
        WearableDevice.user_id == current_user.id,
        WearableDevice.device_type == request.device_type,
        WearableDevice.is_connected == True
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"A {request.device_type} device is already connected. Disconnect it first."
        )

    # Set device capabilities based on type
    capabilities = {
        "apple_watch": {"uv_sensor": False, "location": True, "activity": True, "heart_rate": True},
        "fitbit": {"uv_sensor": False, "location": True, "activity": True, "heart_rate": True},
        "garmin": {"uv_sensor": True, "location": True, "activity": True, "heart_rate": True},
        "samsung": {"uv_sensor": False, "location": True, "activity": True, "heart_rate": True},
        "withings": {"uv_sensor": False, "location": False, "activity": True, "heart_rate": True},
    }

    caps = capabilities.get(request.device_type, {})

    # Generate unique device ID
    import uuid
    device_id = f"{request.device_type}_{current_user.id}_{uuid.uuid4().hex[:8]}"

    # Create device record
    device = WearableDevice(
        user_id=current_user.id,
        device_type=request.device_type,
        device_model=request.device_model,
        device_name=request.device_name or f"My {request.device_type.replace('_', ' ').title()}",
        device_id=device_id,
        is_connected=True,
        connection_status="active",
        access_token=request.access_token,
        refresh_token=request.refresh_token,
        has_uv_sensor=caps.get("uv_sensor", False),
        has_location=caps.get("location", True),
        has_activity_tracking=caps.get("activity", True),
        has_heart_rate=caps.get("heart_rate", True),
        supported_data_types=["activity", "location"] + (["uv_exposure"] if caps.get("uv_sensor") else []),
        connected_at=datetime.utcnow()
    )

    db.add(device)
    db.commit()
    db.refresh(device)

    return {
        "message": f"{request.device_type} connected successfully",
        "device": {
            "id": device.id,
            "device_type": device.device_type,
            "device_name": device.device_name,
            "has_uv_sensor": device.has_uv_sensor,
            "connection_status": device.connection_status,
            "connected_at": device.connected_at.isoformat()
        }
    }


@router.get("/devices")
async def get_connected_devices(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all connected wearable devices for the user."""

    devices = db.query(WearableDevice).filter(
        WearableDevice.user_id == current_user.id
    ).order_by(WearableDevice.connected_at.desc()).all()

    return {
        "devices": [
            {
                "id": d.id,
                "device_type": d.device_type,
                "device_model": d.device_model,
                "device_name": d.device_name,
                "is_connected": d.is_connected,
                "connection_status": d.connection_status,
                "last_sync_at": d.last_sync_at.isoformat() if d.last_sync_at else None,
                "has_uv_sensor": d.has_uv_sensor,
                "total_uv_readings": d.total_uv_readings,
                "auto_sync_enabled": d.auto_sync_enabled,
                "uv_alert_threshold": d.uv_alert_threshold,
                "connected_at": d.connected_at.isoformat() if d.connected_at else None
            }
            for d in devices
        ],
        "total": len(devices),
        "active_count": sum(1 for d in devices if d.is_connected)
    }


@router.patch("/devices/{device_id}")
async def update_device_settings(
    device_id: int,
    settings: DeviceSettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update device settings."""

    device = db.query(WearableDevice).filter(
        WearableDevice.id == device_id,
        WearableDevice.user_id == current_user.id
    ).first()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    if settings.device_name is not None:
        device.device_name = settings.device_name
    if settings.auto_sync_enabled is not None:
        device.auto_sync_enabled = settings.auto_sync_enabled
    if settings.sync_frequency_minutes is not None:
        device.sync_frequency_minutes = settings.sync_frequency_minutes
    if settings.uv_alert_threshold is not None:
        device.uv_alert_threshold = settings.uv_alert_threshold
    if settings.outdoor_detection_enabled is not None:
        device.outdoor_detection_enabled = settings.outdoor_detection_enabled

    device.updated_at = datetime.utcnow()
    db.commit()

    return {"message": "Device settings updated", "device_id": device_id}


@router.delete("/devices/{device_id}")
async def disconnect_device(
    device_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Disconnect a wearable device."""

    device = db.query(WearableDevice).filter(
        WearableDevice.id == device_id,
        WearableDevice.user_id == current_user.id
    ).first()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    device.is_connected = False
    device.connection_status = "disconnected"
    device.disconnected_at = datetime.utcnow()
    device.access_token = None
    device.refresh_token = None

    db.commit()

    return {"message": f"{device.device_name} disconnected successfully"}


# =============================================================================
# UV DATA SYNC ENDPOINTS
# =============================================================================

@router.post("/devices/{device_id}/sync")
async def sync_device_data(
    device_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Trigger a manual sync for a device."""

    device = db.query(WearableDevice).filter(
        WearableDevice.id == device_id,
        WearableDevice.user_id == current_user.id,
        WearableDevice.is_connected == True
    ).first()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found or not connected")

    # Queue background sync task
    background_tasks.add_task(perform_device_sync, device.id, current_user.id)

    return {
        "message": "Sync started",
        "device_id": device_id,
        "device_type": device.device_type
    }


async def perform_device_sync(device_id: int, user_id: int):
    """Background task to sync device data."""
    db = SessionLocal()
    try:
        device = db.query(WearableDevice).filter(WearableDevice.id == device_id).first()
        if not device:
            return

        device.total_syncs += 1

        # In production, this would call the actual device APIs
        # For now, we simulate a successful sync
        device.last_sync_at = datetime.utcnow()
        device.successful_syncs += 1
        device.connection_status = "active"

        db.commit()
    except Exception as e:
        device.failed_syncs += 1
        device.connection_status = "error"
        db.commit()
    finally:
        db.close()


@router.post("/healthkit/sync")
async def sync_healthkit_data(
    request: HealthKitSyncRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Sync health data from Apple HealthKit.

    This endpoint receives workout and UV exposure data directly from
    the iOS HealthKit integration. If workout locations are provided,
    we fetch accurate UV data from OpenUV API.
    """
    # Import OpenUV service
    try:
        from openuv_service import get_openuv_service
        openuv = get_openuv_service()
        openuv_available = openuv.is_configured
    except ImportError:
        openuv = None
        openuv_available = False

    # Find or create Apple Health device for this user
    device = db.query(WearableDevice).filter(
        WearableDevice.user_id == current_user.id,
        WearableDevice.device_type == "apple_health"
    ).first()

    if not device:
        # Create the device if it doesn't exist
        import uuid
        device = WearableDevice(
            user_id=current_user.id,
            device_type="apple_health",
            device_name="Apple Health",
            device_id=f"apple_health_{current_user.id}_{uuid.uuid4().hex[:8]}",
            is_connected=True,
            connection_status="active",
            has_uv_sensor=False,
            has_location=True,
            has_activity_tracking=True,
            has_heart_rate=True,
            supported_data_types=["activity", "workouts", "heart_rate"],
            connected_at=datetime.utcnow()
        )
        db.add(device)
        db.commit()
        db.refresh(device)

    try:
        # Parse dates
        start_date = datetime.fromisoformat(request.startDate.replace('Z', '+00:00'))
        end_date = datetime.fromisoformat(request.endDate.replace('Z', '+00:00'))

        readings_created = 0
        openuv_readings = 0

        # Get UV data from OpenUV for workout locations if available
        location_uv_data = {}
        if openuv_available and request.workoutLocations:
            try:
                uv_results = await openuv.get_uv_for_locations([
                    {"latitude": loc.latitude, "longitude": loc.longitude, "timestamp": loc.timestamp}
                    for loc in request.workoutLocations
                ])
                # Index by timestamp for quick lookup
                for result in uv_results:
                    location_uv_data[result["timestamp"]] = result
            except Exception as e:
                print(f"OpenUV lookup failed: {e}")

        # Process UV readings if any
        for uv_reading in request.uvReadings:
            reading_start = datetime.fromisoformat(uv_reading.startDate.replace('Z', '+00:00'))
            reading_end = datetime.fromisoformat(uv_reading.endDate.replace('Z', '+00:00'))
            duration_seconds = int((reading_end - reading_start).total_seconds())

            reading = WearableUVReading(
                user_id=current_user.id,
                device_id=device.id,
                reading_timestamp=reading_start,
                duration_seconds=duration_seconds,
                reading_date=reading_start.date(),
                uv_index=uv_reading.value,
                uv_dose=uv_reading.value * (duration_seconds / 60),
                uv_source=uv_reading.source,
                is_outdoor=True,
                reading_risk_score=calculate_reading_risk(uv_reading.value, duration_seconds)
            )
            db.add(reading)
            readings_created += 1

        # Process outdoor workouts with UV data
        for workout in request.workouts:
            if workout.isOutdoor and workout.duration > 0:
                workout_start = datetime.fromisoformat(workout.startDate.replace('Z', '+00:00'))

                # Check if we have OpenUV data for this workout
                uv_data = location_uv_data.get(workout.startDate)

                if uv_data and uv_data.get("source") == "openuv":
                    # Use accurate UV data from OpenUV
                    uv_index = uv_data.get("uv_index", 5.0)
                    uv_source = "openuv"
                    openuv_readings += 1
                else:
                    # Fallback to time-based estimation
                    hour = workout_start.hour
                    month = workout_start.month

                    # Seasonal adjustment
                    if month in [6, 7, 8]:  # Summer
                        seasonal_factor = 1.3
                    elif month in [12, 1, 2]:  # Winter
                        seasonal_factor = 0.6
                    else:
                        seasonal_factor = 1.0

                    # Time of day
                    if 10 <= hour <= 14:
                        base_uv = 8.0  # Peak hours
                    elif 8 <= hour <= 16:
                        base_uv = 5.0  # Moderate hours
                    elif 6 <= hour <= 18:
                        base_uv = 2.0  # Low UV
                    else:
                        base_uv = 0.5  # Night

                    uv_index = round(base_uv * seasonal_factor, 1)
                    uv_source = "estimated"

                # Get latitude/longitude if available
                latitude = None
                longitude = None
                if uv_data:
                    latitude = uv_data.get("latitude")
                    longitude = uv_data.get("longitude")

                reading = WearableUVReading(
                    user_id=current_user.id,
                    device_id=device.id,
                    reading_timestamp=workout_start,
                    duration_seconds=int(workout.duration),
                    reading_date=workout_start.date(),
                    uv_index=uv_index,
                    uv_dose=uv_index * (workout.duration / 3600),  # UV dose = UV * hours
                    uv_source=uv_source,
                    activity_type=workout.type,
                    is_outdoor=True,
                    latitude=latitude,
                    longitude=longitude,
                    reading_risk_score=calculate_reading_risk(uv_index, int(workout.duration))
                )
                db.add(reading)
                readings_created += 1

        # Update device stats
        device.total_uv_readings += readings_created
        device.last_sync_at = datetime.utcnow()
        device.total_syncs += 1
        device.successful_syncs += 1
        device.connection_status = "active"

        db.commit()

        # Update daily summaries in background
        if readings_created > 0:
            all_readings = []
            for workout in request.workouts:
                if workout.isOutdoor and workout.duration > 0:
                    workout_start = datetime.fromisoformat(workout.startDate.replace('Z', '+00:00'))
                    uv_data = location_uv_data.get(workout.startDate)
                    uv_index = uv_data.get("uv_index", 5.0) if uv_data else 5.0

                    all_readings.append(UVReadingCreate(
                        reading_timestamp=workout_start,
                        duration_seconds=int(workout.duration),
                        uv_index=uv_index
                    ))

            if all_readings:
                await update_daily_summaries(db, current_user.id, all_readings)

        return {
            "success": True,
            "message": f"Synced {readings_created} readings from HealthKit",
            "readings_created": readings_created,
            "openuv_readings": openuv_readings,
            "estimated_readings": readings_created - openuv_readings - len(request.uvReadings),
            "outdoor_minutes": request.outdoorMinutes,
            "workouts_processed": len(request.workouts),
            "openuv_available": openuv_available,
            "sync_period": {
                "start": request.startDate,
                "end": request.endDate
            }
        }

    except Exception as e:
        device.failed_syncs += 1
        device.connection_status = "error"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to sync HealthKit data: {str(e)}")


@router.post("/uv-readings")
async def submit_uv_readings(
    request: BulkUVReadingsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit UV readings from a device (batch upload)."""

    device = db.query(WearableDevice).filter(
        WearableDevice.id == request.device_id,
        WearableDevice.user_id == current_user.id
    ).first()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    readings_created = 0

    for reading_data in request.readings:
        reading = WearableUVReading(
            user_id=current_user.id,
            device_id=device.id,
            reading_timestamp=reading_data.reading_timestamp,
            duration_seconds=reading_data.duration_seconds,
            reading_date=reading_data.reading_timestamp.date(),
            uv_index=reading_data.uv_index,
            uv_dose=reading_data.uv_index * (reading_data.duration_seconds / 60),  # Simple dose calc
            uv_source=reading_data.uv_source,
            latitude=reading_data.latitude,
            longitude=reading_data.longitude,
            activity_type=reading_data.activity_type,
            is_outdoor=reading_data.is_outdoor,
            weather_condition=reading_data.weather_condition,
            reading_risk_score=calculate_reading_risk(reading_data.uv_index, reading_data.duration_seconds)
        )
        db.add(reading)
        readings_created += 1

    device.total_uv_readings += readings_created
    device.last_sync_at = datetime.utcnow()

    db.commit()

    # Update daily summaries
    await update_daily_summaries(db, current_user.id, request.readings)

    return {
        "message": f"Submitted {readings_created} UV readings",
        "readings_created": readings_created,
        "device_id": device.id
    }


def calculate_reading_risk(uv_index: float, duration_seconds: int) -> float:
    """Calculate risk score for a single UV reading."""
    duration_minutes = duration_seconds / 60

    # Risk increases with UV index and duration
    if uv_index < 3:
        base_risk = 10
    elif uv_index < 6:
        base_risk = 30
    elif uv_index < 8:
        base_risk = 50
    elif uv_index < 11:
        base_risk = 75
    else:
        base_risk = 95

    # Duration multiplier (risk increases with longer exposure)
    duration_multiplier = min(duration_minutes / 30, 2.0)  # Cap at 2x for 30+ minutes

    return min(base_risk * duration_multiplier, 100)


async def update_daily_summaries(db: Session, user_id: int, readings: List[UVReadingCreate]):
    """Update daily UV summaries after new readings."""

    # Group readings by date
    dates = set(r.reading_timestamp.date() for r in readings)

    for summary_date in dates:
        day_readings = db.query(WearableUVReading).filter(
            WearableUVReading.user_id == user_id,
            WearableUVReading.reading_date == summary_date
        ).all()

        if not day_readings:
            continue

        # Check for existing summary
        summary = db.query(WearableDailyUVSummary).filter(
            WearableDailyUVSummary.user_id == user_id,
            WearableDailyUVSummary.summary_date == summary_date
        ).first()

        if not summary:
            summary = WearableDailyUVSummary(
                user_id=user_id,
                summary_date=summary_date
            )
            db.add(summary)

        # Calculate aggregates
        uv_values = [r.uv_index for r in day_readings if r.uv_index is not None]
        outdoor_readings = [r for r in day_readings if r.is_outdoor]

        summary.reading_count = len(day_readings)
        summary.total_outdoor_minutes = sum(r.duration_seconds for r in outdoor_readings) // 60
        summary.total_uv_dose = sum(r.uv_dose or 0 for r in day_readings)

        if uv_values:
            summary.average_uv_index = sum(uv_values) / len(uv_values)
            summary.max_uv_index = max(uv_values)
            summary.min_uv_index = min(uv_values)

        # Time-based breakdown
        morning = [r for r in outdoor_readings if r.reading_timestamp.hour < 10]
        midday = [r for r in outdoor_readings if 10 <= r.reading_timestamp.hour < 16]
        afternoon = [r for r in outdoor_readings if r.reading_timestamp.hour >= 16]

        summary.morning_exposure_minutes = sum(r.duration_seconds for r in morning) // 60
        summary.midday_exposure_minutes = sum(r.duration_seconds for r in midday) // 60
        summary.afternoon_exposure_minutes = sum(r.duration_seconds for r in afternoon) // 60

        # High UV counts
        summary.high_uv_minutes = sum(r.duration_seconds for r in day_readings if r.uv_index and r.uv_index >= 6) // 60
        summary.very_high_uv_minutes = sum(r.duration_seconds for r in day_readings if r.uv_index and r.uv_index >= 8) // 60
        summary.extreme_uv_minutes = sum(r.duration_seconds for r in day_readings if r.uv_index and r.uv_index >= 11) // 60

        # Risk assessment
        summary.daily_risk_score = min(
            (summary.high_uv_minutes * 1.5 +
             summary.very_high_uv_minutes * 3 +
             summary.extreme_uv_minutes * 5 +
             summary.midday_exposure_minutes * 0.5),
            100
        )

        if summary.daily_risk_score < 25:
            summary.risk_category = "low"
        elif summary.daily_risk_score < 50:
            summary.risk_category = "moderate"
        elif summary.daily_risk_score < 75:
            summary.risk_category = "high"
        else:
            summary.risk_category = "very_high"

        summary.updated_at = datetime.utcnow()

    db.commit()


# =============================================================================
# UV ANALYTICS ENDPOINTS
# =============================================================================

@router.get("/uv-exposure/today")
async def get_today_uv_exposure(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get today's UV exposure summary."""

    today = date.today()

    summary = db.query(WearableDailyUVSummary).filter(
        WearableDailyUVSummary.user_id == current_user.id,
        WearableDailyUVSummary.summary_date == today
    ).first()

    if not summary:
        return {
            "date": today.isoformat(),
            "has_data": False,
            "total_outdoor_minutes": 0,
            "average_uv_index": None,
            "risk_category": "unknown",
            "message": "No UV data recorded today. Sync your wearable device."
        }

    return {
        "date": today.isoformat(),
        "has_data": True,
        "total_outdoor_minutes": summary.total_outdoor_minutes,
        "total_uv_dose": summary.total_uv_dose,
        "average_uv_index": summary.average_uv_index,
        "max_uv_index": summary.max_uv_index,
        "morning_minutes": summary.morning_exposure_minutes,
        "midday_minutes": summary.midday_exposure_minutes,
        "afternoon_minutes": summary.afternoon_exposure_minutes,
        "high_uv_minutes": summary.high_uv_minutes,
        "daily_risk_score": summary.daily_risk_score,
        "risk_category": summary.risk_category,
        "reading_count": summary.reading_count
    }


@router.get("/uv-exposure/history")
async def get_uv_exposure_history(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get historical UV exposure summaries."""

    start_date = date.today() - timedelta(days=days)

    summaries = db.query(WearableDailyUVSummary).filter(
        WearableDailyUVSummary.user_id == current_user.id,
        WearableDailyUVSummary.summary_date >= start_date
    ).order_by(WearableDailyUVSummary.summary_date.desc()).all()

    # Calculate period statistics
    total_outdoor_time = sum(s.total_outdoor_minutes or 0 for s in summaries)
    total_uv_dose = sum(s.total_uv_dose or 0 for s in summaries)
    high_risk_days = sum(1 for s in summaries if s.risk_category in ["high", "very_high"])

    avg_daily_outdoor = total_outdoor_time / max(len(summaries), 1)

    return {
        "period_days": days,
        "days_with_data": len(summaries),
        "period_stats": {
            "total_outdoor_hours": round(total_outdoor_time / 60, 1),
            "total_uv_dose": round(total_uv_dose, 1),
            "average_daily_outdoor_minutes": round(avg_daily_outdoor, 1),
            "high_risk_days": high_risk_days,
            "high_risk_percentage": round(high_risk_days / max(len(summaries), 1) * 100, 1)
        },
        "daily_summaries": [
            {
                "date": s.summary_date.isoformat(),
                "outdoor_minutes": s.total_outdoor_minutes,
                "uv_dose": s.total_uv_dose,
                "avg_uv_index": s.average_uv_index,
                "max_uv_index": s.max_uv_index,
                "risk_score": s.daily_risk_score,
                "risk_category": s.risk_category
            }
            for s in summaries
        ]
    }


@router.get("/uv-exposure/trends")
async def get_uv_exposure_trends(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get UV exposure trends and patterns."""

    # Get last 90 days of data
    start_date = date.today() - timedelta(days=90)

    summaries = db.query(WearableDailyUVSummary).filter(
        WearableDailyUVSummary.user_id == current_user.id,
        WearableDailyUVSummary.summary_date >= start_date
    ).all()

    if len(summaries) < 7:
        return {
            "has_sufficient_data": False,
            "message": "Need at least 7 days of data for trend analysis"
        }

    # Weekly averages
    weeks = {}
    for s in summaries:
        week_num = s.summary_date.isocalendar()[1]
        if week_num not in weeks:
            weeks[week_num] = []
        weeks[week_num].append(s)

    weekly_trends = []
    for week_num, week_data in sorted(weeks.items()):
        avg_outdoor = sum(d.total_outdoor_minutes or 0 for d in week_data) / len(week_data)
        avg_uv = sum(d.average_uv_index or 0 for d in week_data) / len(week_data)
        avg_risk = sum(d.daily_risk_score or 0 for d in week_data) / len(week_data)
        weekly_trends.append({
            "week": week_num,
            "avg_outdoor_minutes": round(avg_outdoor, 1),
            "avg_uv_index": round(avg_uv, 2),
            "avg_risk_score": round(avg_risk, 1)
        })

    # Time of day patterns
    total_morning = sum(s.morning_exposure_minutes or 0 for s in summaries)
    total_midday = sum(s.midday_exposure_minutes or 0 for s in summaries)
    total_afternoon = sum(s.afternoon_exposure_minutes or 0 for s in summaries)
    total_time = total_morning + total_midday + total_afternoon

    # Trend direction
    recent = summaries[-14:] if len(summaries) >= 14 else summaries
    older = summaries[:-14] if len(summaries) >= 14 else []

    if older:
        recent_avg_risk = sum(s.daily_risk_score or 0 for s in recent) / len(recent)
        older_avg_risk = sum(s.daily_risk_score or 0 for s in older) / len(older)
        risk_trend = "increasing" if recent_avg_risk > older_avg_risk * 1.1 else \
                     "decreasing" if recent_avg_risk < older_avg_risk * 0.9 else "stable"
    else:
        risk_trend = "insufficient_data"

    return {
        "has_sufficient_data": True,
        "analysis_period_days": 90,
        "days_with_data": len(summaries),
        "weekly_trends": weekly_trends,
        "time_of_day_breakdown": {
            "morning_percentage": round(total_morning / max(total_time, 1) * 100, 1),
            "midday_percentage": round(total_midday / max(total_time, 1) * 100, 1),
            "afternoon_percentage": round(total_afternoon / max(total_time, 1) * 100, 1),
            "recommendation": "Good - mostly avoiding peak UV hours" if total_midday / max(total_time, 1) < 0.4
                             else "Consider reducing midday sun exposure (10am-4pm)"
        },
        "risk_trend": risk_trend,
        "insights": generate_uv_insights(summaries)
    }


def generate_uv_insights(summaries: List[WearableDailyUVSummary]) -> List[Dict]:
    """Generate personalized insights from UV data."""
    insights = []

    if not summaries:
        return insights

    # High midday exposure
    midday_heavy = [s for s in summaries if (s.midday_exposure_minutes or 0) > 60]
    if len(midday_heavy) > len(summaries) * 0.3:
        insights.append({
            "type": "warning",
            "title": "High Midday Exposure",
            "message": f"You spent significant time outdoors during peak UV hours (10am-4pm) on {len(midday_heavy)} days. Consider shifting outdoor activities to morning or evening.",
            "priority": "high"
        })

    # Consistent outdoor activity
    avg_outdoor = sum(s.total_outdoor_minutes or 0 for s in summaries) / len(summaries)
    if avg_outdoor > 30:
        insights.append({
            "type": "info",
            "title": "Active Outdoor Lifestyle",
            "message": f"You average {round(avg_outdoor)} minutes outdoors daily. Great for health, but ensure consistent sun protection.",
            "priority": "medium"
        })

    # High-risk days
    very_high_risk = [s for s in summaries if s.risk_category == "very_high"]
    if very_high_risk:
        insights.append({
            "type": "alert",
            "title": "High UV Exposure Days Detected",
            "message": f"You had {len(very_high_risk)} very high UV exposure days. These significantly increase skin damage risk.",
            "priority": "high"
        })

    # Improvement trend
    recent = summaries[-7:] if len(summaries) >= 7 else summaries
    older = summaries[-14:-7] if len(summaries) >= 14 else []
    if older:
        recent_risk = sum(s.daily_risk_score or 0 for s in recent) / len(recent)
        older_risk = sum(s.daily_risk_score or 0 for s in older) / len(older)
        if recent_risk < older_risk * 0.8:
            insights.append({
                "type": "success",
                "title": "Improving Sun Safety",
                "message": "Your UV exposure risk has decreased compared to the previous week. Keep up the good habits!",
                "priority": "low"
            })

    return insights


# =============================================================================
# LESION CORRELATION ENDPOINTS
# =============================================================================

@router.post("/correlations/analyze")
async def analyze_lesion_uv_correlation(
    request: CorrelationAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Analyze correlation between UV exposure and a specific lesion."""

    # Get the lesion group
    lesion_group = db.query(LesionGroup).filter(
        LesionGroup.id == request.lesion_group_id,
        LesionGroup.user_id == current_user.id
    ).first()

    if not lesion_group:
        raise HTTPException(status_code=404, detail="Lesion group not found")

    # Get UV data for the analysis period
    end_date = date.today()
    start_date = end_date - timedelta(days=request.analysis_days)

    uv_summaries = db.query(WearableDailyUVSummary).filter(
        WearableDailyUVSummary.user_id == current_user.id,
        WearableDailyUVSummary.summary_date >= start_date,
        WearableDailyUVSummary.summary_date <= end_date
    ).order_by(WearableDailyUVSummary.summary_date).all()

    if len(uv_summaries) < 7:
        raise HTTPException(
            status_code=400,
            detail="Insufficient UV data for correlation analysis. Need at least 7 days."
        )

    # Get lesion analyses in the period
    lesion_analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == request.lesion_group_id,
        AnalysisHistory.created_at >= datetime.combine(start_date, datetime.min.time())
    ).order_by(AnalysisHistory.created_at).all()

    # Calculate correlation metrics
    total_uv_dose = sum(s.total_uv_dose or 0 for s in uv_summaries)
    total_outdoor_hours = sum(s.total_outdoor_minutes or 0 for s in uv_summaries) / 60
    high_uv_days = sum(1 for s in uv_summaries if s.risk_category in ["high", "very_high"])
    avg_risk = sum(s.daily_risk_score or 0 for s in uv_summaries) / len(uv_summaries)

    # Determine correlation strength (simplified algorithm)
    # In production, would use proper statistical analysis
    if high_uv_days > len(uv_summaries) * 0.4 and len(lesion_analyses) > 1:
        correlation_strength = "strong"
        correlation_coefficient = 0.75
        uv_contribution_score = 80
    elif high_uv_days > len(uv_summaries) * 0.2:
        correlation_strength = "moderate"
        correlation_coefficient = 0.45
        uv_contribution_score = 50
    else:
        correlation_strength = "weak"
        correlation_coefficient = 0.15
        uv_contribution_score = 20

    # Create or update correlation record
    existing = db.query(WearableLesionCorrelation).filter(
        WearableLesionCorrelation.user_id == current_user.id,
        WearableLesionCorrelation.lesion_group_id == request.lesion_group_id
    ).first()

    if existing:
        correlation = existing
    else:
        correlation = WearableLesionCorrelation(
            user_id=current_user.id,
            lesion_group_id=request.lesion_group_id
        )
        db.add(correlation)

    correlation.analysis_start_date = start_date
    correlation.analysis_end_date = end_date
    correlation.days_analyzed = request.analysis_days
    correlation.lesion_body_location = lesion_group.body_location
    correlation.cumulative_uv_dose = total_uv_dose
    correlation.cumulative_outdoor_hours = total_outdoor_hours
    correlation.correlation_coefficient = correlation_coefficient
    correlation.correlation_strength = correlation_strength
    correlation.uv_contribution_score = uv_contribution_score
    correlation.analyzed_at = datetime.utcnow()

    # Generate recommendations
    recommendations = []
    if uv_contribution_score > 60:
        recommendations.append("Strongly consider using SPF 50+ sunscreen daily")
        recommendations.append(f"Limit outdoor exposure during 10am-4pm for {lesion_group.body_location}")
        recommendations.append("Schedule a dermatologist follow-up to discuss sun protection strategies")
    elif uv_contribution_score > 30:
        recommendations.append("Use SPF 30+ sunscreen when outdoors")
        recommendations.append("Consider wearing protective clothing for extended outdoor activities")
    else:
        recommendations.append("Continue current sun protection habits")
        recommendations.append("Monitor this lesion for any changes")

    correlation.personalized_recommendations = recommendations

    db.commit()
    db.refresh(correlation)

    return {
        "correlation_id": correlation.id,
        "lesion_group_id": request.lesion_group_id,
        "body_location": lesion_group.body_location,
        "analysis_period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days_analyzed": request.analysis_days,
            "days_with_uv_data": len(uv_summaries)
        },
        "uv_exposure_summary": {
            "total_uv_dose": round(total_uv_dose, 2),
            "total_outdoor_hours": round(total_outdoor_hours, 1),
            "high_risk_days": high_uv_days,
            "average_daily_risk": round(avg_risk, 1)
        },
        "correlation_analysis": {
            "correlation_strength": correlation_strength,
            "correlation_coefficient": correlation_coefficient,
            "uv_contribution_score": uv_contribution_score,
            "confidence": "moderate" if len(uv_summaries) > 30 else "low"
        },
        "lesion_timeline": {
            "analyses_in_period": len(lesion_analyses),
            "first_detected": lesion_group.created_at.isoformat() if lesion_group.created_at else None
        },
        "recommendations": recommendations,
        "analyzed_at": correlation.analyzed_at.isoformat()
    }


@router.get("/correlations")
async def get_all_correlations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all UV-lesion correlations for the user."""

    correlations = db.query(WearableLesionCorrelation).filter(
        WearableLesionCorrelation.user_id == current_user.id
    ).order_by(WearableLesionCorrelation.analyzed_at.desc()).all()

    return {
        "correlations": [
            {
                "id": c.id,
                "lesion_group_id": c.lesion_group_id,
                "body_location": c.lesion_body_location,
                "days_analyzed": c.days_analyzed,
                "correlation_strength": c.correlation_strength,
                "uv_contribution_score": c.uv_contribution_score,
                "cumulative_uv_dose": c.cumulative_uv_dose,
                "recommendations": c.personalized_recommendations,
                "analyzed_at": c.analyzed_at.isoformat() if c.analyzed_at else None
            }
            for c in correlations
        ],
        "total": len(correlations)
    }


@router.get("/correlations/{lesion_group_id}")
async def get_lesion_correlation(
    lesion_group_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get UV correlation for a specific lesion."""

    correlation = db.query(WearableLesionCorrelation).filter(
        WearableLesionCorrelation.user_id == current_user.id,
        WearableLesionCorrelation.lesion_group_id == lesion_group_id
    ).first()

    if not correlation:
        return {
            "has_correlation": False,
            "message": "No UV correlation analysis found for this lesion. Run an analysis first."
        }

    return {
        "has_correlation": True,
        "correlation": {
            "id": correlation.id,
            "lesion_group_id": correlation.lesion_group_id,
            "body_location": correlation.lesion_body_location,
            "analysis_period": {
                "start": correlation.analysis_start_date.isoformat() if correlation.analysis_start_date else None,
                "end": correlation.analysis_end_date.isoformat() if correlation.analysis_end_date else None,
                "days": correlation.days_analyzed
            },
            "uv_metrics": {
                "cumulative_dose": correlation.cumulative_uv_dose,
                "outdoor_hours": correlation.cumulative_outdoor_hours,
                "uv_30_day_avg": correlation.uv_30_day_avg,
                "uv_trend": correlation.uv_trend
            },
            "correlation_results": {
                "coefficient": correlation.correlation_coefficient,
                "strength": correlation.correlation_strength,
                "confidence": correlation.correlation_confidence,
                "uv_contribution_score": correlation.uv_contribution_score
            },
            "recommendations": correlation.personalized_recommendations,
            "high_risk_times": correlation.high_risk_times,
            "analyzed_at": correlation.analyzed_at.isoformat() if correlation.analyzed_at else None
        }
    }


# =============================================================================
# OPENUV ENDPOINTS
# =============================================================================

@router.get("/uv/current")
async def get_current_uv(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    skin_type: int = Query(2, ge=1, le=6, description="Fitzpatrick skin type 1-6"),
    outdoor_minutes: int = Query(30, ge=0, description="Planned outdoor duration")
):
    """
    Get current UV index and protection advice for a location.

    Uses OpenUV API for accurate real-time UV data. Falls back to
    time-based estimation if API is not configured.
    """
    try:
        from openuv_service import get_openuv_service
        openuv = get_openuv_service()
    except ImportError:
        openuv = None

    uv_data = None
    source = "estimated"

    if openuv and openuv.is_configured:
        try:
            uv_data = await openuv.get_uv_index(latitude, longitude)
            if uv_data:
                source = "openuv"
        except Exception as e:
            print(f"OpenUV API error: {e}")

    if uv_data:
        uv_index = uv_data.uv_index
        uv_max = uv_data.uv_index_max
        safe_times = uv_data.safe_exposure_times
        sun_info = {k: v.isoformat() if v else None for k, v in uv_data.sun_info.items()}
    else:
        # Fallback estimation
        from datetime import datetime
        now = datetime.now()
        hour = now.hour
        month = now.month

        # Seasonal factor
        if month in [6, 7, 8]:
            seasonal = 1.3
        elif month in [12, 1, 2]:
            seasonal = 0.6
        else:
            seasonal = 1.0

        # Time of day
        if 10 <= hour <= 14:
            uv_index = round(8.0 * seasonal, 1)
        elif 8 <= hour <= 16:
            uv_index = round(5.0 * seasonal, 1)
        elif 6 <= hour <= 18:
            uv_index = round(2.0 * seasonal, 1)
        else:
            uv_index = 0.5

        uv_max = uv_index * 1.2
        safe_times = {}
        sun_info = {}

    # Get protection advice
    if openuv:
        advice = openuv.get_protection_advice(uv_index, skin_type, outdoor_minutes)
        protection = {
            "required": advice.protection_required,
            "safe_exposure_minutes": advice.safe_exposure_minutes,
            "risk_level": advice.risk_level,
            "recommendations": advice.recommendations
        }
    else:
        # Basic advice
        if uv_index < 3:
            risk_level = "low"
            recommendations = ["Minimal protection needed", "Wear sunglasses on bright days"]
        elif uv_index < 6:
            risk_level = "moderate"
            recommendations = ["Apply SPF 30+ sunscreen", "Wear hat and sunglasses"]
        elif uv_index < 8:
            risk_level = "high"
            recommendations = ["Apply SPF 50+ sunscreen", "Seek shade", "Wear protective clothing"]
        elif uv_index < 11:
            risk_level = "very_high"
            recommendations = ["Minimize sun exposure 10am-4pm", "SPF 50+ required", "Full coverage recommended"]
        else:
            risk_level = "extreme"
            recommendations = ["Avoid sun exposure if possible", "Maximum protection required"]

        protection = {
            "required": uv_index >= 3,
            "safe_exposure_minutes": None,
            "risk_level": risk_level,
            "recommendations": recommendations
        }

    return {
        "location": {"latitude": latitude, "longitude": longitude},
        "uv_index": uv_index,
        "uv_max": uv_max,
        "source": source,
        "safe_exposure_times": safe_times,
        "sun_info": sun_info,
        "protection_advice": protection,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/uv/forecast")
async def get_uv_forecast(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180)
):
    """
    Get UV forecast for the next 48 hours.

    Requires OpenUV API to be configured.
    """
    try:
        from openuv_service import get_openuv_service
        openuv = get_openuv_service()
    except ImportError:
        return {"error": "OpenUV service not available", "forecast": []}

    if not openuv.is_configured:
        return {
            "error": "OpenUV API key not configured",
            "message": "Set OPENUV_API_KEY in your .env file. Get a free key at https://www.openuv.io/",
            "forecast": []
        }

    try:
        forecast = await openuv.get_uv_forecast(latitude, longitude)
        return {
            "location": {"latitude": latitude, "longitude": longitude},
            "forecast": [
                {"uv_index": f.uv_index, "time": f.uv_time.isoformat()}
                for f in forecast
            ],
            "count": len(forecast)
        }
    except Exception as e:
        return {"error": str(e), "forecast": []}


@router.get("/openuv/status")
async def get_openuv_status():
    """
    Check OpenUV API configuration and status.
    """
    try:
        from openuv_service import get_openuv_service
        openuv = get_openuv_service()

        return {
            "configured": openuv.is_configured,
            "api_url": "https://api.openuv.io/api/v1",
            "free_tier_limit": "50 requests/day",
            "setup_instructions": {
                "1": "Sign up at https://www.openuv.io/",
                "2": "Get your API key from the dashboard",
                "3": "Add to .env: OPENUV_API_KEY=your_key_here",
                "4": "Restart the backend server"
            }
        }
    except ImportError:
        return {
            "configured": False,
            "error": "OpenUV service module not found"
        }
