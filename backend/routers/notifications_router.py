"""
Notifications and Alerts Router

Endpoints for:
- User notifications management
- High-priority alerts
- Push notification device registration
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from database import get_db, User, Notification
from auth import get_current_active_user

router = APIRouter(prefix="/notifications", tags=["Notifications"])


@router.get("")
async def get_user_notifications(
    unread_only: bool = False,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get notifications for the current user."""
    query = db.query(Notification).filter(Notification.user_id == current_user.id)

    if unread_only:
        query = query.filter(Notification.is_read == False)

    # Filter out expired notifications
    query = query.filter(
        (Notification.expires_at == None) | (Notification.expires_at > datetime.utcnow())
    )

    total = query.count()
    notifications = query.order_by(Notification.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "unread_count": db.query(Notification).filter(
            Notification.user_id == current_user.id,
            Notification.is_read == False
        ).count(),
        "notifications": [
            {
                "id": n.id,
                "type": n.notification_type,
                "title": n.title,
                "message": n.message,
                "data": n.data,
                "is_read": n.is_read,
                "priority": n.priority,
                "created_at": n.created_at.isoformat() if n.created_at else None,
                "read_at": n.read_at.isoformat() if n.read_at else None
            }
            for n in notifications
        ]
    }


@router.post("/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark a notification as read."""
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()

    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    notification.is_read = True
    notification.read_at = datetime.utcnow()
    db.commit()

    return {"success": True, "message": "Notification marked as read"}


@router.post("/read-all")
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark all notifications as read."""
    db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    ).update({
        "is_read": True,
        "read_at": datetime.utcnow()
    })
    db.commit()

    return {"success": True, "message": "All notifications marked as read"}


@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a notification."""
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()

    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    db.delete(notification)
    db.commit()

    return {"success": True, "message": "Notification deleted"}


# =============================================================================
# ALERTS (High-priority notifications)
# =============================================================================

# Note: This uses a separate prefix, will be mounted at /alerts
alerts_router = APIRouter(prefix="/alerts", tags=["Alerts"])


@alerts_router.get("")
async def get_user_alerts(
    unread_only: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get alerts (high-priority notifications) for the current user.
    Alerts are notifications with priority 'high' or 'urgent'.
    """
    try:
        query = db.query(Notification).filter(
            Notification.user_id == current_user.id,
            Notification.priority.in_(["high", "urgent"])
        )

        if unread_only:
            query = query.filter(Notification.is_read == False)

        # Filter out expired notifications
        query = query.filter(
            (Notification.expires_at == None) | (Notification.expires_at > datetime.utcnow())
        )

        alerts = query.order_by(Notification.created_at.desc()).limit(50).all()

        return {
            "alerts": [
                {
                    "id": a.id,
                    "type": a.notification_type,
                    "title": a.title,
                    "message": a.message,
                    "data": a.data,
                    "priority": a.priority,
                    "is_read": a.is_read,
                    "created_at": a.created_at.isoformat() if a.created_at else None
                }
                for a in alerts
            ],
            "unread_count": len([a for a in alerts if not a.is_read])
        }
    except Exception as e:
        print(f"Alerts query error (table may not exist): {e}")
        return {"alerts": [], "unread_count": 0}


# =============================================================================
# PUSH NOTIFICATION DEVICE REGISTRATION
# =============================================================================

@router.post("/register-device")
async def register_push_device(
    device_token: str,
    platform: str = "ios",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Register a device for push notifications."""
    from database import UserProfile

    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()

    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)

    profile.push_token = device_token
    profile.push_platform = platform
    db.commit()

    return {"success": True, "message": "Device registered for push notifications"}


@router.post("/unregister-device")
async def unregister_push_device(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Unregister device from push notifications."""
    from database import UserProfile

    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()

    if profile:
        profile.push_token = None
        profile.push_platform = None
        db.commit()

    return {"success": True, "message": "Device unregistered from push notifications"}
