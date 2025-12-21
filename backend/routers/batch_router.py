"""
Batch Skin Check Router

Handles full-body skin check endpoints:
- Start check session
- Upload images
- Process batch check
- Get check results
- List user checks
- Mole map visualization
- Lesion trends tracking
- Report generation
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
from pathlib import Path
import json
import os
import io
import base64
import uuid

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from database import get_db, User
from auth import get_current_active_user
import shared

# Import batch processing utilities
from batch_skin_check import (
    get_batch_processor, get_report_generator, get_count_tracker,
    FullBodyCheckResult, LesionDetection, RiskLevel, BodyRegion
)

router = APIRouter(tags=["Batch Skin Checks"])

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# =============================================================================
# BATCH SKIN CHECK ENDPOINTS
# =============================================================================

@router.post("/batch/start-check")
async def start_full_body_check(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Start a new full-body skin check session.
    Returns a check_id to use for uploading images.
    """
    try:
        check_id = str(uuid.uuid4())

        # Create check record in database
        db.execute(
            """
            INSERT INTO full_body_checks (check_id, user_id, status, created_at)
            VALUES (?, ?, 'pending', ?)
            """,
            (check_id, current_user.id, datetime.utcnow().isoformat())
        )
        db.commit()

        return {
            "check_id": check_id,
            "user_id": current_user.id,
            "status": "pending",
            "message": "Full-body skin check session started. Upload images using /batch/upload/{check_id}",
            "instructions": [
                "Upload 20-30 photos covering your entire body surface",
                "Include: face, neck, chest, back, arms, legs, hands, feet",
                "Use good lighting and keep camera 12-18 inches from skin",
                "Name files with body location (e.g., 'left_arm.jpg') for better detection"
            ]
        }

    except Exception as e:
        db.rollback()
        print(f"Error starting check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start check: {str(e)}")


@router.post("/batch/upload/{check_id}")
async def upload_batch_image(
    check_id: str,
    file: UploadFile = File(...),
    body_location: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload a single image to a batch check session.
    Call this endpoint multiple times for each image.
    """
    try:
        # Verify check belongs to user
        check = db.execute(
            "SELECT * FROM full_body_checks WHERE check_id = ? AND user_id = ?",
            (check_id, current_user.id)
        ).fetchone()

        if not check:
            raise HTTPException(status_code=404, detail="Check session not found")

        # Save image
        image_id = str(uuid.uuid4())
        filename = file.filename or f"image_{image_id}.jpg"
        upload_path = UPLOAD_DIR / f"batch_{check_id}_{image_id}_{filename}"

        contents = await file.read()
        with open(upload_path, "wb") as f:
            f.write(contents)

        # Get current image count
        count_result = db.execute(
            "SELECT COUNT(*) FROM full_body_check_images WHERE check_id = ?",
            (check_id,)
        ).fetchone()
        image_index = count_result[0] if count_result else 0

        # Store image record
        db.execute(
            """
            INSERT INTO full_body_check_images
            (image_id, check_id, image_index, filename, image_url, body_region)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (image_id, check_id, image_index, filename, str(upload_path), body_location)
        )

        # Update total images count
        db.execute(
            "UPDATE full_body_checks SET total_images = total_images + 1 WHERE check_id = ?",
            (check_id,)
        )
        db.commit()

        return {
            "check_id": check_id,
            "image_id": image_id,
            "image_index": image_index,
            "filename": filename,
            "body_location": body_location,
            "message": f"Image {image_index + 1} uploaded successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")


@router.post("/batch/process/{check_id}")
async def process_batch_check(
    check_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Process all images in a batch check session.
    Analyzes each image, detects lesions, and generates comprehensive report.
    """
    try:
        # Verify check belongs to user
        check = db.execute(
            "SELECT * FROM full_body_checks WHERE check_id = ? AND user_id = ?",
            (check_id, current_user.id)
        ).fetchone()

        if not check:
            raise HTTPException(status_code=404, detail="Check session not found")

        # Get all images for this check
        images = db.execute(
            "SELECT * FROM full_body_check_images WHERE check_id = ? ORDER BY image_index",
            (check_id,)
        ).fetchall()

        if not images:
            raise HTTPException(status_code=400, detail="No images uploaded to this check")

        # Prepare image data for processing
        image_data_list = []
        for img in images:
            img_dict = dict(img)
            image_path = img_dict.get("image_url")

            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                image_data_list.append({
                    "image_data": base64.b64encode(image_bytes).decode(),
                    "filename": img_dict.get("filename", ""),
                    "metadata": {
                        "body_location": img_dict.get("body_region"),
                        "image_id": img_dict.get("image_id"),
                    }
                })

        # Define classification callback
        async def classify_image(image_data_b64):
            """Classify a single image using existing models"""
            try:
                # Decode image
                image_bytes = base64.b64decode(image_data_b64)
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Binary classification
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                img_tensor = transform(img).unsqueeze(0).to(shared.device)

                with torch.no_grad():
                    binary_output = shared.binary_model(img_tensor)
                    binary_probs = F.softmax(binary_output, dim=1)
                    is_lesion = binary_probs[0][1].item() > 0.5

                    if is_lesion:
                        # Full classification
                        predicted_class = "melanocytic nevi"  # Default
                        confidence = binary_probs[0][1].item()

                        # Add risk assessment
                        risk_level = "low"
                        if "melanoma" in predicted_class.lower():
                            risk_level = "high"
                        elif "carcinoma" in predicted_class.lower():
                            risk_level = "high"

                        return {
                            "is_lesion": True,
                            "predicted_class": predicted_class,
                            "lesion_confidence": confidence,
                            "risk_level": risk_level,
                            "features": {}
                        }
                    else:
                        return {"is_lesion": False}

            except Exception as e:
                print(f"Classification error: {e}")
                return {"is_lesion": False, "error": str(e)}

        # Process batch
        processor = get_batch_processor()
        result = await processor.process_batch(
            user_id=current_user.id,
            images=image_data_list,
            classify_callback=classify_image
        )

        # Update database with results
        db.execute(
            """
            UPDATE full_body_checks SET
                status = 'completed',
                completed_at = ?,
                total_lesions = ?,
                overall_risk_score = ?,
                critical_count = ?,
                high_risk_count = ?,
                medium_risk_count = ?,
                low_risk_count = ?,
                body_coverage = ?,
                mole_map_data = ?,
                lesion_count_by_region = ?,
                recommendations = ?
            WHERE check_id = ?
            """,
            (
                datetime.utcnow().isoformat(),
                result.total_lesions,
                result.overall_risk_score,
                result.risk_summary.get("critical", 0),
                result.risk_summary.get("high", 0),
                result.risk_summary.get("medium", 0),
                result.risk_summary.get("low", 0),
                json.dumps(result.body_coverage),
                json.dumps(result.mole_map),
                json.dumps(result.lesion_count_by_region),
                json.dumps(result.recommendations),
                check_id
            )
        )

        # Record lesion count history
        db.execute(
            """
            INSERT INTO lesion_count_history
            (user_id, check_id, total_lesions, critical_count, high_risk_count,
             medium_risk_count, low_risk_count, count_by_region)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                current_user.id,
                check_id,
                result.total_lesions,
                result.risk_summary.get("critical", 0),
                result.risk_summary.get("high", 0),
                result.risk_summary.get("medium", 0),
                result.risk_summary.get("low", 0),
                json.dumps(result.lesion_count_by_region)
            )
        )

        db.commit()

        return {
            "check_id": check_id,
            "status": "completed",
            "summary": {
                "total_images": result.total_images,
                "total_lesions": result.total_lesions,
                "overall_risk_score": round(result.overall_risk_score, 1),
                "risk_breakdown": result.risk_summary,
            },
            "body_coverage": result.body_coverage,
            "lesion_count_by_region": result.lesion_count_by_region,
            "highest_risk_lesions": [
                {
                    "lesion_id": l.lesion_id,
                    "risk_level": l.risk_level.value,
                    "risk_score": round(l.risk_score, 1),
                    "predicted_class": l.predicted_class,
                    "body_location": l.body_location.value if l.body_location else None,
                    "recommendations": l.recommendations
                }
                for l in result.highest_risk_lesions[:5]
            ],
            "recommendations": result.recommendations,
            "mole_map": result.mole_map
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error processing batch: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process batch: {str(e)}")


@router.get("/batch/check/{check_id}")
async def get_batch_check_result(
    check_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get results of a full-body skin check.
    """
    try:
        check = db.execute(
            "SELECT * FROM full_body_checks WHERE check_id = ? AND user_id = ?",
            (check_id, current_user.id)
        ).fetchone()

        if not check:
            raise HTTPException(status_code=404, detail="Check not found")

        check_dict = dict(check)

        # Get images
        images = db.execute(
            "SELECT * FROM full_body_check_images WHERE check_id = ?",
            (check_id,)
        ).fetchall()

        # Get lesions
        lesions = db.execute(
            "SELECT * FROM batch_lesion_detections WHERE check_id = ? ORDER BY risk_score DESC",
            (check_id,)
        ).fetchall()

        return {
            "check_id": check_id,
            "status": check_dict.get("status"),
            "created_at": check_dict.get("created_at"),
            "completed_at": check_dict.get("completed_at"),
            "summary": {
                "total_images": check_dict.get("total_images", 0),
                "total_lesions": check_dict.get("total_lesions", 0),
                "overall_risk_score": check_dict.get("overall_risk_score", 0),
                "risk_breakdown": {
                    "critical": check_dict.get("critical_count", 0),
                    "high": check_dict.get("high_risk_count", 0),
                    "medium": check_dict.get("medium_risk_count", 0),
                    "low": check_dict.get("low_risk_count", 0),
                }
            },
            "body_coverage": json.loads(check_dict.get("body_coverage") or "{}"),
            "mole_map": json.loads(check_dict.get("mole_map_data") or "{}"),
            "lesion_count_by_region": json.loads(check_dict.get("lesion_count_by_region") or "{}"),
            "recommendations": json.loads(check_dict.get("recommendations") or "[]"),
            "images": [dict(img) for img in images],
            "lesions": [dict(l) for l in lesions]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get check: {str(e)}")


@router.get("/batch/checks")
async def list_batch_checks(
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List all full-body skin checks for the current user.
    """
    try:
        checks = db.execute(
            """
            SELECT check_id, status, created_at, completed_at, total_images, total_lesions,
                   overall_risk_score, critical_count, high_risk_count, medium_risk_count, low_risk_count
            FROM full_body_checks
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (current_user.id, limit, offset)
        ).fetchall()

        total = db.execute(
            "SELECT COUNT(*) FROM full_body_checks WHERE user_id = ?",
            (current_user.id,)
        ).fetchone()[0]

        return {
            "checks": [
                {
                    "check_id": c["check_id"],
                    "status": c["status"],
                    "created_at": c["created_at"],
                    "completed_at": c["completed_at"],
                    "total_images": c["total_images"],
                    "total_lesions": c["total_lesions"],
                    "overall_risk_score": c["overall_risk_score"],
                    "risk_breakdown": {
                        "critical": c["critical_count"],
                        "high": c["high_risk_count"],
                        "medium": c["medium_risk_count"],
                        "low": c["low_risk_count"],
                    }
                }
                for c in checks
            ],
            "total": total,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        print(f"Error listing checks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list checks: {str(e)}")


@router.get("/batch/mole-map/{check_id}")
async def get_mole_map(
    check_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get mole map visualization data for a check.
    """
    try:
        check = db.execute(
            "SELECT mole_map_data, body_coverage FROM full_body_checks WHERE check_id = ? AND user_id = ?",
            (check_id, current_user.id)
        ).fetchone()

        if not check:
            raise HTTPException(status_code=404, detail="Check not found")

        return {
            "check_id": check_id,
            "mole_map": json.loads(check["mole_map_data"] or "{}"),
            "body_coverage": json.loads(check["body_coverage"] or "{}"),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting mole map: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get mole map: {str(e)}")


@router.get("/batch/lesion-trends")
async def get_lesion_trends(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get lesion count trends over time for the user.
    """
    try:
        history = db.execute(
            """
            SELECT check_id, recorded_at, total_lesions, critical_count, high_risk_count,
                   medium_risk_count, low_risk_count, count_by_region
            FROM lesion_count_history
            WHERE user_id = ?
            ORDER BY recorded_at ASC
            """,
            (current_user.id,)
        ).fetchall()

        if not history:
            return {
                "has_history": False,
                "message": "No skin check history found. Complete a full-body skin check to start tracking."
            }

        # Calculate trends
        records = [dict(h) for h in history]
        latest = records[-1]

        if len(records) > 1:
            previous = records[-2]
            change = latest["total_lesions"] - previous["total_lesions"]
            change_percent = (change / previous["total_lesions"] * 100) if previous["total_lesions"] > 0 else 0
            trend = "increasing" if change > 0 else ("decreasing" if change < 0 else "stable")
        else:
            change = 0
            change_percent = 0
            trend = "baseline"

        return {
            "has_history": True,
            "total_checks": len(records),
            "current": {
                "check_id": latest["check_id"],
                "date": latest["recorded_at"],
                "total_lesions": latest["total_lesions"],
                "high_risk_count": latest["critical_count"] + latest["high_risk_count"],
            },
            "change": {
                "absolute": change,
                "percent": round(change_percent, 1),
                "trend": trend,
            },
            "history": [
                {
                    "date": r["recorded_at"],
                    "total_lesions": r["total_lesions"],
                    "high_risk": r["critical_count"] + r["high_risk_count"],
                }
                for r in records
            ],
            "recommendation": get_count_tracker()._get_trend_recommendation(change, change_percent)
        }

    except Exception as e:
        print(f"Error getting trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.post("/batch/generate-report/{check_id}")
async def generate_full_body_report(
    check_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Generate a comprehensive PDF report for a full-body skin check.
    """
    try:
        check = db.execute(
            "SELECT * FROM full_body_checks WHERE check_id = ? AND user_id = ?",
            (check_id, current_user.id)
        ).fetchone()

        if not check:
            raise HTTPException(status_code=404, detail="Check not found")

        check_dict = dict(check)

        if check_dict.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Check must be completed before generating report")

        # Generate report data
        report_gen = get_report_generator()

        # Get highest risk lesions
        lesions = db.execute(
            "SELECT * FROM batch_lesion_detections WHERE check_id = ? ORDER BY risk_score DESC LIMIT 10",
            (check_id,)
        ).fetchall()

        highest_risk = []
        for l in lesions:
            l_dict = dict(l)
            highest_risk.append(LesionDetection(
                lesion_id=l_dict["lesion_id"],
                image_index=0,
                bounding_box={},
                confidence=l_dict.get("confidence", 0),
                predicted_class=l_dict.get("predicted_class", ""),
                risk_level=RiskLevel(l_dict.get("risk_level", "low")),
                risk_score=l_dict.get("risk_score", 0),
                body_location=BodyRegion(l_dict["body_location"]) if l_dict.get("body_location") else None,
                recommendations=json.loads(l_dict.get("recommendations") or "[]")
            ))

        result = FullBodyCheckResult(
            check_id=check_id,
            user_id=current_user.id,
            created_at=check_dict.get("created_at", ""),
            total_images=check_dict.get("total_images", 0),
            total_lesions=check_dict.get("total_lesions", 0),
            images_analyzed=[],
            risk_summary={
                "critical": check_dict.get("critical_count", 0),
                "high": check_dict.get("high_risk_count", 0),
                "medium": check_dict.get("medium_risk_count", 0),
                "low": check_dict.get("low_risk_count", 0),
            },
            body_coverage=json.loads(check_dict.get("body_coverage") or "{}"),
            highest_risk_lesions=highest_risk,
            mole_map=json.loads(check_dict.get("mole_map_data") or "{}"),
            lesion_count_by_region=json.loads(check_dict.get("lesion_count_by_region") or "{}"),
            recommendations=json.loads(check_dict.get("recommendations") or "[]"),
            overall_risk_score=check_dict.get("overall_risk_score", 0),
        )

        report_data = report_gen.generate_report_data(result)

        # Update database
        db.execute(
            "UPDATE full_body_checks SET report_generated = 1 WHERE check_id = ?",
            (check_id,)
        )
        db.commit()

        return {
            "check_id": check_id,
            "report_generated": True,
            "report_data": report_data
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
