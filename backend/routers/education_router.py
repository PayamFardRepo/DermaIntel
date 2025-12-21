"""
Patient Education Router

Endpoints for:
- Educational content for conditions
- PDF handout generation
- Content search
- Email/SMS delivery
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import Optional

from database import get_db, User
from auth import get_current_active_user

router = APIRouter(prefix="/education", tags=["Patient Education"])


@router.get("/conditions")
async def get_education_conditions(language: str = "en"):
    """Get list of all available educational conditions."""
    try:
        from patient_education_content import get_patient_education_library
        library = get_patient_education_library()
        conditions = library.get_all_conditions(language)
        return {
            "conditions": conditions,
            "total": len(conditions),
            "language": language
        }
    except Exception as e:
        print(f"Error getting conditions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conditions: {str(e)}")


@router.get("/content/{condition_id}")
async def get_education_content(
    condition_id: str,
    language: str = "en"
):
    """
    Get educational content for a specific condition.

    Returns complete educational material including:
    - Description, Symptoms, Care instructions
    - Treatment options, Warning signs, Prevention tips
    """
    try:
        from patient_education_content import get_patient_education_library
        library = get_patient_education_library()
        content = library.get_content(condition_id, language)

        if not content:
            raise HTTPException(status_code=404, detail=f"Condition '{condition_id}' not found")

        return content

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting education content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get content: {str(e)}")


@router.post("/generate-pdf")
async def generate_education_pdf(
    condition_id: str = Form(...),
    patient_name: str = Form(...),
    language: str = Form("en"),
    include_timeline: bool = Form(True),
    include_images: bool = Form(True),
    current_user: User = Depends(get_current_active_user)
):
    """Generate personalized PDF education handout."""
    try:
        from patient_education_generator import get_patient_education_generator
        generator = get_patient_education_generator()

        pdf_bytes = generator.generate_handout(
            condition_id=condition_id,
            patient_name=patient_name,
            language=language,
            include_timeline=include_timeline,
            include_images=include_images
        )

        filename = f"{condition_id}_education_{patient_name.replace(' ', '_')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(pdf_bytes))
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


@router.get("/search")
async def search_education_conditions(
    query: str,
    language: str = "en"
):
    """Search for conditions by name or symptoms."""
    try:
        if not query or len(query) < 2:
            raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

        from patient_education_content import get_patient_education_library
        library = get_patient_education_library()
        results = library.search_conditions(query, language)

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "language": language
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error searching conditions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/send")
async def send_education_material(
    condition_id: str = Form(...),
    patient_name: str = Form(...),
    delivery_method: str = Form(...),
    contact: str = Form(...),
    language: str = Form("en"),
    current_user: User = Depends(get_current_active_user)
):
    """Send educational material to patient via email or SMS."""
    try:
        from patient_education_delivery import get_patient_education_delivery
        from patient_education_generator import get_patient_education_generator
        from patient_education_content import get_patient_education_library

        if delivery_method not in ["email", "sms"]:
            raise HTTPException(status_code=400, detail="Delivery method must be 'email' or 'sms'")

        library = get_patient_education_library()
        content = library.get_content(condition_id, language)
        if not content:
            raise HTTPException(status_code=404, detail=f"Condition '{condition_id}' not found")

        condition_name = content['condition_name']

        generator = get_patient_education_generator()
        pdf_bytes = generator.generate_handout(
            condition_id=condition_id,
            patient_name=patient_name,
            language=language
        )

        delivery = get_patient_education_delivery()

        if delivery_method == "email":
            pdf_filename = f"{condition_id}_education_{patient_name.replace(' ', '_')}.pdf"
            result = delivery.send_email(
                to_email=contact,
                patient_name=patient_name,
                condition_name=condition_name,
                pdf_bytes=pdf_bytes,
                pdf_filename=pdf_filename,
                language=language
            )
        else:
            result = delivery.send_sms(
                to_phone=contact,
                patient_name=patient_name,
                condition_name=condition_name,
                language=language
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error sending education material: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send: {str(e)}")


@router.get("/delivery-config")
async def get_delivery_config():
    """Get configuration for education delivery methods."""
    import os
    return {
        "email_configured": bool(os.getenv("SENDGRID_API_KEY") or os.getenv("SMTP_HOST")),
        "sms_configured": bool(os.getenv("TWILIO_ACCOUNT_SID")),
        "supported_languages": ["en", "es", "fr", "de", "zh", "pt", "ar"],
        "default_language": "en"
    }
