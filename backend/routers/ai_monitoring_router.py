"""
AI Monitoring Agent Router - Proactive Lesion Monitoring with AI Insights

This agent:
1. Monitors all tracked lesions for changes
2. Correlates data from multiple sources (lesions, sun exposure, family history)
3. Generates personalized, contextual insights
4. Proactively alerts users about concerning changes
5. Explains medical data in plain language
"""

from fastapi import APIRouter, Depends, HTTPException, Form, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import json
import os

from database import (
    get_db, User, AnalysisHistory, LesionGroup, LesionComparison,
    SunExposure, FamilyMember, UserProfile
)
from auth import get_current_active_user

router = APIRouter(prefix="/ai-monitoring", tags=["AI Monitoring Agent"])

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_openai_client():
    """Get OpenAI client with API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


# System prompt for the monitoring agent
MONITORING_AGENT_PROMPT = """You are an AI dermatology monitoring assistant that proactively tracks patients' skin lesions over time.

YOUR ROLE:
- Analyze lesion tracking data and identify concerning changes
- Correlate multiple data sources (lesion changes, sun exposure, family history, symptoms)
- Explain findings in clear, accessible language
- Provide personalized recommendations based on the patient's specific risk factors
- Be proactive but not alarmist - most changes are benign

COMMUNICATION STYLE:
- Warm and reassuring, like a caring healthcare provider
- Use plain language, not medical jargon
- Be specific about what you observe and why it matters
- Always provide actionable next steps
- Acknowledge uncertainty appropriately

WHEN ANALYZING LESIONS:
- Consider the full history, not just the most recent change
- Factor in patient age, skin type, and family history
- Look for patterns (e.g., growth during summer months)
- Prioritize by urgency - highlight the most important findings first

RESPONSE FORMAT:
- Start with overall status (all clear, needs attention, or urgent)
- Summarize key findings in 1-2 sentences
- Provide details for each lesion that needs discussion
- End with clear recommendations

IMPORTANT:
- Never provide definitive diagnoses
- Always recommend professional consultation for concerning findings
- Be clear that AI monitoring supplements, not replaces, dermatologist care
"""


def get_user_context(db: Session, user: User) -> Dict[str, Any]:
    """Gather comprehensive user context for AI analysis."""
    context = {
        "user_id": user.id,
        "age": getattr(user, 'age', None),
        "skin_type": getattr(user, 'skin_type', None),
        "risk_factors": []
    }

    # Get user profile if exists
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    if profile:
        # Calculate age from date_of_birth if available
        if profile.date_of_birth and not context["age"]:
            today = datetime.utcnow()
            age = today.year - profile.date_of_birth.year
            if (today.month, today.day) < (profile.date_of_birth.month, profile.date_of_birth.day):
                age -= 1
            context["age"] = age
        context["skin_type"] = profile.skin_type or context["skin_type"]
        if profile.medical_history:
            context["medical_history"] = profile.medical_history

    # Get family history
    family_members = db.query(FamilyMember).filter(FamilyMember.user_id == user.id).all()
    family_history = []
    for member in family_members:
        relationship = getattr(member, 'relationship_type', None) or getattr(member, 'name', 'family member')
        if getattr(member, 'has_melanoma', False):
            family_history.append(f"{relationship}: melanoma")
            if "family_history_melanoma" not in context["risk_factors"]:
                context["risk_factors"].append("family_history_melanoma")
        if getattr(member, 'has_skin_cancer', False):
            family_history.append(f"{relationship}: skin cancer")
            if "family_history_skin_cancer" not in context["risk_factors"]:
                context["risk_factors"].append("family_history_skin_cancer")

    if family_history:
        context["family_history"] = family_history

    # Determine skin type risk
    if context["skin_type"]:
        skin_type = context["skin_type"]
        if skin_type in ["I", "II", "1", "2", "Type I", "Type II"]:
            context["risk_factors"].append("fair_skin")

    return context


def get_sun_exposure_summary(db: Session, user_id: int, days: int = 90) -> Dict[str, Any]:
    """Get summary of recent sun exposure."""
    cutoff_date = datetime.utcnow() - timedelta(days=days)

    exposures = db.query(SunExposure).filter(
        SunExposure.user_id == user_id,
        SunExposure.exposure_date >= cutoff_date
    ).all()

    if not exposures:
        return {"total_exposures": 0, "message": "No sun exposure data recorded"}

    total_minutes = sum(e.duration_minutes or 0 for e in exposures)
    high_uv_count = sum(1 for e in exposures if (e.uv_index or 0) >= 8)
    unprotected_count = sum(1 for e in exposures if not e.sun_protection_used)
    sunburn_count = sum(1 for e in exposures if e.skin_reaction and 'burn' in (e.skin_reaction or '').lower())

    return {
        "total_exposures": len(exposures),
        "total_hours": round(total_minutes / 60, 1),
        "high_uv_exposures": high_uv_count,
        "unprotected_exposures": unprotected_count,
        "sunburn_events": sunburn_count,
        "average_uv_index": round(sum(e.uv_index or 0 for e in exposures) / len(exposures), 1) if exposures else 0
    }


def get_lesion_details(db: Session, lesion_group: LesionGroup) -> Dict[str, Any]:
    """Get detailed information about a lesion group with its history."""
    # Get all analyses for this lesion
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == lesion_group.id
    ).order_by(desc(AnalysisHistory.created_at)).all()

    # Get comparisons
    comparisons = db.query(LesionComparison).filter(
        LesionComparison.lesion_group_id == lesion_group.id
    ).order_by(desc(LesionComparison.created_at)).all()

    # Build analysis history
    analysis_history = []
    for analysis in analyses[:10]:  # Last 10 analyses
        analysis_history.append({
            "id": analysis.id,
            "date": analysis.created_at.isoformat() if analysis.created_at else None,
            "predicted_class": analysis.predicted_class,
            "risk_level": analysis.risk_level,
            "confidence": analysis.lesion_confidence or analysis.binary_confidence,
            "is_lesion": analysis.is_lesion,
            "body_location": analysis.body_location,
            "symptoms": getattr(analysis, 'symptoms', None),
            "abcde_score": getattr(analysis, 'abcde_total_score', None)
        })

    # Build comparison history
    comparison_history = []
    for comp in comparisons[:5]:  # Last 5 comparisons
        comparison_history.append({
            "date": comp.created_at.isoformat() if comp.created_at else None,
            "change_detected": comp.change_detected,
            "change_severity": comp.change_severity,
            "change_score": comp.change_score,
            "size_change_percent": comp.size_change_percent,
            "color_changed": comp.color_changed,
            "shape_changed": comp.shape_changed,
            "recommendation": comp.recommendation,
            "urgency_level": comp.urgency_level
        })

    # Calculate days since last check
    days_since_last = None
    if lesion_group.last_analyzed_at:
        days_since_last = (datetime.utcnow() - lesion_group.last_analyzed_at).days

    # Check if overdue
    overdue = False
    if lesion_group.next_check_date:
        overdue = datetime.utcnow() > lesion_group.next_check_date

    return {
        "id": lesion_group.id,
        "name": lesion_group.lesion_name,
        "description": lesion_group.lesion_description,
        "body_location": lesion_group.body_location,
        "body_side": lesion_group.body_side,
        "current_risk_level": lesion_group.current_risk_level,
        "requires_attention": lesion_group.requires_attention,
        "attention_reason": lesion_group.attention_reason,
        "total_analyses": lesion_group.total_analyses,
        "growth_rate": lesion_group.growth_rate,
        "change_detected": lesion_group.change_detected,
        "change_summary": lesion_group.change_summary,
        "first_noticed": lesion_group.first_noticed_date.isoformat() if lesion_group.first_noticed_date else None,
        "last_analyzed": lesion_group.last_analyzed_at.isoformat() if lesion_group.last_analyzed_at else None,
        "days_since_last_check": days_since_last,
        "next_check_date": lesion_group.next_check_date.isoformat() if lesion_group.next_check_date else None,
        "overdue_for_check": overdue,
        "monitoring_frequency": lesion_group.monitoring_frequency,
        "analysis_history": analysis_history,
        "comparison_history": comparison_history
    }


@router.get("/status")
async def get_monitoring_status():
    """Check if AI monitoring is available."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    has_api_key = bool(api_key)

    return {
        "available": OPENAI_AVAILABLE and has_api_key,
        "openai_installed": OPENAI_AVAILABLE,
        "api_key_configured": has_api_key,
        "features": [
            "lesion_monitoring",
            "change_detection",
            "risk_correlation",
            "personalized_insights",
            "proactive_alerts"
        ] if has_api_key else []
    }


@router.get("/insights")
async def get_monitoring_insights(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get AI-generated insights for all tracked lesions.

    This is the main endpoint that provides:
    - Overall skin health status
    - Individual lesion assessments
    - Correlations with sun exposure and risk factors
    - Prioritized recommendations
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")

    client = get_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    # Gather all data
    user_context = get_user_context(db, current_user)
    sun_exposure = get_sun_exposure_summary(db, current_user.id)

    # Get all active lesion groups
    lesion_groups = db.query(LesionGroup).filter(
        LesionGroup.user_id == current_user.id,
        LesionGroup.archived == False
    ).order_by(desc(LesionGroup.requires_attention), desc(LesionGroup.last_analyzed_at)).all()

    if not lesion_groups:
        return {
            "status": "no_lesions",
            "message": "You're not tracking any lesions yet. Start by adding a lesion to monitor.",
            "lesions": [],
            "recommendations": ["Add your first lesion to begin AI-powered monitoring"]
        }

    # Get detailed info for each lesion
    lesions_data = [get_lesion_details(db, lg) for lg in lesion_groups]

    # Count alerts
    attention_needed = sum(1 for l in lesions_data if l["requires_attention"])
    overdue_checks = sum(1 for l in lesions_data if l["overdue_for_check"])
    high_risk = sum(1 for l in lesions_data if l["current_risk_level"] == "high")

    # Build context for AI
    ai_context = {
        "user_profile": user_context,
        "sun_exposure_last_90_days": sun_exposure,
        "total_tracked_lesions": len(lesions_data),
        "lesions_needing_attention": attention_needed,
        "overdue_checks": overdue_checks,
        "high_risk_lesions": high_risk,
        "lesions": lesions_data
    }

    # Generate AI insights
    prompt = f"""Analyze this patient's lesion monitoring data and provide personalized insights.

PATIENT DATA:
{json.dumps(ai_context, indent=2, default=str)}

Please provide:
1. An overall status assessment (one of: "all_clear", "needs_attention", "urgent_review")
2. A brief summary (2-3 sentences) of the overall skin health status
3. For each lesion that needs discussion, provide:
   - The lesion name
   - Current status
   - What you've observed (changes, trends)
   - Your recommendation
4. Any correlations you notice (e.g., sun exposure patterns, seasonal trends)
5. Top 3 prioritized action items

Format your response as JSON with this structure:
{{
    "overall_status": "all_clear|needs_attention|urgent_review",
    "summary": "Brief overall summary",
    "lesion_insights": [
        {{
            "lesion_id": 1,
            "lesion_name": "Name",
            "status": "stable|improving|concerning|urgent",
            "status_emoji": "✅|⚠️|🔴",
            "observation": "What you observed",
            "recommendation": "What to do",
            "urgency": "routine|soon|urgent"
        }}
    ],
    "correlations": ["Correlation 1", "Correlation 2"],
    "action_items": [
        {{
            "priority": 1,
            "action": "What to do",
            "reason": "Why",
            "urgency": "routine|soon|urgent"
        }}
    ],
    "next_message": "A friendly, personalized closing message"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": MONITORING_AGENT_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        insights = json.loads(response.choices[0].message.content)

        # Add raw data for drill-downs
        insights["lesions_data"] = lesions_data
        insights["user_context"] = user_context
        insights["sun_exposure_summary"] = sun_exposure
        insights["generated_at"] = datetime.utcnow().isoformat()

        return insights

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.get("/lesion/{lesion_id}/analysis")
async def get_lesion_analysis(
    lesion_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed AI analysis for a specific lesion.

    Provides:
    - Complete history analysis
    - Trend identification
    - Risk assessment
    - Comparison with previous images
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")

    client = get_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    # Get lesion group
    lesion_group = db.query(LesionGroup).filter(
        LesionGroup.id == lesion_id,
        LesionGroup.user_id == current_user.id
    ).first()

    if not lesion_group:
        raise HTTPException(status_code=404, detail="Lesion not found")

    # Get detailed data
    lesion_data = get_lesion_details(db, lesion_group)
    user_context = get_user_context(db, current_user)
    sun_exposure = get_sun_exposure_summary(db, current_user.id)

    # Generate detailed analysis
    prompt = f"""Provide a detailed analysis of this specific lesion's history and current status.

LESION DATA:
{json.dumps(lesion_data, indent=2, default=str)}

PATIENT CONTEXT:
{json.dumps(user_context, indent=2, default=str)}

RECENT SUN EXPOSURE:
{json.dumps(sun_exposure, indent=2, default=str)}

Please provide a comprehensive analysis including:
1. Summary of the lesion's history
2. Trend analysis (is it stable, changing, concerning?)
3. Risk factors specific to this lesion
4. Comparison insights (if multiple analyses exist)
5. Personalized recommendations
6. What to watch for going forward

Format as JSON:
{{
    "lesion_name": "Name",
    "history_summary": "Summary of the lesion's journey",
    "current_status": {{
        "status": "stable|monitoring|concerning|urgent",
        "confidence": "high|medium|low",
        "explanation": "Why this status"
    }},
    "trend_analysis": {{
        "overall_trend": "stable|improving|slowly_changing|rapidly_changing",
        "size_trend": "Description",
        "color_trend": "Description",
        "shape_trend": "Description"
    }},
    "risk_assessment": {{
        "risk_level": "low|medium|high",
        "contributing_factors": ["Factor 1", "Factor 2"],
        "protective_factors": ["Factor 1"]
    }},
    "key_observations": ["Observation 1", "Observation 2"],
    "recommendations": [
        {{
            "action": "What to do",
            "timeframe": "When",
            "reason": "Why"
        }}
    ],
    "watch_for": ["Sign 1 to watch for", "Sign 2"],
    "next_check_suggestion": "When to check next and why"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": MONITORING_AGENT_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        analysis = json.loads(response.choices[0].message.content)
        analysis["lesion_data"] = lesion_data
        analysis["generated_at"] = datetime.utcnow().isoformat()

        return analysis

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate analysis: {str(e)}")


@router.post("/ask")
async def ask_monitoring_agent(
    question: str = Form(...),
    lesion_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Ask the AI monitoring agent a question about your lesions.

    Examples:
    - "How has my shoulder mole changed over the last 3 months?"
    - "Should I be worried about the growth rate?"
    - "What's the connection between my sun exposure and these changes?"
    - "When should I see a dermatologist?"
    """
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI not available")

    client = get_openai_client()
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    # Gather context
    user_context = get_user_context(db, current_user)
    sun_exposure = get_sun_exposure_summary(db, current_user.id)

    # Get lesion data
    if lesion_id:
        lesion_group = db.query(LesionGroup).filter(
            LesionGroup.id == lesion_id,
            LesionGroup.user_id == current_user.id
        ).first()
        if lesion_group:
            lesions_data = [get_lesion_details(db, lesion_group)]
        else:
            lesions_data = []
    else:
        # Get all lesions
        lesion_groups = db.query(LesionGroup).filter(
            LesionGroup.user_id == current_user.id,
            LesionGroup.archived == False
        ).all()
        lesions_data = [get_lesion_details(db, lg) for lg in lesion_groups]

    context = {
        "user_profile": user_context,
        "sun_exposure_last_90_days": sun_exposure,
        "tracked_lesions": lesions_data
    }

    prompt = f"""The patient is asking about their skin monitoring data.

PATIENT QUESTION: {question}

PATIENT DATA:
{json.dumps(context, indent=2, default=str)}

Please answer their question in a warm, helpful manner. Be specific to their data when possible.
If their question requires information you don't have, acknowledge that and provide general guidance.
Always encourage professional consultation for medical decisions."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": MONITORING_AGENT_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )

        return {
            "question": question,
            "answer": response.choices[0].message.content,
            "lesion_id": lesion_id,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


@router.get("/comparison/{lesion_id}")
async def get_lesion_comparison(
    lesion_id: int,
    baseline_id: Optional[int] = Query(None, description="Baseline analysis ID"),
    current_id: Optional[int] = Query(None, description="Current analysis ID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get comparison data between two analyses of a lesion.

    If baseline_id and current_id are not provided, compares the two most recent analyses.
    Returns data needed for side-by-side comparison view.
    """
    # Get lesion group
    lesion_group = db.query(LesionGroup).filter(
        LesionGroup.id == lesion_id,
        LesionGroup.user_id == current_user.id
    ).first()

    if not lesion_group:
        raise HTTPException(status_code=404, detail="Lesion not found")

    # Get analyses
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == lesion_id
    ).order_by(desc(AnalysisHistory.created_at)).all()

    if len(analyses) < 2:
        return {
            "lesion_id": lesion_id,
            "lesion_name": lesion_group.lesion_name,
            "comparison_available": False,
            "message": "Need at least 2 analyses of this lesion to compare changes over time. Analyze this lesion again later to track changes.",
            "total_analyses": len(analyses)
        }

    # Determine which analyses to compare
    if baseline_id and current_id:
        baseline = next((a for a in analyses if a.id == baseline_id), None)
        current = next((a for a in analyses if a.id == current_id), None)
    else:
        # Use two most recent
        current = analyses[0]
        baseline = analyses[1]

    if not baseline or not current:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Get existing comparison if available
    comparison = db.query(LesionComparison).filter(
        LesionComparison.lesion_group_id == lesion_id,
        LesionComparison.baseline_analysis_id == baseline.id,
        LesionComparison.current_analysis_id == current.id
    ).first()

    def format_analysis(analysis):
        return {
            "id": analysis.id,
            "date": analysis.created_at.isoformat() if analysis.created_at else None,
            "image_filename": analysis.image_filename,
            "predicted_class": analysis.predicted_class,
            "risk_level": analysis.risk_level,
            "confidence": analysis.lesion_confidence or analysis.binary_confidence,
            "abcde": {
                "asymmetry": getattr(analysis, 'abcde_asymmetry_score', None),
                "border": getattr(analysis, 'abcde_border_score', None),
                "color": getattr(analysis, 'abcde_color_score', None),
                "diameter": getattr(analysis, 'abcde_diameter_score', None),
                "evolution": getattr(analysis, 'abcde_evolution_score', None),
                "total": getattr(analysis, 'abcde_total_score', None)
            },
            "size_mm": getattr(analysis, 'lesion_size_mm', None),
            "body_location": analysis.body_location
        }

    result = {
        "lesion_id": lesion_id,
        "lesion_name": lesion_group.lesion_name,
        "comparison_available": True,
        "baseline": format_analysis(baseline),
        "current": format_analysis(current),
        "time_difference_days": (current.created_at - baseline.created_at).days if current.created_at and baseline.created_at else None,
        "all_analyses": [{"id": a.id, "date": a.created_at.isoformat() if a.created_at else None} for a in analyses]
    }

    # Add comparison data if available
    if comparison:
        result["comparison"] = {
            "change_detected": comparison.change_detected,
            "change_severity": comparison.change_severity,
            "change_score": comparison.change_score,
            "size_change_percent": comparison.size_change_percent,
            "color_changed": comparison.color_changed,
            "shape_changed": comparison.shape_changed,
            "recommendation": comparison.recommendation,
            "urgency_level": comparison.urgency_level,
            "change_heatmap": comparison.change_heatmap  # Base64 encoded image
        }

    return result


@router.get("/alerts")
async def get_monitoring_alerts(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all active alerts for the user's tracked lesions.

    Returns prioritized list of items needing attention.
    """
    alerts = []

    # Get lesions requiring attention
    attention_lesions = db.query(LesionGroup).filter(
        LesionGroup.user_id == current_user.id,
        LesionGroup.requires_attention == True,
        LesionGroup.archived == False
    ).all()

    for lesion in attention_lesions:
        alerts.append({
            "type": "attention_needed",
            "priority": "high",
            "lesion_id": lesion.id,
            "lesion_name": lesion.lesion_name,
            "message": lesion.attention_reason or "This lesion requires your attention",
            "action": "Review lesion details"
        })

    # Get overdue checks
    overdue_lesions = db.query(LesionGroup).filter(
        LesionGroup.user_id == current_user.id,
        LesionGroup.next_check_date < datetime.utcnow(),
        LesionGroup.archived == False
    ).all()

    for lesion in overdue_lesions:
        days_overdue = (datetime.utcnow() - lesion.next_check_date).days
        alerts.append({
            "type": "overdue_check",
            "priority": "medium" if days_overdue < 14 else "high",
            "lesion_id": lesion.id,
            "lesion_name": lesion.lesion_name,
            "message": f"Check overdue by {days_overdue} days",
            "action": "Take a new photo for comparison"
        })

    # Get high-risk lesions
    high_risk_lesions = db.query(LesionGroup).filter(
        LesionGroup.user_id == current_user.id,
        LesionGroup.current_risk_level == "high",
        LesionGroup.archived == False
    ).all()

    for lesion in high_risk_lesions:
        if not any(a["lesion_id"] == lesion.id for a in alerts):
            alerts.append({
                "type": "high_risk",
                "priority": "high",
                "lesion_id": lesion.id,
                "lesion_name": lesion.lesion_name,
                "message": "This lesion is classified as high risk",
                "action": "Consider scheduling a dermatologist appointment"
            })

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return {
        "total_alerts": len(alerts),
        "high_priority": sum(1 for a in alerts if a["priority"] == "high"),
        "alerts": alerts
    }
