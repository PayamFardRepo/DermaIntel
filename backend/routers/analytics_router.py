"""
Analytics and Risk Calculator Router

Endpoints for:
- Population health dashboard
- Growth forecasts
- Screening schedules
- Risk trends
- Skin cancer risk calculator
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import uuid

from database import get_db, User, AnalysisHistory, GeneticTestResult, GeneticVariant, LabResults
from auth import get_current_active_user
from skin_cancer_risk_calculator import calculate_skin_cancer_risk

router = APIRouter(tags=["Analytics & Risk Calculator"])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_risk_category_description(category: str) -> str:
    """Get human-readable description of risk category."""
    descriptions = {
        "very_low": "Your risk is well below average. Continue standard sun protection practices.",
        "low": "Your risk is below average. Maintain good sun protection habits.",
        "moderate": "Your risk is around average. Be vigilant with sun protection and regular skin checks.",
        "high": "Your risk is elevated. Regular professional skin exams and careful monitoring recommended.",
        "very_high": "Your risk is significantly elevated. Frequent professional monitoring essential."
    }
    return descriptions.get(category, "Risk level determined. Follow personalized recommendations.")


def _interpret_relative_risk(rr: float) -> str:
    """Interpret relative risk value."""
    if rr is None:
        return "Not calculated"
    if rr < 0.5:
        return "Well below average risk"
    elif rr < 0.8:
        return "Below average risk"
    elif rr < 1.2:
        return "Average risk"
    elif rr < 2.0:
        return "Moderately elevated risk"
    elif rr < 4.0:
        return "Significantly elevated risk"
    else:
        return "Very high risk - professional monitoring essential"


# =============================================================================
# POPULATION HEALTH
# =============================================================================

@router.get("/population-health/dashboard")
async def get_population_health_dashboard(
    time_range: str = "30days",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get population-level health analytics and statistics.
    Returns aggregated data on condition prevalence, risk distribution,
    geographic insights, and demographic patterns.
    """
    try:
        from sqlalchemy import func, and_

        # Calculate time range filter
        now = datetime.now()
        time_filters = {
            "7days": now - timedelta(days=7),
            "30days": now - timedelta(days=30),
            "90days": now - timedelta(days=90),
            "1year": now - timedelta(days=365),
            "all": datetime(2000, 1, 1)
        }
        start_date = time_filters.get(time_range, time_filters["30days"])

        # Get all analyses in the time range
        analyses_query = db.query(AnalysisHistory).filter(
            AnalysisHistory.created_at >= start_date
        )

        # Total statistics
        total_analyses = analyses_query.count()
        total_users = db.query(func.count(func.distinct(AnalysisHistory.user_id))).filter(
            AnalysisHistory.created_at >= start_date
        ).scalar() or 0

        total_high_risk = analyses_query.filter(
            AnalysisHistory.risk_level.in_(["high", "very_high"])
        ).count()

        # Calculate average age from users who have analyses in time range
        user_ids = [a.user_id for a in analyses_query.all()]
        avg_age_result = db.query(func.avg(User.age)).filter(
            User.id.in_(user_ids),
            User.age.isnot(None)
        ).scalar()
        average_age = round(float(avg_age_result)) if avg_age_result else None

        # Gender distribution
        gender_counts = db.query(
            User.gender,
            func.count(User.id)
        ).filter(
            User.id.in_(user_ids)
        ).group_by(User.gender).all()

        gender_distribution = {"male": 0, "female": 0, "other": 0}
        for gender, count in gender_counts:
            if gender and gender.lower() in gender_distribution:
                gender_distribution[gender.lower()] = count
            elif gender:
                gender_distribution["other"] += count

        # Condition prevalence with trend analysis
        current_conditions = {}
        for analysis in analyses_query.all():
            condition = analysis.predicted_class or "Unknown"
            current_conditions[condition] = current_conditions.get(condition, 0) + 1

        # Get previous period for trend comparison
        prev_start = start_date - (now - start_date)
        prev_analyses = db.query(AnalysisHistory).filter(
            and_(
                AnalysisHistory.created_at >= prev_start,
                AnalysisHistory.created_at < start_date
            )
        ).all()

        prev_conditions = {}
        for analysis in prev_analyses:
            condition = analysis.predicted_class or "Unknown"
            prev_conditions[condition] = prev_conditions.get(condition, 0) + 1

        # Calculate prevalence with trends
        condition_prevalence = []
        for condition, count in sorted(current_conditions.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_analyses * 100) if total_analyses > 0 else 0

            prev_count = prev_conditions.get(condition, 0)
            if prev_count == 0:
                trend = "up" if count > 0 else "stable"
            else:
                change_pct = ((count - prev_count) / prev_count) * 100
                if change_pct > 10:
                    trend = "up"
                elif change_pct < -10:
                    trend = "down"
                else:
                    trend = "stable"

            condition_prevalence.append({
                "condition": condition,
                "count": count,
                "percentage": round(percentage, 1),
                "trend": trend
            })

        # Risk level distribution
        risk_levels = ["low", "medium", "high", "very_high"]
        risk_distribution = []
        for risk_level in risk_levels:
            count = analyses_query.filter(
                AnalysisHistory.risk_level == risk_level
            ).count()
            percentage = (count / total_analyses * 100) if total_analyses > 0 else 0
            risk_distribution.append({
                "risk_level": risk_level,
                "count": count,
                "percentage": round(percentage, 1)
            })

        # Geographic insights (mock data based on analysis distribution)
        geographic_insights = []
        user_analysis_counts = {}
        user_high_risk_counts = {}
        for analysis in analyses_query.all():
            user_analysis_counts[analysis.user_id] = user_analysis_counts.get(analysis.user_id, 0) + 1
            if analysis.risk_level in ["high", "very_high"]:
                user_high_risk_counts[analysis.user_id] = user_high_risk_counts.get(analysis.user_id, 0) + 1

        regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
        users_per_region = len(user_analysis_counts) // len(regions) + 1

        for i, region in enumerate(regions):
            region_users = list(user_analysis_counts.keys())[i*users_per_region:(i+1)*users_per_region]
            region_analyses = sum(user_analysis_counts.get(uid, 0) for uid in region_users)
            region_high_risk = sum(user_high_risk_counts.get(uid, 0) for uid in region_users)

            if region_analyses > 0:
                geographic_insights.append({
                    "region": region,
                    "total_analyses": region_analyses,
                    "high_risk_percentage": round((region_high_risk / region_analyses * 100), 1) if region_analyses > 0 else 0
                })

        # Demographic insights by age group
        age_groups = [
            {"range": "0-17", "min": 0, "max": 17},
            {"range": "18-29", "min": 18, "max": 29},
            {"range": "30-44", "min": 30, "max": 44},
            {"range": "45-64", "min": 45, "max": 64},
            {"range": "65+", "min": 65, "max": 150}
        ]

        demographic_insights = []
        for age_group in age_groups:
            users_in_group = db.query(User).filter(
                User.id.in_(user_ids),
                User.age >= age_group["min"],
                User.age <= age_group["max"]
            ).all()

            if users_in_group:
                group_user_ids = [u.id for u in users_in_group]
                group_analyses = analyses_query.filter(
                    AnalysisHistory.user_id.in_(group_user_ids)
                ).all()

                group_conditions = {}
                for analysis in group_analyses:
                    condition = analysis.predicted_class or "Unknown"
                    group_conditions[condition] = group_conditions.get(condition, 0) + 1

                top_conditions = sorted(group_conditions.items(), key=lambda x: x[1], reverse=True)[:3]
                common_conditions = [cond for cond, _ in top_conditions]

                demographic_insights.append({
                    "age_range": age_group["range"],
                    "total_analyses": len(group_analyses),
                    "common_conditions": common_conditions
                })

        return {
            "stats": {
                "total_analyses": total_analyses,
                "total_users": total_users,
                "total_high_risk": total_high_risk,
                "average_age": average_age,
                "gender_distribution": gender_distribution
            },
            "condition_prevalence": condition_prevalence,
            "risk_distribution": risk_distribution,
            "geographic_insights": geographic_insights,
            "demographic_insights": demographic_insights,
            "time_range": time_range
        }

    except Exception as e:
        print(f"Error getting population health dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GROWTH FORECASTS & SCREENING SCHEDULE
# =============================================================================

@router.get("/analytics/forecasts")
async def get_all_forecasts(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all growth forecasts for the current user."""
    try:
        from database import GrowthForecast, LesionGroup

        forecasts = db.query(GrowthForecast).filter(
            GrowthForecast.user_id == current_user.id
        ).order_by(GrowthForecast.created_at.desc()).all()

        result = []
        for forecast in forecasts:
            lesion_group = db.query(LesionGroup).filter(
                LesionGroup.id == forecast.lesion_group_id
            ).first()

            result.append({
                "id": forecast.id,
                "lesion_group_id": forecast.lesion_group_id,
                "lesion_name": lesion_group.lesion_name if lesion_group else "Unknown",
                "forecast_date": forecast.forecast_date,
                "growth_trend": forecast.growth_trend,
                "growth_rate_mm_per_month": forecast.growth_rate_mm_per_month,
                "current_size_mm": forecast.current_size_mm,
                "predicted_size_90d": forecast.predicted_size_90d,
                "current_risk_level": forecast.current_risk_level,
                "predicted_risk_level_90d": forecast.predicted_risk_level_90d,
                "risk_escalation_probability": forecast.risk_escalation_probability,
                "change_probability": forecast.change_probability,
                "recommended_action": forecast.recommended_action,
                "next_check_date": forecast.next_check_date,
                "monitoring_frequency": forecast.monitoring_frequency,
                "confidence_score": forecast.confidence_score,
                "primary_risk_factors": forecast.primary_risk_factors,
                "forecast_data": forecast.forecast_data
            })

        return result
    except Exception as e:
        return []


@router.get("/analytics/schedule")
async def get_screening_schedule(
    upcoming_only: bool = True,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get personalized screening schedule."""
    try:
        from database import ScreeningSchedule

        query = db.query(ScreeningSchedule).filter(
            ScreeningSchedule.user_id == current_user.id
        )

        if upcoming_only:
            query = query.filter(ScreeningSchedule.is_completed == False)

        # Update overdue status
        now = datetime.utcnow()
        schedules = query.all()
        for schedule in schedules:
            if schedule.recommended_date and schedule.recommended_date < now and not schedule.is_completed:
                schedule.is_overdue = True
        db.commit()

        schedules = query.order_by(
            ScreeningSchedule.priority.desc(),
            ScreeningSchedule.recommended_date.asc()
        ).all()

        return [
            {
                "id": s.id,
                "schedule_type": s.schedule_type,
                "priority": s.priority,
                "title": s.title,
                "description": s.description,
                "recommended_date": s.recommended_date,
                "is_completed": s.is_completed,
                "is_overdue": s.is_overdue,
                "is_recurring": s.is_recurring,
                "recurrence_frequency": s.recurrence_frequency,
                "based_on_risk_level": s.based_on_risk_level,
                "based_on_genetic_risk": s.based_on_genetic_risk,
                "based_on_lesion_changes": s.based_on_lesion_changes,
                "related_entity_id": s.related_entity_id
            }
            for s in schedules
        ]
    except Exception as e:
        return []


@router.post("/analytics/schedule/generate")
async def generate_screening_schedule(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate personalized screening schedule based on risk profile."""
    try:
        from database import ScreeningSchedule, GeneticRiskProfile, LesionGroup
        from dateutil.relativedelta import relativedelta

        # Get genetic risk profile
        risk_profile = db.query(GeneticRiskProfile).filter(
            GeneticRiskProfile.user_id == current_user.id
        ).first()

        # Clear existing non-completed schedules
        db.query(ScreeningSchedule).filter(
            ScreeningSchedule.user_id == current_user.id,
            ScreeningSchedule.is_completed == False
        ).delete()

        schedules_created = []

        # Self-examination schedule based on genetic risk
        if risk_profile:
            freq = risk_profile.recommended_screening_frequency or "monthly"
            freq_map = {
                "monthly": ("monthly", 1),
                "quarterly": ("quarterly", 3),
                "biannual": ("biannual", 6),
                "annual": ("annual", 12)
            }

            if freq in freq_map:
                recurrence, months = freq_map[freq]
                next_date = datetime.utcnow() + relativedelta(months=months)

                schedule = ScreeningSchedule(
                    user_id=current_user.id,
                    schedule_type="self_exam",
                    priority=8 if risk_profile.overall_risk_level in ['high', 'very_high'] else 5,
                    is_recurring=True,
                    recurrence_frequency=recurrence,
                    title="Full Body Self-Examination",
                    description=f"Perform comprehensive self-examination. Based on your {risk_profile.overall_risk_level} risk level.",
                    recommended_date=next_date,
                    based_on_risk_level=risk_profile.overall_risk_level,
                    based_on_genetic_risk=True
                )
                db.add(schedule)
                schedules_created.append(schedule)

        db.commit()

        return {
            "message": "Schedule generated successfully",
            "schedules_created": len(schedules_created)
        }
    except Exception as e:
        return {
            "message": "Unable to generate schedule",
            "schedules_created": 0
        }


@router.post("/analytics/schedule/{schedule_id}/complete")
async def complete_schedule_item(
    schedule_id: int,
    completion_notes: str = Form(None),
    completion_result: str = Form("normal"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark a schedule item as completed."""
    try:
        from database import ScreeningSchedule

        schedule = db.query(ScreeningSchedule).filter(
            ScreeningSchedule.id == schedule_id,
            ScreeningSchedule.user_id == current_user.id
        ).first()

        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule item not found")

        schedule.is_completed = True
        schedule.completed_date = datetime.utcnow()
        schedule.completion_notes = completion_notes
        schedule.completion_result = completion_result

        db.commit()

        return {"message": "Schedule item marked as completed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/risk-trends")
async def get_risk_trends(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get historical risk trend data for analytics visualization."""
    try:
        from database import RiskTrend, GeneticRiskProfile, LesionGroup

        trends = db.query(RiskTrend).filter(
            RiskTrend.user_id == current_user.id
        ).order_by(RiskTrend.snapshot_date.asc()).all()

        # If no trends exist, create initial snapshot
        if not trends:
            risk_profile = db.query(GeneticRiskProfile).filter(
                GeneticRiskProfile.user_id == current_user.id
            ).first()

            lesion_groups = db.query(LesionGroup).filter(
                LesionGroup.user_id == current_user.id,
                LesionGroup.is_active == True
            ).all()

            if risk_profile:
                trend = RiskTrend(
                    user_id=current_user.id,
                    overall_risk_score=risk_profile.overall_genetic_risk_score,
                    overall_risk_level=risk_profile.overall_risk_level,
                    melanoma_risk_score=risk_profile.melanoma_risk_score,
                    total_lesions_tracked=len(lesion_groups),
                    high_risk_lesions_count=sum(1 for l in lesion_groups if l.current_risk_level == 'high'),
                    genetic_risk_score=risk_profile.family_history_score,
                    risk_trend="stable"
                )
                db.add(trend)
                db.commit()
                trends = [trend]

        return [
            {
                "snapshot_date": t.snapshot_date,
                "overall_risk_score": t.overall_risk_score,
                "overall_risk_level": t.overall_risk_level,
                "melanoma_risk_score": t.melanoma_risk_score,
                "total_lesions_tracked": t.total_lesions_tracked,
                "high_risk_lesions_count": t.high_risk_lesions_count,
                "genetic_risk_score": t.genetic_risk_score,
                "risk_trend": t.risk_trend,
                "predicted_future_risk": t.predicted_future_risk
            }
            for t in trends
        ]
    except Exception as e:
        return []


# =============================================================================
# SKIN CANCER RISK CALCULATOR
# =============================================================================

@router.post("/risk-calculator/calculate")
async def calculate_comprehensive_risk(
    request: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Calculate comprehensive skin cancer risk score.

    Combines multiple validated risk models:
    - Fears & Saraiya Model (phenotype factors)
    - Usher-Smith Model (mole count, sunburn history)
    - Family history assessment
    - Personal medical history
    - AI analysis findings integration
    """
    from database import SkinCancerRiskAssessment, FamilyMember

    try:
        input_data = request.copy()

        # Auto-populate family history from database if not provided
        if "family_history" not in input_data or not input_data["family_history"]:
            family_members = db.query(FamilyMember).filter(
                FamilyMember.user_id == current_user.id
            ).all()

            family_history = []
            for member in family_members:
                if member.has_skin_cancer or member.has_melanoma:
                    cancer_type = "melanoma" if member.has_melanoma else "skin_cancer"
                    family_history.append({
                        "relationship": member.relationship_type,
                        "cancer_type": cancer_type,
                        "age_at_diagnosis": member.earliest_diagnosis_age
                    })
            input_data["family_history"] = family_history

        # Auto-populate genetic test findings from database if not provided
        if "genetic_findings" not in input_data or not input_data["genetic_findings"]:
            # Get the most recent genetic test result for this user
            latest_genetic_test = db.query(GeneticTestResult).filter(
                GeneticTestResult.user_id == current_user.id
            ).order_by(GeneticTestResult.created_at.desc()).first()

            if latest_genetic_test:
                # Get detected variants
                variants = db.query(GeneticVariant).filter(
                    GeneticVariant.test_result_id == latest_genetic_test.id
                ).all()

                # Organize variants by gene and classify risk levels
                variants_by_gene = {}
                high_risk_genes = []
                moderate_risk_genes = []
                pharmacogenomic_flags = []
                pathogenic_count = 0
                likely_pathogenic_count = 0
                vus_count = 0

                # High-risk genes for melanoma/skin cancer
                melanoma_high_risk = ["CDKN2A", "CDK4", "BAP1", "PTCH1", "XPA", "XPB", "XPC", "XPD", "XPE", "XPF", "XPG"]
                moderate_risk = ["MC1R", "MITF"]
                pharmacogenomic = ["TPMT", "DPYD"]

                for variant in variants:
                    gene = variant.gene_symbol
                    if gene not in variants_by_gene:
                        variants_by_gene[gene] = []

                    variants_by_gene[gene].append({
                        "rsid": variant.rsid,
                        "hgvs_c": variant.hgvs_c,
                        "hgvs_p": variant.hgvs_p,
                        "acmg_classification": variant.classification,
                        "zygosity": variant.zygosity,
                    })

                    # Count by classification
                    if variant.classification == "pathogenic":
                        pathogenic_count += 1
                        if gene in melanoma_high_risk and gene not in high_risk_genes:
                            high_risk_genes.append(gene)
                    elif variant.classification == "likely_pathogenic":
                        likely_pathogenic_count += 1
                        if gene in melanoma_high_risk and gene not in high_risk_genes:
                            high_risk_genes.append(gene)
                    elif variant.classification == "vus":
                        vus_count += 1

                    # Check for moderate risk genes
                    if gene in moderate_risk and gene not in moderate_risk_genes:
                        moderate_risk_genes.append(gene)

                    # Check for pharmacogenomic variants
                    if gene in pharmacogenomic and gene not in pharmacogenomic_flags:
                        pharmacogenomic_flags.append(gene)

                # Calculate risk multipliers based on detected genes
                melanoma_multiplier = 1.0
                nmsc_multiplier = 1.0

                if "CDKN2A" in high_risk_genes:
                    melanoma_multiplier *= 10.0
                if "CDK4" in high_risk_genes:
                    melanoma_multiplier *= 8.0
                if "BAP1" in high_risk_genes:
                    melanoma_multiplier *= 5.0
                if "PTCH1" in high_risk_genes:
                    nmsc_multiplier *= 20.0
                if any(g in high_risk_genes for g in ["XPA", "XPB", "XPC", "XPD", "XPE", "XPF", "XPG"]):
                    melanoma_multiplier *= 100.0
                    nmsc_multiplier *= 100.0

                # MC1R variants (cumulative effect)
                if "MC1R" in moderate_risk_genes:
                    mc1r_count = len(variants_by_gene.get("MC1R", []))
                    melanoma_multiplier *= (1.0 + (mc1r_count * 0.5))

                input_data["genetic_findings"] = {
                    "has_genetic_data": True,
                    "test_date": str(latest_genetic_test.test_date) if latest_genetic_test.test_date else None,
                    "lab_name": latest_genetic_test.lab_name,
                    "variants": variants_by_gene,
                    "melanoma_genetic_risk_multiplier": melanoma_multiplier,
                    "nmsc_genetic_risk_multiplier": nmsc_multiplier,
                    "high_risk_genes": high_risk_genes,
                    "moderate_risk_genes": moderate_risk_genes,
                    "pharmacogenomic_flags": pharmacogenomic_flags,
                    "familial_melanoma_syndrome": "CDKN2A" in high_risk_genes or "CDK4" in high_risk_genes,
                    "gorlin_syndrome": "PTCH1" in high_risk_genes,
                    "xeroderma_pigmentosum": any(g in high_risk_genes for g in ["XPA", "XPB", "XPC", "XPD", "XPE", "XPF", "XPG"]),
                    "pathogenic_variant_count": pathogenic_count,
                    "likely_pathogenic_count": likely_pathogenic_count,
                    "vus_count": vus_count,
                }

        # Auto-populate lab result findings from database if not provided
        if "lab_findings" not in input_data or not input_data["lab_findings"]:
            # Get the most recent lab results for this user
            latest_lab_result = db.query(LabResults).filter(
                LabResults.user_id == current_user.id
            ).order_by(LabResults.test_date.desc()).first()

            if latest_lab_result:
                # Calculate vitamin D status
                vitamin_d_status = "unknown"
                vitamin_d_risk_factor = 1.0
                risk_factors_from_labs = []

                if latest_lab_result.vitamin_d is not None:
                    if latest_lab_result.vitamin_d < 20:
                        vitamin_d_status = "deficient"
                        vitamin_d_risk_factor = 1.3  # 30% increased risk
                        risk_factors_from_labs.append("Vitamin D deficiency (<20 ng/mL)")
                    elif latest_lab_result.vitamin_d < 30:
                        vitamin_d_status = "insufficient"
                        vitamin_d_risk_factor = 1.1  # 10% increased risk
                        risk_factors_from_labs.append("Vitamin D insufficiency (20-29 ng/mL)")
                    else:
                        vitamin_d_status = "sufficient"

                # Check for immunosuppression markers
                immunosuppressed_by_labs = False
                if latest_lab_result.wbc is not None and latest_lab_result.wbc < 4.0:
                    immunosuppressed_by_labs = True
                    risk_factors_from_labs.append("Low white blood cell count (leukopenia)")
                if latest_lab_result.lymphocytes is not None and latest_lab_result.lymphocytes < 1.0:
                    immunosuppressed_by_labs = True
                    risk_factors_from_labs.append("Low lymphocyte count (lymphopenia)")

                # Check inflammatory markers
                elevated_inflammation = False
                if latest_lab_result.crp is not None and latest_lab_result.crp > 3.0:
                    elevated_inflammation = True
                    risk_factors_from_labs.append("Elevated CRP (chronic inflammation)")
                if latest_lab_result.esr is not None and latest_lab_result.esr > 20:
                    elevated_inflammation = True
                    risk_factors_from_labs.append("Elevated ESR (inflammation marker)")

                # Check autoimmune markers
                ana_positive = latest_lab_result.ana_positive if latest_lab_result.ana_positive else False
                if ana_positive:
                    risk_factors_from_labs.append("Positive ANA (autoimmune marker)")

                # Check liver function
                liver_function_normal = True
                if latest_lab_result.alt is not None and latest_lab_result.alt > 56:
                    liver_function_normal = False
                if latest_lab_result.ast is not None and latest_lab_result.ast > 40:
                    liver_function_normal = False

                # Calculate overall lab risk multiplier
                lab_risk_multiplier = vitamin_d_risk_factor
                if immunosuppressed_by_labs:
                    lab_risk_multiplier *= 2.0  # Immunosuppression doubles skin cancer risk
                if ana_positive:
                    lab_risk_multiplier *= 1.2  # Some autoimmune conditions increase risk

                input_data["lab_findings"] = {
                    "has_lab_data": True,
                    "test_date": str(latest_lab_result.test_date) if latest_lab_result.test_date else None,
                    "lab_name": latest_lab_result.lab_name,
                    "vitamin_d_level": latest_lab_result.vitamin_d,
                    "vitamin_d_status": vitamin_d_status,
                    "wbc_count": latest_lab_result.wbc,
                    "lymphocyte_count": latest_lab_result.lymphocytes,
                    "immunosuppressed_by_labs": immunosuppressed_by_labs,
                    "crp_level": latest_lab_result.crp,
                    "esr_level": latest_lab_result.esr,
                    "elevated_inflammation": elevated_inflammation,
                    "ana_positive": ana_positive,
                    "liver_function_normal": liver_function_normal,
                    "lab_risk_multiplier": lab_risk_multiplier,
                    "risk_factors_from_labs": risk_factors_from_labs,
                }

        # Include AI findings if requested
        if input_data.get("include_ai_findings", False):
            recent_analyses = db.query(AnalysisHistory).filter(
                AnalysisHistory.user_id == current_user.id
            ).order_by(AnalysisHistory.created_at.desc()).limit(20).all()

            high_risk_count = 0
            malignant_predictions = 0
            ai_analysis_ids = []

            for analysis in recent_analyses:
                ai_analysis_ids.append(analysis.id)
                # Check if predicted class is malignant (mel=melanoma, bcc=basal cell, akiec=actinic keratosis)
                is_malignant = analysis.predicted_class in ["mel", "bcc", "akiec", "melanoma", "basal_cell_carcinoma"]
                if is_malignant:
                    malignant_predictions += 1
                if analysis.lesion_confidence and analysis.lesion_confidence >= 0.7:
                    if analysis.predicted_class in ["mel", "bcc", "akiec", "melanoma", "basal_cell_carcinoma"]:
                        high_risk_count += 1

            input_data["ai_findings"] = {
                "total_analyses": len(recent_analyses),
                "high_risk_lesion_count": high_risk_count,
                "malignant_predictions": malignant_predictions,
                "average_confidence": 0.75,
                "uncertainty_flags": high_risk_count > 0,
                "analysis_ids": ai_analysis_ids
            }

        # Calculate risk
        risk_result = calculate_skin_cancer_risk(input_data)

        # Generate assessment ID
        assessment_id = f"SCRA-{uuid.uuid4().hex[:12].upper()}"

        # Get previous assessment for comparison
        previous_assessment = db.query(SkinCancerRiskAssessment).filter(
            SkinCancerRiskAssessment.user_id == current_user.id
        ).order_by(SkinCancerRiskAssessment.created_at.desc()).first()

        risk_change = None
        risk_trend = None
        if previous_assessment:
            risk_change = risk_result["overall_risk_score"] - previous_assessment.overall_risk_score
            if risk_change > 5:
                risk_trend = "worsening"
            elif risk_change < -5:
                risk_trend = "improving"
            else:
                risk_trend = "stable"

        # Save assessment to database
        new_assessment = SkinCancerRiskAssessment(
            user_id=current_user.id,
            assessment_id=assessment_id,
            overall_risk_score=risk_result["overall_risk_score"],
            risk_category=risk_result["risk_category"],
            melanoma_relative_risk=risk_result.get("relative_risks", {}).get("melanoma", 1.0),
            melanoma_lifetime_risk_percent=risk_result.get("lifetime_risk_percent", {}).get("melanoma", 2.0),
            bcc_relative_risk=risk_result.get("relative_risks", {}).get("basal_cell_carcinoma", 1.0),
            scc_relative_risk=risk_result.get("relative_risks", {}).get("squamous_cell_carcinoma", 1.0),
            genetic_score=risk_result.get("component_scores", {}).get("genetic", 0),
            phenotype_score=risk_result.get("component_scores", {}).get("phenotype", 0),
            sun_exposure_score=risk_result.get("component_scores", {}).get("sun_exposure", 0),
            behavioral_score=risk_result.get("component_scores", {}).get("behavioral", 0),
            medical_history_score=risk_result.get("component_scores", {}).get("medical_history", 0),
            ai_findings_score=risk_result.get("component_scores", {}).get("ai_findings", 0),
            input_data=input_data,
            risk_factors=risk_result["risk_factors"],
            recommendations=risk_result["recommendations"],
            recommended_self_exam_frequency=risk_result.get("screening_recommendations", {}).get("self_exam_frequency", "monthly"),
            recommended_professional_exam_frequency=risk_result.get("screening_recommendations", {}).get("professional_exam_frequency", "annual"),
            urgent_dermatology_referral=risk_result.get("screening_recommendations", {}).get("urgent_referral", False),
            confidence_score=risk_result.get("confidence_score", 0.8),
            methodology_version="1.0",
            models_used=["Fears_Saraiya", "Usher_Smith", "Olsen_Modified"],
            ai_analysis_ids=input_data.get("ai_findings", {}).get("analysis_ids", []),
            ai_high_risk_lesions_count=input_data.get("ai_findings", {}).get("high_risk_lesion_count", 0),
            ai_uncertainty_flag=input_data.get("ai_findings", {}).get("uncertainty_flags", False),
            previous_assessment_id=previous_assessment.assessment_id if previous_assessment else None,
            risk_change=risk_change,
            risk_trend=risk_trend
        )

        db.add(new_assessment)
        db.commit()
        db.refresh(new_assessment)

        return {
            "assessment_id": assessment_id,
            "overall_risk_score": risk_result["overall_risk_score"],
            "risk_category": risk_result["risk_category"],
            "risk_category_description": _get_risk_category_description(risk_result["risk_category"]),
            "melanoma_risk": {
                "relative_risk": risk_result.get("relative_risks", {}).get("melanoma", 1.0),
                "lifetime_risk_percent": risk_result.get("lifetime_risk_percent", {}).get("melanoma", 2.0),
                "interpretation": _interpret_relative_risk(risk_result.get("relative_risks", {}).get("melanoma", 1.0))
            },
            "bcc_risk": {
                "relative_risk": risk_result.get("relative_risks", {}).get("basal_cell_carcinoma", 1.0),
                "interpretation": _interpret_relative_risk(risk_result.get("relative_risks", {}).get("basal_cell_carcinoma", 1.0))
            },
            "scc_risk": {
                "relative_risk": risk_result.get("relative_risks", {}).get("squamous_cell_carcinoma", 1.0),
                "interpretation": _interpret_relative_risk(risk_result.get("relative_risks", {}).get("squamous_cell_carcinoma", 1.0))
            },
            "component_scores": risk_result.get("component_scores", {}),
            "risk_factors": [
                {
                    **factor,
                    "risk_multiplier": factor.get("relative_risk", 1.0)  # Map for frontend compatibility
                }
                for factor in risk_result["risk_factors"]
            ],
            "recommendations": risk_result["recommendations"],
            "screening_recommendations": risk_result.get("screening_recommendations", {}),
            "comparison_to_previous": {
                "has_previous": previous_assessment is not None,
                "risk_change": risk_change,
                "trend": risk_trend
            },
            "confidence_score": risk_result.get("confidence_score", 0.8),
            "created_at": new_assessment.created_at.isoformat()
        }

    except Exception as e:
        print(f"Error calculating risk: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")


@router.get("/risk-calculator/history")
async def get_risk_assessment_history(
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's skin cancer risk assessment history."""
    from database import SkinCancerRiskAssessment

    try:
        assessments = db.query(SkinCancerRiskAssessment).filter(
            SkinCancerRiskAssessment.user_id == current_user.id
        ).order_by(SkinCancerRiskAssessment.created_at.desc()).limit(limit).all()

        return {
            "total_assessments": len(assessments),
            "assessments": [
                {
                    "assessment_id": a.assessment_id,
                    "overall_risk_score": a.overall_risk_score,
                    "risk_category": a.risk_category,
                    "melanoma_relative_risk": a.melanoma_relative_risk,
                    "risk_trend": a.risk_trend,
                    "created_at": a.created_at.isoformat() if a.created_at else None
                }
                for a in assessments
            ]
        }
    except Exception as e:
        return {"total_assessments": 0, "assessments": []}


@router.get("/risk-calculator/latest")
async def get_latest_risk_assessment(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's most recent skin cancer risk assessment."""
    from database import SkinCancerRiskAssessment

    try:
        assessment = db.query(SkinCancerRiskAssessment).filter(
            SkinCancerRiskAssessment.user_id == current_user.id
        ).order_by(SkinCancerRiskAssessment.created_at.desc()).first()

        if not assessment:
            return {
                "has_assessment": False,
                "message": "No risk assessment found. Complete a risk assessment to see your results."
            }

        return {
            "has_assessment": True,
            "assessment_id": assessment.assessment_id,
            "overall_risk_score": assessment.overall_risk_score,
            "risk_category": assessment.risk_category,
            "risk_category_description": _get_risk_category_description(assessment.risk_category),
            "melanoma_risk": {
                "relative_risk": assessment.melanoma_relative_risk,
                "lifetime_risk_percent": assessment.melanoma_lifetime_risk_percent,
                "interpretation": _interpret_relative_risk(assessment.melanoma_relative_risk)
            },
            "component_scores": {
                "genetic": assessment.genetic_score,
                "phenotype": assessment.phenotype_score,
                "sun_exposure": assessment.sun_exposure_score,
                "behavioral": assessment.behavioral_score,
                "medical_history": assessment.medical_history_score,
                "ai_findings": assessment.ai_findings_score
            },
            "risk_factors": assessment.risk_factors or [],
            "recommendations": assessment.recommendations or [],
            "screening_recommendations": {
                "self_exam_frequency": assessment.recommended_self_exam_frequency,
                "professional_exam_frequency": assessment.recommended_professional_exam_frequency,
                "urgent_referral": assessment.urgent_dermatology_referral
            },
            "comparison_to_previous": {
                "risk_change": assessment.risk_change,
                "trend": assessment.risk_trend
            },
            "created_at": assessment.created_at.isoformat() if assessment.created_at else None
        }
    except Exception as e:
        return {"has_assessment": False, "message": "Error retrieving assessment"}


@router.get("/risk-calculator/assessment/{assessment_id}")
async def get_risk_assessment_detail(
    assessment_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific risk assessment."""
    from database import SkinCancerRiskAssessment

    try:
        assessment = db.query(SkinCancerRiskAssessment).filter(
            SkinCancerRiskAssessment.assessment_id == assessment_id,
            SkinCancerRiskAssessment.user_id == current_user.id
        ).first()

        if not assessment:
            raise HTTPException(status_code=404, detail="Assessment not found")

        return {
            "assessment_id": assessment.assessment_id,
            "overall_risk_score": assessment.overall_risk_score,
            "risk_category": assessment.risk_category,
            "risk_category_description": _get_risk_category_description(assessment.risk_category),
            "melanoma_risk": {
                "relative_risk": assessment.melanoma_relative_risk,
                "lifetime_risk_percent": assessment.melanoma_lifetime_risk_percent,
                "interpretation": _interpret_relative_risk(assessment.melanoma_relative_risk)
            },
            "bcc_risk": {
                "relative_risk": assessment.bcc_relative_risk,
                "interpretation": _interpret_relative_risk(assessment.bcc_relative_risk) if assessment.bcc_relative_risk else None
            },
            "scc_risk": {
                "relative_risk": assessment.scc_relative_risk,
                "interpretation": _interpret_relative_risk(assessment.scc_relative_risk) if assessment.scc_relative_risk else None
            },
            "component_scores": {
                "genetic": assessment.genetic_score,
                "phenotype": assessment.phenotype_score,
                "sun_exposure": assessment.sun_exposure_score,
                "behavioral": assessment.behavioral_score,
                "medical_history": assessment.medical_history_score,
                "ai_findings": assessment.ai_findings_score
            },
            "risk_factors": assessment.risk_factors or [],
            "recommendations": assessment.recommendations or [],
            "screening_recommendations": {
                "self_exam_frequency": assessment.recommended_self_exam_frequency,
                "professional_exam_frequency": assessment.recommended_professional_exam_frequency,
                "urgent_referral": assessment.urgent_dermatology_referral
            },
            "methodology": {
                "version": assessment.methodology_version,
                "models_used": assessment.models_used,
                "confidence_score": assessment.confidence_score
            },
            "ai_integration": {
                "analyses_included": len(assessment.ai_analysis_ids) if assessment.ai_analysis_ids else 0,
                "high_risk_lesions": assessment.ai_high_risk_lesions_count,
                "uncertainty_flag": assessment.ai_uncertainty_flag
            },
            "comparison_to_previous": {
                "previous_assessment_id": assessment.previous_assessment_id,
                "risk_change": assessment.risk_change,
                "trend": assessment.risk_trend
            },
            "input_data": assessment.input_data,
            "created_at": assessment.created_at.isoformat() if assessment.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving assessment: {str(e)}")


@router.post("/risk-calculator/quick-assess")
async def quick_risk_assessment(
    request: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Quick risk assessment with minimal required inputs.

    Required inputs:
    - age: int
    - gender: "male" | "female"
    - fitzpatrick_type: 1-6
    - has_family_history: bool
    """
    try:
        input_data = {
            "age": request.get("age", 40),
            "gender": request.get("gender", "male"),
            "fitzpatrick_type": request.get("fitzpatrick_type", 3),
            "natural_hair_color": "dark_brown",
            "natural_eye_color": "brown",
            "freckles": request.get("freckles", "few"),
            "total_mole_count": request.get("mole_count", "some"),
            "sun_exposure_level": 3,
            "sunburn_history": {
                "childhood_severe": 2 if request.get("had_severe_sunburns") else 0,
                "childhood_mild": 5,
                "adult_severe": 1 if request.get("had_severe_sunburns") else 0,
                "adult_mild": 3
            },
            "family_history": [],
            "personal_history": [],
            "tanning_bed_use": False,
            "outdoor_occupation": False,
            "immunosuppressed": False
        }

        if request.get("has_family_history"):
            input_data["family_history"] = [
                {"relationship": "parent", "cancer_type": "skin_cancer", "age_at_diagnosis": 60}
            ]

        risk_result = calculate_skin_cancer_risk(input_data)

        return {
            "overall_risk_score": risk_result["overall_risk_score"],
            "risk_category": risk_result["risk_category"],
            "risk_category_description": _get_risk_category_description(risk_result["risk_category"]),
            "melanoma_relative_risk": risk_result.get("relative_risks", {}).get("melanoma", 1.0),
            "key_risk_factors": risk_result["risk_factors"][:3] if risk_result["risk_factors"] else [],
            "top_recommendation": risk_result["recommendations"][0] if risk_result["recommendations"] else "Perform regular skin self-examinations",
            "note": "This is a quick assessment. Complete the full assessment for comprehensive results."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick assessment failed: {str(e)}")


@router.get("/risk-calculator/risk-factors")
async def get_risk_factor_definitions():
    """
    Get definitions and explanations of all risk factors used in calculations.
    Educational endpoint for understanding the risk model.
    """
    return {
        "fitzpatrick_skin_types": {
            "type_1": {
                "description": "Very fair skin, always burns, never tans",
                "characteristics": "Pale white skin, blue/green eyes, red/blonde hair, many freckles",
                "melanoma_risk_multiplier": 2.0
            },
            "type_2": {
                "description": "Fair skin, usually burns, tans minimally",
                "characteristics": "Fair skin, blue/hazel eyes, blonde/light brown hair",
                "melanoma_risk_multiplier": 1.5
            },
            "type_3": {
                "description": "Medium skin, sometimes burns, tans gradually",
                "characteristics": "Cream white to light brown skin",
                "melanoma_risk_multiplier": 1.0
            },
            "type_4": {
                "description": "Olive skin, rarely burns, tans easily",
                "characteristics": "Light brown to moderate brown skin",
                "melanoma_risk_multiplier": 0.6
            },
            "type_5": {
                "description": "Brown skin, very rarely burns, tans very easily",
                "characteristics": "Brown skin",
                "melanoma_risk_multiplier": 0.3
            },
            "type_6": {
                "description": "Dark brown/black skin, never burns",
                "characteristics": "Dark brown to black skin",
                "melanoma_risk_multiplier": 0.1
            }
        },
        "family_history_impact": {
            "first_degree_relative": {
                "description": "Parent, sibling, or child with skin cancer",
                "melanoma_risk_increase": "2-3x increased risk"
            },
            "second_degree_relative": {
                "description": "Grandparent, aunt, uncle, or cousin with skin cancer",
                "melanoma_risk_increase": "1.5x increased risk"
            },
            "multiple_relatives": {
                "description": "2+ family members with melanoma",
                "melanoma_risk_increase": "4-8x increased risk, consider genetic counseling"
            }
        },
        "mole_count_impact": {
            "few": {"count": "0-10", "risk_multiplier": 1.0},
            "some": {"count": "11-25", "risk_multiplier": 1.5},
            "many": {"count": "26-50", "risk_multiplier": 2.5},
            "very_many": {"count": "50+", "risk_multiplier": 4.0}
        },
        "sunburn_history_impact": {
            "childhood_severe_sunburns": "Each severe childhood sunburn increases melanoma risk by 50-100%",
            "adult_sunburns": "Adult sunburns also increase risk, but less than childhood exposure"
        },
        "other_risk_factors": {
            "immunosuppression": "3-8x increased risk of skin cancer",
            "tanning_bed_use": "59% increased risk of melanoma with any use",
            "outdoor_occupation": "Increased cumulative UV exposure",
            "previous_skin_cancer": "Significantly increased risk of additional skin cancers"
        }
    }
