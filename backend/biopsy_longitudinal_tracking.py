"""
Longitudinal Tracking for Biopsy/Histopathology Results

Tracks histopathology results over time for the same patient/lesion,
enabling analysis of:
- Progression tracking
- Treatment response monitoring
- Recurrence detection
- AI accuracy trends

This module provides functions to query and analyze historical biopsy data.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

logger = logging.getLogger(__name__)


def get_patient_biopsy_history(
    db: Session,
    user_id: int,
    lesion_group_id: Optional[int] = None,
    limit: int = 50
) -> List[Dict]:
    """
    Get biopsy/histopathology history for a patient.

    Args:
        db: Database session
        user_id: Patient user ID
        lesion_group_id: Optional lesion group to filter by
        limit: Maximum results to return

    Returns:
        List of biopsy records with analysis details
    """
    from database import AnalysisHistory

    query = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == user_id,
        AnalysisHistory.histopathology_performed == True
    )

    if lesion_group_id:
        query = query.filter(AnalysisHistory.lesion_group_id == lesion_group_id)

    query = query.order_by(desc(AnalysisHistory.histopathology_date))
    results = query.limit(limit).all()

    history = []
    for r in results:
        history.append({
            'id': r.id,
            'date': r.histopathology_date.isoformat() if r.histopathology_date else None,
            'lesion_group_id': r.lesion_group_id,
            'result': r.histopathology_result,
            'malignant': r.histopathology_malignant,
            'confidence': r.histopathology_confidence,
            'tissue_type': r.histopathology_tissue_type,
            'risk_level': r.histopathology_risk_level,
            'features': _safe_json_load(r.histopathology_features),
            'ai_concordance': r.ai_concordance,
            'ai_concordance_type': r.ai_concordance_type,
            'ai_prediction': r.predicted_class,
            'ai_confidence': r.lesion_confidence
        })

    return history


def get_lesion_progression(
    db: Session,
    lesion_group_id: int,
    include_all_analyses: bool = True
) -> Dict:
    """
    Get progression timeline for a specific lesion.

    Args:
        db: Database session
        lesion_group_id: Lesion group ID
        include_all_analyses: Include non-biopsy analyses too

    Returns:
        Progression data with timeline and status changes
    """
    from database import AnalysisHistory, LesionGroup

    # Get lesion group info
    lesion = db.query(LesionGroup).filter(LesionGroup.id == lesion_group_id).first()
    if not lesion:
        return {'error': 'Lesion group not found'}

    # Get all analyses for this lesion
    query = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == lesion_group_id
    )

    if not include_all_analyses:
        query = query.filter(AnalysisHistory.histopathology_performed == True)

    analyses = query.order_by(AnalysisHistory.created_at).all()

    timeline = []
    status_changes = []
    prev_status = None

    for a in analyses:
        entry = {
            'id': a.id,
            'date': a.created_at.isoformat() if a.created_at else None,
            'type': 'biopsy' if a.histopathology_performed else 'dermoscopy',
            'ai_prediction': a.predicted_class,
            'ai_confidence': a.lesion_confidence,
            'ai_malignant': _is_malignant_class(a.predicted_class)
        }

        if a.histopathology_performed:
            entry.update({
                'biopsy_result': a.histopathology_result,
                'biopsy_malignant': a.histopathology_malignant,
                'biopsy_confidence': a.histopathology_confidence,
                'risk_level': a.histopathology_risk_level,
                'ai_concordance': a.ai_concordance
            })

            current_status = 'malignant' if a.histopathology_malignant else 'benign'
        else:
            current_status = 'malignant_suspected' if entry['ai_malignant'] else 'benign_suspected'

        # Track status changes
        if prev_status and prev_status != current_status:
            status_changes.append({
                'date': entry['date'],
                'from_status': prev_status,
                'to_status': current_status,
                'analysis_id': a.id
            })

        prev_status = current_status
        timeline.append(entry)

    # Analyze progression
    progression_summary = _analyze_progression(timeline)

    return {
        'lesion_group_id': lesion_group_id,
        'lesion_name': lesion.name if hasattr(lesion, 'name') else None,
        'body_location': lesion.body_location if hasattr(lesion, 'body_location') else None,
        'timeline': timeline,
        'status_changes': status_changes,
        'progression_summary': progression_summary,
        'total_analyses': len(timeline),
        'total_biopsies': sum(1 for t in timeline if t['type'] == 'biopsy'),
        'current_status': prev_status
    }


def get_ai_accuracy_over_time(
    db: Session,
    user_id: Optional[int] = None,
    time_period_days: int = 365
) -> Dict:
    """
    Analyze AI prediction accuracy compared to biopsy results over time.

    Args:
        db: Database session
        user_id: Optional user ID to filter by
        time_period_days: Analysis period in days

    Returns:
        Accuracy metrics broken down by time and condition type
    """
    from database import AnalysisHistory

    cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)

    query = db.query(AnalysisHistory).filter(
        AnalysisHistory.histopathology_performed == True,
        AnalysisHistory.histopathology_date >= cutoff_date
    )

    if user_id:
        query = query.filter(AnalysisHistory.user_id == user_id)

    results = query.all()

    if not results:
        return {
            'total_biopsies': 0,
            'message': 'No biopsy data available for analysis'
        }

    # Calculate metrics
    total = len(results)
    concordant = sum(1 for r in results if r.ai_concordance)
    concordance_rate = concordant / total if total > 0 else 0

    # Break down by malignancy
    malignant_biopsies = [r for r in results if r.histopathology_malignant]
    benign_biopsies = [r for r in results if not r.histopathology_malignant]

    # Sensitivity: How often AI correctly identified malignant cases
    true_positives = sum(1 for r in malignant_biopsies
                        if _is_malignant_class(r.predicted_class))
    sensitivity = true_positives / len(malignant_biopsies) if malignant_biopsies else None

    # Specificity: How often AI correctly identified benign cases
    true_negatives = sum(1 for r in benign_biopsies
                        if not _is_malignant_class(r.predicted_class))
    specificity = true_negatives / len(benign_biopsies) if benign_biopsies else None

    # False negative rate (missed malignancies - critical metric)
    false_negatives = len(malignant_biopsies) - true_positives
    fnr = false_negatives / len(malignant_biopsies) if malignant_biopsies else None

    # Break down by condition type
    by_condition = {}
    for r in results:
        condition = r.histopathology_result or 'unknown'
        if condition not in by_condition:
            by_condition[condition] = {
                'total': 0, 'concordant': 0, 'ai_predictions': {}
            }
        by_condition[condition]['total'] += 1
        if r.ai_concordance:
            by_condition[condition]['concordant'] += 1
        # Track what AI predicted for this condition
        pred = r.predicted_class or 'unknown'
        if pred not in by_condition[condition]['ai_predictions']:
            by_condition[condition]['ai_predictions'][pred] = 0
        by_condition[condition]['ai_predictions'][pred] += 1

    # Calculate per-condition accuracy
    for condition in by_condition:
        total_cond = by_condition[condition]['total']
        conc = by_condition[condition]['concordant']
        by_condition[condition]['accuracy'] = conc / total_cond if total_cond > 0 else 0

    # Monthly breakdown
    monthly_stats = _get_monthly_breakdown(results)

    return {
        'time_period_days': time_period_days,
        'total_biopsies': total,
        'overall_concordance_rate': round(concordance_rate, 3),
        'sensitivity': round(sensitivity, 3) if sensitivity is not None else None,
        'specificity': round(specificity, 3) if specificity is not None else None,
        'false_negative_rate': round(fnr, 3) if fnr is not None else None,
        'malignant_cases': len(malignant_biopsies),
        'benign_cases': len(benign_biopsies),
        'by_condition': by_condition,
        'monthly_trends': monthly_stats,
        'critical_metrics': {
            'missed_malignancies': false_negatives,
            'total_malignancies': len(malignant_biopsies),
            'detection_rate': round(sensitivity, 3) if sensitivity is not None else None
        }
    }


def detect_recurrence(
    db: Session,
    lesion_group_id: int,
    threshold_days: int = 180
) -> Dict:
    """
    Detect potential recurrence for a lesion.

    Args:
        db: Database session
        lesion_group_id: Lesion group ID
        threshold_days: Days since treatment to check for recurrence

    Returns:
        Recurrence analysis results
    """
    from database import AnalysisHistory

    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == lesion_group_id,
        AnalysisHistory.histopathology_performed == True
    ).order_by(AnalysisHistory.histopathology_date).all()

    if len(analyses) < 2:
        return {
            'recurrence_detected': False,
            'message': 'Insufficient data for recurrence analysis (need >= 2 biopsies)'
        }

    # Find if there was a malignant -> benign -> malignant pattern
    recurrence_events = []
    was_malignant = False
    was_treated = False

    for i, a in enumerate(analyses):
        if a.histopathology_malignant:
            if was_treated:
                # Malignant after treatment = recurrence
                recurrence_events.append({
                    'date': a.histopathology_date.isoformat() if a.histopathology_date else None,
                    'type': a.histopathology_result,
                    'analysis_id': a.id,
                    'days_since_last': _days_between(
                        analyses[i-1].histopathology_date,
                        a.histopathology_date
                    ) if i > 0 else 0
                })
            was_malignant = True
        elif was_malignant and not a.histopathology_malignant:
            # Benign after malignant = treated/resolved
            was_treated = True

    return {
        'lesion_group_id': lesion_group_id,
        'recurrence_detected': len(recurrence_events) > 0,
        'recurrence_events': recurrence_events,
        'total_biopsies': len(analyses),
        'first_biopsy_date': analyses[0].histopathology_date.isoformat() if analyses else None,
        'last_biopsy_date': analyses[-1].histopathology_date.isoformat() if analyses else None,
        'recommendation': 'Close monitoring recommended' if recurrence_events else 'Continue routine follow-up'
    }


def get_treatment_response(
    db: Session,
    lesion_group_id: int
) -> Dict:
    """
    Analyze treatment response for a lesion based on biopsy results.

    Args:
        db: Database session
        lesion_group_id: Lesion group ID

    Returns:
        Treatment response analysis
    """
    from database import AnalysisHistory

    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == lesion_group_id
    ).order_by(AnalysisHistory.created_at).all()

    if not analyses:
        return {'error': 'No analyses found for this lesion'}

    # Track severity/malignancy over time
    severity_timeline = []
    for a in analyses:
        if a.histopathology_performed:
            severity = _get_severity_score(a.histopathology_result, a.histopathology_risk_level)
        else:
            severity = _get_severity_score(a.predicted_class, None)

        severity_timeline.append({
            'date': a.created_at.isoformat() if a.created_at else None,
            'severity_score': severity,
            'is_biopsy': a.histopathology_performed,
            'result': a.histopathology_result if a.histopathology_performed else a.predicted_class
        })

    # Calculate response
    if len(severity_timeline) >= 2:
        initial = severity_timeline[0]['severity_score']
        current = severity_timeline[-1]['severity_score']
        change = initial - current  # Positive = improvement

        if change > 0.3:
            response = 'excellent'
        elif change > 0.1:
            response = 'good'
        elif change > -0.1:
            response = 'stable'
        elif change > -0.3:
            response = 'progression'
        else:
            response = 'significant_progression'
    else:
        response = 'insufficient_data'
        change = None

    return {
        'lesion_group_id': lesion_group_id,
        'treatment_response': response,
        'severity_change': round(change, 2) if change else None,
        'severity_timeline': severity_timeline,
        'analysis_count': len(analyses),
        'biopsy_count': sum(1 for a in analyses if a.histopathology_performed)
    }


# Helper functions

def _safe_json_load(data):
    """Safely load JSON data."""
    if data is None:
        return None
    if isinstance(data, str):
        try:
            return json.loads(data)
        except:
            return data
    return data


def _is_malignant_class(class_name: str) -> bool:
    """Check if a class name indicates malignancy."""
    if not class_name:
        return False
    malignant_keywords = [
        'melanoma', 'carcinoma', 'malignant', 'mel', 'bcc', 'scc',
        'invasive', 'cancer'
    ]
    class_lower = class_name.lower()
    return any(kw in class_lower for kw in malignant_keywords)


def _analyze_progression(timeline: List[Dict]) -> Dict:
    """Analyze progression from timeline data."""
    if not timeline:
        return {'status': 'unknown'}

    malignant_count = sum(1 for t in timeline
                         if t.get('biopsy_malignant') or t.get('ai_malignant'))
    benign_count = len(timeline) - malignant_count

    if len(timeline) == 1:
        return {
            'status': 'single_observation',
            'current_concern': 'high' if malignant_count > 0 else 'low'
        }

    # Check trend
    recent = timeline[-3:] if len(timeline) >= 3 else timeline
    recent_malignant = sum(1 for t in recent
                          if t.get('biopsy_malignant') or t.get('ai_malignant'))

    if recent_malignant > len(recent) / 2:
        trend = 'worsening'
    elif recent_malignant == 0:
        trend = 'improving'
    else:
        trend = 'stable'

    return {
        'status': 'monitored',
        'trend': trend,
        'malignant_observations': malignant_count,
        'benign_observations': benign_count,
        'current_concern': 'high' if timeline[-1].get('biopsy_malignant') or timeline[-1].get('ai_malignant') else 'low'
    }


def _get_monthly_breakdown(results) -> List[Dict]:
    """Get monthly breakdown of accuracy."""
    monthly = {}

    for r in results:
        if not r.histopathology_date:
            continue
        month_key = r.histopathology_date.strftime('%Y-%m')
        if month_key not in monthly:
            monthly[month_key] = {'total': 0, 'concordant': 0}
        monthly[month_key]['total'] += 1
        if r.ai_concordance:
            monthly[month_key]['concordant'] += 1

    return [
        {
            'month': k,
            'total': v['total'],
            'concordant': v['concordant'],
            'accuracy': round(v['concordant'] / v['total'], 3) if v['total'] > 0 else 0
        }
        for k, v in sorted(monthly.items())
    ]


def _days_between(date1, date2) -> int:
    """Calculate days between two dates."""
    if not date1 or not date2:
        return 0
    delta = date2 - date1
    return abs(delta.days)


def _get_severity_score(result: str, risk_level: str) -> float:
    """Convert result/risk to severity score (0-1)."""
    if not result:
        return 0.5

    result_lower = result.lower()

    # High severity conditions
    if any(kw in result_lower for kw in ['melanoma_invasive', 'invasive']):
        return 1.0
    if any(kw in result_lower for kw in ['melanoma', 'mel']):
        return 0.9
    if any(kw in result_lower for kw in ['carcinoma', 'bcc', 'scc', 'malignant']):
        return 0.8

    # Moderate severity
    if any(kw in result_lower for kw in ['dysplasia', 'in_situ', 'pre']):
        return 0.6
    if risk_level == 'moderate':
        return 0.5

    # Low severity
    if any(kw in result_lower for kw in ['normal', 'benign', 'nevus']):
        return 0.2

    return 0.4  # Default moderate-low
