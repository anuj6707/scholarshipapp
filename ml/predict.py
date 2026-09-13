"""
ml/predict.py — Inference for scholarship recommendation.

Two modes:
  1. predict_scholarships()   — returns ML scores for use with eligibility engine
  2. rank_by_confidence()     — ranks purely by ML confidence (no hardcoded rules)
"""
import logging
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from ml.model_loader import get_system, SCHOLARSHIP_ID_TO_TARGET
from ml.features import engineer_features

logger = logging.getLogger(__name__)

# Branch mapping: form-friendly names → model vocabulary
BRANCH_MAP = {
    "Computer Engineering": "CSE", "Information Technology": "IT",
    "Electronics & Telecommunication": "ECE", "Electrical Engineering": "Electrical",
    "Mechanical Engineering": "Mechanical", "Civil Engineering": "Civil",
    "Artificial Intelligence & Data Science": "AI_DS", "AI_DS": "AI_DS",
    "Chemical Engineering": "Chemical", "Production Engineering": "Production",
    "Instrumentation Engineering": "Instrumentation", "Biotechnology": "Biotechnology",
    "Pharmacy": "Pharmacy", "Medical": "Medical", "Commerce": "Commerce",
    "Management": "Management", "Science": "Science",
}


def _profile_to_dataframe(profile: Dict[str, Any]) -> pd.DataFrame:
    """Convert raw student profile dict to a 1-row DataFrame."""
    def _bool(val): return "Yes" if str(val).lower() in ("true", "yes", "1") else "No"
    branch = BRANCH_MAP.get(str(profile.get("branch", "CSE")).strip(), str(profile.get("branch", "CSE")).strip())
    dis = _bool(profile.get("disability", False))

    return pd.DataFrame([{
        "gender": str(profile.get("gender", "Female")).strip().capitalize(),
        "age": float(profile.get("age", 20)),
        "domicile": str(profile.get("domicile", "Maharashtra")).strip(),
        "category": str(profile.get("category", "General")).strip(),
        "disability": dis,
        "disability_percentage": float(profile.get("disability_percentage", 0)) if dis == "Yes" else 0.0,
        "branch": branch,
        "year": int(profile.get("year", 1)),
        "cgpa": float(profile.get("cgpa", 7.5)),
        "percentage": float(profile.get("percentage", 75)),
        "family_income": float(profile.get("family_income", 300000)),
        "bpl_status": _bool(profile.get("bpl_status", False)),
        "hostel_status": _bool(profile.get("hostel_status", False)),
    }])


def _get_scores(student_profile: Dict[str, Any]) -> Dict[str, float]:
    """Core ML prediction — returns {target_name: probability} for all 32 targets."""
    sys_obj = get_system()
    prep = sys_obj["preprocessor"]
    models = sys_obj.get("models", sys_obj.get("ml_models", {}))

    # Engineer features
    df = engineer_features(_profile_to_dataframe(student_profile))

    # Preprocess
    num = prep["scaler"].transform(df[prep["numerical_features"]])
    cat = prep["encoder"].transform(df[prep["categorical_features"]])
    X = np.hstack([num, cat])

    scores = {}
    for target in sys_obj["target_columns"]:
        info = models.get(target)
        if info is None:
            scores[target] = 0.50
            continue
        try:
            model = info['model'] if isinstance(info, dict) else info
            pca = info.get('pca') if isinstance(info, dict) else None
            X_in = pca.transform(X) if pca else X
            prob = float(model.predict_proba(X_in)[0, 1]) if hasattr(model, 'predict_proba') else 0.5
            scores[target] = round(min(prob, 0.95), 4)  # Cap at 95%
        except Exception as e:
            logger.warning(f"Error predicting {target}: {e}")
            scores[target] = 0.50

    return scores


# ─── Mode 1: Standard (for use with eligibility engine) ─────────────────────

def predict_scholarships(student_profile: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Returns {scholarship_id: {ml_score, recommendation_score}} for eligibility engine."""
    raw_scores = _get_scores(student_profile)
    results = {}
    for sch_id, target in SCHOLARSHIP_ID_TO_TARGET.items():
        s = min(raw_scores.get(target, 0.50), 0.95)
        results[sch_id] = {"ml_score": s, "recommendation_score": s}
    return results


# ─── Mode 2: Confidence-based ranking (no hardcoded rules) ──────────────────

def rank_by_confidence(student_profile: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Rank scholarships purely by ML model confidence.
    No hardcoded eligibility rules — the model decides everything.

    Uses per-scholarship optimized thresholds from training.
    Returns same format as evaluate_and_rank_scholarships() for template compatibility.
    """
    sys_obj = get_system()
    models = sys_obj.get("models", {})
    raw_scores = _get_scores(student_profile)

    # Load scholarship metadata for display
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'scholarships.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        all_scholarships = json.load(f)
    sch_by_id = {s['id']: s for s in all_scholarships}

    eligible, ineligible = [], []

    for sch_id, target in SCHOLARSHIP_ID_TO_TARGET.items():
        scholarship = sch_by_id.get(sch_id, {})
        if not scholarship.get("active", True):
            continue

        prob = raw_scores.get(target, 0.0)

        # Get per-scholarship threshold from trained model
        info = models.get(target, {})
        threshold = info.get('threshold', 0.50) if isinstance(info, dict) else 0.50

        # Confidence threshold: use 0.40 as minimum floor
        # (even if model says 0.30, we want at least moderate confidence)
        effective_threshold = max(threshold, 0.40)

        pct = min(int(round(prob * 100)), 95)
        is_recommended = prob >= effective_threshold

        item = {
            **scholarship,
            "eligible": is_recommended,
            "recommendation_score": prob,
            "recommendation_score_pct": pct if is_recommended else 0,
            "ml_score_pct": pct,
            "match_type": "AI Confidence" if is_recommended else "Low Confidence",
            "match_badge": "ai-match" if is_recommended else "ineligible",
            "has_ml_score": True,
            "score_explanation": f"ML model confidence: {pct}% (threshold: {int(effective_threshold*100)}%)",
            "eligibility_reasons": [f"Model confidence {pct}% exceeds threshold {int(effective_threshold*100)}%"] if is_recommended else [],
            "failed_requirements": [f"Model confidence {pct}% below threshold {int(effective_threshold*100)}%"] if not is_recommended else [],
        }

        if is_recommended:
            eligible.append(item)
        else:
            ineligible.append(item)

    # Sort by score descending
    eligible.sort(key=lambda x: x["recommendation_score"], reverse=True)

    # Stagger tied high scores (same logic as eligibility engine)
    seen = {}
    for item in eligible:
        raw_pct = item["recommendation_score_pct"]
        dup = seen.get(raw_pct, 0)
        seen[raw_pct] = dup + 1
        if dup > 0 and raw_pct >= 75:
            item["recommendation_score_pct"] = max(raw_pct - dup, 70)
            item["ml_score_pct"] = item["recommendation_score_pct"]

    return {
        "eligible": eligible,
        "ineligible": ineligible,
        "total_evaluated": len(eligible) + len(ineligible),
    }
