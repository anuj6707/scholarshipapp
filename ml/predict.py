"""
ml/predict.py — Inference for scholarship recommendation.

Two modes:
  1. predict_scholarships()   — returns ML scores for use with eligibility engine
  2. rank_by_confidence()     — ranks purely by ML confidence with SHAP explainability
"""
import logging
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

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

# Cache for SHAP TreeExplainers to ensure sub-millisecond per-query inference
_TREE_EXPLAINERS: Dict[str, Any] = {}


def _get_tree_explainer(target: str, model: Any):
    """Retrieve or lazily initialize a TreeExplainer for the given target model."""
    if target not in _TREE_EXPLAINERS:
        try:
            import shap
            _TREE_EXPLAINERS[target] = shap.TreeExplainer(model)
        except Exception as e:
            logger.warning(f"Could not initialize TreeExplainer for {target}: {e}")
            _TREE_EXPLAINERS[target] = None
    return _TREE_EXPLAINERS.get(target)


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


def _format_shap_reason(fname: str, sval: float, profile: Dict[str, Any], is_positive: bool) -> str:
    """Translate raw SHAP feature attribution into human-readable, domain-specific text."""
    impact_str = f"+{sval:.2f} SHAP" if is_positive else f"{sval:.2f} SHAP"

    # Financial criteria
    if any(k in fname for k in ("family_income", "log_income", "financial_need", "financial_hardship", "income_bracket")):
        inc = int(profile.get("family_income", 0))
        if is_positive:
            return f"Family annual income (₹{inc:,}) strongly aligns with financial aid criteria ({impact_str})"
        return f"Family annual income (₹{inc:,}) exceeds preferred means-tested threshold ({impact_str})"

    # Merit & CGPA
    if fname == "cgpa":
        cgpa_val = profile.get("cgpa")
        if is_positive:
            return f"Academic CGPA ({cgpa_val}) strongly supports merit qualification ({impact_str})"
        return f"Academic CGPA ({cgpa_val}) falls below high-priority score band ({impact_str})"

    # Percentage
    if fname == "percentage":
        pct_val = profile.get("percentage")
        if is_positive:
            return f"Qualifying percentage ({pct_val}%) elevates academic priority ({impact_str})"
        return f"Qualifying percentage ({pct_val}%) reduces competitive allocation match ({impact_str})"

    # Academic score / merit percentile / high academic performer
    if any(k in fname for k in ("academic_score", "merit_percentile", "high_academic_performer")):
        if is_positive:
            return f"Overall composite academic merit favors award selection ({impact_str})"
        return f"Overall academic standing falls below target competitive percentiles ({impact_str})"

    # Gender
    if "gender" in fname:
        gender_val = profile.get("gender")
        if is_positive:
            return f"Gender profile ({gender_val}) aligns with target beneficiary demographic ({impact_str})"
        return f"Gender profile ({gender_val}) does not match primary demographic quota ({impact_str})"

    # Branch & STEM
    if any(k in fname for k in ("branch", "stem", "cs_it_branch")):
        branch_val = profile.get("branch")
        if is_positive:
            return f"Enrolled branch ({branch_val}) is prioritized for technical awards ({impact_str})"
        return f"Branch of study ({branch_val}) receives lower priority allocation ({impact_str})"

    # Academic Year / Progression
    if any(k in fname for k in ("year", "first_year", "senior_standing", "college_progress")):
        year_val = profile.get("year")
        if is_positive:
            return f"Year of study (Year {year_val}) satisfies intake cohort eligibility ({impact_str})"
        return f"Year of study (Year {year_val}) is outside preferred admission cohort ({impact_str})"

    # Domicile / State
    if "domicile" in fname:
        dom_val = profile.get("domicile")
        if is_positive:
            return f"Domicile state ({dom_val}) satisfies regional quota criteria ({impact_str})"
        return f"Domicile state ({dom_val}) is outside priority regional allocation ({impact_str})"

    # Social Category
    if "category" in fname:
        cat_val = profile.get("category")
        if is_positive:
            return f"Social category ({cat_val}) qualifies under scheme-specific quota ({impact_str})"
        return f"Social category ({cat_val}) is outside designated affirmative quota ({impact_str})"

    # Disability / Inclusion
    if any(k in fname for k in ("disability", "is_pwd")):
        if is_positive:
            return f"Inclusion / PwD status satisfies specialized inclusion parameters ({impact_str})"
        return f"Applicant profile does not qualify under disability quota ({impact_str})"

    # Hostel residency
    if "hostel_status" in fname:
        hostel_val = "Hosteller" if profile.get("hostel_status") else "Day Scholar"
        if is_positive:
            return f"Residency status ({hostel_val}) aligns with scheme accommodation grant ({impact_str})"
        return f"Residency status ({hostel_val}) does not qualify for residential aid ({impact_str})"

    # Need & Merit interaction
    if "need" in fname and "merit" in fname:
        if is_positive:
            return f"Combined merit-and-means interaction strongly elevates application rank ({impact_str})"
        return f"Need-and-merit balance is below priority cutoff index ({impact_str})"

    # Economic / BPL status
    if "bpl" in fname:
        if is_positive:
            return f"Economic necessity status qualifies for prioritized financial aid ({impact_str})"
        return f"Economic necessity profile is outside highest-priority relief tier ({impact_str})"

    # Generic Fallback
    readable_name = fname.replace("_", " ").title()
    if is_positive:
        return f"Attribute '{readable_name}' positively contributes to qualification ({impact_str})"
    return f"Attribute '{readable_name}' reduces qualification match ({impact_str})"


def _compute_shap_reasons(
    target: str,
    model: Any,
    X_in: np.ndarray,
    feat_names: List[str],
    student_profile: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """
    Computes SHAP feature importance attributions for a student profile against a model.
    Returns:
        (qualifying_reasons, unmet_reasons): lists of human-readable explanation strings.
    """
    explainer = _get_tree_explainer(target, model)
    if explainer is None:
        return [], []

    try:
        raw_shap = explainer.shap_values(X_in)
        if isinstance(raw_shap, list):
            sv = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif hasattr(raw_shap, "shape"):
            if len(raw_shap.shape) == 3:
                sv = raw_shap[0, :, 1]
            elif len(raw_shap.shape) == 2:
                sv = raw_shap[0]
            else:
                sv = raw_shap
        else:
            sv = np.asarray(raw_shap)
    except Exception as e:
        logger.warning(f"SHAP attribution computation failed for {target}: {e}")
        return [], []

    # Sort indices by magnitude of impact
    pos_idx = np.where(sv > 0.005)[0]
    neg_idx = np.where(sv < -0.005)[0]

    top_pos = sorted([(feat_names[i], float(sv[i])) for i in pos_idx], key=lambda x: x[1], reverse=True)
    top_neg = sorted([(feat_names[i], float(sv[i])) for i in neg_idx], key=lambda x: x[1])

    qualifying_reasons: List[str] = []
    seen_pos_categories = set()
    for fname, sval in top_pos:
        base_cat = fname.split("_")[0]
        if base_cat in seen_pos_categories and len(qualifying_reasons) >= 2:
            continue
        seen_pos_categories.add(base_cat)
        qualifying_reasons.append(_format_shap_reason(fname, sval, student_profile, is_positive=True))
        if len(qualifying_reasons) >= 3:
            break

    unmet_reasons: List[str] = []
    seen_neg_categories = set()
    for fname, sval in top_neg:
        base_cat = fname.split("_")[0]
        if base_cat in seen_neg_categories and len(unmet_reasons) >= 2:
            continue
        seen_neg_categories.add(base_cat)
        unmet_reasons.append(_format_shap_reason(fname, sval, student_profile, is_positive=False))
        if len(unmet_reasons) >= 3:
            break

    return qualifying_reasons, unmet_reasons


def _prepare_student_features(student_profile: Dict[str, Any]):
    """Transform student profile dictionary into preprocessed feature matrix and column labels."""
    sys_obj = get_system()
    prep = sys_obj["preprocessor"]

    df = engineer_features(_profile_to_dataframe(student_profile))
    num = prep["scaler"].transform(df[prep["numerical_features"]])
    cat = prep["encoder"].transform(df[prep["categorical_features"]])
    X = np.hstack([num, cat])

    feat_names = list(prep["numerical_features"]) + list(
        prep["encoder"].get_feature_names_out(prep["categorical_features"])
    )
    return X, feat_names


def _get_scores(student_profile: Dict[str, Any]) -> Dict[str, float]:
    """Core ML prediction — returns {target_name: probability} for all 32 targets."""
    sys_obj = get_system()
    models = sys_obj.get("models", sys_obj.get("ml_models", {}))
    X, _ = _prepare_student_features(student_profile)

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


# ─── Mode 2: Confidence-based ranking with SHAP Explainability ──────────────

def rank_by_confidence(student_profile: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Rank scholarships purely by ML model confidence with SHAP-based feature attributions.
    No hardcoded eligibility rules — the model decides everything, and SHAP explains why.

    Uses per-scholarship optimized thresholds from training (with 0.40 floor).
    Populates eligibility_reasons (for eligible) and failed_requirements (for ineligible)
    using top positive and negative SHAP feature contributions.
    """
    sys_obj = get_system()
    models = sys_obj.get("models", {})
    target_columns = sys_obj.get("target_columns", [])

    X_base, feat_names = _prepare_student_features(student_profile)

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

        info = models.get(target, {})
        if not info:
            continue

        model = info['model'] if isinstance(info, dict) else info
        pca = info.get('pca') if isinstance(info, dict) else None
        threshold = info.get('threshold', 0.50) if isinstance(info, dict) else 0.50

        # Confidence threshold: use 0.40 as minimum floor
        effective_threshold = max(threshold, 0.40)

        X_input = pca.transform(X_base) if pca else X_base

        try:
            prob = float(model.predict_proba(X_input)[0, 1]) if hasattr(model, 'predict_proba') else 0.5
        except Exception as e:
            logger.warning(f"Prediction failed for {target}: {e}")
            prob = 0.50

        prob = min(prob, 0.95)
        pct = min(int(round(prob * 100)), 95)
        is_recommended = prob >= effective_threshold

        # Compute SHAP explanation reasons
        qualifying_reasons, unmet_reasons = _compute_shap_reasons(
            target=target,
            model=model,
            X_in=X_input,
            feat_names=feat_names,
            student_profile=student_profile
        )

        if is_recommended:
            reasons = [f"Model confidence ({pct}%) exceeds calibrated decision threshold ({int(effective_threshold*100)}%)"]
            if qualifying_reasons:
                reasons.extend(qualifying_reasons)
            else:
                reasons.append("Overall profile metrics satisfy multi-feature predictive criteria")
            failed = []
            score_exp = f"SHAP-Calibrated Match: {pct}% confidence (threshold {int(effective_threshold*100)}%)."
            if qualifying_reasons:
                score_exp += f" Primary positive driver: {qualifying_reasons[0]}"
        else:
            reasons = []
            failed = [f"Model confidence ({pct}%) is below calibrated decision threshold ({int(effective_threshold*100)}%)"]
            if unmet_reasons:
                failed.extend(unmet_reasons)
            else:
                failed.append("Profile metrics did not reach competitive threshold index for this scheme")
            score_exp = f"Low Confidence: {pct}% vs threshold {int(effective_threshold*100)}%."
            if unmet_reasons:
                score_exp += f" Primary constraint: {unmet_reasons[0]}"

        item = {
            **scholarship,
            "eligible": is_recommended,
            "recommendation_score": prob,
            "recommendation_score_pct": pct if is_recommended else 0,
            "ml_score_pct": pct,
            "match_type": "AI Confidence (SHAP Explained)" if is_recommended else "Low Confidence",
            "match_badge": "ai-match" if is_recommended else "ineligible",
            "has_ml_score": True,
            "score_explanation": score_exp,
            "eligibility_reasons": reasons,
            "failed_requirements": failed,
        }

        if is_recommended:
            eligible.append(item)
        else:
            ineligible.append(item)

    # Sort by score descending
    eligible.sort(key=lambda x: x["recommendation_score"], reverse=True)

    # Stagger tied high scores (same presentation logic as eligibility engine)
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
