"""
ml/predict.py — Inference module for scholarship recommendation.

Uses the shared engineer_features() function from ml.features and the
saved artifact (scholarship_system.pkl) to predict scholarship match scores.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from ml.model_loader import (
    get_system,
    SCHOLARSHIP_ID_TO_TARGET,
)
from ml.features import engineer_features

logger = logging.getLogger(__name__)

# Branch mapping: form-friendly names → model vocabulary
BRANCH_MAP = {
    "Computer Engineering": "CSE",
    "Information Technology": "IT",
    "Electronics & Telecommunication": "ECE",
    "Electrical Engineering": "Electrical",
    "Mechanical Engineering": "Mechanical",
    "Civil Engineering": "Civil",
    "Artificial Intelligence & Data Science": "AI_DS",
    "AI_DS": "AI_DS",
    "Chemical Engineering": "Chemical",
    "Production Engineering": "Production",
    "Instrumentation Engineering": "Instrumentation",
    "Biotechnology": "Biotechnology",
    "Pharmacy": "Pharmacy",
    "Medical": "Medical",
    "Commerce": "Commerce",
    "Management": "Management",
    "Science": "Science",
}


def _profile_to_dataframe(profile: Dict[str, Any]) -> pd.DataFrame:
    """Convert raw student profile dict to a 1-row DataFrame with 13 raw features."""
    gender = str(profile.get("gender", "Female")).strip().capitalize()

    branch_input = str(profile.get("branch", "CSE")).strip()
    branch_mapped = BRANCH_MAP.get(branch_input, branch_input)

    disability_raw = profile.get("disability", False)
    disability_str = "Yes" if str(disability_raw).lower() in ("true", "yes", "1") else "No"
    disability_pct = float(profile.get("disability_percentage", 0.0)) if disability_str == "Yes" else 0.0

    bpl_raw = profile.get("bpl_status", False)
    bpl_str = "Yes" if str(bpl_raw).lower() in ("true", "yes", "1") else "No"

    hostel_raw = profile.get("hostel_status", False)
    hostel_str = "Yes" if str(hostel_raw).lower() in ("true", "yes", "1") else "No"

    return pd.DataFrame([{
        "gender": gender,
        "age": float(profile.get("age", 20.0)),
        "domicile": str(profile.get("domicile", "Maharashtra")).strip(),
        "category": str(profile.get("category", "General")).strip(),
        "disability": disability_str,
        "disability_percentage": disability_pct,
        "branch": branch_mapped,
        "year": int(profile.get("year", 1)),
        "cgpa": float(profile.get("cgpa", 7.5)),
        "percentage": float(profile.get("percentage", 75.0)),
        "family_income": float(profile.get("family_income", 300000.0)),
        "bpl_status": bpl_str,
        "hostel_status": hostel_str,
    }])


def predict_scholarships(student_profile: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Predict scholarship match scores for a student profile.

    Pipeline:
      raw dict → DataFrame → engineer_features → preprocess → (PCA) → model → scores

    Returns:
        Dict mapping scholarship_id → {"ml_score": float, "recommendation_score": float}
    """
    system_obj = get_system()
    preprocessor = system_obj["preprocessor"]
    target_columns = system_obj["target_columns"]

    # Detect artifact format: new ("models" dict-of-dicts) vs old ("ml_models" dict-of-models)
    models_dict = system_obj.get("models", system_obj.get("ml_models", {}))
    is_new_format = "models" in system_obj

    # 1. Convert profile to DataFrame and engineer features
    raw_df = _profile_to_dataframe(student_profile)
    featured_df = engineer_features(raw_df)

    # 2. Apply preprocessor (scaler + encoder)
    num_cols = preprocessor["numerical_features"]
    cat_cols = preprocessor["categorical_features"]
    num_data = preprocessor["scaler"].transform(featured_df[num_cols])
    cat_data = preprocessor["encoder"].transform(featured_df[cat_cols])
    X_base = np.hstack([num_data, cat_data])

    # 3. Predict for each scholarship target
    ml_raw_scores: Dict[str, float] = {}
    for target in target_columns:
        model_info = models_dict.get(target)
        if model_info is None:
            ml_raw_scores[target] = 0.70
            continue

        try:
            # New format: dict with 'model', 'pca', 'threshold', etc.
            if is_new_format and isinstance(model_info, dict):
                model = model_info['model']
                pca_obj = model_info.get('pca')
            else:
                # Old format: model object directly
                model = model_info
                pca_obj = None

            X_input = X_base.copy()
            if pca_obj is not None:
                X_input = pca_obj.transform(X_input)

            if hasattr(model, 'predict_proba'):
                proba = float(model.predict_proba(X_input)[0, 1])
            elif hasattr(model, 'classes_') and len(model.classes_) == 1:
                proba = float(model.classes_[0])
            else:
                proba = float(model.predict(X_input)[0])

            # Cap at 0.95 — never show 100% match
            ml_raw_scores[target] = round(min(proba, 0.95), 4)

        except Exception as e:
            logger.warning(f"Error predicting {target}: {e}")
            ml_raw_scores[target] = 0.70

    # 4. Map target column names to scholarship IDs
    results: Dict[str, Dict[str, float]] = {}
    for sch_id, target_key in SCHOLARSHIP_ID_TO_TARGET.items():
        score = min(ml_raw_scores.get(target_key, 0.70), 0.95)
        results[sch_id] = {
            "ml_score": score,
            "recommendation_score": score,
        }

    return results
