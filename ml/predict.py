import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ml.model_loader import get_model, FEATURE_NAMES, TARGET_COLUMNS, TARGET_TO_SCHOLARSHIP_ID

logger = logging.getLogger(__name__)

def format_student_profile(profile: Dict[str, Any]) -> pd.DataFrame:
    """
    Formats raw student profile dictionary into a pandas DataFrame
    with the exact 13 features, order, and data types expected by the ML model.
    """
    missing_fields = [f for f in FEATURE_NAMES if f not in profile]
    if missing_fields:
        error_msg = (
            f"Student profile missing required features: {missing_fields}. "
            f"Expected 13 features: {FEATURE_NAMES}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Cast to expected data types matching synthetic_students.csv
    try:
        data = {
            "gender": [str(profile["gender"])],
            "age": [int(profile["age"])],
            "domicile": [str(profile["domicile"])],
            "category": [str(profile["category"])],
            "disability": [bool(profile["disability"])],
            "disability_percentage": [int(profile.get("disability_percentage", 0) if profile.get("disability") else 0)],
            "branch": [str(profile["branch"])],
            "year": [int(profile["year"])],
            "cgpa": [float(profile["cgpa"])],
            "percentage": [float(profile["percentage"])],
            "family_income": [int(profile["family_income"])],
            "bpl_status": [bool(profile["bpl_status"])],
            "hostel_status": [bool(profile["hostel_status"])],
        }
        df = pd.DataFrame(data, columns=FEATURE_NAMES)
        return df
    except (ValueError, TypeError) as e:
        error_msg = (
            f"Error coercing student profile data types: {e}. "
            f"Received profile: {profile}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

def predict_scholarships(student_profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Takes a single student profile dictionary, formats it for the ML model,
    and returns a dictionary mapping scholarship_id to recommendation score (0.0 to 1.0).

    Example return:
    {
        "cummins": 0.94,
        "siemens": 0.88,
        "reliance": 0.72,
        ...
    }
    """
    model = get_model()
    df = format_student_profile(student_profile)

    try:
        # model is MultiOutputClassifier, predict_proba returns list of length 8
        # Each item is ndarray of shape (1, 2)
        probas = model.predict_proba(df)
        
        scores: Dict[str, float] = {}
        for idx, target_col in enumerate(TARGET_COLUMNS):
            scholarship_id = TARGET_TO_SCHOLARSHIP_ID.get(target_col, target_col)
            # Positive class probability (index 1)
            score = float(probas[idx][0, 1])
            # Round score to 4 decimal places
            scores[scholarship_id] = round(score, 4)

        return scores
    except Exception as e:
        logger.error(f"Prediction failed for profile {student_profile}: {e}")
        raise RuntimeError(f"ML Model prediction failed: {e}")
