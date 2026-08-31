import os
import joblib
import logging

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

# Mapping of target index / column name to scholarship identifier
TARGET_COLUMNS = [
    "target_cummins",
    "target_siemens",
    "target_reliance",
    "target_lila",
    "target_skf",
    "target_katalyst",
    "target_adobe_wit",
    "target_saksham",
]

TARGET_TO_SCHOLARSHIP_ID = {
    "target_cummins": "cummins",
    "target_siemens": "siemens",
    "target_reliance": "reliance",
    "target_lila": "lila_poonawala",
    "target_skf": "skf",
    "target_katalyst": "katalyst",
    "target_adobe_wit": "adobe_wit",
    "target_saksham": "aicte_saksham",
}

FEATURE_NAMES = [
    "gender",
    "age",
    "domicile",
    "category",
    "disability",
    "disability_percentage",
    "branch",
    "year",
    "cgpa",
    "percentage",
    "family_income",
    "bpl_status",
    "hostel_status",
]

def load_model(model_path: str = None):
    """
    Loads and returns the ML model pipeline singleton.
    Loads from disk only once at application startup.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = model_path or _MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"ML Model file not found at path: {path}")

    try:
        _MODEL = joblib.load(path)
        logger.info(f"ML Model successfully loaded from {path}")
        return _MODEL
    except Exception as e:
        logger.error(f"Failed to load ML model from {path}: {e}")
        raise RuntimeError(f"Error loading ML model from {path}: {e}")

def get_model():
    """
    Returns the cached model instance, loading it if not yet loaded.
    """
    return load_model()
