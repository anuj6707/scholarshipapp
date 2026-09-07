import os
import joblib
import logging

logger = logging.getLogger(__name__)

_SYSTEM_OBJ = None
_SYSTEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scholarship_system.pkl")

# Mapping of scholarship IDs (as used in JSON database & web app) to model target keys
SCHOLARSHIP_ID_TO_TARGET = {
    "cybage_khushboo": "cybage_khushboo",
    "skf": "skf",
    "lila_poonawala": "lila_poonawala",
    "katalyst": "katalyst",
    "colgate_keep_india_smiling": "colgate_keep_india_smiling",
    "kiran_girls": "kiran_girls",
    "queens_scholarship": "queens_scholarship",
    "reliance": "reliance_foundation",
    "cummins": "cummins",
    "ffe": "foundation_for_excellence",
    "ieee_wie": "ieee_wie",
    "siemens": "siemens",
    "sitaram_jindal": "sitaram_jindal",
    "hdfc_ecss": "hdfc_ecss",
    "swami_dayanand": "swami_dayanand",
    "yashad_sumedha": "yashad_sumedha",
    "nse_india": "nice_nse",
    "op_jindal_engineering": "opjems",
    "m_scholarship": "magma_scholarship",
    "jspn": "jspn",
    "padala_charitable_trust": "padala_charitable_trust",
    "rajarshi_shahu_maharaj": "rajarshi_shahu",
    "adobe_wit": "adobe_wit",
    "glow_and_lovely": "glow_lovely",
    "loreal_young_women": "loreal_women_science",
    "ugam_legrand": "ugam_legrand",
    "indous_women_stemm": "indous_stemm",
    "aicte_pragati": "aicte_pragati",
    "aicte_saksham": "aicte_saksham",
    "rmd_foundation": "rmd_foundation",
    "disha_parivar": "disha_parivar",
    "iocl_merit": "iocl_merit",
}

TARGET_TO_SCHOLARSHIP_ID = {v: k for k, v in SCHOLARSHIP_ID_TO_TARGET.items()}

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

def load_system(model_path: str = None):
    """
    Loads and returns the combined ML scholarship system dictionary.
    Loaded once as a singleton at application startup.
    """
    global _SYSTEM_OBJ
    if _SYSTEM_OBJ is not None:
        return _SYSTEM_OBJ

    path = model_path or _SYSTEM_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scholarship system pickle file not found at path: {path}")

    try:
        _SYSTEM_OBJ = joblib.load(path)
        logger.info(f"Scholarship ML system successfully loaded from {path}")
        return _SYSTEM_OBJ
    except Exception as e:
        logger.error(f"Failed to load scholarship system from {path}: {e}")
        raise RuntimeError(f"Error loading scholarship system from {path}: {e}")

def get_system():
    """
    Returns the cached system object.
    """
    return load_system()
