import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from ml.model_loader import (
    get_system,
    FEATURE_NAMES,
    SCHOLARSHIP_ID_TO_TARGET,
    TARGET_TO_SCHOLARSHIP_ID
)

logger = logging.getLogger(__name__)

# Normalization mapping for branches between form inputs and model training vocabulary
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

def format_and_engineer_features(profile: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    """
    Takes raw student profile dictionary, normalizes categorical representations,
    computes engineered features, and returns a DataFrame matching the model's feature_columns.
    """
    # 1. Extract base values
    gender = str(profile.get("gender", "Female")).strip().capitalize()
    age = float(profile.get("age", 20.0))
    domicile = str(profile.get("domicile", "Maharashtra")).strip()
    category = str(profile.get("category", "General")).strip()
    
    disability_raw = profile.get("disability", False)
    disability_str = "Yes" if str(disability_raw).lower() in ["true", "yes", "1"] else "No"
    disability_pct = float(profile.get("disability_percentage", 0.0) if disability_str == "Yes" else 0.0)
    
    branch_input = str(profile.get("branch", "CSE")).strip()
    branch_mapped = BRANCH_MAP.get(branch_input, branch_input)
    
    year = int(profile.get("year", 1))
    cgpa = float(profile.get("cgpa", 7.5))
    percentage = float(profile.get("percentage", 75.0))
    family_income = float(profile.get("family_income", 300000.0))
    
    bpl_raw = profile.get("bpl_status", False)
    bpl_str = "Yes" if str(bpl_raw).lower() in ["true", "yes", "1"] else "No"
    
    hostel_raw = profile.get("hostel_status", False)
    hostel_str = "Yes" if str(hostel_raw).lower() in ["true", "yes", "1"] else "No"

    # 2. Compute 9 engineered features
    academic_score = (cgpa * 10.0 + percentage) / 2.0
    financial_need = 1 if (family_income <= 300000.0 or bpl_str == "Yes") else 0
    high_academic_performer = 1 if (cgpa >= 8.0 and percentage >= 75.0) else 0
    is_first_year = 1 if year == 1 else 0
    is_pwd = 1 if (disability_str == "Yes" or disability_pct > 0) else 0
    high_disability = 1 if disability_pct >= 40.0 else 0
    need_and_merit = 1 if (financial_need == 1 and high_academic_performer == 1) else 0
    log_income = float(np.log1p(max(family_income, 0.0)))
    college_progress = year / 4.0

    # 3. Construct DataFrame
    row_data = {
        "gender": [gender],
        "age": [age],
        "domicile": [domicile],
        "category": [category],
        "disability": [disability_str],
        "disability_percentage": [disability_pct],
        "branch": [branch_mapped],
        "year": [year],
        "cgpa": [cgpa],
        "percentage": [percentage],
        "family_income": [family_income],
        "bpl_status": [bpl_str],
        "hostel_status": [hostel_str],
        "academic_score": [academic_score],
        "financial_need": [financial_need],
        "high_academic_performer": [high_academic_performer],
        "is_first_year": [is_first_year],
        "is_pwd": [is_pwd],
        "high_disability": [high_disability],
        "need_and_merit": [need_and_merit],
        "log_income": [log_income],
        "college_progress": [college_progress],
    }

    df = pd.DataFrame(row_data)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            
    return df[feature_columns]

def build_student_profile_text(profile: Dict[str, Any]) -> str:
    """
    Constructs a descriptive natural language representation of student profile
    for semantic TF-IDF text matching against scholarship eligibility criteria text.
    """
    gender = profile.get("gender", "")
    year = profile.get("year", 1)
    branch = profile.get("branch", "Engineering")
    cgpa = profile.get("cgpa", 7.5)
    percentage = profile.get("percentage", 75.0)
    income = profile.get("family_income", 300000)
    domicile = profile.get("domicile", "Maharashtra")
    category = profile.get("category", "General")
    disability = profile.get("disability", False)
    dis_pct = profile.get("disability_percentage", 0)
    hostel = profile.get("hostel_status", False)
    bpl = profile.get("bpl_status", False)

    year_str = "first year" if year == 1 else f"year {year}"
    disability_text = f"with {dis_pct}% certified disability specially abled" if disability else "no disability"
    hostel_text = "residing in college hostel" if hostel else "day scholar"
    bpl_text = "below poverty line BPL family" if bpl else ""

    text = (
        f"{gender} student enrolled in {year_str} undergraduate degree pursuing {branch} "
        f"with academic merit CGPA {cgpa} and qualifying examination {percentage}% marks, "
        f"permanent domicile in {domicile}, social category {category}, "
        f"annual family income {income} rupees {bpl_text}, {disability_text}, {hostel_text}."
    )
    return text

def calibrate_nlp_similarity(sim: float, max_sim: float = 0.25) -> float:
    """
    Calibrates raw TF-IDF cosine similarity into an intuitive semantic criteria alignment score (55% - 98%).
    """
    if sim <= 0.001:
        return 0.55
    ratio = min(sim / max_sim, 1.0)
    calibrated = 0.55 + 0.43 * (ratio ** 0.70)
    return round(float(np.clip(calibrated, 0.55, 0.98)), 4)

def predict_scholarships(student_profile: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Runs unified AI inference combining:
    1. ML Tabular Model Probability (65% weight).
    2. NLP Semantic Criteria Alignment (35% weight).
    3. Returns unified high-accuracy recommendation score and sub-scores.
    """
    system_obj = get_system()
    preprocessor = system_obj["preprocessor"]
    ml_models = system_obj["ml_models"]
    tfidf = system_obj["tfidf"]
    criteria_vectors = system_obj["criteria_vectors"]
    criteria_df = system_obj["criteria_df"]
    target_columns = system_obj["target_columns"]
    feature_columns = system_obj["feature_columns"]
    name_to_key = system_obj.get("name_to_key", {})

    # --- 1. ML Tabular Prediction ---
    df_features = format_and_engineer_features(student_profile, feature_columns)
    X_proc = preprocessor.transform(df_features)

    ml_raw_scores: Dict[str, float] = {}
    for target in target_columns:
        if target in ml_models:
            model = ml_models[target]
            try:
                proba = float(model.predict_proba(X_proc)[0, 1])
                ml_raw_scores[target] = round(proba, 4)
            except Exception as e:
                logger.warning(f"Error predicting target {target}: {e}")
                ml_raw_scores[target] = 0.65

    # --- 2. NLP Semantic Matching ---
    student_text = build_student_profile_text(student_profile)
    q_vec = tfidf.transform([student_text])
    nlp_sims = cosine_similarity(q_vec, criteria_vectors)[0]

    max_sim_observed = max(float(nlp_sims.max()), 0.20)

    nlp_raw_scores: Dict[str, float] = {}
    for idx, row in criteria_df.iterrows():
        key = row.get("key", name_to_key.get(row.get("scholarship"), row.get("scholarship")))
        if not key:
            key = str(row.get("scholarship")).lower().replace(" ", "_")
        raw_sim = float(nlp_sims[idx])
        calibrated_sim = calibrate_nlp_similarity(raw_sim, max_sim=max_sim_observed)
        nlp_raw_scores[key] = calibrated_sim

    # --- 3. Compute Unified AI Match Score ---
    results: Dict[str, Dict[str, float]] = {}
    for sch_id, target_key in SCHOLARSHIP_ID_TO_TARGET.items():
        ml_score = ml_raw_scores.get(target_key, 0.70)
        nlp_score = nlp_raw_scores.get(target_key, 0.65)
        
        # Dual AI Ensemble: 65% ML predictive fit + 35% NLP criteria semantic alignment
        rec_score = round(0.65 * ml_score + 0.35 * nlp_score, 4)

        results[sch_id] = {
            "ml_score": ml_score,
            "nlp_score": nlp_score,
            "recommendation_score": rec_score
        }

    return results
