"""
ml/features.py — Shared feature engineering and constants.

This module contains the SINGLE authoritative feature engineering function
used across training, cross-validation, testing, and inference.
"""
import numpy as np
import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================

RAW_FEATURES = [
    'gender', 'age', 'domicile', 'category', 'disability', 'disability_percentage',
    'branch', 'year', 'cgpa', 'percentage', 'family_income', 'bpl_status', 'hostel_status'
]

TARGET_COLUMNS = [
    'cybage_khushboo', 'skf', 'lila_poonawala', 'katalyst', 'colgate_keep_india_smiling',
    'kiran_girls', 'queens_scholarship', 'reliance_foundation', 'cummins',
    'foundation_for_excellence', 'ieee_wie', 'siemens', 'sitaram_jindal', 'hdfc_ecss',
    'swami_dayanand', 'yashad_sumedha', 'nice_nse', 'opjems', 'magma_scholarship',
    'jspn', 'padala_charitable_trust', 'rajarshi_shahu', 'adobe_wit', 'glow_lovely',
    'loreal_women_science', 'ugam_legrand', 'indous_stemm', 'aicte_pragati',
    'aicte_saksham', 'rmd_foundation', 'disha_parivar', 'iocl_merit'
]

ENGINEERED_FEATURES = [
    'academic_score', 'merit_percentile', 'financial_need', 'financial_hardship_score',
    'high_academic_performer', 'is_first_year', 'senior_standing', 'college_progress',
    'is_pwd', 'high_disability', 'stem_branch', 'cs_it_branch', 'need_and_merit',
    'need_merit_interaction', 'log_income', 'income_bracket'
]

CATEGORICAL_FEATURES = [
    'gender', 'domicile', 'category', 'disability', 'branch', 'bpl_status', 'hostel_status'
]

NUMERICAL_FEATURES = [
    'age', 'disability_percentage', 'year', 'cgpa', 'percentage', 'family_income',
    'academic_score', 'merit_percentile', 'financial_need', 'financial_hardship_score',
    'high_academic_performer', 'is_first_year', 'senior_standing', 'college_progress',
    'is_pwd', 'high_disability', 'stem_branch', 'cs_it_branch', 'need_and_merit',
    'need_merit_interaction', 'log_income', 'income_bracket'
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES  # 7 + 22 = 29


# ============================================================
# FEATURE ENGINEERING — Single authoritative function
# ============================================================

def engineer_features(df_in):
    """
    Compute 16 engineered features from 13 raw student attributes.

    This is the SINGLE authoritative feature engineering function used for:
    - training, cross-validation, final testing, and inference.

    Args:
        df_in: DataFrame with at least the 13 RAW_FEATURES columns.

    Returns:
        DataFrame with original columns PLUS 16 engineered feature columns.
    """
    df = df_in.copy()

    # --- Ensure numeric types and handle missing values ---
    df['age'] = pd.to_numeric(df.get('age'), errors='coerce').fillna(20.0)
    df['cgpa'] = pd.to_numeric(df.get('cgpa'), errors='coerce').fillna(7.5)
    df['percentage'] = pd.to_numeric(df.get('percentage'), errors='coerce').fillna(75.0)
    df['family_income'] = pd.to_numeric(df.get('family_income'), errors='coerce').fillna(300000.0)
    df['disability_percentage'] = pd.to_numeric(df.get('disability_percentage'), errors='coerce').fillna(0.0)
    df['year'] = pd.to_numeric(df.get('year'), errors='coerce').fillna(1).astype(int)

    # --- Normalize categorical string values ---
    df['gender'] = df.get('gender', pd.Series(['Female'])).fillna('Female').astype(str).str.strip().str.capitalize()
    df['domicile'] = df.get('domicile', pd.Series(['Maharashtra'])).fillna('Maharashtra').astype(str).str.strip()
    df['category'] = df.get('category', pd.Series(['General'])).fillna('General').astype(str).str.strip()
    df['branch'] = df.get('branch', pd.Series(['CSE'])).fillna('CSE').astype(str).str.strip()
    df['disability'] = df.get('disability', pd.Series(['No'])).fillna('No').astype(str).str.strip().str.capitalize()
    df['bpl_status'] = df.get('bpl_status', pd.Series(['No'])).fillna('No').astype(str).str.strip().str.capitalize()
    df['hostel_status'] = df.get('hostel_status', pd.Series(['No'])).fillna('No').astype(str).str.strip().str.capitalize()

    cgpa = df['cgpa']
    pct = df['percentage']
    income = df['family_income']
    yr = df['year']
    dis_pct = df['disability_percentage']

    bpl_flag = df['bpl_status'].str.lower().isin(['yes', 'true', '1'])
    pwd_flag = df['disability'].str.lower().isin(['yes', 'true', '1'])
    hostel_flag = df['hostel_status'].str.lower().isin(['yes', 'true', '1'])

    # 1. Academic & Merit
    df['academic_score'] = (cgpa * 10.0 + pct) / 2.0
    df['merit_percentile'] = (cgpa * 10.0 * 0.6 + pct * 0.4)
    df['high_academic_performer'] = ((cgpa >= 8.0) & (pct >= 75.0)).astype(int)

    # 2. Financial
    df['financial_need'] = ((income <= 350000.0) | bpl_flag).astype(int)
    norm_inc = np.clip(1.0 - (income / 1000000.0), 0.0, 1.0)
    df['financial_hardship_score'] = norm_inc + (bpl_flag.astype(float) * 0.25) + (hostel_flag.astype(float) * 0.15)
    df['log_income'] = np.log1p(np.maximum(income, 0.0))
    df['income_bracket'] = np.where(income <= 200000, 1,
                           np.where(income <= 400000, 2,
                           np.where(income <= 800000, 3, 4)))

    # 3. Progression
    df['is_first_year'] = (yr == 1).astype(int)
    df['senior_standing'] = (yr >= 3).astype(int)
    df['college_progress'] = yr / 4.0

    # 4. Disability
    df['is_pwd'] = (pwd_flag | (dis_pct > 0)).astype(int)
    df['high_disability'] = (dis_pct >= 40.0).astype(int)

    # 5. Branch specialization
    stem_list = ['CSE', 'IT', 'AI_DS', 'ECE', 'Electrical', 'Mechanical',
                 'Civil', 'Chemical', 'Production', 'Instrumentation', 'Biotechnology']
    df['stem_branch'] = df['branch'].isin(stem_list).astype(int)
    df['cs_it_branch'] = df['branch'].isin(['CSE', 'IT', 'AI_DS']).astype(int)

    # 6. Interactions
    df['need_and_merit'] = (df['financial_need'] & df['high_academic_performer']).astype(int)
    df['need_merit_interaction'] = df['financial_hardship_score'] * (df['merit_percentile'] / 100.0)

    return df
