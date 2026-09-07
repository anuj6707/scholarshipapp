import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_system():
    sys_obj = joblib.load('ml/scholarship_system.pkl')
    preprocessor = sys_obj['preprocessor']
    ml_models = sys_obj['ml_models']
    tfidf = sys_obj['tfidf']
    criteria_vectors = sys_obj['criteria_vectors']
    criteria_df = sys_obj['criteria_df']
    target_columns = sys_obj['target_columns']
    feature_columns = sys_obj['feature_columns']
    name_to_key = sys_obj['name_to_key']

    print("Target columns count:", len(target_columns))
    print("Criteria df count:", len(criteria_df))

    # Test student feature engineering
    raw_student = pd.DataFrame([{
        'gender': 'Female',
        'age': 19.0,
        'domicile': 'Maharashtra',
        'category': 'OBC',
        'disability': 'No',
        'disability_percentage': 0.0,
        'branch': 'CSE',
        'year': 2,
        'cgpa': 8.7,
        'percentage': 88.0,
        'family_income': 200000.0,
        'bpl_status': 'No',
        'hostel_status': 'Yes'
    }])

    df = raw_student.copy()
    cgpa_val = pd.to_numeric(df['cgpa'], errors='coerce').fillna(7.0)
    pct_val = pd.to_numeric(df['percentage'], errors='coerce').fillna(70.0)
    df['academic_score'] = (cgpa_val * 10 + pct_val) / 2.0
    
    income_val = pd.to_numeric(df['family_income'], errors='coerce').fillna(500000)
    bpl_is_yes = df['bpl_status'].astype(str).str.lower().isin(['yes', 'true', '1'])
    df['financial_need'] = ((income_val <= 300000) | bpl_is_yes).astype(int)
    df['high_academic_performer'] = ((cgpa_val >= 8.0) & (pct_val >= 75.0)).astype(int)
    
    year_val = pd.to_numeric(df['year'], errors='coerce').fillna(1)
    df['is_first_year'] = (year_val == 1).astype(int)
    
    dis_is_yes = df['disability'].astype(str).str.lower().isin(['yes', 'true', '1'])
    dis_pct_val = pd.to_numeric(df['disability_percentage'], errors='coerce').fillna(0)
    df['is_pwd'] = (dis_is_yes | (dis_pct_val > 0)).astype(int)
    df['high_disability'] = (dis_pct_val >= 40).astype(int)
    df['need_and_merit'] = (df['financial_need'] & df['high_academic_performer']).astype(int)
    df['log_income'] = np.log1p(np.maximum(income_val, 0))
    df['college_progress'] = year_val / 4.0

    X_eng = df[feature_columns]
    X_proc = preprocessor.transform(X_eng)
    print("Preprocessed shape:", X_proc.shape)

    ml_scores = {}
    for target in target_columns:
        model = ml_models[target]
        proba = float(model.predict_proba(X_proc)[0, 1])
        ml_scores[target] = proba

    print("\nTop 5 ML Predictions:")
    sorted_ml = sorted(ml_scores.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_ml[:5]:
        print(f"  {k}: {v*100:.1f}%")

    # NLP Similarity
    profile_text = f"Female student pursuing CSE in Year 2 with CGPA 8.7 and 88% from Maharashtra with family income 2 lakh and OBC category residing in hostel"
    q_vec = tfidf.transform([profile_text])
    nlp_sims = cosine_similarity(q_vec, criteria_vectors)[0]

    print("\nTop 5 NLP Similarity Matches:")
    nlp_scores = {}
    for idx, row in criteria_df.iterrows():
        key = row.get('key', name_to_key.get(row['scholarship'], row['scholarship']))
        nlp_scores[key] = float(nlp_sims[idx])

    sorted_nlp = sorted(nlp_scores.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_nlp[:5]:
        print(f"  {k}: {v*100:.1f}% (similarity score)")

if __name__ == "__main__":
    test_system()
