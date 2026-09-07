from ml.predict import predict_scholarships
from eligibility.eligibility_engine import evaluate_and_rank_scholarships

# Test 1: Female engineering student
s1 = {
    'gender': 'Female',
    'age': 19,
    'domicile': 'Maharashtra',
    'category': 'OBC',
    'disability': False,
    'disability_percentage': 0,
    'branch': 'Computer Engineering',
    'year': 2,
    'cgpa': 8.8,
    'percentage': 87.0,
    'family_income': 200000,
    'bpl_status': False,
    'hostel_status': True
}
res1 = evaluate_and_rank_scholarships(s1, predict_scholarships(s1))
print('Test 1 (Female CSE 2nd Yr) Top 5:')
for s in res1['eligible'][:5]:
    print(f"  {s['name']}: Match={s['recommendation_score_pct']}%, ML={s['ml_score_pct']}%, NLP={s['nlp_score_pct']}%")

# Test 2: Male first year student
s2 = {
    'gender': 'Male',
    'age': 18,
    'domicile': 'Maharashtra',
    'category': 'General',
    'disability': False,
    'disability_percentage': 0,
    'branch': 'Mechanical Engineering',
    'year': 1,
    'cgpa': 8.2,
    'percentage': 82.0,
    'family_income': 350000,
    'bpl_status': False,
    'hostel_status': False
}
res2 = evaluate_and_rank_scholarships(s2, predict_scholarships(s2))
print('\nTest 2 (Male Mech 1st Yr) Top 5:')
for s in res2['eligible'][:5]:
    print(f"  {s['name']}: Match={s['recommendation_score_pct']}%, ML={s['ml_score_pct']}%, NLP={s['nlp_score_pct']}%")
