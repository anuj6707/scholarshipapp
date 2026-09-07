import sys
sys.path.insert(0, '.')
from ml.predict import predict_scholarships
from eligibility.eligibility_engine import evaluate_and_rank_scholarships

# Profile 1: High-merit female engineering student in 2nd year
s1 = {
    'gender': 'Female',
    'age': 19,
    'domicile': 'Maharashtra',
    'category': 'OBC',
    'disability': False,
    'disability_percentage': 0,
    'branch': 'Computer Engineering',
    'year': 2,
    'cgpa': 9.2,
    'percentage': 89.0,
    'family_income': 180000,
    'bpl_status': True,
    'hostel_status': True
}

ai_scores = predict_scholarships(s1)
results = evaluate_and_rank_scholarships(s1, ai_scores)

print("=== Profile 1 (Female CSE Yr 2 Merit+Need) ===")
for s in results['eligible'][:8]:
    name = s['name']
    pct = s['recommendation_score_pct']
    print(f"  {name:<50} : Match = {pct}%")

# Profile 2: Male 1st year Mechanical engineering student
s2 = {
    'gender': 'Male',
    'age': 18,
    'domicile': 'Maharashtra',
    'category': 'General',
    'disability': False,
    'disability_percentage': 0,
    'branch': 'Mechanical Engineering',
    'year': 1,
    'cgpa': 7.8,
    'percentage': 76.0,
    'family_income': 400000,
    'bpl_status': False,
    'hostel_status': False
}

ai_scores2 = predict_scholarships(s2)
results2 = evaluate_and_rank_scholarships(s2, ai_scores2)

print("\n=== Profile 2 (Male Mech Yr 1 Mid-Income) ===")
for s in results2['eligible'][:8]:
    name = s['name']
    pct = s['recommendation_score_pct']
    print(f"  {name:<50} : Match = {pct}%")
