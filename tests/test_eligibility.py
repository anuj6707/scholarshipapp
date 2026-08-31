import unittest
from eligibility.eligibility_engine import check_eligibility, evaluate_and_rank_scholarships, get_scholarships, get_scholarship_by_id

class TestEligibilityEngine(unittest.TestCase):
    def setUp(self):
        self.female_eng_student = {
            "gender": "Female",
            "age": 19,
            "domicile": "Maharashtra",
            "category": "General",
            "disability": False,
            "disability_percentage": 0,
            "branch": "Computer Engineering",
            "year": 1,
            "cgpa": 8.5,
            "percentage": 88.0,
            "family_income": 200000,
            "bpl_status": False,
            "hostel_status": True,
        }

        self.male_student = {
            "gender": "Male",
            "age": 20,
            "domicile": "Delhi",
            "category": "OBC",
            "disability": False,
            "disability_percentage": 0,
            "branch": "Mechanical Engineering",
            "year": 2,
            "cgpa": 7.2,
            "percentage": 75.0,
            "family_income": 450000,
            "bpl_status": False,
            "hostel_status": False,
        }

        self.pwd_student = {
            "gender": "Male",
            "age": 21,
            "domicile": "Maharashtra",
            "category": "SC",
            "disability": True,
            "disability_percentage": 50,
            "branch": "Electronics & Telecommunication",
            "year": 2,
            "cgpa": 6.8,
            "percentage": 70.0,
            "family_income": 300000,
            "bpl_status": True,
            "hostel_status": False,
        }

    def test_lila_poonawala_gender_and_domicile_rule(self):
        lila = get_scholarship_by_id("lila_poonawala")
        self.assertIsNotNone(lila)

        # Female from Maharashtra with low income should be eligible
        res_female = check_eligibility(self.female_eng_student, lila)
        self.assertTrue(res_female["eligible"])
        self.assertEqual(len(res_female["failed_requirements"]), 0)

        # Male should NOT be eligible
        res_male = check_eligibility(self.male_student, lila)
        self.assertFalse(res_male["eligible"])
        self.assertTrue(any("Female" in f for f in res_male["failed_requirements"]))

    def test_saksham_disability_rule(self):
        saksham = get_scholarship_by_id("aicte_saksham")
        self.assertIsNotNone(saksham)

        # Student with 50% disability should be eligible
        res_pwd = check_eligibility(self.pwd_student, saksham)
        self.assertTrue(res_pwd["eligible"])

        # Non-disabled student should NOT be eligible
        res_non_pwd = check_eligibility(self.female_eng_student, saksham)
        self.assertFalse(res_non_pwd["eligible"])
        self.assertTrue(any("specially-abled" in f.lower() or "pwd" in f.lower() for f in res_non_pwd["failed_requirements"]))

    def test_income_limit_rule(self):
        cummins = get_scholarship_by_id("cummins")
        self.assertIsNotNone(cummins)

        # High income student
        rich_student = self.female_eng_student.copy()
        rich_student["family_income"] = 1500000  # 15 Lakh > 6 Lakh max

        res = check_eligibility(rich_student, cummins)
        self.assertFalse(res["eligible"])
        self.assertTrue(any("income" in f.lower() for f in res["failed_requirements"]))

    def test_year_of_study_rule(self):
        siemens = get_scholarship_by_id("siemens")  # Year 1 only
        self.assertIsNotNone(siemens)

        year3_student = self.female_eng_student.copy()
        year3_student["year"] = 3

        res = check_eligibility(year3_student, siemens)
        self.assertFalse(res["eligible"])
        self.assertTrue(any("Year" in f for f in res["failed_requirements"]))

    def test_hostel_requirement(self):
        hostel_sch = get_scholarship_by_id("panjabrao_deshmukh")
        self.assertIsNotNone(hostel_sch)

        # Non-hostel student
        res = check_eligibility(self.male_student, hostel_sch)
        self.assertFalse(res["eligible"])
        self.assertTrue(any("hostel" in f.lower() for f in res["failed_requirements"]))

    def test_hybrid_ranking_output(self):
        ml_scores = {
            "cummins": 0.95,
            "siemens": 0.88,
            "reliance": 0.80,
            "lila_poonawala": 0.92,
            "skf": 0.70,
            "katalyst": 0.85,
            "adobe_wit": 0.10,
            "aicte_saksham": 0.05
        }
        res = evaluate_and_rank_scholarships(self.female_eng_student, ml_scores)
        self.assertIn("eligible", res)
        self.assertIn("ineligible", res)
        self.assertGreater(len(res["eligible"]), 0)

        # Highest score should be at index 0
        scores = [s["recommendation_score"] for s in res["eligible"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

if __name__ == "__main__":
    unittest.main()
