import unittest
from app import app
from eligibility.eligibility_engine import get_scholarships, get_scholarship_by_id

class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        app.config["SECRET_KEY"] = "test-secret-key-12345"
        self.client = app.test_client()

    def test_1_homepage_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Find Scholarships You're Eligible For", response.data)
        self.assertIn(b"Find My Scholarships", response.data)

    def test_2_questionnaire_renders(self):
        response = self.client.get("/questionnaire")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Student Profile Questionnaire", response.data)
        self.assertIn(b"Step 1: Basic Information", response.data)

    def test_3_valid_recommendation_and_results(self):
        valid_form_data = {
            "gender": "Female",
            "age": "19",
            "domicile": "Maharashtra",
            "category": "General",
            "branch": "Computer Engineering",
            "year": "1",
            "cgpa": "8.75",
            "percentage": "89.5",
            "family_income": "200000",
            "bpl_status": "No",
            "disability": "No",
            "disability_percentage": "0",
            "hostel_status": "Yes"
        }

        # Submit form -> expect redirect to /results
        post_res = self.client.post("/recommend", data=valid_form_data, follow_redirects=False)
        self.assertEqual(post_res.status_code, 302)
        self.assertEqual(post_res.headers["Location"], "/results")

        # Follow redirect to /results
        get_res = self.client.get("/results")
        self.assertEqual(get_res.status_code, 200)
        self.assertIn(b"Your Scholarship Matches", get_res.data)
        self.assertIn(b"Cummins Scholarship", get_res.data)
        self.assertIn(b"Recommendation Score", get_res.data)

    def test_4_invalid_recommendation_handling(self):
        # Missing required fields
        invalid_data = {
            "gender": "",
            "age": "invalid_age",
            "branch": ""
        }
        res = self.client.post("/recommend", data=invalid_data, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Please fill in all required", res.data)

    def test_5_scholarships_directory(self):
        res = self.client.get("/scholarships")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Browse All College Scholarships", res.data)
        self.assertIn(b"Cummins Scholarship Program", res.data)
        self.assertIn(b"Siemens Scholarship Program", res.data)

    def test_6_scholarship_detail_page(self):
        res = self.client.get("/scholarships/cummins")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Cummins Scholarship Program", res.data)
        self.assertIn(b"Mandatory Documents Checklist", res.data)
        self.assertIn(b"Apply on Official Portal", res.data)
        self.assertIn(b"https://www.cummins.com", res.data)

    def test_7_missing_scholarship_404(self):
        res = self.client.get("/scholarships/nonexistent_scholarship_id")
        self.assertEqual(res.status_code, 404)
        self.assertIn(b"404", res.data)
        self.assertIn(b"Scholarship or Page Not Found", res.data)

    def test_8_about_page(self):
        res = self.client.get("/about")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"How ScholarMatch Works", res.data)
        self.assertIn(b"Hybrid Recommendation Architecture", res.data)
        self.assertIn(b"Research & Data Limitations", res.data)

    def test_9_scholarship_with_null_url_handled_gracefully(self):
        # e.g. queens_scholarship has official_url = None
        res = self.client.get("/scholarships/queens_scholarship")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Official application link currently unavailable", res.data)

    def test_10_multiple_scholarships_qualified(self):
        student_data = {
            "gender": "Female",
            "age": "18",
            "domicile": "Maharashtra",
            "category": "OBC",
            "branch": "Computer Engineering",
            "year": "1",
            "cgpa": "9.2",
            "percentage": "92.0",
            "family_income": "150000",
            "bpl_status": "Yes",
            "disability": "No",
            "disability_percentage": "0",
            "hostel_status": "Yes"
        }
        self.client.post("/recommend", data=student_data, follow_redirects=True)
        res = self.client.get("/results")
        self.assertEqual(res.status_code, 200)
        # Should be eligible for multiple (Cummins, Siemens, Lila, SKF, Katalyst, etc.)
        self.assertIn(b"Cummins Scholarship", res.data)
        self.assertIn(b"Lila Poonawala", res.data)
        self.assertIn(b"SKF India Scholarship", res.data)

    def test_11_zero_scholarships_qualified_edge_case(self):
        # Out of bounds profile (e.g. 58 year old, 1.0 CGPA, 25 Lakh income)
        extreme_student = {
            "gender": "Other",
            "age": "58",
            "domicile": "Other",
            "category": "General",
            "branch": "Civil Engineering",
            "year": "4",
            "cgpa": "2.0",
            "percentage": "35.0",
            "family_income": "9900000",
            "bpl_status": "No",
            "disability": "No",
            "disability_percentage": "0",
            "hostel_status": "No"
        }
        self.client.post("/recommend", data=extreme_student, follow_redirects=True)
        res = self.client.get("/results")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"0 matching opportunities", res.data)
        self.assertIn(b"Ineligible", res.data)

if __name__ == "__main__":
    unittest.main()
