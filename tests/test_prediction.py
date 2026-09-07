import unittest
import pandas as pd
from ml.model_loader import get_system, FEATURE_NAMES, SCHOLARSHIP_ID_TO_TARGET
from ml.predict import format_and_engineer_features, build_student_profile_text, predict_scholarships

class TestMLPrediction(unittest.TestCase):
    def setUp(self):
        self.valid_student = {
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

    def test_system_loading(self):
        sys_obj = get_system()
        self.assertIsNotNone(sys_obj)
        self.assertIn("preprocessor", sys_obj)
        self.assertIn("ml_models", sys_obj)
        self.assertIn("tfidf", sys_obj)
        self.assertIn("criteria_vectors", sys_obj)
        self.assertIn("criteria_df", sys_obj)
        self.assertEqual(len(sys_obj["ml_models"]), 32)

    def test_format_and_engineer_features(self):
        sys_obj = get_system()
        feature_cols = sys_obj["feature_columns"]
        df = format_and_engineer_features(self.valid_student, feature_cols)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), feature_cols)
        self.assertEqual(df["gender"].iloc[0], "Female")
        self.assertEqual(df["is_first_year"].iloc[0], 1)
        self.assertEqual(df["financial_need"].iloc[0], 1)
        self.assertEqual(df["high_academic_performer"].iloc[0], 1)
        self.assertEqual(df["academic_score"].iloc[0], (8.5 * 10 + 88.0) / 2.0)

    def test_build_student_profile_text(self):
        text = build_student_profile_text(self.valid_student)
        self.assertIsInstance(text, str)
        self.assertIn("Female", text)
        self.assertIn("Computer Engineering", text)
        self.assertIn("Maharashtra", text)

    def test_predict_scholarships(self):
        scores = predict_scholarships(self.valid_student)
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), len(SCHOLARSHIP_ID_TO_TARGET))
        self.assertIn("cummins", scores)
        self.assertIn("siemens", scores)
        self.assertIn("reliance", scores)
        self.assertIn("lila_poonawala", scores)

        for sch_id, score_data in scores.items():
            self.assertIn("ml_score", score_data)
            self.assertIn("nlp_score", score_data)
            self.assertIn("recommendation_score", score_data)
            self.assertGreaterEqual(score_data["ml_score"], 0.0)
            self.assertLessEqual(score_data["ml_score"], 1.0)
            self.assertGreaterEqual(score_data["nlp_score"], 0.0)
            self.assertLessEqual(score_data["nlp_score"], 1.0)

if __name__ == "__main__":
    unittest.main()
