import unittest
import pandas as pd
from ml.model_loader import get_model, FEATURE_NAMES, TARGET_COLUMNS
from ml.predict import format_student_profile, predict_scholarships

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

    def test_model_loading(self):
        model = get_model()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict_proba"))
        self.assertTrue(hasattr(model, "steps"))

    def test_format_student_profile(self):
        df = format_student_profile(self.valid_student)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), FEATURE_NAMES)
        self.assertEqual(df["gender"].iloc[0], "Female")
        self.assertEqual(df["age"].iloc[0], 19)
        self.assertEqual(df["cgpa"].iloc[0], 8.5)

    def test_missing_feature_raises_error(self):
        incomplete_profile = self.valid_student.copy()
        del incomplete_profile["family_income"]
        with self.assertRaises(ValueError):
            format_student_profile(incomplete_profile)

    def test_predict_scholarships(self):
        scores = predict_scholarships(self.valid_student)
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), len(TARGET_COLUMNS))
        self.assertIn("cummins", scores)
        self.assertIn("siemens", scores)
        self.assertIn("reliance", scores)
        self.assertIn("lila_poonawala", scores)

        for sch_id, score in scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()
