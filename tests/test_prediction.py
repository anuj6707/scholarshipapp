import unittest
import pandas as pd
from ml.model_loader import get_system, FEATURE_NAMES, SCHOLARSHIP_ID_TO_TARGET
from ml.predict import predict_scholarships
from ml.features import engineer_features, ALL_FEATURES


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
        # Support both old ("ml_models") and new ("models") artifact key
        models = sys_obj.get("models", sys_obj.get("ml_models", {}))
        self.assertGreaterEqual(len(models), 30)  # at least 30 of 32 targets
        self.assertIn("target_columns", sys_obj)
        self.assertIn("feature_columns", sys_obj)

    def test_engineer_features(self):
        raw_df = pd.DataFrame([{
            "gender": "Female", "age": 19, "domicile": "Maharashtra",
            "category": "General", "disability": "No", "disability_percentage": 0,
            "branch": "CSE", "year": 1, "cgpa": 8.5, "percentage": 88.0,
            "family_income": 200000, "bpl_status": "No", "hostel_status": "Yes",
        }])
        df = engineer_features(raw_df)
        self.assertEqual(len(df), 1)
        # Check engineered values
        self.assertEqual(df["is_first_year"].iloc[0], 1)
        self.assertEqual(df["financial_need"].iloc[0], 1)
        self.assertEqual(df["high_academic_performer"].iloc[0], 1)
        self.assertAlmostEqual(df["academic_score"].iloc[0], (8.5 * 10 + 88.0) / 2.0)
        self.assertEqual(df["cs_it_branch"].iloc[0], 1)
        self.assertEqual(df["stem_branch"].iloc[0], 1)
        # All expected features present
        for feat in ALL_FEATURES:
            self.assertIn(feat, df.columns)

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
            self.assertIn("recommendation_score", score_data)
            self.assertGreaterEqual(score_data["ml_score"], 0.0)
            self.assertLessEqual(score_data["ml_score"], 1.0)
            self.assertGreaterEqual(score_data["recommendation_score"], 0.0)
            self.assertLessEqual(score_data["recommendation_score"], 1.0)
    def test_rank_by_confidence(self):
        from ml.predict import rank_by_confidence
        result = rank_by_confidence(self.valid_student)
        self.assertIn("eligible", result)
        self.assertIn("ineligible", result)
        self.assertIn("total_evaluated", result)
        self.assertGreater(result["total_evaluated"], 0)
        # At least some scholarships should be recommended
        total = len(result["eligible"]) + len(result["ineligible"])
        self.assertEqual(total, result["total_evaluated"])
        # Check structure of eligible items
        for item in result["eligible"]:
            self.assertTrue(item["eligible"])
            self.assertGreater(item["recommendation_score_pct"], 0)
            self.assertIsInstance(item.get("eligibility_reasons"), list)
            self.assertGreater(len(item["eligibility_reasons"]), 0)

        # Check structure of ineligible items
        for item in result["ineligible"]:
            self.assertFalse(item["eligible"])
            self.assertIsInstance(item.get("failed_requirements"), list)
            self.assertGreater(len(item["failed_requirements"]), 0)

if __name__ == "__main__":
    unittest.main()
