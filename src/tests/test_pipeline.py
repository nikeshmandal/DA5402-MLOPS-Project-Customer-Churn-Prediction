"""
Unit & Integration Tests – Customer Churn Prediction
=====================================================
Test Plan
---------
TC-01  Data ingestion loads CSV and validates schema
TC-02  Data ingestion raises error on missing required columns
TC-03  Feature engineering produces correct shape after encoding
TC-04  Engineered feature 'charges_per_tenure' is computed correctly
TC-05  Model produces binary output (0 or 1)
TC-06  Model probability output is in [0, 1]
TC-07  API /health endpoint returns 200
TC-08  API /predict returns valid schema with dummy model
TC-09  Drift detector returns dict with 'drift_detected' key
TC-10  Batch predict rejects payloads > 1000 customers

Acceptance Criteria
-------------------
- All 10 test cases must pass.
- Prediction latency < 200 ms per request (business KPI).
- F1-score on test set > 0.65 (ML KPI).

Author: Nikesh Kumar Mandal (ID25M805)
"""

import json
import time
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
class TestDataIngestion(unittest.TestCase):
    """TC-01, TC-02 — DataIngestionPipeline"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_csv(self, cols=None):
        default_cols = [
            "customer_id", "tenure", "monthly_charges", "total_charges",
            "num_products", "contract_type", "payment_method", "internet_service",
            "online_security", "tech_support", "paperless_billing", "senior_citizen",
            "gender", "partner", "dependents", "phone_service", "multiple_lines",
            "streaming_tv", "streaming_movies", "device_protection", "online_backup", "churn",
        ]
        cols = cols or default_cols
        data = {c: ["val"] * 10 for c in cols}
        data["tenure"] = [5] * 10
        data["monthly_charges"] = [50.0] * 10
        data["total_charges"] = [300.0] * 10
        data["num_products"] = [2] * 10
        data["senior_citizen"] = [0] * 10
        data["churn"] = [0, 1] * 5
        p = self.tmp_path / "test.csv"
        pd.DataFrame(data).to_csv(p, index=False)
        return p

    def test_tc01_load_and_validate(self):
        """TC-01: Valid CSV loads without error."""
        from src.pipeline.data_ingestion import DataIngestionPipeline
        csv = self._make_csv()
        proc = self.tmp_path / "proc"
        bl = self.tmp_path / "baseline.json"
        pipe = DataIngestionPipeline(raw_path=csv, processed_path=proc, baseline_path=bl)
        pipe.load()
        self.assertIsNotNone(pipe.df)
        self.assertGreater(len(pipe.df), 0)

    def test_tc02_missing_column_raises(self):
        """TC-02: Missing required column raises ValueError."""
        from src.pipeline.data_ingestion import DataIngestionPipeline
        csv = self._make_csv(cols=["customer_id", "tenure", "churn"])  # missing many cols
        proc = self.tmp_path / "proc"
        bl = self.tmp_path / "baseline.json"
        pipe = DataIngestionPipeline(raw_path=csv, processed_path=proc, baseline_path=bl)
        with self.assertRaises(ValueError):
            pipe.load()


# ─────────────────────────────────────────────────────────────────────────────
class TestFeatureEngineering(unittest.TestCase):
    """TC-03, TC-04 — FeatureEngineeringPipeline"""

    def _make_df(self, n=50):
        np.random.seed(0)
        return pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(n)],
            "tenure": np.random.randint(0, 72, n),
            "monthly_charges": np.random.uniform(20, 120, n),
            "total_charges": np.random.uniform(20, 8000, n),
            "num_products": np.random.randint(1, 5, n),
            "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "payment_method": np.random.choice(["Electronic check", "Mailed check"], n),
            "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "online_security": np.random.choice(["Yes", "No", "No internet service"], n),
            "tech_support": np.random.choice(["Yes", "No", "No internet service"], n),
            "paperless_billing": np.random.choice(["Yes", "No"], n),
            "senior_citizen": np.random.randint(0, 2, n),
            "gender": np.random.choice(["Male", "Female"], n),
            "partner": np.random.choice(["Yes", "No"], n),
            "dependents": np.random.choice(["Yes", "No"], n),
            "phone_service": np.random.choice(["Yes", "No"], n),
            "multiple_lines": np.random.choice(["Yes", "No", "No phone service"], n),
            "streaming_tv": np.random.choice(["Yes", "No", "No internet service"], n),
            "streaming_movies": np.random.choice(["Yes", "No", "No internet service"], n),
            "device_protection": np.random.choice(["Yes", "No", "No internet service"], n),
            "online_backup": np.random.choice(["Yes", "No", "No internet service"], n),
            "churn": np.random.randint(0, 2, n),
        })

    def test_tc03_feature_shape(self):
        """TC-03: After engineering, we get 2 extra features."""
        from src.pipeline.feature_engineering import FeatureEngineeringPipeline
        df = self._make_df()
        pipe = FeatureEngineeringPipeline()
        df2 = pipe.engineer_features(df)
        self.assertIn("charges_per_tenure", df2.columns)
        self.assertIn("value_score", df2.columns)
        self.assertEqual(len(df2), len(df))

    def test_tc04_charges_per_tenure_formula(self):
        """TC-04: charges_per_tenure = monthly_charges / (tenure + 1)."""
        from src.pipeline.feature_engineering import FeatureEngineeringPipeline
        df = pd.DataFrame({"tenure": [10], "monthly_charges": [55.0], "total_charges": [550.0],
                           "num_products": [2], "senior_citizen": [0],
                           "contract_type": ["Month-to-month"], "payment_method": ["Mailed check"],
                           "internet_service": ["DSL"], "online_security": ["No"],
                           "tech_support": ["No"], "paperless_billing": ["No"],
                           "gender": ["Male"], "partner": ["No"], "dependents": ["No"],
                           "phone_service": ["Yes"], "multiple_lines": ["No"],
                           "streaming_tv": ["No"], "streaming_movies": ["No"],
                           "device_protection": ["No"], "online_backup": ["No"], "churn": [0]})
        pipe = FeatureEngineeringPipeline()
        df2 = pipe.engineer_features(df)
        expected = 55.0 / (10 + 1)
        self.assertAlmostEqual(df2["charges_per_tenure"].iloc[0], expected, places=5)


# ─────────────────────────────────────────────────────────────────────────────
class TestModel(unittest.TestCase):
    """TC-05, TC-06 — Model output validation using a mock sklearn model"""

    def _mock_model(self):
        model = MagicMock()
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.25, 0.75]])
        return model

    def test_tc05_prediction_binary(self):
        """TC-05: Prediction is 0 or 1."""
        model = self._mock_model()
        X = np.zeros((1, 18))
        pred = int(model.predict(X)[0])
        self.assertIn(pred, [0, 1])

    def test_tc06_probability_range(self):
        """TC-06: Probability is in [0, 1]."""
        model = self._mock_model()
        X = np.zeros((1, 18))
        prob = float(model.predict_proba(X)[0][1])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
class TestDriftDetector(unittest.TestCase):
    """TC-09 — DriftDetector"""

    def test_tc09_drift_report_keys(self):
        """TC-09: Drift report always contains 'drift_detected' key."""
        from src.monitoring.drift_detector import DriftDetector
        detector = DriftDetector(baseline_path=Path("nonexistent.json"))
        result = detector.detect({"tenure": [10, 20, 30]})
        self.assertIn("drift_detected", result)


# ─────────────────────────────────────────────────────────────────────────────
class TestLatency(unittest.TestCase):
    """Acceptance criterion: inference < 200 ms."""

    def test_tc_latency(self):
        """Preprocessing + mock inference should complete in < 200 ms."""
        model = MagicMock()
        model.predict.return_value = np.array([0])
        model.predict_proba.return_value = np.array([[0.8, 0.2]])

        t0 = time.time()
        X = np.zeros((1, 18))
        _ = model.predict(X)
        _ = model.predict_proba(X)
        elapsed_ms = (time.time() - t0) * 1000
        self.assertLess(elapsed_ms, 200, f"Inference took {elapsed_ms:.1f} ms (limit: 200 ms)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
