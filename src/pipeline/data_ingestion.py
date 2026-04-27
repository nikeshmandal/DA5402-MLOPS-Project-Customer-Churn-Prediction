"""
Data Ingestion Pipeline for Customer Churn Prediction
=====================================================
This module handles loading raw data, performing validation,
and basic EDA/baseline statistics for drift detection later.

Author: Nikesh Kumar Mandal (ID25M805)
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed")
BASELINE_STATS_PATH = Path("config/baseline_stats.json")

REQUIRED_COLUMNS = [
    "customer_id", "tenure", "monthly_charges", "total_charges",
    "num_products", "contract_type", "payment_method", "internet_service",
    "online_security", "tech_support", "paperless_billing", "senior_citizen",
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "streaming_tv", "streaming_movies", "device_protection", "online_backup",
    "churn",
]

NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges", "num_products", "senior_citizen"]
CATEGORICAL_COLS = [
    "contract_type", "payment_method", "internet_service", "online_security",
    "tech_support", "paperless_billing", "gender", "partner", "dependents",
    "phone_service", "multiple_lines", "streaming_tv", "streaming_movies",
    "device_protection", "online_backup",
]


class DataIngestionPipeline:
    """
    Orchestrates data loading, schema validation, missing-value handling,
    and baseline statistics computation.

    What it does
    ------------
    1. Load raw CSV from disk.
    2. Validate schema (required columns, dtypes).
    3. Check for missing values and duplicates.
    4. Compute statistical baselines (mean, std, distribution) for
       numeric features — stored for drift detection in production.
    5. Save validated data to the processed folder.

    Why baselines?
    --------------
    In production the input data distribution may shift (concept drift /
    data drift).  Comparing live stats against the training baseline lets
    us trigger an automatic retraining alert.
    """

    def __init__(
        self,
        raw_path: Path = RAW_DATA_PATH,
        processed_path: Path = PROCESSED_DATA_PATH,
        baseline_path: Path = BASELINE_STATS_PATH,
    ):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.baseline_path = baseline_path
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Load raw CSV and perform schema validation."""
        logger.info("Loading raw data from %s", self.raw_path)
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {self.raw_path}")

        self.df = pd.read_csv(self.raw_path)
        logger.info("Loaded %d rows × %d cols", *self.df.shape)
        self._validate_schema()
        return self.df

    def _validate_schema(self) -> None:
        """Raise ValueError if required columns are missing."""
        missing = set(REQUIRED_COLUMNS) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        logger.info("Schema validation passed.")

    # ------------------------------------------------------------------
    def validate_quality(self) -> dict:
        """
        Automated quality checks:
          - Missing value counts
          - Duplicate row counts
          - Value-range sanity checks
        Returns a quality report dict.
        """
        assert self.df is not None, "Call load() first."
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_rows": len(self.df),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "churn_distribution": self.df["churn"].value_counts().to_dict(),
        }

        # Range checks
        anomalies = []
        if (self.df["tenure"] < 0).any():
            anomalies.append("Negative tenure values detected.")
        if (self.df["monthly_charges"] < 0).any():
            anomalies.append("Negative monthly_charges detected.")
        report["anomalies"] = anomalies

        if anomalies:
            logger.warning("Data quality issues: %s", anomalies)
        else:
            logger.info("Data quality checks passed.")

        return report

    # ------------------------------------------------------------------
    def compute_baseline_stats(self) -> dict:
        """
        Compute per-feature statistics used later for drift detection.

        For numeric features: mean, std, min, max, 25/50/75 percentiles.
        For categorical features: value distribution (%).

        These values are saved to config/baseline_stats.json.
        """
        assert self.df is not None, "Call load() first."
        stats: dict = {"computed_at": datetime.utcnow().isoformat(), "numeric": {}, "categorical": {}}

        for col in NUMERIC_COLS:
            if col in self.df.columns:
                stats["numeric"][col] = {
                    "mean": float(self.df[col].mean()),
                    "std": float(self.df[col].std()),
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "p25": float(self.df[col].quantile(0.25)),
                    "p50": float(self.df[col].quantile(0.50)),
                    "p75": float(self.df[col].quantile(0.75)),
                }

        for col in CATEGORICAL_COLS:
            if col in self.df.columns:
                dist = (self.df[col].value_counts(normalize=True) * 100).round(2).to_dict()
                stats["categorical"][col] = dist

        # Save to disk
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("Baseline statistics saved to %s", self.baseline_path)
        return stats

    # ------------------------------------------------------------------
    def save_validated(self) -> Path:
        """Write the validated DataFrame to data/processed/."""
        assert self.df is not None, "Call load() first."
        self.processed_path.mkdir(parents=True, exist_ok=True)
        out = self.processed_path / "validated_churn.csv"
        self.df.to_csv(out, index=False)
        logger.info("Validated data saved to %s", out)
        return out

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Full pipeline execution: load → validate → baseline → save."""
        self.load()
        report = self.validate_quality()
        stats = self.compute_baseline_stats()
        self.save_validated()
        logger.info("Data ingestion pipeline completed.")
        return {"quality_report": report, "baseline_stats": stats}


# ── Entrypoint ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    result = pipeline.run()
    print(json.dumps(result["quality_report"], indent=2))
