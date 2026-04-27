"""
Feature Engineering & Preprocessing Pipeline
=============================================
Transforms raw validated data into model-ready features.
Implements the 'Feature Store Concept' from the MLOps guidelines.

Author: Nikesh Kumar Mandal (ID25M805)
"""

import logging
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

VALIDATED_DATA = Path("data/processed/validated_churn.csv")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("src/api/artifacts")

NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges", "num_products", "senior_citizen"]
CATEGORICAL_COLS = [
    "contract_type", "payment_method", "internet_service", "online_security",
    "tech_support", "paperless_billing", "gender", "partner", "dependents",
    "phone_service", "multiple_lines", "streaming_tv", "streaming_movies",
    "device_protection", "online_backup",
]
TARGET = "churn"
DROP_COLS = ["customer_id"]


class FeatureEngineeringPipeline:
    """
    Handles all preprocessing steps needed before model training.

    Steps
    -----
    1. Drop irrelevant columns (customer_id).
    2. Encode categorical variables with LabelEncoder.
       (One-hot encoding is also valid but LabelEncoder keeps it compact
        for gradient boosting models.)
    3. Standardize numeric features with StandardScaler so that no single
       feature dominates the model due to scale differences.
    4. Create engineered features:
       - charges_per_tenure   = monthly_charges / (tenure + 1)
       - value_score          = total_charges / (monthly_charges + 1)
    5. Split into train / validation / test sets (70 / 15 / 15).
    6. Save fitted encoders and scaler as pickle artifacts so the API
       can apply the SAME transformations at inference time.

    Why save the scaler/encoders?
    ------------------------------
    The model was trained on scaled/encoded data.  At inference the raw
    user input must undergo the IDENTICAL transformation.  Saving fitted
    artifacts ensures reproducibility and prevents training-serving skew.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        logger.info("Loading validated data from %s", VALIDATED_DATA)
        df = pd.read_csv(VALIDATED_DATA)
        logger.info("Shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-driven derived features."""
        df = df.copy()
        # Avoid division by zero with +1
        df["charges_per_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)
        df["value_score"] = df["total_charges"] / (df["monthly_charges"] + 1)
        logger.info("Engineered 2 new features: charges_per_tenure, value_score")
        return df

    # ------------------------------------------------------------------
    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode all categorical columns."""
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories gracefully
                df[col] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )
        return df

    # ------------------------------------------------------------------
    def scale_numerics(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Standard-scale numeric columns."""
        df = df.copy()
        numeric_present = [c for c in NUMERIC_COLS + ["charges_per_tenure", "value_score"] if c in df.columns]
        if fit:
            df[numeric_present] = self.scaler.fit_transform(df[numeric_present])
        else:
            df[numeric_present] = self.scaler.transform(df[numeric_present])
        return df

    # ------------------------------------------------------------------
    def split(self, df: pd.DataFrame):
        """Stratified 70/15/15 split preserving churn ratio."""
        X = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
        y = df[TARGET]
        self.feature_names = list(X.columns)

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
        )
        logger.info("Split — train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    def save_artifacts(self):
        """Persist fitted scaler and encoders for inference-time use."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        with open(ARTIFACTS_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(ARTIFACTS_DIR / "label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)
        with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)

        logger.info("Saved scaler, label_encoders, and feature_names to %s", ARTIFACTS_DIR)

    # ------------------------------------------------------------------
    def save_splits(self, X_train, X_val, X_test, y_train, y_val, y_test):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            X.assign(churn=y.values).to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        logger.info("Saved train/val/test CSVs to %s", PROCESSED_DIR)

    # ------------------------------------------------------------------
    def run(self):
        """Execute the full feature engineering pipeline."""
        df = self.load()
        df = self.engineer_features(df)
        df = self.encode_categoricals(df, fit=True)
        df = self.scale_numerics(df, fit=True)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split(df)
        self.save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
        self.save_artifacts()
        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    pipe = FeatureEngineeringPipeline()
    pipe.run()
    print("Feature engineering complete.")
