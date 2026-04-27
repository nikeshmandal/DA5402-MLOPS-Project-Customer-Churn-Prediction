"""
Customer Churn Prediction – FastAPI Backend
===========================================
Exposes REST endpoints for:
  - Single prediction  POST /predict
  - Batch prediction   POST /predict/batch
  - Health checks      GET  /health  |  GET /ready
  - Model metadata     GET  /model/info
  - Feature importance GET  /model/features
  - Drift report       GET  /monitoring/drift
  - Prometheus metrics GET  /metrics

Author: Nikesh Kumar Mandal (ID25M805)
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Optional Prometheus instrumentation ───────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# ── Prometheus metrics ─────────────────────────────────────────────────────────
if PROMETHEUS_AVAILABLE:
    PREDICTION_COUNTER    = Counter("churn_predictions_total", "Total predictions made", ["result"])
    PREDICTION_LATENCY    = Histogram("churn_prediction_latency_seconds", "Prediction latency")
    ERROR_COUNTER         = Counter("churn_api_errors_total", "Total API errors", ["endpoint"])
    ACTIVE_REQUESTS       = Gauge("churn_active_requests", "Current active requests")
    DRIFT_GAUGE           = Gauge("churn_drift_detected", "1 if drift detected, 0 otherwise")

# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLOps-compliant REST API for predicting customer churn.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory stores for drift detection ──────────────────────────────────────
_recent_inputs: list[dict] = []   # last 500 inference inputs
_MAX_DRIFT_WINDOW = 500

# ── Model loading ──────────────────────────────────────────────────────────────
def load_artifacts():
    """Load model, encoders, scaler, and metadata at startup."""
    artifacts = {}
    try:
        with open(ARTIFACTS_DIR / "best_model.pkl", "rb") as f:
            artifacts["model"] = pickle.load(f)
        with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
            artifacts["scaler"] = pickle.load(f)
        with open(ARTIFACTS_DIR / "label_encoders.pkl", "rb") as f:
            artifacts["label_encoders"] = pickle.load(f)
        with open(ARTIFACTS_DIR / "feature_names.json") as f:
            artifacts["feature_names"] = json.load(f)
        with open(ARTIFACTS_DIR / "model_metadata.json") as f:
            artifacts["metadata"] = json.load(f)
        logger.info("All artifacts loaded successfully.")
    except FileNotFoundError as e:
        logger.error("Artifact missing: %s. Run the training pipeline first.", e)
        artifacts["model"] = None
    return artifacts


ARTIFACTS = load_artifacts()


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    """
    Input schema for a single customer.
    All fields mirror the raw CSV columns.
    """
    tenure: float = Field(..., ge=0, le=120, description="Months as customer (0-120)")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill in USD")
    total_charges: float = Field(..., ge=0, description="Total lifetime spend in USD")
    num_products: int = Field(..., ge=1, le=10, description="Number of subscribed products")
    contract_type: str = Field(..., description="Month-to-month | One year | Two year")
    payment_method: str = Field(..., description="Payment method used")
    internet_service: str = Field(..., description="DSL | Fiber optic | No")
    online_security: str = Field(default="No")
    tech_support: str = Field(default="No")
    paperless_billing: str = Field(default="No")
    senior_citizen: int = Field(default=0, ge=0, le=1)
    gender: str = Field(default="Male")
    partner: str = Field(default="No")
    dependents: str = Field(default="No")
    phone_service: str = Field(default="Yes")
    multiple_lines: str = Field(default="No")
    streaming_tv: str = Field(default="No")
    streaming_movies: str = Field(default="No")
    device_protection: str = Field(default="No")
    online_backup: str = Field(default="No")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 5,
                "monthly_charges": 89.99,
                "total_charges": 450.0,
                "num_products": 2,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "tech_support": "No",
                "paperless_billing": "Yes",
                "senior_citizen": 0,
                "gender": "Male",
                "partner": "No",
                "dependents": "No",
                "phone_service": "Yes",
                "multiple_lines": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes",
                "device_protection": "No",
                "online_backup": "No",
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: int           # 0 or 1
    churn_probability: float        # 0.0–1.0
    risk_level: str                 # Low / Medium / High
    confidence: str                 # Confident / Moderate
    inference_time_ms: float


class BatchInput(BaseModel):
    customers: list[CustomerInput]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_customers: int
    churn_count: int
    churn_rate: float


# ── Helper functions ───────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "contract_type", "payment_method", "internet_service", "online_security",
    "tech_support", "paperless_billing", "gender", "partner", "dependents",
    "phone_service", "multiple_lines", "streaming_tv", "streaming_movies",
    "device_protection", "online_backup",
]
NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges", "num_products", "senior_citizen"]


def preprocess_input(data: dict) -> pd.DataFrame:
    """Apply the same transformations used during training."""
    df = pd.DataFrame([data])

    # Engineered features
    df["charges_per_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)
    df["value_score"] = df["total_charges"] / (df["monthly_charges"] + 1)

    # Label encode
    label_encoders = ARTIFACTS.get("label_encoders", {})
    for col in CATEGORICAL_COLS:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Scale numerics
    scaler = ARTIFACTS.get("scaler")
    numeric_present = [c for c in NUMERIC_COLS + ["charges_per_tenure", "value_score"] if c in df.columns]
    if scaler:
        df[numeric_present] = scaler.transform(df[numeric_present])

    # Reorder columns
    feature_names = ARTIFACTS.get("feature_names", [])
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)

    return df


def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def confidence_label(prob: float) -> str:
    return "Confident" if prob >= 0.70 or prob <= 0.30 else "Moderate"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    """Liveness probe – returns 200 if the API process is alive."""
    return {"status": "ok", "service": "churn-prediction-api"}


@app.get("/ready", tags=["Health"])
def ready():
    """Readiness probe – returns 200 only if the model is loaded."""
    if ARTIFACTS.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    return {"status": "ready", "model": ARTIFACTS["metadata"].get("model_name", "unknown")}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns model metadata: algorithm, hyperparameters, val metrics."""
    if not ARTIFACTS.get("metadata"):
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    return ARTIFACTS["metadata"]


@app.get("/model/features", tags=["Model"])
def feature_importance():
    """Returns feature importances from the trained model."""
    path = ARTIFACTS_DIR / "feature_importances.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Feature importances not computed.")
    with open(path) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: CustomerInput):
    """
    Predict churn for a single customer.

    Returns
    -------
    - churn_prediction : 0 = will not churn, 1 = will churn
    - churn_probability: probability of churning (0–1)
    - risk_level       : Low / Medium / High
    - confidence       : Confident / Moderate
    - inference_time_ms: how long the inference took
    """
    if ARTIFACTS.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if PROMETHEUS_AVAILABLE:
        ACTIVE_REQUESTS.inc()

    try:
        t0 = time.time()
        df = preprocess_input(customer.model_dump())
        model = ARTIFACTS["model"]
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        # Store for drift detection
        _recent_inputs.append(customer.model_dump())
        if len(_recent_inputs) > _MAX_DRIFT_WINDOW:
            _recent_inputs.pop(0)

        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNTER.labels(result="churn" if pred == 1 else "no_churn").inc()
            PREDICTION_LATENCY.observe(elapsed_ms / 1000)

        return PredictionResponse(
            churn_prediction=pred,
            churn_probability=round(prob, 4),
            risk_level=risk_label(prob),
            confidence=confidence_label(prob),
            inference_time_ms=elapsed_ms,
        )

    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNTER.labels(endpoint="/predict").inc()
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if PROMETHEUS_AVAILABLE:
            ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(batch: BatchInput):
    """Batch prediction for multiple customers."""
    if ARTIFACTS.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(batch.customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size must not exceed 1000.")

    predictions = []
    for customer in batch.customers:
        try:
            df = preprocess_input(customer.model_dump())
            pred = int(ARTIFACTS["model"].predict(df)[0])
            prob = float(ARTIFACTS["model"].predict_proba(df)[0][1])
            predictions.append(PredictionResponse(
                churn_prediction=pred,
                churn_probability=round(prob, 4),
                risk_level=risk_label(prob),
                confidence=confidence_label(prob),
                inference_time_ms=0,
            ))
        except Exception as e:
            logger.error("Batch item error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    churn_count = sum(p.churn_prediction for p in predictions)
    return BatchResponse(
        predictions=predictions,
        total_customers=len(predictions),
        churn_count=churn_count,
        churn_rate=round(churn_count / len(predictions), 4),
    )


@app.get("/monitoring/drift", tags=["Monitoring"])
def drift_report():
    """
    Returns the latest drift detection report.
    Uses the last 500 inference inputs vs training baseline.
    """
    if len(_recent_inputs) < 30:
        return {"message": "Not enough data for drift analysis (need ≥30 predictions).", "count": len(_recent_inputs)}

    try:
        from src.monitoring.drift_detector import DriftDetector
        detector = DriftDetector()
        live_data = {
            col: [inp.get(col, 0) for inp in _recent_inputs]
            for col in ["tenure", "monthly_charges", "total_charges", "num_products"]
        }
        report = detector.detect(live_data)
        if PROMETHEUS_AVAILABLE:
            DRIFT_GAUGE.set(1 if report["drift_detected"] else 0)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if PROMETHEUS_AVAILABLE:
    @app.get("/metrics", tags=["Monitoring"])
    def metrics():
        """Prometheus scrape endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
