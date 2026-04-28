import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

if PROMETHEUS_AVAILABLE:
    PREDICTION_COUNTER = Counter("churn_predictions_total", "Total predictions made", ["result"])
    PREDICTION_LATENCY = Histogram("churn_prediction_latency_seconds", "Prediction latency")
    ERROR_COUNTER      = Counter("churn_api_errors_total", "Total API errors", ["endpoint"])
    ACTIVE_REQUESTS    = Gauge("churn_active_requests", "Current active requests")
    DRIFT_GAUGE        = Gauge("churn_drift_detected", "1 if drift detected, 0 otherwise")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLOps-compliant REST API for predicting customer churn.",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_recent_inputs = []
_MAX_DRIFT_WINDOW = 500

CATEGORICAL_COLS = [
    "contract_type", "payment_method", "internet_service", "online_security",
    "tech_support", "paperless_billing", "gender", "partner", "dependents",
    "phone_service", "multiple_lines", "streaming_tv", "streaming_movies",
    "device_protection", "online_backup",
]
NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges", "num_products", "senior_citizen"]


def load_artifacts():
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
        threshold_path = ARTIFACTS_DIR / "threshold.json"
        artifacts["threshold"] = json.load(open(threshold_path))["threshold"] if threshold_path.exists() else 0.5
        logger.info("All artifacts loaded. Threshold=%.2f", artifacts["threshold"])
    except FileNotFoundError as e:
        logger.error("Artifact missing: %s. Run training pipeline first.", e)
        artifacts["model"] = None
    return artifacts


ARTIFACTS = load_artifacts()


def sanitise(obj):
    """Recursively convert numpy types to native Python so FastAPI can JSON-serialise."""
    if isinstance(obj, dict):
        return {k: sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


class CustomerInput(BaseModel):
    tenure: float = Field(..., ge=0, le=200)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_products: int = Field(..., ge=1, le=10)
    contract_type: str
    payment_method: str
    internet_service: str
    online_security: str = "No"
    tech_support: str = "No"
    paperless_billing: str = "No"
    senior_citizen: int = Field(default=0, ge=0, le=1)
    gender: str = "Male"
    partner: str = "No"
    dependents: str = "No"
    phone_service: str = "Yes"
    multiple_lines: str = "No"
    streaming_tv: str = "No"
    streaming_movies: str = "No"
    device_protection: str = "No"
    online_backup: str = "No"

    class Config:
        json_schema_extra = {"example": {
            "tenure": 5, "monthly_charges": 89.99, "total_charges": 450.0,
            "num_products": 2, "contract_type": "Month-to-month",
            "payment_method": "Electronic check", "internet_service": "Fiber optic",
        }}


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    confidence: str
    inference_time_ms: float


class BatchInput(BaseModel):
    customers: list[CustomerInput]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_customers: int
    churn_count: int
    churn_rate: float


def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df["charges_per_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)
    df["value_score"] = df["total_charges"] / (df["monthly_charges"] + 1)
    for col in CATEGORICAL_COLS:
        if col in df.columns and col in ARTIFACTS.get("label_encoders", {}):
            le = ARTIFACTS["label_encoders"][col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )
    scaler = ARTIFACTS.get("scaler")
    numeric_present = [c for c in NUMERIC_COLS + ["charges_per_tenure", "value_score"] if c in df.columns]
    if scaler:
        df[numeric_present] = scaler.transform(df[numeric_present])
    feature_names = ARTIFACTS.get("feature_names", [])
    if feature_names:
        df = df.reindex(columns=feature_names, fill_value=0)
    return df


def store_for_drift(customer_dict: dict):
    _recent_inputs.append(customer_dict)
    if len(_recent_inputs) > _MAX_DRIFT_WINDOW:
        _recent_inputs.pop(0)


def risk_label(prob: float) -> str:
    if prob >= 0.70: return "High"
    if prob >= 0.40: return "Medium"
    return "Low"


def confidence_label(prob: float) -> str:
    return "Confident" if prob >= 0.70 or prob <= 0.30 else "Moderate"


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "churn-prediction-api"}


@app.get("/ready", tags=["Health"])
def ready():
    if ARTIFACTS.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ready", "model": ARTIFACTS["metadata"].get("model_name", "unknown")}


@app.get("/model/info", tags=["Model"])
def model_info():
    if not ARTIFACTS.get("metadata"):
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    return ARTIFACTS["metadata"]


@app.get("/model/features", tags=["Model"])
def feature_importance():
    path = ARTIFACTS_DIR / "feature_importances.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Feature importances not available.")
    with open(path) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: CustomerInput):
    if ARTIFACTS.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if PROMETHEUS_AVAILABLE:
        ACTIVE_REQUESTS.inc()
    try:
        t0 = time.time()
        df = preprocess_input(customer.model_dump())
        threshold = ARTIFACTS.get("threshold", 0.5)
        prob = float(ARTIFACTS["model"].predict_proba(df)[0][1])
        pred = int(prob >= threshold)
        elapsed_ms = round((time.time() - t0) * 1000, 2)
        store_for_drift(customer.model_dump())
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
    if ARTIFACTS.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(batch.customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size must not exceed 1000.")
    predictions = []
    threshold = ARTIFACTS.get("threshold", 0.5)
    for customer in batch.customers:
        try:
            df = preprocess_input(customer.model_dump())
            prob = float(ARTIFACTS["model"].predict_proba(df)[0][1])
            pred = int(prob >= threshold)
            predictions.append(PredictionResponse(
                churn_prediction=pred,
                churn_probability=round(prob, 4),
                risk_level=risk_label(prob),
                confidence=confidence_label(prob),
                inference_time_ms=0,
            ))
            store_for_drift(customer.model_dump())
            if PROMETHEUS_AVAILABLE:
                PREDICTION_COUNTER.labels(result="churn" if pred == 1 else "no_churn").inc()
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
    if len(_recent_inputs) < 30:
        return {"message": "Need >=30 predictions for drift analysis.", "count": len(_recent_inputs)}
    try:
        try:
            from src.monitoring.drift_detector import DriftDetector
        except ImportError:
            from monitoring.drift_detector import DriftDetector

        detector = DriftDetector()
        live_data = {
            col: [inp.get(col, 0) for inp in _recent_inputs]
            for col in ["tenure", "monthly_charges", "total_charges", "num_products", "senior_citizen"]
        }
        report = detector.detect(live_data)

        # Convert all numpy types to native Python before returning
        report = sanitise(report)
        report["inputs_analysed"] = len(_recent_inputs)

        if PROMETHEUS_AVAILABLE:
            DRIFT_GAUGE.set(1 if report["drift_detected"] else 0)

        return report
    except Exception as e:
        logger.exception("Drift detection error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if PROMETHEUS_AVAILABLE:
    @app.get("/metrics", tags=["Monitoring"])
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)