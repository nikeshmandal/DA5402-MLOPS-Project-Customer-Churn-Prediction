# High-Level Design (HLD) Document
## Customer Churn Prediction – MLOps System
**Author:** Nikesh Kumar Mandal (ID25M805) | IIT Madras
**Version:** 1.0 | April 2026

---

## 1. System Overview

ChurnGuard is an end-to-end MLOps system that predicts whether a telecom customer is likely to cancel their subscription (churn). It is built with full separation of concerns between the data pipeline, model training, inference API, and monitoring systems — all orchestrated via Docker Compose and DVC.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER LAYER                                  │
│        Streamlit Frontend (port 8501) — non-technical UI            │
└────────────────────────────┬────────────────────────────────────────┘
                             │ REST API (HTTP/JSON)
┌────────────────────────────▼────────────────────────────────────────┐
│                     INFERENCE LAYER                                  │
│          FastAPI Backend (port 8000)                                 │
│   /predict  /predict/batch  /health  /ready  /metrics  /model/info  │
└────────┬───────────────────────────────────────────┬────────────────┘
         │ loads artifacts                           │ exposes /metrics
┌────────▼─────────────────┐          ┌─────────────▼──────────────┐
│    MODEL ARTIFACTS        │          │   MONITORING LAYER          │
│  best_model.pkl           │          │  Prometheus (port 9090)     │
│  scaler.pkl               │          │  Grafana     (port 3000)    │
│  label_encoders.pkl       │          │  Drift Detector             │
│  feature_names.json       │          └────────────────────────────┘
└──────────────────────────┘
         ▲
         │ produced by
┌────────┴─────────────────────────────────────────────────────────────┐
│                       ML PIPELINE LAYER                              │
│  Airflow DAG / DVC pipeline:                                         │
│  data_ingestion → feature_engineering → model_training               │
└────────┬─────────────────────────────────────────────────────────────┘
         │ reads from
┌────────▼──────────────────┐   ┌───────────────────────────────────┐
│     DATA LAYER             │   │    EXPERIMENT TRACKING             │
│  data/raw/telco_churn.csv  │   │    MLflow (port 5000)             │
│  data/processed/*.csv      │   │    metrics, params, artifacts      │
│  config/baseline_stats.json│   └───────────────────────────────────┘
└───────────────────────────┘
```

---

## 3. Key Design Decisions

### 3.1 Loose Coupling (Frontend ↔ Backend)
The Streamlit UI and FastAPI backend are **completely independent services** connected only via configurable REST API calls. The frontend never imports Python ML libraries — it only sends HTTP requests. This means the frontend can be rebuilt or replaced without touching the model logic.

### 3.2 Artifact-Based Inference
All preprocessing artifacts (scaler, label encoders) are trained once and saved as pickle files. The API loads these at startup. This eliminates training-serving skew — the exact same transformations are applied both during training and inference.

### 3.3 Containerised Environments (Docker Compose)
Every service runs in its own Docker container. The `docker-compose.yml` defines 6 services: API, frontend, MLflow, Prometheus, Grafana, and Airflow. This ensures identical environments across dev, staging, and production (environment parity).

### 3.4 Multi-Model Experimentation
The training script runs 4 algorithms (Logistic Regression, Random Forest ×2, Gradient Boosting) in a loop, logs all metrics to MLflow, and automatically selects the best model by validation F1-score. New algorithms can be added by appending to the `experiments` list.

### 3.5 Drift Detection
During training, per-feature statistical baselines (mean, std, percentiles) are saved to `config/baseline_stats.json`. The `/monitoring/drift` endpoint computes a Kolmogorov-Smirnov test between recent inference inputs and these baselines. A p-value < 0.05 triggers a drift alert.

---

## 4. Data Flow

```
Raw CSV (telco_churn.csv)
  ↓  DataIngestionPipeline
Validated CSV + Baseline Stats JSON
  ↓  FeatureEngineeringPipeline
Train / Val / Test CSVs + Scaler + LabelEncoders
  ↓  TrainingPipeline (4 models)
best_model.pkl + model_metadata.json + feature_importances.json
  ↓  FastAPI loads artifacts at startup
Inference: CustomerInput JSON → preprocess → model.predict → PredictionResponse JSON
  ↓  Prometheus scrapes /metrics every 15s
Grafana visualises prediction rate, latency, errors, drift
```

---

## 5. Security Considerations
- All inter-service communication happens within a private Docker network (`churn_net`), not exposed to the internet.
- The API validates all inputs via Pydantic schemas before passing to the model.
- In production, sensitive environment variables (DB credentials, API keys) should be managed via Docker Secrets, not environment variables.

---

## 6. Scalability
- The FastAPI API uses `uvicorn` with multiple workers (`--workers 2`). Workers can be increased as traffic grows.
- Batch prediction endpoint supports up to 1,000 customers per request.
- The architecture can be extended to Kubernetes with minimal changes — each Docker Compose service maps directly to a K8s Deployment.
