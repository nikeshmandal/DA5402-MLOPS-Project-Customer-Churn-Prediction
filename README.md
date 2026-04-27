# ChurnGuard – Customer Churn Prediction with MLOps
**IIT Madras | Building an AI Application with MLOps**
**Student:** Nikesh Kumar Mandal | **Roll:** ID25M805 | **Mode:** Individual

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Running the Pipeline](#running-the-pipeline)
4. [Running the Application](#running-the-application)
5. [Running Tests](#running-tests)
6. [MLOps Tools & Ports](#mlops-tools--ports)
7. [API Reference](#api-reference)

---

## Quick Start

### Prerequisites
- Docker Desktop ≥ 24.0
- Docker Compose ≥ 2.20
- Git & DVC

### One-Command Launch
```bash
# Clone and start all services
git clone <repo_url>
cd churn_prediction
docker-compose up --build
```

### Access the Application
| Service | URL |
|---|---|
| 🔮 Streamlit UI | http://localhost:8501 |
| ⚡ FastAPI Docs | http://localhost:8000/docs |
| 📊 MLflow UI | http://localhost:5000 |
| 📈 Grafana | http://localhost:3000 (admin/admin) |
| 🔥 Prometheus | http://localhost:9090 |
| 🌬️ Airflow | http://localhost:8080 (admin/admin) |

---

## Project Structure
```
churn_prediction/
├── data/
│   ├── raw/                     # Raw input CSV
│   └── processed/               # Train/Val/Test splits
├── src/
│   ├── pipeline/
│   │   ├── data_ingestion.py    # Load, validate, baseline stats
│   │   ├── feature_engineering.py # Encode, scale, split
│   │   ├── train.py             # Multi-model training + MLflow
│   │   └── churn_dag.py         # Airflow DAG definition
│   ├── api/
│   │   ├── main.py              # FastAPI app (inference backend)
│   │   └── artifacts/           # Trained model + preprocessing artifacts
│   ├── monitoring/
│   │   └── drift_detector.py    # KS-test drift detection
│   └── tests/
│       └── test_pipeline.py     # 10 unit/integration tests
├── frontend/
│   └── app.py                   # Streamlit multi-page UI
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.frontend
├── grafana/
│   ├── dashboards/              # Pre-built Grafana dashboard JSON
│   └── provisioning/            # Auto-loaded datasource/dashboard config
├── prometheus/
│   └── prometheus.yml           # Scrape config
├── docs/
│   ├── HLD.md                   # High-Level Design
│   ├── LLD.md                   # Low-Level Design + API specs
│   └── TEST_REPORT.md           # Test plan + results
├── config/
│   └── baseline_stats.json      # Training feature baselines for drift
├── dvc.yaml                     # DVC pipeline stages
├── params.yaml                  # Hyperparameters (tracked by DVC)
├── metrics.json                 # Final model metrics
├── docker-compose.yml           # All 6 services
├── requirements.txt
└── README.md
```

---

## Running the Pipeline

### Option A: DVC (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (ingestion → features → training)
dvc repro

# See the pipeline DAG
dvc dag

# Check metrics
dvc metrics show
```

### Option B: Manual Step-by-Step
```bash
export PYTHONPATH=.

# Step 1: Data ingestion & validation
python src/pipeline/data_ingestion.py

# Step 2: Feature engineering
python src/pipeline/feature_engineering.py

# Step 3: Model training
python src/pipeline/train.py
```

### Option C: Airflow (when running via Docker Compose)
- Navigate to http://localhost:8080
- Enable the `churn_prediction_pipeline` DAG
- Trigger a run manually or let it run on its `@daily` schedule

---

## Running the Application

### With Docker Compose (Recommended)
```bash
docker-compose up --build

# To stop all services
docker-compose down

# To view logs
docker-compose logs -f api
docker-compose logs -f frontend
```

### Without Docker (Development Mode)
```bash
# Terminal 1 – Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 – Start Frontend
streamlit run frontend/app.py

# Terminal 3 – Start MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## Running Tests

```bash
# Run all unit tests
python -m pytest src/tests/test_pipeline.py -v

# Run with coverage
python -m pytest src/tests/ --cov=src --cov-report=html
```

Expected output:
```
test_pipeline.py::TestDataIngestion::test_tc01_load_and_validate  PASSED
test_pipeline.py::TestDataIngestion::test_tc02_missing_column_raises  PASSED
test_pipeline.py::TestFeatureEngineering::test_tc03_feature_shape  PASSED
test_pipeline.py::TestFeatureEngineering::test_tc04_charges_per_tenure_formula  PASSED
test_pipeline.py::TestModel::test_tc05_prediction_binary  PASSED
test_pipeline.py::TestModel::test_tc06_probability_range  PASSED
test_pipeline.py::TestDriftDetector::test_tc09_drift_report_keys  PASSED
test_pipeline.py::TestLatency::test_tc_latency  PASSED
========== 8 passed in 1.23s ==========
```

---

## MLOps Tools & Ports

### MLflow – Experiment Tracking
Track every training run with metrics, params, and model artifacts.
```python
import mlflow
mlflow.set_experiment("Customer_Churn_Prediction")
with mlflow.start_run():
    mlflow.log_params({"n_estimators": 200, "max_depth": 12})
    mlflow.log_metrics({"f1": 0.525, "roc_auc": 0.665})
    mlflow.sklearn.log_model(model, "model")
```
UI: http://localhost:5000

### DVC – Pipeline & Data Versioning
```bash
dvc repro          # Reproduce the pipeline
dvc dag            # Visualise the DAG
dvc params show    # Show current hyperparameters
dvc metrics show   # Show model metrics
```

### Prometheus + Grafana – Monitoring
- Prometheus scrapes `/metrics` from the API every 15 seconds.
- Grafana displays: prediction rate, latency (p99), error count, drift status.
- Pre-built dashboard auto-loaded at http://localhost:3000.

---

## API Reference

Full documentation available at: http://localhost:8000/docs (Swagger UI)

### Predict a single customer
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 5,
    "monthly_charges": 89.99,
    "total_charges": 450.0,
    "num_products": 2,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic"
  }'
```

Response:
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7821,
  "risk_level": "High",
  "confidence": "Confident",
  "inference_time_ms": 2.31
}
```

### Health check
```bash
curl http://localhost:8000/health
```

### Batch prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"customers": [<customer1>, <customer2>]}'
```
