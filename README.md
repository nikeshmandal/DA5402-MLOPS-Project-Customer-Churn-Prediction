# ChurnGuard – Customer Churn Prediction with MLOps

IIT Madras | Building an AI Application with MLOps
Student: Nikesh Kumar Mandal | Roll: ID25M805 | Mode: Individual

---

## Table of Contents

1. Quick Start
2. Project Structure
3. Running the Pipeline
4. Running the Application
5. Running Tests
6. MLOps Tools
7. API Reference
8. Project Artifacts
9. AI Disclosure

---


---

## Project Artifacts

| Category      | Artifact                 | Description                            | Status    |
| ------------- | ------------------------ | -------------------------------------- | --------- |
| Data          | data/raw/telco_churn.csv | Raw dataset                            | Completed |
| Data          | data/processed/          | Train/Validation/Test splits           | Completed |
| Pipeline      | data_ingestion.py        | Data loading and validation            | Completed |
| Pipeline      | feature_engineering.py   | Feature processing and transformations | Completed |
| Model         | train.py                 | Model training and MLflow logging      | Completed |
| Orchestration | churn_dag.py             | Airflow DAG definition                 | Completed |
| Versioning    | dvc.yaml                 | Pipeline versioning                    | Completed |
| Versioning    | params.yaml              | Hyperparameter tracking                | Completed |
| Metrics       | metrics.json             | Final evaluation metrics               | Completed |
| API           | src/api/main.py          | FastAPI inference service              | Completed |
| Frontend      | frontend/app.py          | Streamlit interface                    | Completed |
| Monitoring    | prometheus.yml           | Metrics configuration                  | Completed |
| Monitoring    | grafana dashboards       | Visualization dashboards               | Completed |
| Tracking      | MLflow                   | Experiment tracking                    | Completed |
| Testing       | test_pipeline.py         | Unit and integration tests             | Completed |
| Documentation | HLD.md                   | High-level design                      | Completed |
| Documentation | LLD.md                   | Low-level design                       | Completed |
| Documentation | TEST_REPORT.md           | Testing report                         | Completed |
| Deployment    | docker-compose.yml       | Service orchestration                  | Completed |

---

## Quick Start

### Prerequisites

* Docker Desktop ≥ 24.0
* Docker Compose ≥ 2.20
* Git and DVC

### One-Command Launch

```bash
git clone <repo_url>
cd churn_prediction
docker-compose up --build
```

### Access Services

| Service      | URL                        |
| ------------ | -------------------------- |
| Streamlit UI | http://localhost:8501      |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI    | http://localhost:5000      |
| Grafana      | http://localhost:3000      |
| Prometheus   | http://localhost:9090      |
| Airflow      | http://localhost:8080      |

---

## Project Structure

```
churn_prediction/
├── data/
├── src/
├── frontend/
├── docker/
├── grafana/
├── prometheus/
├── docs/
├── config/
├── dvc.yaml
├── params.yaml
├── metrics.json
├── docker-compose.yml
└── README.md
```

---

## Running the Pipeline

### Using DVC

```bash
pip install -r requirements.txt
dvc repro
dvc dag
dvc metrics show
```

### Manual Execution

```bash
export PYTHONPATH=.

python src/pipeline/data_ingestion.py
python src/pipeline/feature_engineering.py
python src/pipeline/train.py
```

---

## Running the Application

### Using Docker

```bash
docker-compose up --build
docker-compose down
```

### Development Mode

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
streamlit run frontend/app.py
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## Running Tests

```bash
python -m pytest src/tests/test_pipeline.py -v
```

---

## MLOps Tools

| Tool       | Purpose                                |
| ---------- | -------------------------------------- |
| MLflow     | Experiment tracking and model registry |
| DVC        | Data and pipeline versioning           |
| Airflow    | Workflow orchestration                 |
| FastAPI    | Model serving                          |
| Streamlit  | User interface                         |
| Prometheus | Metrics collection                     |
| Grafana    | Monitoring dashboards                  |
| Docker     | Containerization                       |

---

## API Reference

### Predict Endpoint

POST /predict

Example request:

```json
{
  "tenure": 5,
  "monthly_charges": 89.99,
  "total_charges": 450.0,
  "num_products": 2,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic"
}
```

Example response:

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7821,
  "risk_level": "High",
  "confidence": "Confident"
}
```

---
## AI Disclosure

This project was developed with assistance from AI tools such as ChatGPT. The role of AI was limited to:

* Improving documentation clarity and formatting
* Assisting in debugging and code refinement
* Providing guidance on best practices

The following points clarify authorship and responsibility:

* All core logic, architecture design, pipeline construction, and model development were completed by me
* All technical decisions, including tool selection and system design, were independently made
* I fully understand and can explain every part of the implementation

This project represents my original work, with AI used strictly as a supporting tool and not as a substitute for knowledge or implementation.

## Final Note

This project demonstrates a complete end-to-end machine learning system with production-oriented MLOps practices, including reproducibility, monitoring, deployment, and testing.

All components have been implemented with a focus on clarity, modularity, and real-world applicability.

---
