"""
Airflow DAG – Customer Churn Prediction Pipeline
=================================================
Fix history
-----------
v1: ModuleNotFoundError: No module named 'sklearn'
v2: ModuleNotFoundError: No module named 'dags.train'
v3: ModuleNotFoundError: No module named 'train'
    → actual files are train_v1.py and train_v2.py, not train.py

v4 (this file): imports train_v2 (GridSearchCV hypertuned version)

Author: Nikesh Kumar Mandal (ID25M805)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

AIRFLOW_HOME = Path(os.environ.get("AIRFLOW_HOME", "/opt/airflow"))
DAGS_DIR     = AIRFLOW_HOME / "dags"
DATA_ROOT    = AIRFLOW_HOME / "data"


def _ensure_path():
    for p in [str(AIRFLOW_HOME), str(DAGS_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)


default_args = {
    "owner": "id25m805",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="churn_prediction_pipeline",
    default_args=default_args,
    description="Customer Churn Prediction ML Pipeline",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "churn", "iitm"],
) as dag:

    # ── Task 1: Data Ingestion ─────────────────────────────────────────────────
    def run_ingestion(**ctx):
        _ensure_path()
        import data_ingestion as di
        p = di.DataIngestionPipeline(
            raw_path       = DATA_ROOT / "raw"       / "telco_churn.csv",
            processed_path = DATA_ROOT / "processed",
            baseline_path  = DATA_ROOT / "processed" / "baseline_stats.json",
        )
        r = p.run()
        print("rows:", r["quality_report"]["total_rows"])

    ingest_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_ingestion,
    )

    # ── Task 2: Feature Engineering ───────────────────────────────────────────
    def run_feature_engineering(**ctx):
        _ensure_path()
        import feature_engineering as fe
        fe.VALIDATED_DATA = DATA_ROOT / "processed" / "validated_churn.csv"
        fe.PROCESSED_DIR  = DATA_ROOT / "processed"
        fe.ARTIFACTS_DIR  = DATA_ROOT / "processed"
        fe.FeatureEngineeringPipeline().run()
        print("Feature engineering done.")

    feature_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_feature_engineering,
    )

    # ── Task 3: Model Training (v2 — GridSearchCV hypertuned) ─────────────────
    def run_training(**ctx):
        _ensure_path()
        import train_v2 as tr                             # ← train_v2.py
        tr.PROCESSED_DIR = DATA_ROOT / "processed"
        tr.ARTIFACTS_DIR = DATA_ROOT / "processed"
        best = tr.main()
        print("Best model :", best["name"])
        print("Val F1     :", best["metrics"]["f1"])
        print("Val ROC-AUC:", best["metrics"]["roc_auc"])

    train_task = PythonOperator(
        task_id="model_training",
        python_callable=run_training,
    )

    # ── Task 4: API Health Check ───────────────────────────────────────────────
    health_task = BashOperator(
        task_id="api_health_check",
        bash_command=(
            "curl -sf http://api:8000/ready "
            "&& echo 'API ready' "
            "|| echo 'API not ready — restart after model update'"
        ),
    )

    ingest_task >> feature_task >> train_task >> health_task