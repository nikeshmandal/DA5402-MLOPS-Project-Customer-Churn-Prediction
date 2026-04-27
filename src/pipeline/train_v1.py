import json
import logging
import os
import pickle
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

IN_DOCKER = os.path.exists("/opt/airflow")

PROCESSED_DIR  = Path("/opt/airflow/data/processed")   if IN_DOCKER else Path("data/processed")
ARTIFACTS_DIR  = Path("/opt/airflow/data/artifacts")   if IN_DOCKER else Path("src/api/artifacts")
MLFLOW_TRACKING_URI = "sqlite:////opt/airflow/data/mlruns/mlflow.db" if IN_DOCKER else "sqlite:///mlruns/mlflow.db"

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Customer_Churn_Prediction")


def load_splits():
    X_train = pd.read_csv(PROCESSED_DIR / "train.csv")
    X_val   = pd.read_csv(PROCESSED_DIR / "val.csv")
    X_test  = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_train = X_train.pop("churn")
    y_val   = X_val.pop("churn")
    y_test  = X_test.pop("churn")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall":    round(float(recall_score(y_true, y_pred)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def save_confusion_matrix_plot(y_true, y_pred, run_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No Churn", "Churn"],
        colorbar=False, ax=ax
    )
    ax.set_title("Confusion Matrix")
    path = os.path.join(run_dir, "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_roc_curve_plot(clf, X_val, y_val, run_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(clf, X_val, y_val, ax=ax)
    ax.set_title("ROC Curve")
    path = os.path.join(run_dir, "roc_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_feature_importance_plot(clf, feature_names, run_dir):
    if not hasattr(clf, "feature_importances_"):
        return None
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(indices)), importances[indices])
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Top 15 Feature Importances")
    ax.set_ylabel("Importance")
    path = os.path.join(run_dir, "feature_importances.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def train_model(name, clf, X_train, y_train, X_val, y_val, params):
    logger.info("Training %s...", name)
    start = time.time()
    clf.fit(X_train, y_train)
    duration = round(time.time() - start, 2)

    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    metrics = compute_metrics(y_val, y_pred, y_prob)
    metrics["training_time_s"] = duration

    report = classification_report(y_val, y_pred, target_names=["No Churn", "Churn"])

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("version", "v1")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            with tempfile.TemporaryDirectory() as tmp:
                cm_path = save_confusion_matrix_plot(y_val, y_pred, tmp)
                mlflow.log_artifact(cm_path, artifact_path="plots")

                roc_path = save_roc_curve_plot(clf, X_val, y_val, tmp)
                mlflow.log_artifact(roc_path, artifact_path="plots")

                fi_path = save_feature_importance_plot(clf, list(X_train.columns), tmp)
                if fi_path:
                    mlflow.log_artifact(fi_path, artifact_path="plots")

                report_path = os.path.join(tmp, "classification_report.txt")
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path, artifact_path="reports")

                meta = {"model_name": name, "params": params, "val_metrics": metrics}
                meta_path = os.path.join(tmp, "model_metadata.json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                mlflow.log_artifact(meta_path, artifact_path="reports")

                pkl_path = os.path.join(tmp, "model.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(clf, f)
                mlflow.log_artifact(pkl_path, artifact_path="pickle")

            mlflow.sklearn.log_model(clf, artifact_path="model")

    return {"name": name, "model": clf, "metrics": metrics, "params": params}


def select_best(results):
    return max(results, key=lambda r: r["metrics"]["f1"])


def evaluate_test(best, X_test, y_test):
    clf = best["model"]
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(confusion_matrix(y_test, y_pred))
    return metrics


def save_best_model(best):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    metadata = {
        "model_name": best["name"],
        "params": best["params"],
        "val_metrics": best["metrics"],
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    experiments = [
        {
            "name": "LogisticRegression",
            "clf": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params": {"model": "logistic", "class_weight": "balanced"},
        },
        {
            "name": "RandomForest_100",
            "clf": RandomForestClassifier(n_estimators=100, random_state=42),
            "params": {"n_estimators": 100},
        },
        {
            "name": "GradientBoosting",
            "clf": GradientBoostingClassifier(random_state=42),
            "params": {"model": "gb", "n_estimators": 100},
        },
    ]

    results = []
    for exp in experiments:
        results.append(
            train_model(
                exp["name"], exp["clf"],
                X_train, y_train, X_val, y_val,
                exp["params"],
            )
        )

    best = select_best(results)
    evaluate_test(best, X_test, y_test)
    save_best_model(best)
    logger.info("Best model: %s (val F1=%.4f)", best["name"], best["metrics"]["f1"])


if __name__ == "__main__":
    main()
