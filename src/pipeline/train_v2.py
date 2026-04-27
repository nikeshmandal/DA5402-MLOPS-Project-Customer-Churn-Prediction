"""
Model Training v2 – Hypertuned with GridSearchCV
==================================================
Version 2 of the training pipeline. Improves on v1 by:

  1. GridSearchCV with 5-fold stratified cross-validation on every model
  2. Threshold optimisation — instead of always using 0.5, we find the
     probability threshold that maximises F1 on the validation set
  3. SMOTE oversampling of the minority class (churners) so the model
     sees a balanced training set
  4. Added XGBoost (if installed) and a tuned SVM as new candidates
  5. All runs logged to MLflow with tags marking them as v2
  6. Registered model versions incremented to v2 in MLflow Model Registry

Baseline to beat (v1 best):
  RF_Tuned_Exp2  →  F1=0.5462  ROC-AUC=0.6859

Author: Nikesh Kumar Mandal (ID25M805)
"""

import json
import logging
import os
import pickle
import time
import warnings
from pathlib import Path

import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

IN_DOCKER = os.path.exists("/opt/airflow")

PROCESSED_DIR  = Path("/opt/airflow/data/processed")   if IN_DOCKER else Path("data/processed")
ARTIFACTS_DIR  = Path("/opt/airflow/data/artifacts")   if IN_DOCKER else Path("src/api/artifacts")
VERSION = "v2"

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed — tracking disabled.")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    logger.warning("imbalanced-learn not installed — SMOTE disabled.")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.info("XGBoost not installed — skipping XGB candidate.")

# ── CV strategy ───────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:////opt/airflow/data/mlruns/mlflow.db" if IN_DOCKER else "sqlite:///mlruns/mlflow.db"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
def load_splits():
    X_train = pd.read_csv(PROCESSED_DIR / "train.csv")
    X_val   = pd.read_csv(PROCESSED_DIR / "val.csv")
    X_test  = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_train = X_train.pop("churn")
    y_val   = X_val.pop("churn")
    y_test  = X_test.pop("churn")
    logger.info("Loaded splits — train:%d val:%d test:%d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    """
    SMOTE — Synthetic Minority Over-sampling Technique.

    Why: Our dataset has ~32% churners (minority) and ~68% non-churners
    (majority). Many classifiers learn to predict 'no churn' for everything
    to achieve high accuracy while being useless for the actual task.
    SMOTE generates synthetic examples of churners by interpolating between
    real ones, giving the model a balanced view during training.

    Note: SMOTE is applied ONLY to the training set. Validation and test
    sets are never touched — they must reflect the real class distribution.
    """
    if not SMOTE_AVAILABLE:
        logger.info("SMOTE unavailable — using original imbalanced training data.")
        return X_train, y_train

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(
        "SMOTE applied — before: %d samples (churn=%.1f%%), after: %d samples (churn=%.1f%%)",
        len(y_train), y_train.mean() * 100,
        len(y_res), y_res.mean() * 100
    )
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
def find_best_threshold(clf, X_val, y_val):
    """
    Threshold optimisation.

    By default sklearn uses 0.5 as the decision threshold. But for imbalanced
    data, a lower threshold (e.g. 0.35) catches more churners at the cost of
    more false alarms. We sweep thresholds from 0.20 to 0.60 and pick the
    one that maximises F1 on the validation set.

    This is done AFTER training and ONLY evaluated on the validation set —
    never the test set — to avoid data leakage.
    """
    probs = clf.predict_proba(X_val)[:, 1]
    best_threshold, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.65, 0.05):
        preds = (probs >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = round(t, 2)
    logger.info("Best threshold: %.2f  (val F1=%.4f)", best_threshold, best_f1)
    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
def run_grid_search(name, estimator, param_grid, X_train, y_train):
    """
    Run GridSearchCV to find the best hyperparameter combination.

    GridSearchCV tries every combination of hyperparameters in param_grid
    using 5-fold cross-validation. It fits n_combinations × 5 models in
    total and returns the best one by val F1.

    scoring='f1' means the CV selection criterion is F1, not accuracy —
    important for our imbalanced problem.
    """
    logger.info("GridSearchCV for %s — %d combinations × 5 folds",
                name, np.prod([len(v) for v in param_grid.values()]))
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=CV,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info("%s best params: %s  (CV F1=%.4f)", name, gs.best_params_, gs.best_score_)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ─────────────────────────────────────────────────────────────────────────────
def train_and_log(name, estimator, param_grid, X_train, y_train, X_val, y_val):
    """
    Full train → tune → evaluate → MLflow log cycle for one model.
    """
    t0 = time.time()

    # Grid search
    best_clf, best_params, cv_f1 = run_grid_search(
        name, estimator, param_grid, X_train, y_train
    )

    # Threshold tuning
    threshold = find_best_threshold(best_clf, X_val, y_val)

    # Evaluate with optimised threshold
    probs = best_clf.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)
    metrics = compute_metrics(y_val, preds, probs)
    metrics["threshold"] = threshold
    metrics["cv_f1"]     = round(float(cv_f1), 4)
    metrics["training_time_s"] = round(time.time() - t0, 2)

    logger.info("%s val metrics: %s", name, metrics)

    report = classification_report(y_val, preds, target_names=["No Churn", "Churn"])

    if MLFLOW_AVAILABLE:
        tags = {
            "version": VERSION,
            "model_family": name.split("_")[0],
            "smote": str(SMOTE_AVAILABLE),
            "threshold_tuned": "true",
        }
        with mlflow.start_run(run_name=f"{name}_{VERSION}"):
            mlflow.set_tags(tags)
            mlflow.log_params({**best_params, "threshold": threshold})
            mlflow.log_metrics(metrics)

            with tempfile.TemporaryDirectory() as tmp:
                cm_path = save_confusion_matrix_plot(y_val, preds, tmp)
                mlflow.log_artifact(cm_path, artifact_path="plots")

                roc_path = save_roc_curve_plot(best_clf, X_val, y_val, tmp)
                mlflow.log_artifact(roc_path, artifact_path="plots")

                fi_path = save_feature_importance_plot(best_clf, list(X_val.columns), tmp)
                if fi_path:
                    mlflow.log_artifact(fi_path, artifact_path="plots")

                report_path = os.path.join(tmp, "classification_report.txt")
                with open(report_path, "w") as f:
                    f.write(report)
                mlflow.log_artifact(report_path, artifact_path="reports")

                meta = {
                    "model_name": name, "version": VERSION,
                    "params": {k: str(v) for k, v in best_params.items()},
                    "val_metrics": metrics, "threshold": threshold,
                }
                meta_path = os.path.join(tmp, "model_metadata.json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                mlflow.log_artifact(meta_path, artifact_path="reports")

                pkl_path = os.path.join(tmp, "model.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(best_clf, f)
                mlflow.log_artifact(pkl_path, artifact_path="pickle")

            mlflow.sklearn.log_model(best_clf, artifact_path="model")
            logger.info("MLflow run logged for %s_%s", name, VERSION)

    return {
        "name": name,
        "model": best_clf,
        "threshold": threshold,
        "metrics": metrics,
        "params": best_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
def build_experiment_grid():
    """
    Define all models and their hyperparameter search grids.

    Why these grids:
    ─────────────────
    LogisticRegression:
      C         = regularisation strength (lower = more regularised)
      solver    = 'saga' supports all penalties and is fast on large datasets
      penalty   = 'l1' does feature selection, 'l2' is the standard

    RandomForest:
      n_estimators = more trees = more stable but slower
      max_depth    = controls overfitting; None = fully grown
      min_samples_leaf = prevents tiny leaves that memorise noise
      max_features = 'sqrt' is standard for classification

    GradientBoosting:
      n_estimators   = number of boosting rounds
      learning_rate  = shrinkage — lower = better generalisation but needs more trees
      max_depth      = depth of each tree; shallower = less overfitting
      subsample      = fraction of samples per tree (adds randomness like RF)

    SVM:
      C     = margin penalty; higher = tighter fit
      gamma = kernel width; 'scale' is safe default
      kernel = 'rbf' handles non-linear boundaries well
    """
    experiments = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=42
            ),
            "param_grid": {
                "C":       [0.01, 0.1, 1.0, 10.0],
                "solver":  ["saga"],
                "penalty": ["l1", "l2"],
            },
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "param_grid": {
                "n_estimators":    [100, 200, 300],
                "max_depth":       [6, 10, 15, None],
                "min_samples_leaf":[1, 2, 5],
                "max_features":    ["sqrt", "log2"],
            },
        },
        {
            "name": "GradientBoosting",
            "estimator": GradientBoostingClassifier(random_state=42),
            "param_grid": {
                "n_estimators":  [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth":     [3, 5, 7],
                "subsample":     [0.8, 1.0],
            },
        },
        {
            "name": "SVM",
            "estimator": SVC(
                class_weight="balanced", probability=True, random_state=42
            ),
            "param_grid": {
                "C":      [0.1, 1.0, 10.0],
                "gamma":  ["scale", "auto"],
                "kernel": ["rbf", "poly"],
            },
        },
    ]

    # Add XGBoost if available
    if XGB_AVAILABLE:
        experiments.append({
            "name": "XGBoost",
            "estimator": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            "param_grid": {
                "n_estimators":  [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth":     [3, 5, 7],
                "subsample":     [0.8, 1.0],
                "scale_pos_weight": [2, 3],   # handles class imbalance
            },
        })

    return experiments


# ─────────────────────────────────────────────────────────────────────────────
def select_best(results):
    """Pick model with highest validation F1."""
    best = max(results, key=lambda r: r["metrics"]["f1"])
    logger.info(
        "Best model: %s  (val F1=%.4f, ROC-AUC=%.4f, threshold=%.2f)",
        best["name"], best["metrics"]["f1"],
        best["metrics"]["roc_auc"], best["threshold"]
    )
    return best


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_test(best, X_test, y_test):
    """Final unbiased evaluation on the held-out test set."""
    clf       = best["model"]
    threshold = best["threshold"]
    probs     = clf.predict_proba(X_test)[:, 1]
    preds     = (probs >= threshold).astype(int)
    metrics   = compute_metrics(y_test, preds, probs)
    metrics["threshold"] = threshold

    print("\n" + "="*60)
    print(f"TEST SET RESULTS — {best['name']} (threshold={threshold})")
    print("="*60)
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print(f"\nROC-AUC : {metrics['roc_auc']}")
    print(f"F1      : {metrics['f1']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall  : {metrics['recall']}")
    print("="*60)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def save_best_model(best, test_metrics):
    """Save the best model pickle + updated metadata."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best["model"], f)

    # Save threshold alongside the model so the API can use it
    with open(ARTIFACTS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": best["threshold"]}, f)

    metadata = {
        "model_name":    best["name"],
        "version":       VERSION,
        "params":        {k: str(v) for k, v in best["params"].items()},
        "val_metrics":   best["metrics"],
        "test_metrics":  test_metrics,
        "threshold":     best["threshold"],
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Feature importances (RF / GB / XGB only)
    if hasattr(best["model"], "feature_importances_"):
        with open(ARTIFACTS_DIR / "feature_names.json") as f:
            feature_names = json.load(f)
        importances = dict(zip(
            feature_names,
            best["model"].feature_importances_.tolist()
        ))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        with open(ARTIFACTS_DIR / "feature_importances.json", "w") as f:
            json.dump(importances, f, indent=2)

    logger.info("Best model saved → %s (version=%s)", model_path, VERSION)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Customer_Churn_Prediction")

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # Apply SMOTE to training set only
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Build experiment grid
    experiments = build_experiment_grid()
    logger.info("Running %d models with GridSearchCV...", len(experiments))

    results = []
    for exp in experiments:
        try:
            result = train_and_log(
                name=exp["name"],
                estimator=exp["estimator"],
                param_grid=exp["param_grid"],
                X_train=X_train_res,
                y_train=y_train_res,
                X_val=X_val,
                y_val=y_val,
            )
            results.append(result)
            logger.info("✓ %s done — val F1=%.4f", exp["name"], result["metrics"]["f1"])
        except Exception as e:
            logger.error("✗ %s failed: %s", exp["name"], e)

    # Select and evaluate best
    best = select_best(results)
    test_metrics = evaluate_test(best, X_test, y_test)
    save_best_model(best, test_metrics)

    # Save metrics.json for DVC / CI gate
    with open("metrics.json", "w") as f:
        json.dump({**test_metrics, "model": best["name"], "version": VERSION}, f, indent=2)

    print(f"\n✅ v2 Training complete.")
    print(f"   Best model  : {best['name']}")
    print(f"   Threshold   : {best['threshold']}")
    print(f"   Val F1      : {best['metrics']['f1']}")
    print(f"   Val ROC-AUC : {best['metrics']['roc_auc']}")
    print(f"   Test F1     : {test_metrics['f1']}")
    print(f"   Test ROC-AUC: {test_metrics['roc_auc']}")

    return best


if __name__ == "__main__":
    main()