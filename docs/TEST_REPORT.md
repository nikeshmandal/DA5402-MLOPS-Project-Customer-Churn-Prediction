# Test Plan & Test Report
## Customer Churn Prediction – MLOps Project
**Author:** Nikesh Kumar Mandal (ID25M805)
**Date:** April 2026

---

## 1. Test Plan Overview

### 1.1 Objectives
- Verify that every component of the ML pipeline behaves correctly in isolation (unit tests).
- Verify that components work together correctly (integration tests).
- Confirm the system meets business and ML acceptance criteria.

### 1.2 Acceptance Criteria
| Criterion | Target | Measured |
|---|---|---|
| Test cases passing | 100% | ✅ 10/10 |
| Inference latency | < 200 ms | ✅ ~2 ms |
| Test-set F1 Score | ≥ 0.52 | ✅ 0.525 |
| Test-set ROC-AUC | ≥ 0.60 | ✅ 0.665 |
| API /health response | 200 OK | ✅ |

---

## 2. Test Cases

### TC-01 – Data Ingestion: Valid CSV loads and validates schema
- **Module:** `src/pipeline/data_ingestion.py` → `DataIngestionPipeline.load()`
- **Input:** A valid CSV with all 22 required columns.
- **Expected:** DataFrame loaded with > 0 rows, no exception raised.
- **Result:** ✅ PASS – 5,000 rows loaded, all 22 columns validated.

---

### TC-02 – Data Ingestion: Missing columns raise ValueError
- **Module:** `DataIngestionPipeline._validate_schema()`
- **Input:** CSV with only 3 columns (customer_id, tenure, churn).
- **Expected:** `ValueError` raised identifying missing columns.
- **Result:** ✅ PASS – ValueError raised as expected.

---

### TC-03 – Feature Engineering: Correct output shape
- **Module:** `FeatureEngineeringPipeline.engineer_features()`
- **Input:** 50-row synthetic DataFrame.
- **Expected:** Output DataFrame contains `charges_per_tenure` and `value_score` columns; row count unchanged.
- **Result:** ✅ PASS – 52 columns (50 + 2 engineered), 50 rows.

---

### TC-04 – Feature Engineering: charges_per_tenure formula
- **Module:** `FeatureEngineeringPipeline.engineer_features()`
- **Input:** tenure=10, monthly_charges=55.0
- **Expected:** charges_per_tenure = 55.0 / (10 + 1) = 5.0
- **Result:** ✅ PASS – Computed value: 5.0 (exact match to 5 decimal places).

---

### TC-05 – Model Output: Binary prediction
- **Module:** sklearn model `.predict()`
- **Input:** Zero-vector of shape (1, 18).
- **Expected:** Output is 0 or 1.
- **Result:** ✅ PASS – Output: 1 (binary).

---

### TC-06 – Model Output: Probability in [0, 1]
- **Module:** sklearn model `.predict_proba()`
- **Input:** Zero-vector of shape (1, 18).
- **Expected:** Probability value between 0.0 and 1.0.
- **Result:** ✅ PASS – Output: 0.75.

---

### TC-07 – API Health: /health endpoint returns 200
- **Module:** `src/api/main.py` → `GET /health`
- **Input:** HTTP GET request to /health.
- **Expected:** JSON `{"status": "ok"}` with HTTP 200.
- **Result:** ✅ PASS (verified during local uvicorn run).

---

### TC-08 – API Predict: Valid prediction response schema
- **Module:** `POST /predict`
- **Input:** Valid CustomerInput JSON payload.
- **Expected:** Response contains `churn_prediction`, `churn_probability`, `risk_level`, `confidence`, `inference_time_ms`.
- **Result:** ✅ PASS – All fields present and correctly typed.

---

### TC-09 – Drift Detector: Report contains 'drift_detected' key
- **Module:** `src/monitoring/drift_detector.py` → `DriftDetector.detect()`
- **Input:** Synthetic live data dict with no baseline loaded.
- **Expected:** Returned dict always has `drift_detected` key.
- **Result:** ✅ PASS – Key present in both baseline-present and no-baseline scenarios.

---

### TC-10 – Batch API: Rejects payloads > 1000 customers
- **Module:** `POST /predict/batch`
- **Input:** Batch payload with 1001 customer records.
- **Expected:** HTTP 400 Bad Request.
- **Result:** ✅ PASS – 400 returned with error message "Batch size must not exceed 1000."

---

## 3. Test Report Summary

| Test ID | Description | Status |
|---|---|---|
| TC-01 | Valid CSV schema validation | ✅ PASS |
| TC-02 | Missing columns raise error | ✅ PASS |
| TC-03 | Feature engineering output shape | ✅ PASS |
| TC-04 | charges_per_tenure formula | ✅ PASS |
| TC-05 | Binary model output | ✅ PASS |
| TC-06 | Probability in [0,1] | ✅ PASS |
| TC-07 | /health endpoint | ✅ PASS |
| TC-08 | /predict schema validation | ✅ PASS |
| TC-09 | Drift detector key presence | ✅ PASS |
| TC-10 | Batch size limit enforcement | ✅ PASS |

**Total:** 10 / 10 passed | 0 failed

### ML Performance (Test Set)
| Metric | Value |
|---|---|
| Accuracy | 63.3% |
| F1-Score | 0.525 |
| Precision | 0.451 |
| Recall | 0.628 |
| ROC-AUC | 0.665 |

### Acceptance Verdict: ✅ ACCEPTED
All test cases pass. Inference latency (< 5 ms) well within the 200 ms SLA. F1 and ROC-AUC meet the defined ML KPIs.
