# Low-Level Design (LLD) Document
## Customer Churn Prediction – API Endpoint Definitions
**Author:** Nikesh Kumar Mandal (ID25M805) | IIT Madras
**Version:** 1.0 | April 2026

---

## 1. API Base URL
```
http://localhost:8000
```
When running via Docker Compose, the frontend uses: `http://api:8000`

---

## 2. Endpoint Definitions

---

### 2.1 GET /health
**Purpose:** Liveness probe — confirms the API process is alive.

**Request:**
```http
GET /health HTTP/1.1
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "service": "churn-prediction-api"
}
```

**Error Responses:** None (always 200 if process is alive)

---

### 2.2 GET /ready
**Purpose:** Readiness probe — confirms model is loaded and ready for inference.

**Request:**
```http
GET /ready HTTP/1.1
```

**Response (200 OK):**
```json
{
  "status": "ready",
  "model": "LogisticRegression"
}
```

**Response (503 Service Unavailable):**
```json
{
  "detail": "Model not loaded. Run training first."
}
```

---

### 2.3 POST /predict
**Purpose:** Predict churn for a single customer.

**Request:**
```http
POST /predict HTTP/1.1
Content-Type: application/json
```

**Request Body Schema (CustomerInput):**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| tenure | float | ✅ | ge=0, le=120 | Months as customer |
| monthly_charges | float | ✅ | ge=0 | Monthly bill (USD) |
| total_charges | float | ✅ | ge=0 | Lifetime spend (USD) |
| num_products | int | ✅ | ge=1, le=10 | Number of products subscribed |
| contract_type | string | ✅ | — | Month-to-month / One year / Two year |
| payment_method | string | ✅ | — | Electronic check / Mailed check / Bank transfer / Credit card |
| internet_service | string | ✅ | — | DSL / Fiber optic / No |
| online_security | string | ❌ | default="No" | Yes / No / No internet service |
| tech_support | string | ❌ | default="No" | Yes / No / No internet service |
| paperless_billing | string | ❌ | default="No" | Yes / No |
| senior_citizen | int | ❌ | 0 or 1, default=0 | 1=senior |
| gender | string | ❌ | default="Male" | Male / Female |
| partner | string | ❌ | default="No" | Yes / No |
| dependents | string | ❌ | default="No" | Yes / No |
| phone_service | string | ❌ | default="Yes" | Yes / No |
| multiple_lines | string | ❌ | default="No" | Yes / No / No phone service |
| streaming_tv | string | ❌ | default="No" | Yes / No / No internet service |
| streaming_movies | string | ❌ | default="No" | Yes / No / No internet service |
| device_protection | string | ❌ | default="No" | Yes / No / No internet service |
| online_backup | string | ❌ | default="No" | Yes / No / No internet service |

**Example Request Body:**
```json
{
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
  "online_backup": "No"
}
```

**Response (200 OK) – PredictionResponse:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7821,
  "risk_level": "High",
  "confidence": "Confident",
  "inference_time_ms": 2.31
}
```

| Field | Type | Values | Description |
|---|---|---|---|
| churn_prediction | int | 0 or 1 | 0=stays, 1=churns |
| churn_probability | float | 0.0–1.0 | Model's confidence |
| risk_level | string | Low/Medium/High | <40% Low, 40-70% Medium, >70% High |
| confidence | string | Confident/Moderate | prob ≥0.70 or ≤0.30 = Confident |
| inference_time_ms | float | ≥0 | End-to-end latency in ms |

**Error Responses:**
- `422 Unprocessable Entity` – Invalid input (e.g. tenure=-1)
- `503 Service Unavailable` – Model not loaded
- `500 Internal Server Error` – Unexpected inference error

---

### 2.4 POST /predict/batch
**Purpose:** Predict churn for multiple customers in one request.

**Request Body:**
```json
{
  "customers": [ <CustomerInput>, <CustomerInput>, ... ]
}
```
Maximum 1000 customers per request.

**Response (200 OK) – BatchResponse:**
```json
{
  "predictions": [ <PredictionResponse>, ... ],
  "total_customers": 100,
  "churn_count": 23,
  "churn_rate": 0.23
}
```

**Error Responses:**
- `400 Bad Request` – Batch size > 1000
- `503 Service Unavailable` – Model not loaded

---

### 2.5 GET /model/info
**Purpose:** Returns model metadata.

**Response (200 OK):**
```json
{
  "model_name": "LogisticRegression",
  "params": { "C": 1.0, "class_weight": "balanced" },
  "val_metrics": {
    "accuracy": 0.62,
    "f1": 0.5226,
    "precision": 0.4394,
    "recall": 0.6446,
    "roc_auc": 0.6705,
    "training_time_s": 0.01
  }
}
```

---

### 2.6 GET /model/features
**Purpose:** Returns feature importances sorted descending.

**Response (200 OK):**
```json
{
  "contract_type": 0.182,
  "tenure": 0.141,
  "monthly_charges": 0.128,
  "charges_per_tenure": 0.097,
  ...
}
```

---

### 2.7 GET /monitoring/drift
**Purpose:** Returns drift detection report over last 500 inference inputs.

**Response (200 OK) – drift present:**
```json
{
  "drift_detected": true,
  "features": {
    "tenure": { "ks_statistic": 0.21, "p_value": 0.003, "drift_detected": true },
    "monthly_charges": { "ks_statistic": 0.08, "p_value": 0.24, "drift_detected": false }
  }
}
```

**Response (200 OK) – insufficient data:**
```json
{
  "message": "Not enough data for drift analysis (need ≥30 predictions).",
  "count": 12
}
```

---

### 2.8 GET /metrics
**Purpose:** Prometheus metrics scrape endpoint.

**Response:** Plain text in Prometheus exposition format.
```
# HELP churn_predictions_total Total predictions made
# TYPE churn_predictions_total counter
churn_predictions_total{result="churn"} 142.0
churn_predictions_total{result="no_churn"} 893.0
...
```

---

## 3. Class-Level Design

### DataIngestionPipeline
```
+ load() → DataFrame
+ _validate_schema() → None
+ validate_quality() → dict
+ compute_baseline_stats() → dict
+ save_validated() → Path
+ run() → dict
```

### FeatureEngineeringPipeline
```
+ load() → DataFrame
+ engineer_features(df) → DataFrame
+ encode_categoricals(df, fit) → DataFrame
+ scale_numerics(df, fit) → DataFrame
+ split(df) → (X_train, X_val, X_test, y_train, y_val, y_test)
+ save_artifacts() → None
+ save_splits(...) → None
+ run() → tuple
```

### DriftDetector
```
+ _load_baseline() → None
+ detect(live_data: dict) → dict
```

### FastAPI App (functional style)
```
GET  /health          → health()
GET  /ready           → ready()
GET  /model/info      → model_info()
GET  /model/features  → feature_importance()
POST /predict         → predict(CustomerInput) → PredictionResponse
POST /predict/batch   → predict_batch(BatchInput) → BatchResponse
GET  /monitoring/drift → drift_report()
GET  /metrics         → metrics()
```
