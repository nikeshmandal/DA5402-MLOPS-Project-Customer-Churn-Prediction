"""
Customer Churn Prediction – Streamlit Frontend
===============================================
A non-technical-user-friendly web interface for:
  1. Predicting churn for a single customer
  2. Uploading a CSV for batch predictions
  3. Viewing model information and feature importances
  4. Viewing the monitoring/drift dashboard

Author: Nikesh Kumar Mandal (ID25M805)

To run:
    streamlit run frontend/app.py
"""

import json
import time
from typing import Optional
import requests
import pandas as pd
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://api:8000"  # Change to docker service name when in compose

st.set_page_config(
    page_title="ChurnGuard – Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/crystal-ball.png", width=80)
st.sidebar.title("ChurnGuard")
st.sidebar.caption("Customer Churn Prediction System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔮 Single Prediction", "📊 Batch Prediction", "🤖 Model Info", "📡 Monitoring", "📖 User Manual"],
)
st.sidebar.markdown("---")
st.sidebar.info("**IIT Madras | MLOps Project**\nNikesh Kumar Mandal\nID25M805")


# ── Helper ────────────────────────────────────────────────────────────────────
def call_api(method: str, endpoint: str, payload: Optional[dict]= None ):
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=10)
        else:
            r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to backend API. Make sure Docker services are running."
    except requests.exceptions.HTTPError as e:
        return None, f"❌ API Error {r.status_code}: {r.text}"
    except Exception as e:
        return None, f"❌ Unexpected error: {e}"


def risk_color(risk: str) -> str:
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk, "⚪")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Single Prediction
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Single Prediction":
    st.title("🔮 Customer Churn Prediction")
    st.markdown("Fill in the customer's details below and click **Predict** to get an instant churn risk assessment.")
    st.markdown("---")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📋 Account Details")
            tenure = st.slider("Tenure (months)", 0, 120, 12,
                               help="How many months the customer has been with the company.")
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 75.0, step=1.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 900.0, step=10.0)
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4, 5], index=1)
            contract_type = st.selectbox("Contract Type",
                                         ["Month-to-month", "One year", "Two year"],
                                         help="Month-to-month contracts have higher churn risk.")
            payment_method = st.selectbox("Payment Method",
                                          ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

        with col2:
            st.subheader("🌐 Services")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            st.subheader("👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_products": num_products,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "online_security": online_security,
            "tech_support": tech_support,
            "paperless_billing": paperless_billing,
            "senior_citizen": 1 if senior_citizen == "Yes" else 0,
            "gender": gender,
            "partner": partner,
            "dependents": dependents,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "device_protection": device_protection,
            "online_backup": online_backup,
        }

        with st.spinner("Analyzing customer profile…"):
            result, err = call_api("POST", "/predict", payload)

        if err:
            st.error(err)
        else:
            st.markdown("---")
            st.subheader("📊 Prediction Result")

            c1, c2, c3, c4 = st.columns(4)
            verdict = "⚠️ WILL CHURN" if result["churn_prediction"] == 1 else "✅ WILL NOT CHURN"
            c1.metric("Prediction", verdict)
            c2.metric("Churn Probability", f"{result['churn_probability']*100:.1f}%")
            c3.metric("Risk Level", f"{risk_color(result['risk_level'])} {result['risk_level']}")
            c4.metric("Inference Time", f"{result['inference_time_ms']} ms")

            prob = result["churn_probability"]
            bar_color = "🔴" if prob >= 0.7 else "🟡" if prob >= 0.4 else "🟢"
            st.progress(prob, text=f"{bar_color} Churn Risk: {prob*100:.1f}%")

            if result["churn_prediction"] == 1:
                st.warning("**Recommended Action:** This customer is at risk. Consider offering a loyalty discount, extended contract, or personalized support call.")
            else:
                st.success("**Customer is likely to stay.** Continue standard engagement practices.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Batch Prediction":
    st.title("📊 Batch Churn Prediction")
    st.markdown("Upload a CSV file with multiple customers to get predictions for all of them at once.")

    with st.expander("📋 Required CSV Columns"):
        st.code(
            "tenure, monthly_charges, total_charges, num_products, contract_type, "
            "payment_method, internet_service, online_security, tech_support, "
            "paperless_billing, senior_citizen, gender, partner, dependents, "
            "phone_service, multiple_lines, streaming_tv, streaming_movies, "
            "device_protection, online_backup"
        )

    uploaded = st.file_uploader("Upload Customer CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**Loaded {len(df)} customers.** Preview:")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            payload = {"customers": df.to_dict(orient="records")}
            with st.spinner(f"Predicting for {len(df)} customers…"):
                result, err = call_api("POST", "/predict/batch", payload)

            if err:
                st.error(err)
            else:
                preds = result["predictions"]
                df["churn_prediction"] = [p["churn_prediction"] for p in preds]
                df["churn_probability"] = [p["churn_probability"] for p in preds]
                df["risk_level"] = [p["risk_level"] for p in preds]

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Customers", result["total_customers"])
                c2.metric("Predicted to Churn", result["churn_count"])
                c3.metric("Churn Rate", f"{result['churn_rate']*100:.1f}%")

                st.dataframe(df, use_container_width=True)

                csv_out = df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Results", csv_out, "churn_predictions.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Info
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Info":
    st.title("🤖 Model Information")

    info, err = call_api("GET", "/model/info")
    if err:
        st.error(err)
    else:
        st.subheader("Model Metadata")
        st.json(info)

    st.markdown("---")
    st.subheader("Feature Importances")
    feat, err2 = call_api("GET", "/model/features")
    if err2:
        st.warning(err2)
    else:
        feat_df = pd.DataFrame(list(feat.items()), columns=["Feature", "Importance"])
        feat_df = feat_df.sort_values("Importance", ascending=True).tail(15)
        st.bar_chart(feat_df.set_index("Feature"))


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Monitoring
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Monitoring":
    st.title("📡 System Monitoring")
    st.markdown("Live metrics and drift detection status.")

    col1, col2 = st.columns(2)
    with col1:
        health, _ = call_api("GET", "/health")
        ready, _ = call_api("GET", "/ready")
        st.subheader("Service Health")
        if health:
            st.success(f"🟢 API is **alive** — {health['status']}")
        else:
            st.error("🔴 API is **unreachable**")
        if ready:
            st.success(f"🟢 Model is **ready** — {ready.get('model', 'unknown')}")
        else:
            st.warning("⚠️ Model is **not ready** (training might be needed)")

    with col2:
        st.subheader("Data Drift Report")
        drift, err = call_api("GET", "/monitoring/drift")
        if err:
            st.warning(err)
        elif drift:
            if drift.get("drift_detected"):
                st.error("🚨 Data drift detected! Model retraining recommended.")
            else:
                st.success("✅ No significant drift detected.")
            if "features" in drift:
                for feat, info in drift["features"].items():
                    icon = "🔴" if info["drift_detected"] else "🟢"
                    st.write(f"{icon} **{feat}** — KS stat: {info['ks_statistic']}, p-value: {info['p_value']}")

    st.markdown("---")
    st.subheader("External Dashboards")
    st.info("🔗 **Grafana Dashboard:** http://localhost:3005 (admin / admin)\n\n🔗 **Prometheus:** http://localhost:9090\n\n🔗 **MLflow Tracking UI:** http://localhost:5005\n\n🔗 **Airflow UI:** http://localhost:8081")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: User Manual
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📖 User Manual":
    st.title("📖 User Manual")
    st.markdown("""
## What is ChurnGuard?

**ChurnGuard** is an AI-powered tool that predicts whether a telecom customer is likely to cancel their subscription (**churn**).  
It analyses customer profile data — such as contract type, monthly bill, and service usage — and returns an instant risk score.

---

## How to use Single Prediction

1. Navigate to **🔮 Single Prediction** from the left sidebar.
2. Fill in the customer's details across three sections:
   - **Account Details** — Tenure, charges, contract type, payment.
   - **Services** — Internet, security, tech support etc.
   - **Demographics** — Gender, age, family status.
3. Click **🔮 Predict Churn**.
4. The result shows:
   - ✅ **WILL NOT CHURN** or ⚠️ **WILL CHURN**
   - Probability score (0–100%)
   - Risk level: 🟢 Low / 🟡 Medium / 🔴 High

---

## How to use Batch Prediction

1. Prepare a CSV file with the required columns (see the list in that page).
2. Navigate to **📊 Batch Prediction**.
3. Upload your CSV file.
4. Click **🚀 Run Batch Prediction**.
5. Download the results CSV with churn predictions appended.

---

## Understanding the Results

| Metric | Meaning |
|---|---|
| Churn Prediction | 0 = stays, 1 = leaves |
| Churn Probability | How confident the model is (higher = more likely to churn) |
| Risk Level | Low (<40%), Medium (40–70%), High (>70%) |

---

## What to do with High-Risk Customers?

- Offer a **loyalty discount** or upgraded plan.
- Assign a **personal account manager**.
- Send a **proactive satisfaction survey**.
- Offer a **long-term contract incentive**.

---

## Troubleshooting

- **"Cannot connect to API"** — Ensure Docker containers are running: `docker-compose up`
- **"Model not ready"** — Run the training pipeline: `python src/pipeline/train.py`
- For support, contact: id25m805@smail.iitm.ac.in
""")
