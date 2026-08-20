"""
Step 11: Streamlit Frontend
Neural Network from Scratch + Fraud Detection

A simple demo UI that calls the FastAPI backend's /predict and /model-info
endpoints. Run the FastAPI server first (uvicorn app:app --reload), then
run this in a separate terminal.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🔍", layout="centered")

st.title("🔍 Real-Time Fraud Detection")
st.caption(
    "A neural network implemented from scratch in NumPy, rebuilt and tuned in PyTorch "
    "(RMSprop + L2 regularization + SMOTE oversampling), served live via FastAPI."
)

# ---- Model info panel ----
with st.expander("ℹ️ Model details", expanded=False):
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=5).json()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Architecture", info["architecture"])
            st.metric("Optimizer", info["optimizer"])
        with col2:
            st.metric("Imbalance strategy", info["imbalance_strategy"])
            st.metric("Decision threshold", info["decision_threshold"])

        st.write("**Held-out test set performance:**")
        m = info["test_metrics"]
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Precision", f"{m['precision']:.3f}")
        mcol2.metric("Recall", f"{m['recall']:.3f}")
        mcol3.metric("F1", f"{m['f1']:.3f}")
        mcol4.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the API. Make sure `uvicorn app:app --reload` is running.")

st.divider()

# ---- Transaction input form ----
st.subheader("Score a transaction")

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Amount ($)", min_value=0.0, value=245.50, step=10.0)
        transaction_hour = st.slider("Transaction hour (0-23)", 0, 23, 3)
        merchant_category = st.selectbox(
            "Merchant category",
            ["Clothing", "Electronics", "Food", "Grocery", "Travel"],
            index=1,
        )
        cardholder_age = st.number_input("Cardholder age", min_value=18, max_value=120, value=34)

    with col2:
        foreign_transaction = st.radio("Foreign transaction?", ["No", "Yes"], index=1, horizontal=True)
        location_mismatch = st.radio("Location mismatch?", ["No", "Yes"], index=1, horizontal=True)
        device_trust_score = st.slider("Device trust score (0-100)", 0, 100, 28)
        velocity_last_24h = st.number_input("Transactions in last 24h", min_value=0, value=6)

    submitted = st.form_submit_button("Check for fraud", use_container_width=True)

if submitted:
    payload = {
        "amount": amount,
        "transaction_hour": transaction_hour,
        "merchant_category": merchant_category,
        "foreign_transaction": 1 if foreign_transaction == "Yes" else 0,
        "location_mismatch": 1 if location_mismatch == "Yes" else 0,
        "device_trust_score": device_trust_score,
        "velocity_last_24h": velocity_last_24h,
        "cardholder_age": cardholder_age,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        probability = result["fraud_probability"]
        flagged = result["flagged_as_fraud"]

        st.divider()
        if flagged:
            st.error(f"### 🚨 Flagged as FRAUD — {probability:.1%} probability")
        else:
            st.success(f"### ✅ Looks legitimate — {probability:.1%} fraud probability")

        st.progress(min(probability, 1.0))
        st.caption(f"Decision threshold: {result['decision_threshold']}")

    except requests.exceptions.ConnectionError:
        st.error("Can't reach the API. Make sure `uvicorn app:app --reload` is running on port 8000.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}")
        