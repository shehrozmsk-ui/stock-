
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "stock_price_model.joblib"
META_PATH = ROOT / "model_metadata.json"

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction")
st.write("Machine Learning project using historical stock prices.")

# Load trained model and metadata
if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error(
        "Model files are missing. Upload stock_price_model.joblib "
        "and model_metadata.json to the same GitHub repository as app.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)
col1.metric("Algorithm", metadata.get("model_name", "Unknown"))
col2.metric("MAE", f"{metadata.get('mae', 0):.4f}")
col3.metric("RMSE", f"{metadata.get('rmse', 0):.4f}")

st.caption(
    "This model predicts the next observed closing price using "
    "the previous closing prices. Predictions are estimates, not "
    "guaranteed future market prices."
)

st.divider()
st.subheader("🔮 Predict the Next Closing Price")

feature_cols = metadata.get(
    "feature_columns",
    ["close_lag_1", "close_lag_2", "close_lag_3",
     "close_lag_5", "close_lag_10"]
)

if len(feature_cols) != 5:
    st.error("Unexpected model features. Please check model_metadata.json.")
    st.stop()

st.write(
    "Enter the previous closing prices in the same currency and "
    "price scale used by the training dataset."
)

labels = {
    "close_lag_1": "Most recent close (last observed day)",
    "close_lag_2": "Close 2 observations ago",
    "close_lag_3": "Close 3 observations ago",
    "close_lag_5": "Close 5 observations ago",
    "close_lag_10": "Close 10 observations ago"
}

with st.form("prediction_form"):
    values = {}

    for feature in feature_cols:
        values[feature] = st.number_input(
            labels.get(feature, feature),
            min_value=0.000001,
            value=100.0,
            step=1.0,
            format="%.4f"
        )

    submitted = st.form_submit_button(
        "Predict Next Closing Price",
        type="primary"
    )

if submitted:
    try:
        input_df = pd.DataFrame(
            [[values[c] for c in feature_cols]],
            columns=feature_cols
        )

        prediction = float(model.predict(input_df)[0])

        st.success("Prediction completed!")
        st.metric(
            "Predicted Next Closing Price",
            f"{prediction:,.4f}"
        )

        last_close = metadata.get("last_close")
        if last_close is not None:
            st.write(f"Last training close: {float(last_close):,.4f}")

        st.info(
            "For a meaningful prediction, enter actual historical "
            "closing prices from the same company used to train the model."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

with st.expander("About this project"):
    st.write("""
    - Dataset: historical stock prices from Kaggle
    - Task: predict the next observed closing price
    - Models evaluated: Linear Regression and Random Forest
    - Evaluation: MAE, RMSE, and R²
    - Platform: Streamlit
    """)
