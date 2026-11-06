import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model and preprocessing objects

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_detection_model.pkl"
)

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
scaler = saved_data["scaler"]
feature_names = saved_data["feature_names"]

st.success("Fraud Detection Model Loaded Successfully!")

#page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("💳 Welcome to Credit Card Fraud Detection!")
st.subheader("🔐 Smart Detection. Safer Transactions.")
# Sidebar
st.sidebar.header("💳 Transaction Information")

# Input fields
amt = st.sidebar.number_input(
    "Transaction Amount ($)",
    min_value=1.0,
    max_value=30000.0,
    value=100.0
)

category = st.sidebar.selectbox(
    "Transaction Category",
    [
        "misc_net", "grocery_pos", "entertainment",
        "gas_transport", "misc_pos", "grocery_net",
        "shopping_net", "shopping_pos", "food_dining",
        "personal_care", "health_fitness", "travel",
        "kids_pets", "home"
    ]
)

lat = st.sidebar.number_input(
    "Customer Latitude",
    value=39.7189
)

long = st.sidebar.number_input(
    "Customer Longitude",
    value=-110.6223
)

city_pop = st.sidebar.number_input(
    "City Population",
    min_value=1,
    value=50000
)

merch_lat = st.sidebar.number_input(
    "Merchant Latitude",
    value=39.7200
)

merch_long = st.sidebar.number_input(
    "Merchant Longitude",
    value=-110.6200
)

trans_hour = st.sidebar.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12
)

trans_day = st.sidebar.slider(
    "Transaction Day",
    min_value=1,
    max_value=31,
    value=15
)

trans_month = st.sidebar.selectbox(
    "Transaction Month",
    list(range(1, 13)),
    index=5
)
# Transaction Details
st.subheader("🔍 Transaction Details")

input_values = {
    "amt": amt,
    "lat": lat,
    "long": long,
    "city_pop": city_pop,
    "merch_lat": merch_lat,
    "merch_long": merch_long,
    "trans_hour": trans_hour,
    "trans_day": trans_day,
    "trans_month": trans_month,
    "category": category
}

input_df = pd.DataFrame([input_values])

st.dataframe(input_df, use_container_width=True)

# Fraud Prediction

if st.button("🔍 Predict Transaction", type="primary"):

    # Convert category into dummy variables
    prediction_df = pd.get_dummies(
        input_df,
        columns=["category"],
        dtype=int
    )

    # Match the exact training feature columns and order
    prediction_df = prediction_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Apply the saved scaler
    scaled_input = scaler.transform(prediction_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("🚨 Potential Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction Classified as Legitimate.")