import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import timedelta

# --- CONFIG ---
st.set_page_config(page_title="NeerDrishti - Flood Forecast", layout="centered", page_icon="🌊")

st.markdown("""
<style>
body {
    background-color: #121212;
    color: #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

st.title("🌊 NeerDrishti - Flood Risk Forecast")
st.markdown("Predicting Floods, Protecting Lives in Odisha")

# --- LOAD MODEL AND ARTIFACTS ---
@st.cache_resource
def load_resources():
    model = load_model("mahanadi_bilstm_model.h5")  # Use the binary classification model
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, scaler, encoders

model, scaler, label_encoders = load_resources()

# --- USE DISTRICTS FROM TRAINING ---
all_districts = sorted(label_encoders["District"].classes_)
district_input = st.selectbox("Select Your District", options=all_districts)
today = pd.to_datetime("today").normalize()

# --- LOAD AND PREPARE DATA FOR SELECTED DISTRICT ---
@st.cache_data
def simulate_district_data(district):
    df = pd.read_csv("C:/Users/LENOVO/Desktop/NeerDrishti/data/mahanadi_flood_data_2001_2024.csv", parse_dates=["Date"])
    df = df[df["District"] == district]
    df = df.sort_values("Date").copy()
    df["Rain_7day_sum"] = df["Rainfall (mm)"].rolling(7).sum().fillna(0)
    df["Water_vs_Elevation"] = df["Water Level (m)"] / (df["Elevation (m)"] + 1)
    df = df.dropna()
    return df

district_df = simulate_district_data(district_input)

if len(district_df) < 30:
    st.error("Not enough recent data to make a prediction for this district.")
    st.stop()

# --- PREPARE INPUT SEQUENCE ---
features = ['District', 'Latitude', 'Longitude', 'Rainfall (mm)', 'Temperature (°C)', 
            'Humidity (%)', 'River Discharge (m³/s)', 'Water Level (m)', 
            'Rain_7day_sum', 'Water_vs_Elevation']

encoded_district = label_encoders["District"].transform([district_input])[0]
district_df["District"] = encoded_district

X_input = district_df[features].tail(30).values
X_input = X_input.reshape(1, 30, len(features))
X_input = scaler.transform(X_input.reshape(-1, len(features))).reshape(1, 30, len(features))

# --- MAKE PREDICTION ---
pred = model.predict(X_input)[0]
predicted_class = np.argmax(pred)
confidence = pred[predicted_class] * 100

# Binary risk labels
risk_label = ["🟢 No Flood", "🔴 Flood"][predicted_class]

# --- DISPLAY OUTPUT ---
st.subheader(f"📍 District: {district_input}")
st.metric("🧠 Predicted Risk Level", risk_label, f"{confidence:.2f}% confidence")
st.markdown("🗓️ **Forecast Date**: " + today.strftime("%d %B %Y"))

if predicted_class == 1:
    st.error("🔴 Flood Risk: Please stay alert and follow local authority advisories.")
else:
    st.success("🟢 No Flood: No immediate flood threat.")

st.markdown("---")
st.markdown("Predict. Prepare. Protect. | NeerDrishti 2025")
