import streamlit as st
import pickle
import pandas as pd

# --- Load trained model ---
with open("classifier.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="F1 DNF Predictor", page_icon="🏎️", layout="centered")

st.title("🏁 F1 DNF Predictor")
st.write("Predict whether a driver will **Finish** or **DNF** based on race details.")

# --- User Inputs ---
st.header("Enter Race Details")

year = st.number_input("Year", min_value=1950, max_value=2030, value=2023)
round_ = st.number_input("Round", min_value=1, max_value=25, value=1)
grid = st.number_input("Grid Position", min_value=0, max_value=100, value=1)
positionOrder = st.number_input("Position Order (finishing position)", min_value=1, max_value=100, value=1)
points = st.number_input("Points Scored", min_value=0.0, max_value=100.0, value=0.0)
laps = st.number_input("Laps Completed", min_value=0, max_value=1000, value=50)
circuitId = st.number_input("Circuit ID", min_value=1, max_value=100, value=1)
lat = st.number_input("Circuit Latitude", value=0.0, format="%.6f")
lng = st.number_input("Circuit Longitude", value=0.0, format="%.6f")
alt = st.number_input("Circuit Altitude (m)", value=0.0, format="%.2f")
age_at_race = st.number_input("Driver Age at Race", min_value=15.0, max_value=60.0, value=28.0, format="%.1f")

# --- Prepare input for model ---
input_data = pd.DataFrame({
    "year": [year],
    "round": [round_],
    "grid": [grid],
    "positionOrder": [positionOrder],
    "points": [points],
    "laps": [laps],
    "circuitId": [circuitId],
    "lat": [lat],
    "lng": [lng],
    "alt": [alt],
    "age_at_race": [age_at_race]
})

st.write("### Input Summary:")
st.dataframe(input_data)

# --- Prediction ---
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        result = "🚩 DNF (Did Not Finish)" if prediction == 1 else "🏆 Finished"
        st.success(f"**Prediction:** {result}")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
