import streamlit as st
import joblib
import os
import numpy as np

# Get the absolute path to the model file
model_path = os.path.join(os.getcwd(), "models", "best_model.pkl")

# Check if the model file exists before loading
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("❌ Model file not found! Please train and save the model first.")
    st.stop()  # Stop execution if the model is missing

st.title("Credit Card Fraud Detection")

# Input fields for user to enter transaction details
st.write("Enter transaction details:")

# Assuming the model expects 30 numerical inputs (like in the credit card dataset)
input_values = []
for i in range(30):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    input_values.append(value)

# Convert to NumPy array
input_data = np.array(input_values).reshape(1, -1)

# Predict fraud or not
if st.button("Check for Fraud"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
