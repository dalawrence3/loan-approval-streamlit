import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

st.title("Loan Approval Prediction App")

# User inputs
fico = st.number_input("FICO Score", 300, 850, 650)
income = st.number_input("Monthly Income", 1, 10000, 5000)
loan = st.number_input("Requested Loan Amount", 1000, 100000, 20000)
housing = st.number_input("Housing Payment", 0, 5000, 1500)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "FICO_score": [fico],
        "Monthly_Gross_Income": [income],
        "Requested_Loan_Amount": [loan],
        "Monthly_Housing_Payment": [housing]
    })

    # Create engineered features
    input_df["Loan_to_Income"] = input_df["Requested_Loan_Amount"] / input_df["Monthly_Gross_Income"]
    input_df["Payment_to_Income"] = input_df["Monthly_Housing_Payment"] / input_df["Monthly_Gross_Income"]

    # Align with training features
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Approval Probability: {prob:.2%}")