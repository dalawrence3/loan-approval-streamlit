import streamlit as st
import pandas as pd
import pickle

with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

st.title("Loan Approval Prediction App")
st.write("Enter applicant information below to estimate approval probability.")

# Inputs
fico = st.number_input("FICO Score", 300, 850, 650)
income = st.number_input("Monthly Gross Income ($)", 1, 20000, 5000)
requested_loan = st.number_input("Requested Loan Amount ($)", 1000, 2500000, 20000)
housing = st.number_input("Housing Payment (Monthly) ($)", 0, 50000, 1500)

employment_status = st.selectbox("Employment Status", [
    "full_time",
    "part_time",
    "unemployed"
])

# 👇 User sees Yes/No, model gets 0/1
bankrupt_label = st.selectbox("Ever Bankrupt or Foreclosed?", ["No", "Yes"])
bankrupt = 1 if bankrupt_label == "Yes" else 0

lender = st.selectbox("Select Lender", ["A", "B", "C"])

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Granted_Loan_Amount": [requested_loan],  # assumed equal
        "Requested_Loan_Amount": [requested_loan],
        "FICO_score": [fico],
        "Employment_Status": [employment_status],
        "Monthly_Gross_Income": [income],
        "Monthly_Housing_Payment": [housing],
        "Ever_Bankrupt_or_Foreclose": [bankrupt],
        "Lender": [lender]
    })

    # Engineered features
    input_df["Loan_to_Income"] = input_df["Requested_Loan_Amount"] / input_df["Monthly_Gross_Income"]
    input_df["Payment_to_Income"] = input_df["Monthly_Housing_Payment"] / input_df["Monthly_Gross_Income"]
    input_df["Loan_Gap"] = input_df["Requested_Loan_Amount"] - input_df["Granted_Loan_Amount"]

    # Encode + align
input_encoded = pd.get_dummies(input_df)

# Ensure all expected columns exist
for col in features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns (OUTSIDE the loop)
input_encoded = input_encoded[features]

# Scale (OUTSIDE the loop)
input_scaled = scaler.transform(input_encoded)

# Reorder columns
input_encoded = input_encoded[features]

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prob = model.predict_proba(input_scaled)[0][1]

    threshold = 0.65
    prediction = int(prob >= threshold)

    # Output
    st.subheader("Prediction Result")
    st.write(f"Selected Lender: {lender}")
    st.write(f"Approval Probability: {prob:.2%}")

    if prediction == 1:
        st.success("Predicted: Approved")
    else:
        st.error("Predicted: Not Approved")
