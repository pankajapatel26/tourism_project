import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join("tourism_project/deployment/", "best_machine_tourism.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction App")
st.write("This tool predicts whether a customer is likely to purchase the Wellness Tourism Package.")
st.write("Kindly enter the customer details below:")

# Collect numerical inputs
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=15.0)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
NumberOfTrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=2)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=20000.0)

# Collect categorical inputs
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Super Deluxe", "Standard", "King"])
MaritalStatus = st.selectbox("MaritalStatus", ["Married", "Divorced", "Unmarried", "Single"])
Passport = st.selectbox("Passport", ["Yes", "No"])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
Designation = st.selectbox("Designation", ["Executive", "VP", "Manager", "Senior Manager", "AVP"])

# Convert categorical/binary inputs to match model training format
# (Adjust 1/0 mapping if your training set expects numeric 1/0 for Passport/OwnCar)
input_data = pd.DataFrame([{
    "Age": Age,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Passport": 1 if Passport == "Yes" else 0,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "Designation": Designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "likely to purchase" if prediction == 1 else "not likely to purchase"
    st.write(f"Based on the information provided, the customer is **{result}** the package.")
