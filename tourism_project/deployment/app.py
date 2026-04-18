import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# ── Load Model from Hugging Face Model Hub ─────────────────────────────────────

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="nikhileshmehta89/tourism-best-model",
        filename="best_model.pkl",
        token=os.getenv("HF_TOKEN"),
    )
    return joblib.load(model_path)

model = load_model()

# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.title("Tourism Wellness Package - Purchase Predictor")
st.markdown("Fill in the customer details below to predict if they will purchase the package.")

col1, col2 = st.columns(2)

with col1:
    age                     = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact         = st.selectbox("Type of Contact", [0, 1], format_func=lambda x: "Company Invited" if x == 0 else "Self Enquiry")
    city_tier               = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch       = st.number_input("Duration of Pitch (mins)", min_value=0, max_value=100, value=15)
    occupation              = st.selectbox("Occupation", [0, 1, 2, 3], format_func=lambda x: ["Free Lancer", "Large Business", "Salaried", "Small Business"][x])
    gender                  = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    number_of_person        = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    number_of_followups     = st.number_input("Number of Followups", min_value=1, max_value=10, value=3)

with col2:
    product_pitched         = st.selectbox("Product Pitched", [0, 1, 2, 3, 4], format_func=lambda x: ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"][x])
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status          = st.selectbox("Marital Status", [0, 1, 2], format_func=lambda x: ["Divorced", "Married", "Single"][x])
    number_of_trips         = st.number_input("Number of Trips per Year", min_value=1, max_value=20, value=3)
    passport                = st.selectbox("Passport", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    pitch_satisfaction      = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car                 = st.selectbox("Owns a Car", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    children_visiting       = st.number_input("Number of Children Visiting (< 5 yrs)", min_value=0, max_value=5, value=0)
    designation             = st.selectbox("Designation", [0, 1, 2, 3, 4], format_func=lambda x: ["AVP", "Executive", "Manager", "Senior Manager", "VP"][x])
    monthly_income          = st.number_input("Monthly Income (₹)", min_value=1000, max_value=100000, value=25000)

# ── Build Input DataFrame ──────────────────────────────────────────────────────

input_data = pd.DataFrame([{
    "Age":                      age,
    "TypeofContact":            type_of_contact,
    "CityTier":                 city_tier,
    "DurationOfPitch":          duration_of_pitch,
    "Occupation":               occupation,
    "Gender":                   gender,
    "NumberOfPersonVisiting":   number_of_person,
    "NumberOfFollowups":        number_of_followups,
    "ProductPitched":           product_pitched,
    "PreferredPropertyStar":    preferred_property_star,
    "MaritalStatus":            marital_status,
    "NumberOfTrips":            number_of_trips,
    "Passport":                 passport,
    "PitchSatisfactionScore":   pitch_satisfaction,
    "OwnCar":                   own_car,
    "NumberOfChildrenVisiting": children_visiting,
    "Designation":              designation,
    "MonthlyIncome":            monthly_income,
}])

# ── Predict ────────────────────────────────────────────────────────────────────

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"This customer is likely to purchase the package. (Confidence: {probability:.1%})")
    else:
        st.warning(f"This customer is unlikely to purchase the package. (Confidence: {1 - probability:.1%})")
