import streamlit as st
import pandas as pd
import prediction_helper

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Insurance Premium Predictor",
    layout="centered"
)

st.markdown(
    "<h2 style='text-align:center;'>🏥 Insurance Premium Prediction</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Fill customer details to predict annual premium</p>",
    unsafe_allow_html=True
)

# ======================
# Input Form
# ======================
with st.form("insurance_form"):

    # -------- Personal Details --------
    st.markdown("### 👤 Personal Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 0, 100, 30)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col3:
        marital_status = st.selectbox("Marital Status", ["Unmarried", "Married"])

    st.markdown("---")

    # -------- Lifestyle & Health --------
    st.markdown("### 🏠 Lifestyle & Health")

    col4, col5 = st.columns(2)
    with col4:
        bmi_category = st.selectbox(
            "BMI Category",
            ['Normal', 'Obesity', 'Overweight', 'Underweight']
        )

    with col5:
        smoking_status = st.selectbox(
            "Smoking Status",
            ['No Smoking', 'Regular', 'Occasional']
        )

    st.write("")

    col6, col7 = st.columns(2)
    with col6:
        medical_history = st.selectbox(
            "Medical History",
            [
                'No Disease',
                'Diabetes',
                'High blood pressure',
                'Thyroid',
                'Heart disease',
                'Diabetes & High blood pressure',
                'Diabetes & Thyroid',
                'Diabetes & Heart disease',
                'High blood pressure & Heart disease'
            ]
        )

    with col7:
        genetic_risk = st.selectbox(
            "Genetic Risk",
            ['No Risk', 'Very Low Risk', 'Low Risk', 'Moderate Risk','High Risk','Very High Risk']
        )

    st.markdown("---")

    # -------- Employment & Income --------
    st.markdown("### 💼 Employment & Income")
    col8, col9, col10 = st.columns(3)

    with col8:
        employment_status = st.selectbox(
            "Employment Status",
            ['Salaried', 'Self-Employed', 'Freelancer']
        )

    with col9:
        income_level = st.selectbox(
            "Income Level",
            ['<10L', '10L - 25L', '25L - 40L', '> 40L']
        )

    with col10:
        income_lakhs = st.number_input(
            "Income (₹ Lakhs)",
            min_value=0.0,
            max_value=100.0,
            step=0.5
        )

    st.markdown("---")

    # -------- Insurance Details --------
    st.markdown("### 🛡 Insurance Details")
    col11, col12, col13 = st.columns(3)

    with col11:
        region = st.selectbox(
            "Region",
            ['Northwest', 'Southeast', 'Northeast', 'Southwest']
        )

    with col12:
        dependants = st.number_input(
            "Number of Dependants",
            min_value=0,
            max_value=10,
            value=0
        )

    with col13:
        insurance_plan = st.selectbox(
            "Insurance Plan",
            ["Bronze", "Silver", "Gold"]
        )

    st.markdown("---")
    submit = st.form_submit_button("🔍 Predict Premium")

# ======================
# Prediction Output
# ======================
if submit:

    input_data = {
        "Age": age,
        "Gender": gender,
        "Region": region,
        "Marital_status": marital_status,
        "Number Of Dependants": dependants,
        "BMI_Category": bmi_category,
        "Smoking_Status": smoking_status,
        "Employment_Status": employment_status,
        "Income_Level": income_level,
        "Income_Lakhs": income_lakhs,
        "Medical History": medical_history,
        "Genetic_risk": genetic_risk,
        "Insurance_Plan": insurance_plan
    }

    st.success("✅ Prediction Successful")

    colA, colB = st.columns([2, 1])

    with colA:
        st.subheader("📋 Input Summary")
        st.dataframe(
            pd.DataFrame([input_data]),
            use_container_width=True
        )

    with colB:
        predicted_premium = prediction_helper.predict(input_data)  # placeholder
        st.metric(
            label="💰 Annual Premium",
            value=f"₹ {predicted_premium:,}"
        )
