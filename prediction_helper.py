import pandas as pd
import joblib


model_rest = joblib.load("artifacts/model_rest.joblib")
model_young = joblib.load("artifacts/model_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")
scaler_young = joblib.load("artifacts/scaler_young.joblib")


def assign_values_and_scaling(df, input_dict):
    scaler = None

    # Decide the values for model and scaler
    if input_dict.get("Age") :
        if input_dict["Age"] <= 25 :
            scaler = scaler_young
        else:
            scaler = scaler_rest

    # Age
    if input_dict.get("Age") :
        df["age"] = input_dict["Age"]

    # Number Of Dependants
    if input_dict.get("Number Of Dependants") :
        df["number_of_dependants"] = input_dict["Number Of Dependants"]

    # Income_Lakhs
    if input_dict.get("Income_Lakhs") :
        df["income_lakhs"] = input_dict["Income_Lakhs"]

    # Insurance_Plan
    if input_dict.get("Insurance_Plan"):
        insurance_plans = {"Bronze": 1, "Silver":2, "Gold":3 }
        df["insurance_plan"] = insurance_plans[input_dict["Insurance_Plan"]]

    # genetical_risk
    if input_dict.get("Genetic_risk"):
        genetic_risks = { 'No Risk': 0 , 'Very Low Risk': 1, 'Low Risk': 2, 'Moderate Risk' : 3,'High Risk': 4,'Very High Risk':5 }
        df["genetical_risk"] = genetic_risks[input_dict["Genetic_risk"]]

    # normalized_risk_score
    if input_dict.get("Medical History"):
        medical_history = input_dict["Medical History"]
        diseases = medical_history.lower().split(" & ")
        risk_scores = {
            "diabetes": 6,
            "heart disease": 8,
            "high blood pressure": 6,
            "thyroid": 5,
            "no disease": 0,
            "none": 0
        }
        total_score = 0
        for disease in diseases:
            total_score += risk_scores[disease]
        max_score = 14
        min_score = 0
        normalized_risk_score = (total_score - min_score) / (max_score - min_score)
        df["normalized_risk_score"] = normalized_risk_score

    # Gender
    if input_dict.get("Gender"):
        df["gender_Male"] = 1

    # Region
    if input_dict.get("Region"):
        if input_dict["Region"] == "Northwest":
            df["region_Northwest"] = 1
        elif input_dict["Region"] == "Southeast":
            df["region_Southeast"] = 1
        elif input_dict["Region"] == "Southwest":
            df["region_Southwest"] = 1

    # Marital_status
    if input_dict.get("Marital_status"):
        if input_dict["Marital_status"] == "Unmarried":
            df["marital_status_Unmarried"] = 1

    # BMI Category
    if input_dict.get("BMI_Category"):
        if input_dict["BMI_Category"] == "Obesity":
            df["bmi_category_Obesity"] = 1
        elif input_dict["BMI_Category"] == "Overweight":
            df["bmi_category_Overweight"] = 1
        elif input_dict["BMI_Category"] == "Underweight":
            df["bmi_category_Underweight"] = 1

    # Smoking Status
    if input_dict.get("Smoking Status"):
        if input_dict["Smoking Status"] == "Regular":
            df["smoking_status_Regular"] = 1
        elif input_dict["Smoking Status"] == "Occasional":
            df["smoking_status_Occasional"] = 1

    # Employment_Status
    if input_dict.get("Employment_Status"):
        if input_dict["Employment_Status"] == "Salaried":
            df["employment_status_Salaried"] = 1
        elif input_dict["Employment_Status"] == "Self-Employed":
            df["employment_status_Self-Employed"] = 1

    ## Handle Scaling
    cols_to_scale = scaler['cols_to_scale']
    scaler = scaler['scaler']

    df['income_level'] = None  # since scaler object expects income_level supply it. This will have no impact on anything
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def preprocessing_input(input_dict):
    # Define the expected columns and initialize the DataFrame with zeros
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]
    # Create empty data frame
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Assign values to that
    return assign_values_and_scaling(df, input_dict)


def predict(input_dict):
    input_df = preprocessing_input(input_dict)
    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])
