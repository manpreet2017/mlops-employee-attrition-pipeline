import streamlit as st
import numpy as np
import pandas as pd
import joblib

from src.data_preprocessing import load_data, preprocess_data


# -----------------------------
# Load model
# -----------------------------

model = joblib.load("model.pkl")


# Load training data to get same feature columns
df = load_data()

X, y = preprocess_data(df)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Employee Attrition AI Assistant")

st.write(
    "Predict whether an employee will leave the company"
)


# Inputs

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)


monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=5000
)


years = st.number_input(
    "Years At Company",
    min_value=0,
    value=5
)


overtime = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)


job_level = st.number_input(
    "Job Level",
    min_value=1,
    max_value=5,
    value=2
)


total_working_years = st.number_input(
    "Total Working Years",
    min_value=0,
    value=10
)



# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):


    # Create empty dataframe with 47 features

    input_df = pd.DataFrame(
        np.zeros((1, X.shape[1])),
        columns=X.columns
    )


    # Fill values

    input_df["Age"] = age

    input_df["MonthlyIncome"] = monthly_income

    input_df["YearsAtCompany"] = years

    input_df["JobLevel"] = job_level

    input_df["TotalWorkingYears"] = total_working_years


    if "OverTime_Yes" in input_df.columns:

        input_df["OverTime_Yes"] = (
            1 if overtime == "Yes" else 0
        )


    prediction = model.predict(input_df)[0]


    # -----------------------------
    # Result
    # -----------------------------

    if prediction == 1:

        result = "Employee is likely to leave the company"

        st.error("⚠ " + result)


    else:

        result = "Employee is likely to stay with the company"

        st.success("✅ " + result)



    # -----------------------------
    # AI HR Explanation
    # -----------------------------

    st.subheader(
        "AI HR Assistant Explanation"
    )


    explanation = f"""
### Prediction

{result}


### Employee Details

- Age: {age}
- Monthly Income: ${monthly_income}
- Years At Company: {years}
- OverTime: {overtime}
- Job Level: {job_level}
- Total Working Years: {total_working_years}


### HR Recommendation

"""


    if prediction == 1:

        explanation += """
The model detected a higher attrition risk.

Recommended actions:

- Review workload and overtime
- Discuss career growth opportunities
- Check employee satisfaction
- Consider retention strategies
"""


    else:

        explanation += """
The model detected a lower attrition risk.

Recommended actions:

- Continue employee engagement
- Provide career development opportunities
- Maintain a positive work environment
"""


    st.info(explanation)