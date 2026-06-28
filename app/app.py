import streamlit as st
import joblib
import pandas as pd

from llm_parser import parse_employee_text
pipeline = joblib.load("pipeline.pkl")


st.title("Employee Attrition AI Assistant")

st.write(
    "Predict whether an employee will leave the company"
)
st.subheader("AI Employee Assistant")

text = st.text_area(
    "Describe employee in natural language"
)

if st.button("Analyze Text"):

    employee = parse_employee_text(text)

    df = pd.DataFrame([employee])

    prediction = pipeline.predict(df)[0]

    if prediction == 1:
        st.error("⚠ Employee is likely to leave")

        st.write(
            "HR Recommendation: Review workload, career growth, and engagement."
        )

    else:
        st.success("✅ Employee is likely to stay")

        st.write(
            "HR Recommendation: Continue engagement and development."
        )

    st.write("Parsed employee information:")
    st.dataframe(df)


# Required training features

age = st.number_input("Age", value=30)
business_travel = st.selectbox(
    "BusinessTravel",
    ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
)

daily_rate = st.number_input("DailyRate", value=800)

department = st.selectbox(
    "Department",
    ["Sales", "Research & Development", "Human Resources"]
)

distance = st.number_input(
    "DistanceFromHome",
    value=5
)

education = st.number_input(
    "Education",
    value=3
)

education_field = st.selectbox(
    "EducationField",
    [
        "Life Sciences",
        "Medical",
        "Marketing",
        "Technical Degree",
        "Other"
    ]
)

employee_count = 1

employee_number = st.number_input(
    "EmployeeNumber",
    value=1000
)

environment = st.number_input(
    "EnvironmentSatisfaction",
    value=3
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

hourly_rate = st.number_input(
    "HourlyRate",
    value=60
)

job_involvement = st.number_input(
    "JobInvolvement",
    value=3
)

job_level = st.number_input(
    "JobLevel",
    value=2
)

job_role = st.selectbox(
    "JobRole",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manager"
    ]
)

job_satisfaction = st.number_input(
    "JobSatisfaction",
    value=3
)

marital = st.selectbox(
    "MaritalStatus",
    ["Single", "Married", "Divorced"]
)

monthly_income = st.number_input(
    "MonthlyIncome",
    value=5000
)

monthly_rate = st.number_input(
    "MonthlyRate",
    value=20000
)

num_companies = st.number_input(
    "NumCompaniesWorked",
    value=2
)

over_time = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)

total_years = st.number_input(
    "TotalWorkingYears",
    value=10
)

years_company = st.number_input(
    "YearsAtCompany",
    value=5
)


if st.button("Predict"):


    data = pd.DataFrame({

        "Age":[age],
        "BusinessTravel":[business_travel],
        "DailyRate":[daily_rate],
        "Department":[department],
        "DistanceFromHome":[distance],
        "Education":[education],
        "EducationField":[education_field],
        "EmployeeCount":[employee_count],
        "EmployeeNumber":[employee_number],
        "EnvironmentSatisfaction":[environment],
        "Gender":[gender],
        "HourlyRate":[hourly_rate],
        "JobInvolvement":[job_involvement],
        "JobLevel":[job_level],
        "JobRole":[job_role],
        "JobSatisfaction":[job_satisfaction],
        "MaritalStatus":[marital],
        "MonthlyIncome":[monthly_income],
        "MonthlyRate":[monthly_rate],
        "NumCompaniesWorked":[num_companies],
        "OverTime":[over_time],
        "TotalWorkingYears":[total_years],
        "YearsAtCompany":[years_company],
        "PercentSalaryHike":[15],
        "WorkLifeBalance":[3],
        "StandardHours":[80],
        "StockOptionLevel":[1],
        "RelationshipSatisfaction":[3],
        "YearsSinceLastPromotion":[1],
        "TrainingTimesLastYear":[3],
        "Over18":["Y"],
        "PerformanceRating":[3],
        "YearsInCurrentRole":[3],
        "YearsWithCurrManager":[3],

    })


    prediction = pipeline.predict(data)[0]


    if prediction == 1:

        st.error(
            "⚠ Employee is likely to leave"
        )

        st.write(
            "HR Recommendation: Review workload, career growth, and engagement."
        )

    else:

        st.success(
            "✅ Employee is likely to stay"
        )

        st.write(
            "HR Recommendation: Continue engagement and development."
        )