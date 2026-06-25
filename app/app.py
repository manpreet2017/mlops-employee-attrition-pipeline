from fastapi import FastAPI
import joblib
import os
import pandas as pd
from llm.assistant import explain_prediction
from pydantic import BaseModel


app = FastAPI()


# Input schema for Swagger
class Employee(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EmployeeCount: int
    EmployeeNumber: int
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    Over18: str
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StandardHours: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Employee Attrition AI Assistant Running"
    }


@app.post("/predict")
def predict(employee: Employee):

    data = employee.dict()

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    model_features = model.feature_names_in_

    df = df.reindex(
        columns=model_features,
        fill_value=0
    )

    prediction = model.predict(df)

    result = str(prediction[0])

    explanation = explain_prediction(
        result,
        data
    )

    return {
        "attrition_prediction": result,
        "explanation": explanation
    }