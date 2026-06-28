import re


def parse_employee_text(text):

    data = {
        "Age": 30,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Sales",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 1000,
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 60,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 3,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5000,
        "MonthlyRate": 20000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "TotalWorkingYears": 10,
        "YearsAtCompany": 5,
        "PercentSalaryHike": 15,
        "WorkLifeBalance": 3,
        "StandardHours": 80,
        "StockOptionLevel": 1,
        "RelationshipSatisfaction": 3,
        "YearsSinceLastPromotion": 1,
        "TrainingTimesLastYear": 3,
        "Over18": "Y",
        "PerformanceRating": 3,
        "YearsInCurrentRole": 3,
        "YearsWithCurrManager": 3
    }


    age = re.search(r"age\s*(\d+)", text.lower())

    if age:
        data["Age"] = int(age.group(1))


    income = re.search(r"income\s*(\d+)", text.lower())

    if income:
        data["MonthlyIncome"] = int(income.group(1))


    if "overtime yes" in text.lower():
        data["OverTime"] = "Yes"


    if "sales" in text.lower():
        data["Department"] = "Sales"


    return data