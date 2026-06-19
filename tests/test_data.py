import pandas as pd


def test_dataset_exists():
    df = pd.read_csv("data/employee_attrition.csv")
    assert len(df) > 0


def test_attrition_column_exists():
    df = pd.read_csv("data/employee_attrition.csv")
    assert "Attrition" in df.columns


def test_expected_columns_exist():
    df = pd.read_csv("data/employee_attrition.csv")

    expected_columns = [
        "Age",
        "Department",
        "MonthlyIncome",
        "Attrition"
    ]

    for col in expected_columns:
        assert col in df.columns