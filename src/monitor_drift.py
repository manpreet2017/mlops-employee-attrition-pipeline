import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


reference_data = pd.read_csv(
    "data/employee_attrition.csv"
)

production_data = reference_data.sample(
    frac=0.5,
    random_state=42
)

report = Report(
    [
        DataDriftPreset()
    ]
)

result = report.run(
    reference_data=reference_data,
    current_data=production_data
)

result.save_html("reports/drift_report.html")

print("Drift report created successfully")