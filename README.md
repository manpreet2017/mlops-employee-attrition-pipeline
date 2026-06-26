Employee Attrition Prediction MLOps Pipeline

This project develops an end-to-end machine learning pipeline to predict employee attrition. The pipeline includes data preprocessing, model training, model evaluation, experiment tracking with MLflow, data versioning with DVC, automated testing with pytest, CI/CD automation using GitHub Actions, and data drift monitoring.

Technologies Used
Python
Scikit-learn
MLflow
DVC
Pytest
GitHub Actions
Pandas
NumPy
## How to Run

Install dependencies:

pip install -r requirements.txt

Start MLflow server:

mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

Train the model:

python src/train.py

The trained model and MLflow experiment logs will be generated after training.
## Streamlit Application

Run the interactive employee attrition prediction app:

```bash
python -m streamlit run app/app.py
