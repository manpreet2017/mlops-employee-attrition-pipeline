def explain_prediction(prediction, employee_data):

    if prediction == "Yes":
        explanation = (
            "This employee has a higher risk of attrition. "
            "Factors such as job satisfaction, overtime, income, "
            "and years at company may contribute to this prediction."
        )
    else:
        explanation = (
            "This employee has a lower risk of attrition. "
            "The model suggests stable employment factors "
            "reduce the likelihood of leaving."
        )

    return explanation