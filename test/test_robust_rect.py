import pytest
from src.robust_radius_sklearn import RobustRadiusSKLearn
import pandas as pd
import joblib
import numpy as np


# private features: age (index 1), salary (index 6)
# choose a user from the dataset as x
user_row = [619,0.22,2,0.0,1,1,0.4067444,2,464]
model = joblib.load('../experiments/bank_attrition/classifiers/bank_rf.pkl')

def test_decision_boundary():
    """
    Draw the decision boundary, projected to the age and salary
    """
    # perturb the age and salary
    perturb_age = np.linspace(0, 1, 100)
    perturb_salary = np.linspace(0, 1, 100)
    perturb_age, perturb_salary = np.meshgrid(perturb_age, perturb_salary)
    perturb_age = perturb_age.flatten()
    perturb_salary = perturb_salary.flatten()
    perturbed_rows = np.array([user_row] * len(perturb_age))
    # index of age and salary
    perturbed_rows[:,1] = perturb_age
    perturbed_rows[:,6] = perturb_salary
    perturbed_df = pd.DataFrame(data=perturbed_rows, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])
    pred = model.predict(perturbed_df)
    # draw the decision boundary
    import matplotlib.pyplot as plt
    plt.scatter(perturb_age, perturb_salary, c=pred)
    plt.xlabel('Perturbed Age')
    plt.ylabel('Perturbed Salary')
    plt.title('Decision Boundary')
    plt.show()

def test_robust_radius():
    x_df = pd.DataFrame(data=[user_row], columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])
    robust_rec = RobustRadiusSKLearn(model, x_df, ['Age','EstimatedSalary'], 0.05, 0.1)
    radius = robust_rec.binary_search()
    assert radius > 0
    print(f"Robust radius: {radius}")

def test_robust_rect():
    x_df = pd.DataFrame(data=[user_row], columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])
    robust_rec = RobustRadiusSKLearn(model, x_df, ['Age','EstimatedSalary'], 0.05, 0.1)
    robust_rec.binary_search()
    print(robust_rec.clipped_robust_area)
    founded_rec = robust_rec.adjust_step_rate([(0.2, 0.5), (0.15, 0.5), (0.1, 0.5)])
    if founded_rec is not None:
        print(founded_rec)


