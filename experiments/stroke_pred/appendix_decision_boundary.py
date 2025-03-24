from src.robust_radius_sklearn import RobustRadiusSKLearn
import pandas as pd
import joblib
import numpy as np


# private features: age (index 1), salary (index 6)
# choose a user from the dataset as x
user_row = [0.67,0.0,1.0,1.0,0.7623,0.732]
private_ind_1, private_ind_2 = 0, 5
private_values = [user_row[private_ind_1], user_row[private_ind_2]] # age, bmi
data_columns = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi']
x_df = pd.DataFrame(data=[user_row], columns=data_columns)
model = joblib.load('classifiers/stroke_lr.pkl')


def projected_decision_boundary():
    """
    Draw the decision boundary, projected to the age and salary
    """
    # perturb the age and salary
    perturb_age = np.linspace(0, 1, 300)
    perturb_salary = np.linspace(0, 1, 300)
    perturb_age, perturb_salary = np.meshgrid(perturb_age, perturb_salary)
    perturb_age = perturb_age.flatten()
    perturb_salary = perturb_salary.flatten()
    perturbed_rows = np.array([user_row] * len(perturb_age))
    # index of age and salary
    perturbed_rows[:,private_ind_1] = perturb_age
    perturbed_rows[:,private_ind_2] = perturb_salary
    perturbed_df = pd.DataFrame(data=perturbed_rows, columns=['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi'])
    pred = model.predict(perturbed_df)

    # draw the decision boundary
    import matplotlib.pyplot as plt
    # times new roman font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(5, 5))
    # color map: green for 1, gray for 0
    # alpha: 0.5 for transparency
    color_map = {0: 'pink', 1: 'green'}
    pred_colors = [color_map[p] for p in pred]
    plt.scatter(perturb_age, perturb_salary, c=pred_colors)
    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.title('Projected decision boundary')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig('./projected_decision_boundary_lr.png')
    plt.show()


if __name__ == "__main__":
    projected_decision_boundary()
