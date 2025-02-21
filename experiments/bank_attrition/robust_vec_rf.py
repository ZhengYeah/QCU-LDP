from src.robust_radius_sklearn import RobustRadiusSKLearn
import pandas as pd
import joblib
import numpy as np

# private features: age (index 1), salary (index 6)
# choose a user from the dataset as x
user_row = [619,0.22,2,0.0,1,1,0.4067444,2,464]
x_df = pd.DataFrame(data=[user_row], columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])
model = joblib.load('classifiers/bank_rf.pkl')


robust_rec = RobustRadiusSKLearn(model, x_df, ['Age','EstimatedSalary'], 0.05, 0.1)
radius = robust_rec.binary_search()
print(f"Robust radius: {radius}")

