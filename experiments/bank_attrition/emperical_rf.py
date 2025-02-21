import joblib
from src.samples_from_mechanism import samples_of_mechanism
import numpy as np
import pandas as pd


# private features: age (index 1), salary (index 6)
# choose a user from the dataset as x
user_row = [619,0.42,2,0.0,1,1,0.5067444,2,464]
private_values = [0.42, 0.567444]

sample_num = 1000
mechanism = "pm"
epsilon = 5

samples = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)

perturbed_age = samples[:,0]
perturbed_salary = samples[:,1]

# fill the perturbed rows with the original row
perturbed_rows = user_row * np.ones((sample_num, len(user_row)))
perturbed_rows[:,1] = perturbed_age
perturbed_rows[:,6] = perturbed_salary

# form dataframes
# CreditScore,Age,Tenure,Balance,NumOfProducts,IsActiveMember,EstimatedSalary,Exited,Satisfaction Score,Point Earned
original_df = pd.DataFrame(data=[user_row], columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])
perturbed_df = pd.DataFrame(data=perturbed_rows, columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned'])

# feed to the trained model
model = joblib.load('classifiers/bank_rf.pkl')
ground_truth = model.predict(original_df)
pred = model.predict(perturbed_df)
print(ground_truth)

# calculate the empirical accuracy
accuracy = np.sum(ground_truth == pred) / sample_num
print(f"Empirical accuracy: {accuracy}")

