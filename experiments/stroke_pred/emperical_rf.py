import joblib
from src.samples_from_mechanism import samples_of_mechanism
import numpy as np
import pandas as pd


# private features: age (index 0), bmi (index -1)
# choose a user from the dataset as x
user_row = [0.4,1,0,1,0.4,0.25]
private_values = [0.4, 0.25]

sample_num = 1000
mechanism = "pm"
epsilon = 1

samples = samples_of_mechanism(private_values, sample_num, mechanism, epsilon)

perturbed_age = samples[:,0]
perturbed_bmi = samples[:,-1]

# fill the perturbed rows with the original row
perturbed_rows = user_row * np.ones((sample_num, len(user_row)))
perturbed_rows[:,0] = perturbed_age
perturbed_rows[:,-1] = perturbed_bmi

# form dataframes
original_df = pd.DataFrame(data=[user_row], columns=["age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", "bmi"])
perturbed_df = pd.DataFrame(data=perturbed_rows, columns=["age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", "bmi"])

# feed to the trained model
model = joblib.load('./classifiers/stroke_lr.pkl')
ground_truth = model.predict(original_df)
pred = model.predict(perturbed_df)
print(ground_truth)

# calculate the empirical accuracy
accuracy = np.sum(ground_truth == pred) / sample_num
print(f"Empirical accuracy: {accuracy}")

