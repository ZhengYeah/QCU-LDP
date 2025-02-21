import pandas as pd
import numpy as np


df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# drop columns: id, gender, work_type, residence_type, smoking_status
# age: age / 100
# hypertension: 0 -> 0, 1 -> 1
# heart_disease: 0 -> 0, 1 -> 1
# ever_married: no -> 0, yes -> 1
# avg_glucose_level: avg_glucose_level / 300
# bmi: bmi / 50

df = df.drop(columns=['id', 'gender', 'work_type', 'Residence_type', 'smoking_status'])
df['age'] = df['age'] / 100
df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
df['avg_glucose_level'] = df['avg_glucose_level'] / 300
df['bmi'] = df['bmi'] / 50

# fill missing values in bmi with the mean
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# output the processed data
df.to_csv('processed_stroke_data.csv', index=False)
