import pandas as pd
import numpy as np


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
# first 1000 rows to avoid trivial classification for stroke = 0
df = df.iloc[:1000]

# drop columns: id, gender, work_type, residence_type, smoking_status
# age: age / 100
# hypertension: 0 -> 0, 1 -> 1
# heart_disease: 0 -> 0, 1 -> 1
# ever_married: no -> 0, yes -> 1
# avg_glucose_level: avg_glucose_level / 300
# bmi: bmi / 50

df = df.drop(columns=['id', 'gender', 'work_type', 'Residence_type', 'smoking_status'])
df['age'] = df['age'] / 100
df['hypertension'] = df['hypertension'] / 1
df['heart_disease'] = df['heart_disease'] / 1
df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1}) / 1
df['avg_glucose_level'] = df['avg_glucose_level'] / 300
df['bmi'] = df['bmi'] / 50

# fill missing values in bmi with the mean
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# output the processed data
df.to_csv('processed_stroke_data_normalized.csv', index=False)
