import pandas as pd
import numpy as np


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()
df.drop(columns=['id', 'Residence_type', 'gender', 'work_type'], inplace=True)

# age: age / 100
# hypertension: 0 -> 0, 1 -> 1
# heart_disease: 0 -> 0, 1 -> 1
# ever_married: no -> 0, yes -> 1
# avg_glucose_level: avg_glucose_level / 300
# bmi: bmi / 50
# smoking_status: never smoked -> 0, formerly smoked -> 1, smokes -> 2, Unknown -> 3
