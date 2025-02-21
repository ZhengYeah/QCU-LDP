import pandas as pd
import numpy as np


df = pd.read_csv('Bank-Customer-Attrition-Insights-Data.csv')

# drop columns: RowNumber, CustomerId, Surname, Geography, HasCrCard, Card Type
# Age: Age / 100
# EstimatedSalary: EstimatedSalary / 200,000
# label: Exited


df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender', 'HasCrCard', 'Complain', 'Card Type'])
df['Age'] = df['Age'] / 100
df['EstimatedSalary'] = df['EstimatedSalary'] / 200000

# output the processed data
df.to_csv('processed_bank_data.csv', index=False)
