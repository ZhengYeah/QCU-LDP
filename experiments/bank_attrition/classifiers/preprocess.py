import pandas as pd
import numpy as np


df = pd.read_csv('Bank-Customer-Attrition-Insights-Data.csv')

# Tree-based models do not require normalization
# drop columns: RowNumber, CustomerId, Surname, Geography, HasCrCard, Card Type
# Age: Age / 100
# EstimatedSalary: EstimatedSalary / 200,000
# label: Exited
# df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender', 'HasCrCard', 'Complain', 'Card Type'])
# df['Age'] = df['Age'] / 100
# df['EstimatedSalary'] = df['EstimatedSalary'] / 200000
# df.to_csv('processed_bank_data.csv', index=False)


# distance-based models require normalization
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender', 'HasCrCard', 'Complain', 'Card Type'])
df['CreditScore'] = df['CreditScore'] / 1000
df['Age'] = df['Age'] / 100
df['Tenure'] = df['Tenure'] / 10
df['Balance'] = df['Balance'] / 250000
df['NumOfProducts'] = df['NumOfProducts'] / 4
df['IsActiveMember'] = df['IsActiveMember'] / 1
df['EstimatedSalary'] = df['EstimatedSalary'] / 200000
df['Satisfaction Score'] = df['Satisfaction Score'] / 5
df['Point Earned'] = df['Point Earned'] / 1000
df.to_csv('processed_bank_data_normalized.csv', index=False)
