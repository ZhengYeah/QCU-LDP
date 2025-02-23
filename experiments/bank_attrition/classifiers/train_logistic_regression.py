from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

df = pd.read_csv('processed_bank_data_normalized.csv')
X = df.drop(columns=['Exited'])
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")

# export the model
import joblib
joblib.dump(lr, 'bank_lr.pkl')
