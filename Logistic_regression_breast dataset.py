import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

df.head()
df.isna().sum()

corr_matrix = df.corr()

plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

threshold = 0.9
correlated_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range (i):
        if abs(corr_matrix.iloc[i,j]) > threshold:
            row_name = corr_matrix.columns[i]
            if row_name in correlated_features:
                continue
            column_name = corr_matrix.columns[i]
            correlated_features.add(column_name)

df_reduced = df.drop(columns= correlated_features)
X_reduced = df_reduced.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, df_reduced['target'], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)

