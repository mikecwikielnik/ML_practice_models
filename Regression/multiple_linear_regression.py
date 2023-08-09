"""
multiple linear regression with toy coffee_images dataset
"""

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/coffee_images.csv')

print(df)

X = df.drop('images analyzed', axis='columns')
y = df[['images analyzed']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_predictions = model.predict(X_test_scaled)

model.score(X_test_scaled, y_predictions) # the model overfit the data




