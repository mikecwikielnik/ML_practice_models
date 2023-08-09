"""
sklearn.datasets.load_diabetes toy regression example
"""

from sklearn import datasets
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y = datasets.load_diabetes(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)

y_predictions = model.predict(X_test)

model.score(X_test, y_predictions)

# plot outputs

plt.scatter(X_train, y_train, color='black')
plt.plot(X_test, y_predictions, color='blue')

plt.xticks(())
plt.yticks(())

plt.show()

