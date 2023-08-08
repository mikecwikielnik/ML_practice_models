"""
a simple linear regression via a toy nhl dataset
"""

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/nhl_goals.csv")
print(df)

X = df.drop('Goals', axis='columns')
y = df.Goals # y is an int 64, not an object like X 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)     # training the model (fitting a line)

y_predictions = model.predict(X_test)

# y_test = y_test.drop('Goal predict', axis='columns')

print(y_test, y_predictions)

model.score(X_train, y_predictions)

print("the mean square error is:\n", mean_squared_error(y_test, y_predictions))

print("the mean absolute error is:\n", mean_absolute_error(y_test, y_predictions))

################################################

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()

tree.fit(X_train, y_train)

tree_y_predictions = tree.predict(X_test)

print(y_train, tree_y_predictions)

tree.score(X_train, tree_y_predictions)

print("MSE (loss) for decision tree:\n", mean_squared_error(y_train, tree_y_predictions))
print("MAE (accuracy) for decision tree:\n", mean_absolute_error(y_train, tree_y_predictions))


###################################################

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train, y_train)

forest_y_predictions = forest.predict(X_test)

print(y_train, forest_y_predictions)

forest.score(X_train, forest_y_predictions)

print("MSE (loss) for random forest:\n", mean_squared_error(y_train, forest_y_predictions))
print("MAE (accuracy) for random forest:\n", mean_absolute_error(y_train, forest_y_predictions))
      
