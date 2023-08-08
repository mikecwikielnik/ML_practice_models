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
      
################################################

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim = 1, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

history = model.fit(X_train, y_train, validation_split=0.2, epochs=100)

import matplotlib.pyplot as plt

loss = history.history['loss'] # training
val_loss = history.history['val_loss']  # validation
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



acc = history.history['mae']    # training
val_acc = history.history['val_mae']    # validation
plt.plot(epochs, acc, 'y', label='Training MAE (accuracy)')
plt.plot(epochs, val_acc, 'r', label = 'Validation MAE (accuracy)')
plt.title = ('Training and Validation MAE (accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (accuracy)')
plt.legend()
plt.show()

# ------- predict on test data -----------

predictions = model.predict(X_test)
print("Predicted values are:\n", predictions)
print("Real values are:\n", y_test)
