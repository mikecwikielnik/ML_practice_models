"""
Regression using Neural Networks and comparison to other models

https://www.youtube.com/watch?v=2yhLEx2FKoY&list=PLPtfMJjpfeunb60QAdr7e1yJJe_KlNmiF&index=2
"""

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# load the data and arrange into pd df

df = pd.read_csv("data/housing.csv", delim_whitespace = True, header = None)

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                  'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df.columns = feature_names
print(df.head())

df = df.rename(columns={'MEDV': 'PRICE'})
print(df.describe())

# split into features and target (price)

X = df.drop('PRICE', axis=1)
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the data, otherwise it will fail
# standardize features by removing the mean and scaling to unit variance

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define the model 
# experiment with deeper and wider networks

model = Sequential()
model.add(Dense(128, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))

# ouput layer

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100)

# plot the mean absolute error (accuracy), and the loss at each epoch

import matplotlib.pyplot as plt

loss = history.history['loss'] # training
val_loss = history.history['val_loss']  # validation
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
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


