"""

Logistic Regression using Scikit-learn in python

https://www.youtube.com/watch?v=xl8ljyE66jM&list=PLZsOBAyNTZwaQB9nUTYUYNhz7b22bAJYY&index=7
"""

import pandas as pd
from matplotlib import pyplot as plt

# 1) read in the data
df = pd.read_csv("data\coffee_images.csv")

print(df.head())

# plt.scatter(df.age, df.result, marker='+', color='r')

# sizes = df['images_analyzed'].value_counts(sort=1)
# plt.pie(sizes, autopct='%1.1f')

# 2) drop irrelevant data

df2 = df.drop(['age'], axis="columns")
df2 = df2.drop(['time'], axis="columns")

print(df2.head())

# 3) deal with missing values

df2 = df2.dropna()

# 4) convert non-numeric to numeric

# so so so so important

df2.result[df2.result == 'good'] = 1 # important
df2.result[df2.result == 'bad'] = 0 # imporant

print(df2.head())

# 5) define independent, dependent variables

Y = df2['result'].values
Y=Y.astype('int32')

Y.dtype

# remember to drop your TARGET (Y) variable! 

X = df2.drop(labels=['result'], axis="columns")
print(X)

# 6) split the data

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

print(X_test)

# 7) define the model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


