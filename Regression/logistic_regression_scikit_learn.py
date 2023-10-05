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

