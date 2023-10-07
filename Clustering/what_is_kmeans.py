"""
What is k-mens clustering and how to code it in Python?

https://www.youtube.com/watch?v=d7NJGLevmwA&list=PLZsOBAyNTZwaQB9nUTYUYNhz7b22bAJYY&index=9
"""

import pandas as pd

df = pd.read_excel('data\k_means.xlsx')

# print(df.head())

# import seaborn as sns

# sns.regplot(x=df['x'], y=df['y'], fit_reg=False)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init='k-means++',
                max_iter=300, n_init=10, random_state=42)

model = kmeans.fit(df)

predicted_values = kmeans.predict(df)

from matplotlib import pyplot as plt

plt.scatter(df['x'], df['y'], c=predicted_values,
            s=50, cmap='viridis')

plt.scatter(kmeans.cluster_centers_[:,0], # run both plt.scatter chunks
            kmeans.cluster_centers_[:,1],
            s=200, c='black', alpha=0.5)

