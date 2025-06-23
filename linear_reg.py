import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
df = pd.read_csv('data.csv')

print(df)

x= df[['height']]
y = df[['weight']]
print(x)
print(y)

plt.scatter(x,y)
plt.show()

linear_regression = linear_model.LinearRegression()

linear_regression.fit(x,y)

input = np.array([[160]])

result = linear_regression.predict(input)

print(result)


