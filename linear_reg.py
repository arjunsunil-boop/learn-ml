import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data.csv')

print(df)

x= df[['height']]
y = df[['weight']]
print(x)
print(y)

plt.scatter(x,y)
plt.show()

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42) #splitting the data into 80,20
print(x_train)
linear_regression = linear_model.LinearRegression()

linear_regression.fit(x_train,y_train)#training

joblib.dump(linear_regression,'height_model.pkl')

input = np.array([[160]])

result = linear_regression.predict(input)

print(result)


