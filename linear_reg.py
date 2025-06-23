import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error,r2_score
import math

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

print("training completed!")

pred_y = linear_regression.predict(x_test)

mse = mean_squared_error(y_test,pred_y)
print(mse)

rmse = math.sqrt(mse)
print(rmse)

r2score = r2_score(y_test,pred_y)

print(r2score)




