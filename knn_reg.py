from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import joblib

df = pd.read_csv('new_data.csv')
x = df[['height']]
y=df[['weight']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

knn_model = KNeighborsRegressor(n_neighbors=3)

knn_model.fit(x_train,y_train)
print("Training complete")
joblib.dump(knn_model,'knn_model.pkl')

y_pred = knn_model.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
print(mse)

rmse = math.sqrt(mse)
print(rmse)

r2 = r2_score(y_test,y_pred)
print(r2)
