import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Datasets\diabetes.csv')
print(df.head())

#x= df.iloc[:,:8]
x = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y = df["Outcome"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(x_train,y_train)

pred_values = knn_model.predict(x_test)

acc = accuracy_score(y_test,pred_values)
print(acc)

# test_value = pd.DataFrame([[2,109,92,0,0,42.7,0.845,54]])

# result = knn_model.predict(test_value)

# print(result)