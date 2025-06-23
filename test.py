import joblib
import numpy as np

l_r = joblib.load("height_model.pkl")
predvalue = np.array([[160]])
result = l_r.predict(predvalue)
print(result)