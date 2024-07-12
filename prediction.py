import numpy as np
import pandas as pd
import pickle

loaded_model = pickle.load(open('trained_model.pkl', 'rb'))

input = np.array([39.0, 6.0, 77516.0, 9.0, 13.0, 4.0, 0.0, 1.0, 4.0, 1.0, 2174.0, 0.0, 40.0, 38.0]).reshape(1, -1)


prediction = loaded_model.predict(input)

print(prediction)

if prediction[0] == 0:
    print("Annual income is less than 50K")
else:
    print("Annual income is more than 50K")