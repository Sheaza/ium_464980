import pandas as pd
from tensorflow import keras
import numpy as np
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

data = pd.read_csv("MLFLOW/df_test.csv")
X_test = data.drop("Performance Index", axis=1)
y_test = data["Performance Index"]

model = keras.models.load_model("model.keras")

predictions = model.predict(X_test)

with open("predictions.txt", "w") as f:
    f.write(str(predictions))

rmse = root_mean_squared_error(y_test, predictions)
with open("rmse.txt", 'a') as file:
    file.write(str(rmse)+"\n")

with open("rmse.txt", 'r') as file:
    lines = file.readlines()
    num_lines = len(lines)
    lines = [float(line.replace("\n", "")) for line in lines]

plt.plot(range(1, num_lines+1), lines)
plt.xlabel("Build number")
plt.ylabel("RMSE value")
plt.title("RMSE")
plt.savefig("rmse.jpg")
