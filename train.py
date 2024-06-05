import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import argparse

class RegressionModel:
    def __init__(self, optimizer="adam", loss="mean_squared_error"):
        self.model = keras.Sequential([
            layers.Input(shape=(5,)),  # Input layer
            layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons and ReLU activation
            layers.Dense(1)  # Output layer with a single neuron (for regression)
        ])
        self.optimizer = optimizer
        self.loss = loss
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, train_path, test_path):
        data_train = pd.read_csv(train_path)
        data_test = pd.read_csv(test_path)
        self.X_train = data_train.drop("Performance Index", axis=1)
        self.y_train = data_train["Performance Index"]
        self.X_test = data_test.drop("Performance Index", axis=1)
        self.y_test = data_test["Performance Index"]

    def train(self, epochs=30):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=32, validation_data=(self.X_test, self.y_test))

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction

    def evaluate(self):
        test_loss = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss: {test_loss:.4f}")
        return test_loss

    def save_model(self):
        self.model.save("model.keras")


parser = argparse.ArgumentParser()
parser.add_argument('--epochs')

args = parser.parse_args()
model = RegressionModel()
model.load_data("df_train.csv", "df_test.csv")
model.train(epochs=int(args.epochs))
model.save_model()
