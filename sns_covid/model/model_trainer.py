from tensorflow.python.keras import Sequential
from sns_covid.model.model_structures import *
import numpy as np
import pandas as pd


class CovidPredictionModel:
    def __init__(self, model_name, layers):
        self.model = Sequential(name=model_name)
        # Builds layers based on the structure in model_structures
        for layer in layers:
            self.model.add(layer)

    def compile(self):
        """Compile the model and print the structure"""
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        self.model.summary()

    def test_model(self, x_data, y_data):
        """Calculate the model's accuracy on the input dataset"""
        score = self.model.evaluate(x_data, y_data, verbose=0)
        accuracy = 100 * score[1]
        return accuracy

    def train_model(self, x_train, y_train, x_val, y_val):
        """Train and save the model"""
        pass


def train_and_test_model(features, model):
    pass


def trainer():
    pass
