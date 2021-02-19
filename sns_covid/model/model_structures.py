from tensorflow.python.keras.layers import *

"""Module which contains different neural network structures"""


def cnn_multi(n_timesteps, n_features, n_outputs):
    return [
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=16, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(n_outputs)]


def cnn_uni(n_timesteps, n_features, n_outputs):
    return [
        Conv1D(filters=16, kernel_size=3, activation='relu',
               input_shape=(n_timesteps, n_features)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(10, activation='relu'),
        Dense(n_outputs)
    ]
