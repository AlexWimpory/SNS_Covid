from tensorflow.python.keras.layers import *

"""
Module which contains different neural network structures
•	n_timesteps = The number of previous days used to make the prediction
•	n_features = The number of data columns used to make the prediction
•	n_outputs = The number of days to predict
"""


def cnn_multi(n_timesteps, n_features, n_outputs):
    """
    Multivariate CNN layers
    """
    return [
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
        Conv1D(filters=16, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=16, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(n_outputs)]


def cnn_uni(n_timesteps, n_features, n_outputs):
    """
    Univariate CNN layers
    """
    return [
        Conv1D(filters=16, kernel_size=3, activation='relu',
               input_shape=(n_timesteps, n_features)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(10, activation='relu'),
        Dense(n_outputs)
    ]


def lstm_simple_uni(n_timesteps, n_features, n_outputs):
    """
    Simple univariate LSTM layers
    """
    return [
        LSTM(100, activation='relu', input_shape=(n_timesteps, n_features)),
        Dense(10, activation='relu'),
        Dense(n_outputs)
    ]


def lstm_enc_dec(n_timesteps, n_features, n_outputs):
    """
    LSTM encoder-decoder layers for univariate or multivariate data
    """
    return [
        LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)),
        RepeatVector(n_outputs),
        LSTM(200, activation='relu', return_sequences=True),
        TimeDistributed(Dense(100, activation='relu')),
        TimeDistributed(Dense(1))
    ]
