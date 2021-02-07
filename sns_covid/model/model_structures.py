from tensorflow.python.keras.layers import Dense, Dropout, Activation, LSTM, Conv1D, Lambda
import tensorflow as tf
from sns_covid import config

"""Module which contains different neural network structures"""


def linear():
    return [
        Dense(units=1)
    ]


def dense():
    return [
        Dense(64, input_shape=(config.input_width, len(config.columns_used))),
        Activation('relu'),
        Dropout(0.2),

        Dense(64),
        Activation('relu'),
        Dropout(0.2),

        Dense(1)
    ]


def recurrent():
    return [
        LSTM(32, return_sequences=True),
        Dense(1)
    ]


def cnn():
    return [
        Conv1D(filters=32, kernel_size=(config.conv_width,), activation='relu', input_shape=(1, 7, 4)),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ]


def multi_dense():
    return [
        Lambda(lambda x: x[:, -1:, :]),
        Dense(512, activation='relu'),
        Dense(7*4, kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Reshape([7, 4])
    ]