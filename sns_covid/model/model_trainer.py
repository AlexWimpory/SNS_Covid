from tensorflow.python.keras import Sequential
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sns_covid import config
from numpy import array
from sklearn.metrics import mean_squared_error
from math import sqrt
import abc


class CovidPredictionModel:
    def __init__(self, model_name, layers):
        self.model = Sequential(name=model_name)
        self.name = model_name
        # Builds layers based on the structure in model_structures
        for layer in layers:
            self.model.add(layer)

    def compile(self):
        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam())
        try:
            self.model.summary()
        except ValueError:
            print('Unable to produce summary')

    def fit(self, train_x, train_y):
        checkpointer = ModelCheckpoint(filepath=f'data/{self.model.name}.hdf5',
                                       verbose=1)
        history = self.model.fit(train_x, train_y,
                                 epochs=config.epochs,
                                 batch_size=config.batch_size,
                                 callbacks=[checkpointer])
        return history

    @staticmethod
    def evaluate_forecasts(actual, predicted):
        scores = list()
        # calculate an RMSE score for each day
        for i in range(actual.shape[1]):
            # calculate mse
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            # calculate rmse
            rmse = sqrt(mse)
            # store
            scores.append(rmse)
            # calculate overall RMSE
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                s += (actual[row, col] - predicted[row, col]) ** 2
        score = sqrt(s / (actual.shape[0] * actual.shape[1]))
        return score, scores

    @abc.abstractmethod
    def forecast(self, history):
        raise NotImplementedError

    def evaluate_model(self, train, test):
        # history is a list of weekly data
        history = [x for x in train]
        # walk-forward validation over each week
        predictions = list()
        for i in range(len(test)):
            # predict the week
            yhat_sequence = self.forecast(history)
            # store the predictions
            predictions.append(yhat_sequence)
            # get real observation and add to history for predicting the next week
            history.append(test[i, :])
        predictions = array(predictions)
        # evaluate predictions days for each week
        score, scores = self.evaluate_forecasts(test[:, :, 0], predictions)
        return score, scores


class CovidPredictionModelCNN(CovidPredictionModel):
    def forecast(self, history):
        # flatten data
        data = array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[-config.n_input:, :]
        # reshape into [1, n_input, n]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        # forecast the next week
        yhat = self.model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    @staticmethod
    def to_supervised(train):
        # flatten data
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + config.n_input
            out_end = in_end + config.n_out
            # ensure we have enough data for this instance
            if out_end < len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)