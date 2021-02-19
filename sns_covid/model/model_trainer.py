from functools import partial

from tensorflow.python.keras import Sequential
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sns_covid import config
from numpy import array
from sklearn.metrics import mean_squared_error
from math import sqrt
from sns_covid.visulisation.plotter import plot_loss
import abc


class CovidPredictionModel:
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


class CovidPredictionSequentialModel(CovidPredictionModel):
    def __init__(self, model_name, f_layers, train):
        self.model = Sequential(name=model_name)
        self.name = model_name
        self.train_x, self.train_y = self.to_supervised(train)
        n_timesteps, n_features, n_outputs = self.train_x.shape[1], self.train_x.shape[2], self.train_y.shape[1]
        pf_layers = partial(f_layers, n_timesteps, n_features, n_outputs)
        # Builds layers based on the structure in model_structures
        for layer in pf_layers():
            self.model.add(layer)

    def compile(self):
        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam())
        try:
            self.model.summary()
        except ValueError:
            print('Unable to produce summary')

    def fit(self):
        checkpointer = ModelCheckpoint(filepath=f'data/{self.model.name}.hdf5',
                                       verbose=1)
        history = self.model.fit(self.train_x, self.train_y,
                                 epochs=config.epochs,
                                 batch_size=config.batch_size,
                                 callbacks=[checkpointer])
        plot_loss(history)

    @abc.abstractmethod
    def to_supervised(self, train):
        raise NotImplementedError


class CovidPredictionModelCNNMulti(CovidPredictionSequentialModel):
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

    def to_supervised(self, train):
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


class CovidPredictionModelCNNUni(CovidPredictionSequentialModel):
    def forecast(self, history):
        # flatten data
        data = array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[-config.n_input:, 0]
        # reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, len(input_x), 1))
        # forecast the next week
        yhat = self.model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    def to_supervised(self, train):
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
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)


class CovidPredictionModelNaiveDaily(CovidPredictionModel):
    def __init__(self, model_name, train):
        self.name = model_name
        self.train = train

    def fit(self):
        pass

    def compile(self):
        pass

    def forecast(self, history):
        # get the data for the prior week
        last_week = history[-1]
        # get the total active power for the last day
        value = last_week[-1, 0]
        # prepare 7 day forecast
        yhat = [value for _ in range(7)]
        return yhat


class CovidPredictionModelNaiveWeekly(CovidPredictionModel):
    def __init__(self, model_name, train):
        self.name = model_name
        self.train = train

    def fit(self):
        pass

    def compile(self):
        pass

    def forecast(self, history):
        # get the data for the prior week
        last_week = history[-1]
        yhat = last_week[:, 0]
        return yhat
