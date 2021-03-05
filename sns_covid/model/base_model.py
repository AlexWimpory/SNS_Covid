from sklearn.metrics import mean_squared_error
from math import sqrt
from sns_covid.visulisation.plotter import plot_prediction_vs_actual
from numpy import array
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
            y_pred_sequence = self.forecast(history)
            # store the predictions
            predictions.append(y_pred_sequence)
            # get real observation and add to history for predicting the next week
            history.append(test[i, :])
        # Predictions an x by 7 array for each week in the test set
        predictions = array(predictions)
        # Actual is an x by 7 of the real results for each week in the test set
        actual = test[:, :, 0]
        # Compare the predictions to the actual data through the RMSE
        score, scores = self.evaluate_forecasts(actual, predictions)
        return score, scores, predictions, actual

    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def compile(self):
        raise NotImplementedError
