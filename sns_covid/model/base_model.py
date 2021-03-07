from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import array
import abc


class CovidPredictionModel:
    """
    Base class
    """
    @staticmethod
    def evaluate_forecasts(actual, predicted):
        """
        Calculate the RMSE (root mean squared error) for each day and across all days
        :param actual: Real life data to be compared to predictions
        :param predicted: Model output predictions
        :return: score = RMSE of each day
                 scores = RMSE across all of the days
        """
        scores = list()
        # Calculate the RMSE for each day which has been predicted
        for i in range(actual.shape[1]):
            # Calculate the MSE
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            # Calculate RMSE from the MSE
            rmse = sqrt(mse)
            scores.append(rmse)
            # Calculate RMSE across all of the predicted days
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                s += (actual[row, col] - predicted[row, col]) ** 2
        score = sqrt(s / (actual.shape[0] * actual.shape[1]))
        return score, scores

    @abc.abstractmethod
    def forecast(self, history):
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError

    @abc.abstractmethod
    def compile(self):
        raise NotImplementedError

    def evaluate_model(self, dataset):
        # history is a list of weekly data
        history = [x for x in dataset.train_df]
        # walk-forward validation over each week
        predictions = list()
        for i in range(len(dataset.test_df)):
            # predict the week
            y_pred_sequence = self.forecast(history)
            # store the predictions
            predictions.append(y_pred_sequence)
            # get real observation and add to history for predicting the next week
            history.append(dataset.test_df[i, :])
        # Predictions is an x by 7 array for each week in the test set
        predictions = dataset.denormalise_data(array(predictions))
        # Actual is an x by 7 of the real results for each week in the test set
        actual = dataset.denormalise_data(dataset.test_df[:, :, 0])
        # Compare the predictions to the actual data through the RMSE
        score, scores = self.evaluate_forecasts(actual, predictions)
        return score, scores, predictions, actual
