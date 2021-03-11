from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import array
import abc


class CovidPredictionModel:
    """
    Base class for all covid prediction models
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
        sqr_diff = 0
        for row in range(actual.shape[0]):
            for column in range(actual.shape[1]):
                sqr_diff += (actual[row, column] - predicted[row, column]) ** 2
        score = sqrt(sqr_diff / (actual.shape[0] * actual.shape[1]))
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
        """
        Generate predictions for the dataset using walk forward validation and evaluate the models performance with
        evaluate_forecast
        :param dataset:
        :return: score = RMSE of each day
                 scores = RMSE across all of the days
                 actual = Array of expected values
                 prediction = Array of predictions
        """
        # History is a list of weekly data
        history = [data for data in dataset.train_df]
        # Perform walk-forward validation for each week
        predictions = list()
        for i in range(len(dataset.test_df)):
            # Predict each week
            pred_sequence = self.forecast(history)
            predictions.append(pred_sequence)
            # Get actual values
            history.append(dataset.test_df[i, :])
        # Predictions is an x by 7 array for each week in the test set
        predictions = dataset.destandardise_data(array(predictions))
        # Actual is an x by 7 of the real results for each week in the test set
        actual = dataset.destandardise_data(dataset.test_df[:, :, 0])
        # Compare the predictions to the actual data through the RMSE
        score, scores = self.evaluate_forecasts(actual, predictions)
        return score, scores, predictions, actual
