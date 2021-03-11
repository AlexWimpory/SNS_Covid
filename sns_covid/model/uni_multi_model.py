from sns_covid import config
from numpy import array
from sns_covid.model.sequential_model import CovidPredictionSequentialModel


class CovidPredictionModelMulti(CovidPredictionSequentialModel):
    """
    Class which implements multivariate sequential Keras models.  Inherits from sequential model
    """
    def forecast(self, history):
        """
        Predict the values for the next week
        """
        # Flatten the data
        data = array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        # Retrieve the last observations for the input data
        input_x = data[-config.n_input:, :]
        # Reshape into [1, n_input, n]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        # Predict the next week
        y_pred = self.model.predict(input_x, verbose=0)
        y_pred = y_pred[0]
        return y_pred

    def to_supervised(self, train):
        """
        Uses the list of weeks and the number of time steps to create overlapping windows of data
        """
        # Flatten the data
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # Step over the history
        for _ in range(len(data)):
            # Define the end of the input sequence
            in_end = in_start + config.n_input
            out_end = in_end + config.n_out
            # Ensure that there is enough data for this instance
            if out_end < len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # Increment by 1 time step
            in_start += 1
        return array(X), array(y)


class CovidPredictionModelUni(CovidPredictionSequentialModel):
    """
    Class which implements univariate sequential Keras models.  Inherits from sequential model
    """
    def forecast(self, history):
        """
        Predict the values for the next week
        """
        # Flatten the data
        data = array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        # retrieve the last observations for the input data
        input_x = data[-config.n_input:, 0]
        # Reshape into [1, n_input, 1]
        input_x = input_x.reshape((1, len(input_x), 1))
        # Predict the next week
        y_pred = self.model.predict(input_x, verbose=0)
        y_pred = y_pred[0]
        return y_pred

    def to_supervised(self, train):
        """
        Uses the list of weeks and the number of time steps to create overlapping windows of data
        """
        # Flatten the data
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # Step over the history
        for _ in range(len(data)):
            # Define the end of the input sequence
            in_end = in_start + config.n_input
            out_end = in_end + config.n_out
            # Ensure that there is enough data for this instance
            if out_end < len(data):
                x_input = data[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])
            # Increment by 1 time step
            in_start += 1
        return array(X), array(y)
