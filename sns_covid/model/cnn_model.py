from sns_covid import config
from numpy import array
from sns_covid.model.base_model import CovidPredictionModel
from sns_covid.model.sequential_model import CovidPredictionSequentialModel


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
        y_pred = self.model.predict(input_x, verbose=0)
        # we only want the vector forecast
        y_pred = y_pred[0]
        return y_pred

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
        y_pred = self.model.predict(input_x, verbose=0)
        # we only want the vector forecast
        y_pred = y_pred[0]
        return y_pred

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



