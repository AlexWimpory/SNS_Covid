from sns_covid.model.base_model import CovidPredictionModel


class CovidPredictionModelNaiveDaily(CovidPredictionModel):
    def __init__(self, model_name, train):
        self.name = model_name
        self.train = train

    def fit(self):
        return None

    def compile(self):
        pass

    def forecast(self, history):
        # get the data for the prior week
        last_week = history[-1]
        # get the total active power for the last day
        value = last_week[-1, 0]
        # prepare 7 day forecast
        y_pred = [value for _ in range(7)]
        return y_pred


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
        y_pred = last_week[:, 0]
        return y_pred