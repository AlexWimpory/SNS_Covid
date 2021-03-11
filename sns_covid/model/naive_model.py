from sns_covid.model.base_model import CovidPredictionModel


class CovidPredictionModelNaiveDaily(CovidPredictionModel):
    """
    Daily naive covid model which inherits from the base model
    """
    def __init__(self, model_name, dataset):
        self.name = model_name
        self.train = dataset.train_df

    def fit(self):
        # No model to be fit
        return None

    def compile(self):
        # No model to be compiled
        pass

    def forecast(self, history):
        """
        Predict the values for the next week
        """
        # Get the data for the prior week
        last_week = history[-1]
        value = last_week[-1, 0]
        # Prepare the 7 day forecast
        pred = [value for _ in range(7)]
        return pred


class CovidPredictionModelNaiveWeekly(CovidPredictionModel):
    """
    Weekly naive covid model which inherits from the base model
    """
    def __init__(self, model_name, dataset):
        self.name = model_name
        self.train = dataset.train_df

    def fit(self):
        # No model to be fit
        return None

    def compile(self):
        # No model to be compiled
        pass

    def forecast(self, history):
        # Get the data for the prior week
        last_week = history[-1]
        # Prepare the 7 day forecast
        pred = last_week[:, 0]
        return pred
