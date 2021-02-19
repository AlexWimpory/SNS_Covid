from sns_covid import config
from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import generate_train_test
from sns_covid.model.model_structures import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import array
from matplotlib import pyplot



# TODO Set frequency on dataframe to fill in any gaps (prob aren't any but good practice)(uses date column)
# TODO Reverse normalisation
# TODO More complex model
# TODO Logging
# TODO Predictions
# TODO Command line actions
# TODO Document code

# evaluate one or more weekly forecasts against expected values
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
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# evaluate a single model
def evaluate_model(model_func, train, test):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = model_func(history)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = array(predictions)
    # evaluate predictions days for each week
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores



# daily persistence model
def daily_persistence(history):
    # get the data for the prior week
    last_week = history[-1]
    # get the total active power for the last day
    value = last_week[-1, 0]
    # prepare 7 day forecast
    forecast = [value for _ in range(7)]
    return forecast


# weekly persistence model
def weekly_persistence(history):
    # get the data for the prior week
    last_week = history[-1]
    return last_week[:, 0]


def main():
    # Load the data into a dataframe
    df = load_country(config.country_iso_code, download=False)
    # Generate the train, validation and test dataframes
    train_df, test_df = generate_train_test(df)
    models = dict()
    models['daily'] = daily_persistence
    models['weekly'] = weekly_persistence
    days = ['1', '2', '3', '4', '5', '6', '7']
    for name, func in models.items():
        score, scores = evaluate_model(func, train_df, test_df)
        summarize_scores(name, score, scores)
        pyplot.plot(days, scores, marker='o', label=name)
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    main()
