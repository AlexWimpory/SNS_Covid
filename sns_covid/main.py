from functools import partial
from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import generate_train_test
from sns_covid.model.model_structures import *
from sns_covid.model.cnn_model import *
from sns_covid.model.naive_model import CovidPredictionModelNaiveDaily, CovidPredictionModelNaiveWeekly
from sns_covid.visulisation.plotter import visualise
from consolemenu import ConsoleMenu

# TODO Set frequency on dataframe to fill in any gaps (prob aren't any but good practice)(uses date column)
# TODO Reverse normalisation
# TODO Download data if currently empty
# TODO Set output column
# TODO Train and test multiple models and save the best
# TODO Plot predictions and actual
# TODO Fill new vaccinations in with 0s
# TODO Move test,val,train ratio into config
# TODO Logging
# TODO Comments and docstrings


def run_model(f_model):
    # Load the data into a dataframe
    df = load_country(config.country_iso_code)
    # Generate the train and test dataframes
    train_df, test_df = generate_train_test(df)
    model = f_model(train_df)
    model.compile()
    model.fit()
    # Test the model
    score, scores = model.evaluate_model(train_df, test_df)
    # Plot the loss
    visualise(model.name, score, scores)


def run_naive_daily_model():
    f_model = partial(CovidPredictionModelNaiveDaily, 'daily_naive')
    run_model(f_model)


def run_naive_weekly_model():
    f_model = partial(CovidPredictionModelNaiveWeekly, 'weekly_naive')
    run_model(f_model)


def run_cnn_uni_model():
    f_model = partial(CovidPredictionModelCNNUni, 'cnn_uni', cnn_uni)
    run_model(f_model)


def run_cnn_multi_model():
    f_model = partial(CovidPredictionModelCNNMulti, 'cnn_multi', cnn_multi)
    run_model(f_model)


def load_data():
    load_country(config.country_iso_code, download=True)


if __name__ == '__main__':
    model_menu = ConsoleMenu('Choose model', {
        'daily_naive': run_naive_daily_model,
        'weekly_naive': run_naive_weekly_model,
        'uni_cnn': run_cnn_uni_model,
        'multi_cnn': run_cnn_multi_model
    })
    menu = ConsoleMenu('Covid-19', {
        'Download data file': load_data,
        'Choose model': model_menu
    })
    menu.execute()
