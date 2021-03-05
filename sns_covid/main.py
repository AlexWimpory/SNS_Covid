from functools import partial
from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import generate_train_test
from sns_covid.logging_config import get_logger
from sns_covid.model.model_structures import *
from sns_covid.model.uni_multi_model import *
from sns_covid.model.naive_model import CovidPredictionModelNaiveDaily, CovidPredictionModelNaiveWeekly
from sns_covid.visulisation.plotter import visualise, plot_loss, plot_prediction_vs_actual, print_scores
from consolemenu import ConsoleMenu

logger = get_logger(__name__)


# TODO Reverse normalisation
# Fill new vaccinations in with 0s
# TODO Set output column
# TODO Set frequency on dataframe to fill in any gaps (prob aren't any but good practice)(uses date column)
# Train and test multiple models and save the best
# TODO Logging
# TODO Comments and docstrings
# TODO Combine sources of data


def run_model(f_model, model_runs=1):
    # Load the data into a dataframe
    df = load_country(config.country_iso_code)
    # Generate the train and test dataframes
    train_df, test_df = generate_train_test(df)
    results = []
    for i in range(0, model_runs):
        result = dict()
        model = f_model(train_df)
        result['model_name'] = model.name
        model.compile()
        result['history'] = model.fit()
        # Test the model
        evaluate_results = model.evaluate_model(train_df, test_df)
        result = capture_evaluate(result, *evaluate_results)
        results.append(result)
        print_scores(logger, result['model_name'], result['score'], result['scores'])
    plot_best_result(results)


def capture_evaluate(result, score, scores, predictions, actual):
    result['score'] = score
    result['scores'] = scores
    result['predictions'] = predictions
    result['actual'] = actual
    return result


def plot_best_result(results):
    best_result = None
    for result in results:
        if not best_result or best_result['score'] > result['score']:
            best_result = result
    plot_loss(best_result['history'])
    # Plot the loss
    print_scores(logger, best_result['model_name'], best_result['score'], best_result['scores'])
    visualise(best_result['scores'])
    plot_prediction_vs_actual(best_result['predictions'], best_result['actual'])


def iterations():
    return int(input('Enter the number of iterations: '))


# Partial functions which run when selecting an option from the menu
def run_naive_daily_model():
    f_model = partial(CovidPredictionModelNaiveDaily, 'daily_naive')
    run_model(f_model)


def run_naive_weekly_model():
    f_model = partial(CovidPredictionModelNaiveWeekly, 'weekly_naive')
    run_model(f_model)


def run_cnn_uni_model():
    f_model = partial(CovidPredictionModelUni, 'cnn_uni', cnn_uni)
    model_runs = iterations()
    run_model(f_model, model_runs)


def run_cnn_multi_model():
    f_model = partial(CovidPredictionModelMulti, 'cnn_multi', cnn_multi)
    model_runs = iterations()
    run_model(f_model, model_runs)


def run_lstm_simple_uni_model():
    f_model = partial(CovidPredictionModelUni, 'lstm_simple_uni', lstm_simple_uni)
    model_runs = iterations()
    run_model(f_model, model_runs)


def run_lstm_enc_dec_uni_model():
    f_model = partial(CovidPredictionModelUni, 'lstm_enc_dec_uni', lstm_enc_dec)
    model_runs = iterations()
    run_model(f_model, model_runs)


def run_lstm_enc_dec_multi_model():
    f_model = partial(CovidPredictionModelMulti, 'lstm_enc_dec_multi', lstm_enc_dec)
    model_runs = iterations()
    run_model(f_model, model_runs)


def load_data():
    load_country(config.country_iso_code, download=True)


if __name__ == '__main__':
    # Menu using the ConsoleMenu package
    # Define sub-menu
    model_menu = ConsoleMenu('Choose model', {
        'daily_naive': run_naive_daily_model,
        'weekly_naive': run_naive_weekly_model,
        'uni_cnn': run_cnn_uni_model,
        'multi_cnn': run_cnn_multi_model,
        'lstm_simple_uni': run_lstm_simple_uni_model,
        'lstm_enc_dec_uni': run_lstm_enc_dec_uni_model,
        'lstm_enc_dec_multi': run_lstm_enc_dec_multi_model
    })
    # Define main menu
    menu = ConsoleMenu('Covid-19', {
        'Download data file': load_data,
        'Choose model': model_menu
    })
    menu.execute()
