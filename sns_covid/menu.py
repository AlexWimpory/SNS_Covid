from consolemenu import ConsoleMenu

from sns_covid import config
from sns_covid.main import run_model
from sns_covid.model.naive_model import CovidPredictionModelNaiveDaily, CovidPredictionModelNaiveWeekly
from functools import partial
from sns_covid.model.model_structures import *
from sns_covid.data_processing.data_loader import load_data
from sns_covid.model.uni_multi_model import CovidPredictionModelUni, CovidPredictionModelMulti


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


def download_data_owid():
    load_data(config.owid_url, download=True)


def download_data_gstatic():
    load_data(config.gstatic_url, download=True)


def run_menu():
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
        'Download OWID data file': download_data_owid,
        'Download Google mobility data file': download_data_gstatic,
        'Choose model': model_menu
    })
    menu.execute()
