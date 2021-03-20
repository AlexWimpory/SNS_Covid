from sns_covid.data_processing.data_loader import load_country_owid, load_country_gstatic
from sns_covid.data_processing.data_pre_processor import Dataset
from sns_covid.logging_config import get_logger
from sns_covid.model.uni_multi_model import *
from sns_covid.visulisation.plotter import visualise, plot_loss, plot_prediction_vs_actual, print_scores

logger = get_logger(__name__)
file_logger = get_logger('file_logger')


def run_model(f_model, model_runs=1):
    # Load the data into a dataframe
    df_owid = load_country_owid(config.country_iso_code)
    df_gstatic = load_country_gstatic(config.country_alpha_2_code)
    df_all = df_owid.join(df_gstatic)
    # Generate the train and test dataframes
    dataset = Dataset(df_all)
    results = []
    for i in range(0, model_runs):
        result = dict()
        model = f_model(dataset)
        result['model_name'] = model.name
        model.compile()
        logger.info(f'Training model {i+1}')
        result['history'] = model.fit()
        # Test the model
        evaluate_results = model.evaluate_model(dataset)
        result = capture_evaluate(result, *evaluate_results)
        results.append(result)
        print_scores(logger, file_logger, result['model_name'], result['score'],
                     result['scores'], f'RMSE for model {i+1}: ')
    plot_best_result(results)


def capture_evaluate(result, score, scores, predictions, actual):
    """
    Remember score,scores, actual and predictions for a model
    """
    result['score'] = score
    result['scores'] = scores
    result['predictions'] = predictions
    result['actual'] = actual
    return result


def plot_best_result(results):
    """
    Plots:
    1. Loss during training
    2. The daily forecast error
    3. A comparison of the predictions to the actual value
    For the best model
    """
    best_result = None
    for result in results:
        if not best_result or best_result['score'] > result['score']:
            best_result = result
    plot_loss(best_result['history'])
    print_scores(logger, file_logger, best_result['model_name'],
                 best_result['score'], best_result['scores'], 'Best model: ')
    visualise(best_result['scores'])
    plot_prediction_vs_actual(best_result['predictions'], best_result['actual'])


if __name__ == '__main__':
    from sns_covid.menu import run_menu
    run_menu()
