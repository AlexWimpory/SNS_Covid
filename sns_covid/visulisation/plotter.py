from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sns_covid import config
from sns_covid.data_processing.data_loader import load_country_owid
from sns_covid.data_processing.data_pre_processor import smooth_data
import seaborn as sns
import numpy as np


def visualise(scores):
    """
    Plot RMSE scores
    """
    # plot scores
    days = ['1', '2', '3', '4', '5', '6', '7']
    plt.plot(days, scores, marker='o', label='cnn')
    plt.show()


def print_scores(logger, file_logger, model_name, score, scores, prefix):
    """
    Print RMSE scores to the terminal and to file
    """
    # summarize scores
    s_scores = ', '.join(['%.3f' % s for s in scores])
    logger.info('%s %s: [%.3f] %s' % (prefix, model_name, score, s_scores))
    file_logger.info('%s %s: [%.3f] %s' % (prefix, model_name, score, s_scores))


def plot_time_indexed_data(dataframe, categories):
    fig, ax = plt.subplots(figsize=(20, 10))
    # Add x-axis and y-axis
    for category in categories:
        ax.plot(dataframe.index.values, dataframe[category], label=category)
    # Set title and labels for axes
    ax.set(xlabel="Date")
    plt.legend()
    plt.savefig('graph.png')
    plt.show()


def plot_correlation(dataframe, column):
    """
    Plot the auto-correlation for a column of data
    """
    dataframe = dataframe[column].dropna(axis=0)
    plot_acf(dataframe, lags=50)
    plot_pacf(dataframe, lags=50)
    plt.show()


def show_heatmap(data):
    """
    Plot a correlation heatmap
    """
    corr = data.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        annot=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.show()


def plot_loss(history):
    """
    Plot the training and the validation loss
    """
    if history is None:
        return
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history.history['val_loss'], marker='o', markersize=3, label='val')
    plt.plot(history.history['loss'], marker='o', markersize=3, label='train')
    plt.legend()
    plt.grid()
    plt.show()


def plot_prediction_vs_actual(prediction, actual):
    """
    PLot model predictions vs the expected values
    """
    shape = prediction.shape
    prediction.reshape(shape[0], shape[1])
    plt.plot(actual.flatten(), marker='o')
    x = np.array(range(0, shape[1]))
    for i in range(0, shape[0]):
        plt.plot(x, prediction[i], marker='o', color='orange')
        x = x + shape[1]
    plt.legend(['actual', 'predictions'])
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # Overriding config as this module is at a different level and can't find the data
    # Could implement something more complicated but not worth the time
    config.output_directory = '../data'
    df = load_country_owid(config.country_iso_code, download=False)
    df = smooth_data(df, 'new_deaths')
    plot_time_indexed_data(df, ['new_deaths_smoothed', 'new_deaths', 'new_deaths_smoothed_manual'])
    plot_correlation(df, 'new_deaths_smoothed')
    show_heatmap(df[df.columns[df.columns.isin(config.input_columns)]])
