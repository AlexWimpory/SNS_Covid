from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sns_covid import config
from sns_covid.data_processing.data_loader import load_country_owid, load_country_gstatic
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
    plt.title('RMSE for each of the 7 Days')
    plt.grid()
    plt.xlabel('Day')
    plt.ylabel('RMSE')
    plt.savefig(f'{config.output_directory}/RMSE.png')
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
    plt.legend(fontsize='x-large')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f'{config.output_directory}/graph.png')
    plt.show()


def plot_correlation(dataframe, column):
    """
    Plot the auto-correlation for a column of data
    """
    dataframe = dataframe[column].dropna(axis=0)
    plot_pacf(dataframe, lags=50)
    plot_acf(dataframe, lags=50)
    plt.savefig(f'{config.output_directory}/auto_correlation.png')
    plt.show()


def show_heatmap(data):
    """
    Plot a correlation heatmap
    """
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        annot_kws={'size': 8},
        square=True,
        annot=True,
        ax=ax,
        cbar_kws={'shrink': 0.6}
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    plt.tight_layout()
    plt.savefig(f'{config.output_directory}/heatmap.png')
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
    plt.title('Learning Curves')
    plt.savefig(f'{config.output_directory}/loss.png')
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
    plt.title('Predictions vs Actual Values')
    plt.xlabel('Day')
    plt.ylabel('New Deaths')
    plt.savefig(f'{config.output_directory}/pred.png')
    plt.show()


if __name__ == '__main__':
    # Overriding config as this module is at a different level and can't find the data
    # Could implement something more complicated but not worth the time
    config.output_directory = '../data'
    df_owid = load_country_owid(config.country_iso_code, download=False)
    df_gstatic = load_country_gstatic(config.country_alpha_2_code, download=False)
    df_all = df_owid.join(df_gstatic)
    df = smooth_data(df_all, 'new_deaths')
    plot_time_indexed_data(df_all, ['new_deaths_smoothed', 'new_deaths', 'new_deaths_smoothed_manual'])
    plot_correlation(df_all, 'new_deaths_smoothed')
    show_heatmap(df_all[df_all.columns[df_all.columns.isin(config.input_columns)]])
