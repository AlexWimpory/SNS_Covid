from matplotlib import pyplot as plt

from sns_covid import config
from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import smooth_data


def visualise_results(test_predictions, test_df):
    x = list(range(0, len(test_predictions)))
    plt.plot(x, test_df[config.label_columns])
    plt.scatter(x, test_predictions)
    plt.show()


def plot_time_indexed_data(dataframe, categories):
    fig, ax = plt.subplots(figsize=(20, 10))
    # Add x-axis and y-axis
    for category in categories:
        ax.plot(dataframe.index.values, dataframe[category], label=category)
    # Set title and labels for axes
    ax.set(xlabel="Date")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df = load_country(config.country_iso_code, download=False)
    df = smooth_data(df, 'new_deaths')
    plot_time_indexed_data(df, ['new_deaths_smoothed', 'new_deaths', 'new_deaths_smoothed_manual'])
