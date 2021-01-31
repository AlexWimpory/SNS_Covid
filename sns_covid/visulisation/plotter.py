from matplotlib import pyplot as plt

from sns_covid import config


def visualise(test_predictions, test_df):
    x = list(range(0, len(test_predictions)))
    plt.plot(x, test_df[config.label_columns])
    plt.scatter(x, test_predictions)
    plt.show()