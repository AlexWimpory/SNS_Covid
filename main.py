from data_loader import load_country
import matplotlib.pyplot as plt


def main():
    # download_data('https://covid.ourworldindata.org/data/owid-covid-data.json')
    df = load_country('owid-covid-data.json', 'GBR')
    df.plot(x='date', y='new_deaths', kind='line')
    plt.show()
