from data_loader import load_country


def main():
    # download_data('https://covid.ourworldindata.org/data/owid-covid-data.csv')
    df = load_country('owid-covid-data.json', 'GBR')
