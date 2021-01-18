import json
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt


def is_downloadable(url):
    """
    Checks if the url contains a downloadable file
    """
    file_head = requests.head(url, allow_redirects=True)
    content_type = file_head.headers.get('content-type')
    if 'json' in content_type.lower():
        return True
    else:
        print('Invalid url')
        return False


def download_data(url):
    if is_downloadable(url):
        file = requests.get(url, allow_redirects=True)
        basename = os.path.splitext(os.path.basename(url))[0]
        open(f'data/{basename}.json', 'wb').write(file.content)


def load_country(file_name, iso_code):
    with open(f'data/{file_name}') as json_file:
        data = json.load(json_file)
        country_data = data[iso_code]['data']
        dataframe = pd.DataFrame(country_data)
        return dataframe


if __name__ == '__main__':
    # download_data('https://covid.ourworldindata.org/data/owid-covid-data.json')
    df = load_country('owid-covid-data.json', 'GBR')
    df.plot(x='date', y='new_deaths', kind='line')
    plt.show()
