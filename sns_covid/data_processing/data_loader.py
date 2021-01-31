import json
import os
from urllib.error import URLError

import requests
import pandas as pd
import matplotlib.pyplot as plt

from sns_covid import config


def check_downloadable(url):
    """
    Checks if the url contains a downloadable file
    """
    file_head = requests.head(url, allow_redirects=True)
    content_type = file_head.headers.get('content-type')
    if 'json' not in content_type.lower():
        raise URLError('Invalid Url')


def load_data(download=False):
    check_downloadable(config.data_source_url)
    file = requests.get(config.data_source_url, allow_redirects=True)
    basename = os.path.splitext(os.path.basename(config.data_source_url))[0]
    file_name = f'{config.output_directory}/{basename}.json'
    if download:
        with open(file_name, 'wb') as fin:
            fin.write(file.content)
    return file_name


def load_country(iso_code, download=False):
    file_name = load_data(download)
    with open(f'{file_name}') as json_file:
        data = json.load(json_file)
        country_data = data[iso_code]['data']
        dataframe = pd.DataFrame(country_data)
        return dataframe
