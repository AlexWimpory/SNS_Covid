import json
import os
from urllib.error import URLError
import requests
import pandas as pd
from sns_covid import config


def check_downloadable(url):
    """
    Checks if the url contains a downloadable file
    """
    # Get the header for the file (not its contents)
    file_head = requests.head(url, allow_redirects=True)
    # Check if it is a json file
    content_type = file_head.headers.get('content-type')
    if 'json' not in content_type.lower():
        raise URLError('Invalid Url')


def load_data(download=False):
    """
    Find the file and download if necessary
    """
    basename = os.path.splitext(os.path.basename(config.data_source_url))[0]
    file_name = f'{config.output_directory}/{basename}.json'
    # Downloads the data if user wants to or if the data directory is empty
    if download or not os.path.isfile(file_name):
        # Check if the url contains a downloadable file
        check_downloadable(config.data_source_url)
        file = requests.get(config.data_source_url, allow_redirects=True)
        # Write to a new file
        with open(file_name, 'wb') as fin:
            fin.write(file.content)
    return file_name


def load_country(iso_code, download=False):
    """
    Downloads the data if needed and extracts the content for the specified country into a dataframe
    :param iso_code: code for required country, 'GBR'=United Kingdom
    :param download: Set to True to download the data
    :return: dataframe containing data
    """
    # Find the file name, if download is set to True the data will be downloaded
    file_name = load_data(download)
    # Open the file and load its contents
    with open(f'{file_name}') as json_file:
        data = json.load(json_file)
        # Find the data for the country specified in config. Bits of data (deaths, cases) that we want are in 'data'
        # whereas other data such as population and life expectancy are not in this section
        country_data = data[iso_code]['data']
        # Create a dataframe containing data
        dataframe = pd.DataFrame(country_data)
        # Create date time object from date column which is a string object
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        # Set the date time as the index for the dataframe
        dataframe.set_index('date', inplace=True)
        return dataframe
