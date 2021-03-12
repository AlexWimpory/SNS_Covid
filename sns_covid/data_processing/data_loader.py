import json
import os
import zipfile
from urllib.error import URLError
import requests
import pandas as pd
from sns_covid import config, logging_config

logger = logging_config.get_logger(__name__)


def check_downloadable(url, extension):
    """
    Checks if the url contains a downloadable file
    """
    # Get the header for the file (not its contents)
    logger.info(f'Checking download')
    file_head = requests.head(url, allow_redirects=True)
    # Check if it is a json file
    content_type = file_head.headers.get('content-type')
    if extension[1:] not in content_type.lower():
        raise URLError('Invalid Url')


def load_data(url, download=False):
    """
    Find the file and download if necessary
    """
    basename, extension = os.path.splitext(os.path.basename(url))
    file_name = f'{config.output_directory}/{basename}{extension}'
    # Downloads the data if user wants to or if the data directory is empty
    if download or not os.path.isfile(file_name):
        # Check if the url contains a downloadable file
        check_downloadable(url, extension)
        logger.info(f'Downloading data from {url}')
        file = requests.get(url, allow_redirects=True)
        # Write to a new file
        with open(file_name, 'wb') as fin:
            fin.write(file.content)
    else:
        logger.info(f'Data not downloaded from {url}')
    return file_name


def load_country_owid(iso_code, download=False):
    """
    Downloads the data if needed and extracts the content for the specified country into a dataframe from the json file
    :param iso_code: code for required country, 'GBR'=United Kingdom
    :param download: Set to True to download the data
    :return: dataframe containing data
    """
    # Find the file name, if download is set to True the data will be downloaded
    file_name = load_data(config.owid_url, download)
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
        # Set frequency of dataframe to daily to fill in missing dates with nan
        dataframe.asfreq('d')
        logger.info(f'Data loaded for {iso_code}')
        return dataframe


def load_country_gstatic(alpha_2_code, download=False):
    """
    Downloads the data if needed and extracts the content for the specified country into a dataframe from the zip file
    :param alpha_2_code: code for required country, 'GB'=United Kingdom
    :param download: Set to True to download the data
    :return: dataframe containing data
    """
    # Find the file name, if download is set to True the data will be downloaded
    file_name = load_data(config.gstatic_url, download)
    country_file_name = config.gstatic_zip_name.format(code=alpha_2_code)
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        extracted_file_name = zip_ref.extract(country_file_name, path=config.output_directory)
    # Open the file and load its contents
    dataframe = pd.read_csv(extracted_file_name)
    dataframe = dataframe.loc[dataframe[config.gstatic_filter].isnull()]
    # Create date time object from date column which is a string object
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    # Set the date time as the index for the dataframe
    dataframe.set_index('date', inplace=True)
    # Set frequency of dataframe to daily to fill in missing dates with nan
    dataframe.asfreq('d')
    logger.info(f'Data loaded for {alpha_2_code}')
    return dataframe
