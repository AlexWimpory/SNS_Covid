from sns_covid import config, logging_config
import pandas as pd
import numpy as np
from numpy import array_split
from numpy import array

logger = logging_config.get_logger(__name__)


def process_date(df):
    """
    Insert a column which converts the date-time to a integer from 1 to x
    """
    anchor_date = pd.to_datetime(df[config.date_column_name].iloc[0])
    df[f'{config.date_column_name}_as_days'] = (pd.to_datetime(df[config.date_column_name]) - anchor_date).dt.days
    return df


def filter_data(df):
    """
    Remove any columns in the dataframe which will not be trained on as defined in config by input_columns
    """
    df = df[df.columns[df.columns.isin(config.input_columns)]]
    # Order the columns as they appear in config
    df = df.reindex(columns=config.input_columns)
    return df


def insert_missing_dates():
    pass


def remove_nan(df):
    """
    The model can't deal with missing data (nan), so any missing entries are filled in
    Unless the column name is in fill_override back propagation and forward propagation are used with a limit of 7
    fill_override fills the entire column with linear interpolation
    """
    for column in config.fill_override:
        df[column].interpolate(method='linear', inplace=True, limit_direction='forward')
        df[column].fillna(0, inplace=True)
    # Back fill and forward fill with a limit of 7 rows
    df = df.bfill(axis='rows', limit=7).ffill(axis='rows', limit=7)
    # Drop any rows which still contain a nan
    return df.dropna(axis=0)


def smooth_data(df, column_name):
    """
    Replaces nans with 0s and calculates a rolling 7-day mean
    """
    df[f'{column_name}_smoothed_manual'] = df[column_name].replace(np.nan, 0).rolling(7).mean()
    return df


def split_data_train_test(data):
    """
    Splits the data into a training and test set.  The last 10% is the test set
    Also ensures that the training and the test set can be divided equally into weeks
    """
    n = len(data)
    # Split into train and test
    train = data[0: int(n * 0.9)]
    test = data[int(n * 0.9): n]
    # Ensure number of data entries is divisible by 7
    n = (len(train) // 7) * 7
    train = train[len(train) - n:]
    n = (len(test) // 7) * 7
    test = test[0: n]
    return train, test


def split_data_7_days(data):
    """
    Splits the data into groups of 7-days
    """
    data = array_split(data, len(data) / 7, axis=0)
    data_split = array(data)
    return data_split


def normalise_data(train, test):
    """
    Normalise the data through using the mean and the standard deviation of the training set
    """
    train_mean = train.mean()
    train_std = train.std()
    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, test


def generate_train_test(df):
    """
    Perform all of the required preprocessing techniques to the dataframe, returning a train and a test set
    """
    processed_df = filter_data(df)
    processed_df = remove_nan(processed_df)
    train_df, test_df = split_data_train_test(processed_df)
    train_df, test_df = normalise_data(train_df, test_df)
    train_df_7 = split_data_7_days(train_df)
    test_df_7 = split_data_7_days(test_df)
    logger.info(f'The training data shape is: {train_df_7.shape}')
    logger.info(f'The testing data shape is: {test_df_7.shape}')
    return train_df_7, test_df_7
