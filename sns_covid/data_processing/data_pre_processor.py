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


def smooth_data(df, column_name):
    """
    Replaces nans with 0s and calculates a rolling 7-day mean
    """
    df[f'{column_name}_smoothed_manual'] = df[column_name].replace(np.nan, 0).rolling(7).mean()
    return df


class Dataset:
    def __init__(self, df):
        self.df = df
        self.train_df, self.test_df = self.__generate_train_test()

    def __generate_train_test(self):
        """
        Perform all of the required preprocessing techniques to the dataframe, returning a train and a test set
        :return: Preprocessed train and test set
        """
        processed_df = self.__filter_data(self.df)
        processed_df = self.__remove_nan(processed_df)
        train_df, test_df = self.__split_data_train_test(processed_df)
        train_df, test_df = self.__standardise_data(train_df, test_df)
        train_df_7 = self.__split_data_7_days(train_df)
        test_df_7 = self.__split_data_7_days(test_df)
        logger.info(f'The training data shape is: {train_df_7.shape}')
        logger.info(f'The testing data shape is: {test_df_7.shape}')
        return train_df_7, test_df_7

    def __normalise_data(self, train, test):
        """
        Normalise the data through using the max and the min values of the training set
        """
        self.train_max = train.max()
        self.train_min = train.min()
        norm_train = (train - self.train_min) / (self.train_max - self.train_min)
        norm_test = (test - self.train_min) / (self.train_max - self.train_min)
        return norm_train, norm_test

    def denormalise_data(self, ndarray):
        """
        Reverse the normalisation process
        """
        denorm_ndarray = (ndarray * (self.train_max[config.output_column] - self.train_min[config.output_column])) + self.train_min[config.output_column]
        return denorm_ndarray

    def __standardise_data(self, train, test):
        """
        Standardise the data through using the mean and the standard deviation of the training set
        """
        self.train_mean = train.mean()
        self.train_std = train.std()
        norm_train = (train - self.train_mean) / self.train_std
        norm_test = (test - self.train_mean) / self.train_std
        return norm_train, norm_test

    def destandardise_data(self, ndarray):
        """
        Reverse the standardisation process
        """
        denorm_ndarray = (ndarray * self.train_std[config.output_column]) + self.train_mean[config.output_column]
        return denorm_ndarray

    @staticmethod
    def __split_data_7_days(data):
        """
        Splits the data into groups of 7-days
        """
        data = array_split(data, len(data) / 7, axis=0)
        data_split = array(data)
        return data_split

    @staticmethod
    def __split_data_train_test(data):
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

    @staticmethod
    def __remove_nan(df):
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

    @staticmethod
    def __filter_data(df):
        """
        Remove any columns in the dataframe which will not be trained on as defined in config by input_columns
        """
        df = df[df.columns[df.columns.isin(config.input_columns)]]
        # Ensure that the output column in config is the first column of the dataframe
        process_columns = config.input_columns.copy()
        process_columns.remove(config.output_column)
        process_columns.insert(0, config.output_column)
        df = df.reindex(columns=process_columns)
        logger.info(f'Columns to be trained on: {config.input_columns}')
        return df
