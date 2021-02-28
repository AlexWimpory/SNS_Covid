from sns_covid import config
import pandas as pd
import numpy as np
from numpy import array_split
from numpy import array


def process_date(df):
    anchor_date = pd.to_datetime(df[config.date_column_name].iloc[0])
    df[f'{config.date_column_name}_as_days'] = (pd.to_datetime(df[config.date_column_name]) - anchor_date).dt.days
    return df


def filter_data(df):
    df = df[df.columns[df.columns.isin(config.input_columns)]]
    df = df.reindex(columns=config.input_columns)
    return df


def insert_missing_dates():
    pass


def remove_nan(df):
    df = df.bfill(axis='rows', limit=7).ffill(axis='rows', limit=7)
    return df.dropna(axis=0)


def smooth_data(df, column_name):
    df[f'{column_name}_smoothed_manual'] = df[column_name].replace(np.nan, 0).rolling(7).mean()
    return df


def split_data_train_test(data):
    n = len(data)
    train = data[0: int(n * 0.9)]
    test = data[int(n * 0.9): n]
    n = (len(train) // 7) * 7
    train = train[len(train) - n:]
    n = (len(test) // 7) * 7
    test = test[0: n]
    return train, test


def split_data_7_days(data):
    data = array_split(data, len(data) / 7, axis=0)
    data_split = array(data)
    return data_split


def normalise_data(train, test):
    train_mean = train.mean()
    train_std = train.std()
    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, test


def generate_train_test(df):
    processed_df = filter_data(df)
    processed_df = remove_nan(processed_df)
    train_df, test_df = split_data_train_test(processed_df)
    train_df, test_df = normalise_data(train_df, test_df)
    train_df_7 = split_data_7_days(train_df)
    test_df_7 = split_data_7_days(test_df)
    print(train_df_7.shape)
    print(test_df_7.shape)
    return train_df_7, test_df_7
