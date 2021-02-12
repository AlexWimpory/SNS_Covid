from sns_covid import config
import pandas as pd
import numpy as np

def process_date(df):
    anchor_date = pd.to_datetime(df[config.date_column_name].iloc[0])
    df[f'{config.date_column_name}_as_days'] = (pd.to_datetime(df[config.date_column_name]) - anchor_date).dt.days
    return df


def filter_data(df):
    return df[df.columns[df.columns.isin(config.columns_used)]]


def insert_missing_dates():
    pass


def remove_nan(df):
    df = df.bfill(axis='rows', limit=7).ffill(axis='rows', limit=7)
    return df.dropna(axis=0)


def smooth_data(df, column_name):
    df[f'{column_name}_smoothed_manual'] = df[column_name].replace(np.nan, 0).rolling(7).mean()
    return df


def save_data():
    pass


def split_data(df):
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]
    return train_df, val_df, test_df


def normalise_data(train, val, test):
    # TODO Use moving averages
    train_mean = train.mean()
    train_std = train.std()
    train = (train - train_mean) / train_std
    val = (val - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, val, test


def generate_train_val_test(df):
    processed_df = filter_data(df)
    processed_df = remove_nan(processed_df)
    train_df, val_df, test_df = split_data(processed_df)
    return normalise_data(train_df, val_df, test_df)


def generate_train_val_test_2(df):
    processed_df = filter_data(df)
    processed_df = remove_nan(processed_df)
    train_df, val_df, test_df = split_data(processed_df)
    train_df, val_df, test_df = normalise_data(train_df, val_df, test_df)
    return pd.concat([train_df, val_df, test_df])
