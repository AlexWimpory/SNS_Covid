from sns_covid import config
import pandas as pd


def process_date(df):
    anchor_date = pd.to_datetime(df[config.date_column_name].iloc[0])
    df[f'{config.date_column_name}_as_days'] = (pd.to_datetime(df[config.date_column_name]) - anchor_date).dt.days
    return df


def filter_data(df):
    return df[df.columns[df.columns.isin(config.columns_used)]]


def insert_missing_dates():
    pass


def remove_nan():
    pass


def save_data():
    pass


def split_data(df, start_at, end_at):
    n = end_at - start_at
    train_df = df[start_at:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):end_at]
    return train_df, val_df, test_df


def normalise_data(train, val, test):
    # TODO Use moving averages
    train_mean = train.mean()
    train_std = train.std()
    train = (train - train_mean) / train_std
    val = (val - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, val, test
