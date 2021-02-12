from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint

from sns_covid import config
from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import generate_train_val_test, generate_train_val_test_2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def main():
    df = load_country(config.country_iso_code, download=False)
    train_df, val_df, test_df = generate_train_val_test(df)
    test_df_2 = generate_train_val_test_2(df)
    time_steps = 14
    # reshape to [samples, time_steps, n_features]
    X_train, y_train = create_dataset(train_df, train_df.new_deaths_smoothed, time_steps)
    X_val, y_val = create_dataset(val_df, val_df.new_deaths_smoothed, time_steps)
    X_test, y_test = create_dataset(test_df, test_df.new_deaths_smoothed, time_steps)
    X_test_2, y_test_2 = create_dataset(test_df_2, test_df_2.new_deaths_smoothed, time_steps)

    print(X_train.shape, y_train.shape)

    #### RNN ######
    model = keras.Sequential()

    model.add(
            keras.layers.LSTM(
                units=128,
                input_shape=(X_train.shape[1], X_train.shape[2])
            )
        )
    model.add(keras.layers.Dropout(rate=0.2))

    # model.add(
    #     keras.layers.LSTM(
    #         units=128,
    #         return_sequences=True,
    #         input_shape=(X_train.shape[1], X_train.shape[2])
    #     )
    # )
    # model.add(keras.layers.Dropout(rate=0.2))
    # model.add(keras.layers.LSTM(units=128))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=1))

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.metrics.MeanAbsoluteError()])
    checkpointer = ModelCheckpoint(filepath='data/test.hdf5',
                                   verbose=1,
                                   save_best_only=True)
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_data=(X_val, y_val),
        shuffle=False,
        callbacks=[checkpointer]
    )

    y_pred = model.predict(X_test)
    df = test_df.iloc[14:]
    plt.plot(df.index.values, y_pred, label='prediction')
    plt.plot(df.index.values, df.new_deaths_smoothed, label='actual')
    plt.legend()
    plt.show()

    # y_pred = model.predict(X_test_2)
    # df = test_df_2.iloc[14:]
    # plt.plot(df.index.values, y_pred, label='prediction')
    # plt.plot(df.index.values, df.new_deaths_smoothed, label='actual')
    # plt.legend()
    # plt.show()

    ###### Dense ######
    #
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=1)
    # ])
    #
    #
    # model.compile(loss=tf.losses.MeanSquaredError(), optimizer='adam', metrics=[tf.metrics.MeanAbsoluteError()])
    # checkpointer = ModelCheckpoint(filepath='data/test.hdf5',
    #                                verbose=1,
    #                                save_best_only=True)
    # history = model.fit(
    #     X_train, y_train,
    #     epochs=1000,
    #     batch_size=32,
    #     validation_data=(X_val, y_val),
    #     shuffle=False,
    #     callbacks=[checkpointer]
    # )
    #
    # y_pred = model.predict(test_df)
    # plt.plot(test_df.index.values, y_pred, label='prediction')
    # plt.plot(test_df.index.values, test_df.new_deaths_smoothed, label='actual')
    # plt.legend()
    # plt.show()
    # y_pred_2 = model.predict(val_df)
    # plt.plot(val_df.index.values, y_pred_2, label='prediction')
    # plt.plot(val_df.index.values, val_df.new_deaths_smoothed, label='actual')
    # plt.legend()
    # plt.show()




if __name__ == '__main__':
    main()
