from sns_covid import config
from sns_covid.data_processing.data_loader import load_country, load_data
from sns_covid.data_processing.data_pre_processor import process_date, split_data, filter_data, normalise_data
from sns_covid.model.model_structures import model_1
from sns_covid.model.model_trainer import CovidPredictionModel
from sns_covid.model.window_generator import WindowGenerator
import tensorflow as tf
from matplotlib import pyplot as plt

# TODO Keep days in dataframe but don't train on them for plotting graphs
# TODO Set frequency on dataframe to fill in any gaps (prob aren't any but good practice)(uses date column)
# TODO Decide on what to do with nan - maybe delete some at end/start with little data otherwise use LOCF/NOCB/linear interpolation/ spline interpolation
# TODO Reverse normalisation
# TODO maybe save the data as so the modified training data can be seen afterwards

MAX_EPOCHS = 100


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def main():
    data_file_name = load_data()
    df = load_country(data_file_name, config.country_iso_code)
    processed_df = filter_data(process_date(df))
    train_df, val_df, test_df = split_data(processed_df, 62, 340)
    train_df, val_df, test_df = normalise_data(train_df, val_df, test_df)
    w1 = WindowGenerator(input_width=7, label_width=1, shift=1,
                         train_df=train_df, val_df=val_df, test_df=test_df,
                         label_columns=['new_deaths'])
    linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
    history = compile_and_fit(linear, w1)
    val_performance = {}
    performance = {}
    val_performance['Linear'] = linear.evaluate(w1.val)
    performance['Linear'] = linear.evaluate(w1.test)
    print(performance)

    test_predictions = linear.predict(test_df)
    x = list(range(0, len(test_predictions)))
    plt.plot(x, test_df['new_deaths'].tolist())
    plt.scatter(x, test_predictions)
    plt.show()

    # model = CovidPredictionModel('model_1', model_1())


if __name__ == '__main__':
    main()
