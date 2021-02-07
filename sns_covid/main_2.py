from sns_covid import config
from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import generate_train_val_test
from sns_covid.model.data_generator import DataGenerator
from sns_covid.model.data_generator_2 import WindowGenerator
from sns_covid.model.model_predictor import ModelPredictor
from sns_covid.model.model_structures import *
from sns_covid.model.model_trainer import CovidPredictionModel
from sns_covid.visulisation.plotter import visualise_results


# TODO Set frequency on dataframe to fill in any gaps (prob aren't any but good practice)(uses date column)
# TODO Reverse normalisation
# TODO More complex model
# TODO Logging
# TODO Predictions
# TODO Command line actions
# TODO Document code

def main():
    # Load the data into a dataframe
    df = load_country(config.country_iso_code, download=False)
    train_df, val_df, test_df = generate_train_val_test(df)

    # CONV_WIDTH = 7
    # LABEL_WIDTH = 7
    # INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    wide_conv_window = WindowGenerator(
        input_width=7,
        label_width=7,
        shift=7,
        label_columns=['new_deaths_smoothed'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df)
    # Generate the train, validation and test dataframes
    # Turn the data into tensorflow readable data
    # Now for the model
    model = CovidPredictionModel('multi_dense', multi_dense())
    model.compile()
    model.fit(wide_conv_window.train, wide_conv_window.val)
    # Finally evaluate the model
    print(model.test(wide_conv_window.val))
    print(model.test(wide_conv_window.test))
    wide_conv_window.plot(model.model)



if __name__ == '__main__':
    main()
