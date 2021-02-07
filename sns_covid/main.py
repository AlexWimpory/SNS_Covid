from sns_covid import config
from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import generate_train_val_test
from sns_covid.model.data_generator import DataGenerator
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
    # Generate the train, validation and test dataframes
    train_df, val_df, test_df = generate_train_val_test(df)
    # Turn the data into tensorflow readable data
    data_generator = DataGenerator(train_df.columns)
    train_ds, val_ds, test_ds = data_generator.make_datasets(train_df, val_df, test_df)
    # Now for the model
    model = CovidPredictionModel('dense', dense())
    model.compile()
    model.fit(train_ds, val_ds)
    # Finally evaluate the model
    print(model.test(val_ds))
    print(model.test(test_ds))
    # Do some predictions
    predictor = ModelPredictor('dense')
    test_predictions = predictor.predict(test_df)
    visualise_results(test_predictions, test_df)


if __name__ == '__main__':
    main()
