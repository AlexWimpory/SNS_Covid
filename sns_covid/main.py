from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import generate_train_test
from sns_covid.model.model_structures import *



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
    train_df, test_df = generate_train_test(df)
    print('sss')


if __name__ == '__main__':
    main()
