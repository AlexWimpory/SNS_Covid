from sns_covid import config
from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import generate_train_test
from sns_covid.model.model_structures import *
from sns_covid.model.model_trainer import *
from sns_covid.visulisation.plotter import visualise


def build_model(train):
    # model = CovidPredictionModelCNNMulti('cnn_multi', cnn_multi, train)
    # model = CovidPredictionModelCNNUni('cnn_uni', cnn_uni, train)
    # model = CovidPredictionModelNaiveDaily('daily_naive', train)
    model = CovidPredictionModelNaiveWeekly('weekly_naive', train)
    model.compile()
    model.fit()
    return model


def main():
    # Load the data into a dataframe
    df = load_country(config.country_iso_code, download=False)
    # Generate the train, validation and test dataframes
    train_df, test_df = generate_train_test(df)
    model = build_model(train_df)
    score, scores = model.evaluate_model(train_df, test_df)
    visualise(model.name, score, scores)


if __name__ == '__main__':
    main()
