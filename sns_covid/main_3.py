from sns_covid import config
from sns_covid.data_processing.data_loader import load_country
from sns_covid.data_processing.data_pre_processor import generate_train_test
from sns_covid.model.model_structures import *
from sns_covid.model.model_trainer import CovidPredictionModelCNN
from sns_covid.visulisation.plotter import visualise


# Different - large
def build_model(train):
    # prepare data
    train_x, train_y = CovidPredictionModelCNN.to_supervised(train)
    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = CovidPredictionModelCNN('cnn_multi', cnn_multi(n_timesteps, n_features, n_outputs))
    # compile model
    model.compile()
    # fit network
    model.fit(train_x, train_y)
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
