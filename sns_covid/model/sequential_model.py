from functools import partial
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sns_covid import config
from sns_covid.logging_config import get_logger
from sns_covid.model.base_model import CovidPredictionModel
import abc

logger = get_logger(__name__)
file_logger = get_logger('file_logger')


class CovidPredictionSequentialModel(CovidPredictionModel):
    def __init__(self, model_name, f_layers, dataset):
        self.model = Sequential(name=model_name)
        self.name = model_name
        self.train_x, self.train_y = self.to_supervised(dataset.train_df)
        n_timesteps, n_features, n_outputs = self.train_x.shape[1], self.train_x.shape[2], self.train_y.shape[1]
        pf_layers = partial(f_layers, n_timesteps, n_features, n_outputs)
        # Builds layers based on the structure in model_structures
        for layer in pf_layers():
            self.model.add(layer)

    def compile(self):
        self.model.compile(loss='mse',
                           optimizer='adam')
        try:
            # Printing model structure to a logging file
            self.model.summary(print_fn=file_logger.info)
        except ValueError:
            logger.info('Unable to produce summary')

    def fit(self):
        checkpointer = ModelCheckpoint(filepath=f'data/{self.model.name}.hdf5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       verbose=0)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
        history = self.model.fit(self.train_x, self.train_y,
                                 epochs=config.epochs,
                                 batch_size=config.batch_size,
                                 callbacks=[checkpointer, es],
                                 shuffle=False,
                                 validation_split=0.2,
                                 verbose=0)
        return history

    @abc.abstractmethod
    def to_supervised(self, train):
        raise NotImplementedError
