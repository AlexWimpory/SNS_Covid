from functools import partial
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from sns_covid import config
from sns_covid.model.base_model import CovidPredictionModel
from sns_covid.visulisation.plotter import plot_loss
import abc


class CovidPredictionSequentialModel(CovidPredictionModel):
    def __init__(self, model_name, f_layers, train):
        self.model = Sequential(name=model_name)
        self.name = model_name
        self.train_x, self.train_y = self.to_supervised(train)
        n_timesteps, n_features, n_outputs = self.train_x.shape[1], self.train_x.shape[2], self.train_y.shape[1]
        pf_layers = partial(f_layers, n_timesteps, n_features, n_outputs)
        # Builds layers based on the structure in model_structures
        for layer in pf_layers():
            self.model.add(layer)

    def compile(self):
        self.model.compile(loss='mse',
                           optimizer='adam')
        try:
            self.model.summary()
        except ValueError:
            print('Unable to produce summary')

    def fit(self):
        checkpointer = ModelCheckpoint(filepath=f'data/{self.model.name}.hdf5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       verbose=1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        history = self.model.fit(self.train_x, self.train_y,
                                 epochs=config.epochs,
                                 batch_size=config.batch_size,
                                 callbacks=[checkpointer, es],
                                 shuffle=False,
                                 validation_split=0.2)
        plot_loss(history)

    @abc.abstractmethod
    def to_supervised(self, train):
        raise NotImplementedError
