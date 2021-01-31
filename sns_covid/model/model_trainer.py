from tensorflow.python.keras import Sequential
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

from sns_covid import config



class CovidPredictionModel:
    def __init__(self, model_name, layers):
        self.model = Sequential(name=model_name)
        # Builds layers based on the structure in model_structures
        for layer in layers:
            self.model.add(layer)

    def compile(self):
        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])
        # self.model.summary()

    def test(self, dataframe):
        """Calculate the model's accuracy on the input dataset"""
        score = self.model.evaluate(dataframe, verbose=0)
        return score

    def fit(self, train_ds, val_ds, patience=2):
        checkpointer = ModelCheckpoint(filepath=f'data/{self.model.name}.hdf5',
                                       verbose=1,
                                       save_best_only=True)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')
        history = self.model.fit(train_ds,
                                 epochs=config.epochs,
                                 validation_data=val_ds,
                                 callbacks=[early_stopping, checkpointer])
        return history


def train_and_test_model(features, model):
    pass


def trainer():
    pass
