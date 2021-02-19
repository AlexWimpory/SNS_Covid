from tensorflow.python.keras.models import load_model


class ModelPredictor:
    def __init__(self, model_name):
        self._model = load_model(f'data/{model_name}.hdf5')

    def predict(self, df):
        results = self._model.predict(df)
        return results
