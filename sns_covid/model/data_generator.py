import tensorflow as tf
import numpy as np

from sns_covid import config


class DataGenerator:

    def __init__(self, columns):
        # Work out the window parameters.
        self.input_width = config.input_width
        self.label_columns = config.label_columns
        self.label_width = len(self.label_columns)
        self.shift = config.shift
        # Create the slices
        self.input_slice = slice(0, self.input_width)
        self.labels_slice = slice(self.input_width + self.shift - self.label_width, None)
        self.column_indices = {name: i for i, name in enumerate(columns)}

    # This is a function that is passed to the tensorflow engine and generates the split data
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = tf.stack(
            [
                features[:, self.labels_slice, :]
                [:, :, self.column_indices[name]
            ] for name in self.label_columns],
            axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    # This function makes a tensorflow dataset
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width + self.shift,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )
        ds = ds.map(self.split_window)
        return ds

    # Just a helper function for converting test/train/validate data
    def make_datasets(self, data1, data2, data3):
        return self.make_dataset(data1), self.make_dataset(data2), self.make_dataset(data3)