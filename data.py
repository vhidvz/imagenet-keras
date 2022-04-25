import glob
import math

import numpy as np
import tensorflow as tf

from tensorflow import keras


def load_imagenet_dataset(directory, image_size=(224, 224), batch_size=32, mode: "tf" or "torch" = "tf", **kwargs):
    def resize_with_crop(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, *image_size)
        image = tf.keras.applications.imagenet_utils.preprocess_input(
            image, mode=mode)
        return (image, label)

    ds = keras.utils.image_dataset_from_directory(
        directory, image_size=image_size, batch_size=batch_size, **kwargs)
    ds = ds.map(resize_with_crop)
    return ds


class ImageNetSequence(keras.utils.Sequence):
    def __init__(self, X_set, y_set, batch_size, image_size=(224, 224), mode: "tf" or "torch" = "tf"):
        self.X, self.y = X_set, y_set
        self.batch_size = batch_size
        self.image_size = image_size
        self.mode = mode

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = [tf.io.read_file(image_path) for image_path in batch_X]
        X = [tf.image.decode_jpeg(image, channels=3) for image in X]

        X = [tf.cast(image, tf.float32) for image in X]
        X = [tf.image.resize_with_crop_or_pad(
            image, *self.image_size) for image in X]
        X = [keras.applications.imagenet_utils.preprocess_input(
            image, mode=self.mode) for image in X]

        return np.array(X), np.array(batch_y)


if __name__ == '__main__':
    # tf.data.Dataset
    imagenet_train_ds = load_imagenet_dataset(
        './imagenet/train', crop_to_aspect_ratio=True)
    imagenet_valid_ds = load_imagenet_dataset(
        './imagenet/valid', crop_to_aspect_ratio=False)

    batches_train = list(imagenet_train_ds.as_numpy_iterator())
    batches_valid = list(imagenet_valid_ds.as_numpy_iterator())

    # keras.utils.Sequence
    data_train = glob.glob('./imagenet/train/n*/*.JPEG')
    data_valid = glob.glob('./imagenet/valid/n*/*.JPEG')

    def export(data):
        X, y = [], []
        for datum in data:
            X.append(datum)
            y.append(datum.split('/')[-2])
        return X, y

    X_train, y_train = export(data_train)
    X_valid, y_valid = export(data_valid)

    imagenet_train_seq = ImageNetSequence(X_train, y_train, batch_size=32)
    imagenet_valid_seq = ImageNetSequence(X_valid, y_valid, batch_size=32)

    def export_seq(seq):
        batches = []
        for i in range(len(seq)):
            batches.append(seq[i])
        return batches

    batches_train_seq = export_seq(imagenet_train_seq)
    batches_valid_seq = export_seq(imagenet_valid_seq)
