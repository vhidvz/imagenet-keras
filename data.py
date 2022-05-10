import glob
import math
import random

import numpy as np
import tensorflow as tf

from tensorflow import keras


def feature(type: "byte" or "float" or "int", key, value):
    if type == "byte":
        return {key: tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))}
    elif type == "float":
        return {key: tf.train.Feature(float_list=tf.train.FloatList(value=[value]))}
    elif type == "int":
        return {key: tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))}


def example(*features):
    feature = {}
    for key, value in features.items():
        feature.update({key: value})
    return tf.train.Example(features=tf.train.Features(feature=feature))


def sorted_labels(directory):
    labels = glob.glob(directory + "/*")
    labels.sort()
    return labels


def get_class_num(class_name, labels):
    for i, label in enumerate(labels):
        if class_name in label:
            return i + 1  # 1-indexed


def twisted_pair(pattern, size=None):
    files = glob.glob(pattern)
    random.shuffle(files)

    if not size:
        size = len(files)

    result = []
    for i in range(size):
        while True:
            file = random.choice(files)
            if not files[i] == file:
                result.append((files[i], file))
                break

    return result


def convert_twisted_pair(pairs, labels):
    result = []
    for pair in pairs:
        pool = []
        for item in pair:
            file_path = item
            class_name = file_path.split("/")[-1]
            class_label = get_class_num(class_name, labels)
            pool.append((file_path, class_label))
        result.append(pool)
    return result


def converted_twisted_pair_to_example(pairs):
    for pair in pairs:
        result = []
        for i in range(len(pair)):
            file_path, class_label = pair[i]
            result.append(feature("byte", "image"+str(i+1),
                          tf.io.decode_image(file_path)))
            result.append(feature("int", "label"+str(i+1), class_label))
        yield example(*result)


def converted_twisted_pair_to_tfrecord(ds_example, filename, size=None):
    if not size:
        with tf.io.TFRecordWriter(filename + '.tfrecord') as tf_writer:
            for example in ds_example:
                tf_writer.write(example.SerializeToString())
    else:
        i = 0
        count = 0
        tf_writer = tf.io.TFRecordWriter(filename + '_' + str(i) + '.tfrecord')
        for example in ds_example:
            tf_writer.write(example.SerializeToString())
            count += 1
            if count == size:
                i += 1
                count = 0
                tf_writer = tf.io.TFRecordWriter(
                    filename + '_' + str(i) + '.tfrecord')


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
