import glob
import random
import numpy as np

import tensorflow as tf

from tensorflow import keras


def to_feature(type: "byte" or "float" or "int", key, value):
    if type == "byte":
        return {key: tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))}
    elif type == "float":
        return {key: tf.train.Feature(float_list=tf.train.FloatList(value=[value]))}
    elif type == "int":
        return {key: tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))}


def features_to_example(*features):
    feature = {}
    for item in features:
        feature.update(item)
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
            if not files[i % len(files)] == file:
                result.append((files[i % len(files)], file))
                break

    return result


def convert_twisted_pair(pairs, labels):
    result = []
    for pair in pairs:
        pool = []
        for item in pair:
            file_path = item
            class_name = file_path.split("/")[-2]
            class_label = get_class_num(class_name, labels)
            pool.append((file_path, class_label))
        result.append(pool)
    return result


def converted_twisted_pairs_to_example(pairs):
    for pair in pairs:
        result = []
        for i in range(len(pair)):
            file_path, class_label = pair[i]
            result.append(to_feature("byte", "image"+str(i+1),
                          open(file_path, 'rb').read()))
            result.append(to_feature("int", "label"+str(i+1), class_label))
        yield features_to_example(*result)


def write_examples_to_tfrecord(ds_example, filename, batch_size=None):
    if not batch_size:
        with tf.io.TFRecordWriter(filename + '.tfrecord') as tf_writer:
            for data in ds_example:
                tf_writer.write(data.SerializeToString())
    else:
        i = 0
        count = 0
        tf_writer = None
        for data in ds_example:
            if not count % batch_size:
                i += 1
                count = 0
                if tf_writer:
                    tf_writer.flush()
                    tf_writer.close()
                tf_writer = tf.io.TFRecordWriter(
                    filename + '_' + '{0:04d}'.format(i) + '.tfrecord')
            tf_writer.write(data.SerializeToString())
            count += 1


if __name__ == "__main__":
    labels = sorted_labels("imagenet/train")

    print(get_class_num("n01484850", labels))
    print(get_class_num("n02281787", labels))
    print(get_class_num("n03028079", labels))

    pairs = twisted_pair("imagenet/train/*/*.JPEG")
    pairs = twisted_pair("imagenet/train/*/*.JPEG", size=10)
    pairs = twisted_pair("imagenet/train/*/*.JPEG", size=1000)

    converted_pairs = convert_twisted_pair(pairs, labels)

    ds_example = converted_twisted_pairs_to_example

    write_examples_to_tfrecord(
        ds_example(converted_pairs), "imagenet/train")
    write_examples_to_tfrecord(
        ds_example(converted_pairs), "imagenet/train", batch_size=100)

    # read tf records
    filenames = glob.glob('imagenet/train/train_*.tfrecord')
    raw_dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        'image1': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image2': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label2': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    def _parse_function(example, shape=(480, 480), to_categorical=None):
        example = tf.io.parse_single_example(example, feature_description)

        image1 = tf.image.resize_with_crop_or_pad(
            tf.io.decode_jpeg(example['image1']), *shape)
        image2 = tf.image.resize_with_crop_or_pad(
            tf.io.decode_jpeg(example['image2']), *shape)

        label1 = example['label1'] if not to_categorical else keras.utils.to_categorical(
            example['label1'], num_classes=to_categorical)
        label2 = example['label2'] if not to_categorical else keras.utils.to_categorical(
            example['label2'], num_classes=to_categorical)

        return (image1/255.0, image2/2550.0), tf.stack((label1, label2), axis=-1)

    parsed_dataset = raw_dataset.map(_parse_function)

    print(parsed_dataset.take(1))
