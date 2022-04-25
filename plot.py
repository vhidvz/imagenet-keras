import os
import sys
import glob
import pickle

import matplotlib.pyplot as plt

from data import load_imagenet_dataset, ImageNetSequence


def plot_nxn(images, labels, title, n=5, figsize=(19.20, 19.20), path='./logs/images/'):
    if not os.path.exists(path):
        os.makedirs(path)

    fig, ax = plt.subplots(n, n, figsize=figsize)
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(images[i*n+j])
            ax[i, j].set_title(labels[i*n+j])
            ax[i, j].axis('off')
    fig.savefig(path + '/' + title + '.png')
    plt.close()


def plot_history(history, title, figsize=(19.20, 10.80), path='./logs/history/'):
    if not os.path.exists(path):
        os.makedirs(path)

    for key in history.keys():
        if not key.startswith('val_'):
            fig, ax = plt.subplots(figsize=figsize)
            x = range(1, len(history[key])+1)
            ax.plot(x, history[key], label=key)
            has_val = False
            if 'val_' + key in history.keys():
                ax.plot(x, history['val_' + key], label='val_' + key)
                has_val = True
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_ylabel(key)
            ax.set_xlabel('epoch')
            ax.legend(['train', 'valid'] if has_val else ['train'])
            fig.savefig(path + '/{}_{}.png'.format(title, key))
            fig.savefig(path + '/{}_{}.pdf'.format(title, key))
            plt.close()


if __name__ == "__main__":
    # ploting history
    history_path = './logs/history.pkl'
    history = pickle.load(open(history_path, 'rb'))

    plot_history(history, 'vgg16_imagenet')

    # tf.data.Dataset
    imagenet_train_ds = load_imagenet_dataset(
        './imagenet/train', crop_to_aspect_ratio=True)
    imagenet_valid_ds = load_imagenet_dataset(
        './imagenet/valid', crop_to_aspect_ratio=False)

    batches_train = list(imagenet_train_ds.as_numpy_iterator())
    batches_valid = list(imagenet_valid_ds.as_numpy_iterator())

    plot_nxn(batches_train[0][0], batches_train[0]
             [1], 'train_batch_0', n=5)
    plot_nxn(batches_train[1][0], batches_train[1]
             [1], 'train_batch_1', n=3)

    plot_nxn(batches_valid[0][0], batches_valid[0]
             [1], 'valid_batch_0', n=5)
    plot_nxn(batches_valid[1][0], batches_valid[1]
             [1], 'valid_batch_1', n=3)

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

    plot_nxn(batches_train_seq[0][0],
             batches_train_seq[0][1], 'train_batch_seq_0', n=5)
    plot_nxn(batches_train_seq[1][0],
             batches_train_seq[1][1], 'train_batch_seq_1', n=3)

    plot_nxn(batches_valid_seq[0][0],
             batches_valid_seq[0][1], 'valid_batch_seq_0', n=5)
    plot_nxn(batches_valid_seq[1][0],
             batches_valid_seq[1][1], 'valid_batch_seq_1', n=3)
