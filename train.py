import pickle

import tensorflow as tf

from tensorflow import keras

from model import VGG16
from plot import plot_history
from data import load_imagenet_dataset


def train_imagenet(model: keras.Model, train_ds: tf.data.Dataset, valid_ds: tf.data.Dataset, epochs=1000, optimizer="adam", verbose=1, append=True, use_multiprocessing=False,
                   loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], run_eagerly=False, jit_compile=True, backup_dir="./logs/backup",
                   patience=10, filename='./logs/training.log', save_best_only=True, save_weights_only=True, checkpoint_filepath='./logs/checkpoint/weights'):
    # define callbacks
    callbacks = [
        keras.callbacks.BackupAndRestore(backup_dir),
        keras.callbacks.EarlyStopping(patience=patience),
        keras.callbacks.CSVLogger(filename, append=append),
        keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, save_best_only=save_best_only, save_weights_only=save_weights_only),
    ]

    # model compilation
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                  run_eagerly=run_eagerly, jit_compile=jit_compile)

    # model training
    return model.fit(train_ds, epochs=epochs, verbose=verbose, validation_data=valid_ds,
                     callbacks=callbacks, use_multiprocessing=use_multiprocessing).history


if __name__ == "__main__":
    # loading dataset (this sample dataset only contains 5 classes)
    imagenet_train_ds = load_imagenet_dataset('./imagenet/train')
    imagenet_valid_ds = load_imagenet_dataset('./imagenet/valid')

    # model definition
    base_model = VGG16(include_top=False,
                       input_shape=(224, 224, 3), weights=None)

    # Classification block
    x = keras.layers.Flatten(name='flatten')(base_model.output)
    x = keras.layers.Dense(4096, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(4096, activation='relu', name='fc2')(x)

    x = keras.layers.Dense(5, activation='softmax', name='predictions')(x)

    full_model = keras.models.Model(inputs=base_model.input, outputs=x)

    # model summary
    full_model.summary()

    keras.utils.plot_model(full_model, to_file="./logs/vgg16_imagenet_model.pdf",
                           show_shapes=True, expand_nested=True)
    keras.utils.plot_model(full_model, to_file="./logs/vgg16_imagenet_model.png",
                           show_layer_activations=True, show_dtype=True)

    # training phase
    history = train_imagenet(full_model, imagenet_train_ds, imagenet_valid_ds,
                             epochs=3, run_eagerly=True, jit_compile=False,
                             filename='./logs/vgg16_imagenet_training.log')

    # ploting history
    plot_history(history, 'vgg16_imagenet')

    # saving model
    full_model.save('./logs/vgg16_imagenet_model.h5')
    base_model.save('./logs/vgg16_imagenet_model_base.h5')

    # saving model weights
    full_model.save_weights('./logs/vgg16_imagenet_weights.h5')
    base_model.save_weights('./logs/vgg16_imagenet_weights_base.h5')

    # saving history
    pickle.dump(history, open('./logs/vgg16_imagenet_history.pkl', 'wb'))

    # load pretrained model
    model = VGG16(include_top=False, input_shape=(224, 224, 3),
                  weights='./logs/vgg16_imagenet_weights_base.h5')
    model.trainable = False
