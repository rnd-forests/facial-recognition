import cv2
import keras
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.examples.tutorials.mnist import input_data

import os
import glob
import shutil

import config


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def create_summary_writer(base, path, graph):
    path = "{}/{}".format(base, path)
    if os.path.exists(path):
        shutil.rmtree(path)
    return tf.summary.FileWriter(path, graph)


def random_batch(x_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(x_train), batch_size)
    x_batch = x_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return x_batch, y_batch


def neuron_layer(X, n_neurons, name, training=None, activation=None,
                 dropout=True, dropout_rate=0.2,
                 batch_norm=True, batch_momentum=0.9):

    with tf.name_scope(name=name):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        dc = tf.layers.dense(X, units=n_neurons, kernel_initializer=he_init)
        if batch_norm:
            dc = tf.layers.batch_normalization(dc, training=training, momentum=batch_momentum)
        if activation is not None:
            act = activation(dc)
            if dropout:
                return tf.layers.dropout(act, rate=dropout_rate, training=training)
            return act
        else:
            return dc


def load_minist():
    mnist = input_data.read_data_sets("/tmp/data/")
    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels
    return X_train, X_test, y_train, y_test


def load_faces(zipfile="faces", channels=1):
    def load_images(dir):
        X = []
        y = []
        folders = glob.glob(os.path.join(dir, '*/*'))
        for folder in folders:
            person = folder.split('/')[-2]
            images = glob.glob(os.path.join(folder, '*.jpg'))
            for image in images:
                temp = cv2.imread(image, 0)
                temp = cv2.resize(temp, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                if channels == 3:
                    temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB)
                temp = np.reshape(temp, config.IMAGE_SIZE * config.IMAGE_SIZE * channels)
                X.append(temp)
                y.append(person)
        return X, y

    def load_original_data():
        X_train, y_train = load_images(config.TRAIN_DIR)
        X_test, y_test = load_images(config.TEST_DIR)

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

        np.savez(zipfile, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    zipfile_with_ext = zipfile + '.npz'
    if not os.path.exists(zipfile_with_ext):
        load_original_data()
    data = np.load(zipfile_with_ext)

    return data['X_train'], data['X_test'], data['y_train'], data['y_test']


def load_preprocessed_faces(zipfile="faces", channels=1):
    size = config.IMAGE_SIZE
    n_classes = config.N_CLASSES

    X_train, X_test, y_train, y_test = load_faces(zipfile, channels)

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    X_train = X_train.astype('float32')
    X_train /= 255
    X_train = X_train.reshape(X_train.shape[0], size, size, 3)

    X_test = X_test.astype('float32')
    X_test /= 255
    X_test = X_test.reshape(X_test.shape[0], size, size, 3)

    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
