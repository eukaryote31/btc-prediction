from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution1D, LeakyReLU, MaxPooling1D, ELU, Input, Add, Concatenate
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend
from keras import optimizers
from keras.models import Sequential

import numpy as np
import keras.backend as K
import tensorflow as tf


import os
import sys

import indicators

# Number of candles to consider in input
INPUT_LENGTH = 25

TESTING_SIZE = 100
TRAIN_EXISTING = True

inds = [indicators.ema(12), indicators.ema(26), indicators.ema(65), indicators.ema(200),
        indicators.sma(50), indicators.rsi(14), indicators.accdistdelt()]


SIGN_PENALTY = 250.


def conv_net():
    inputs = Input(shape=(INPUT_LENGTH, 5 + len(inds)))
    l = Convolution1D(filters=128, kernel_size=3, padding='same')(inputs)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Convolution1D(filters=128, kernel_size=3, padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Convolution1D(filters=128, kernel_size=3, padding='same')(l)
    l = BatchNormalization()(l)
    l = Activation('relu')(l)
    l = Flatten()(l)

    l = Dense(16)(l)
    l = Activation('relu')(l)
    predictions = Dense(1)(l)
    return Model(inputs=inputs, outputs=predictions)


def fc_net():
    inputs = Input(shape=(INPUT_LENGTH, 5 + len(inds)))
    l = Flatten()(inputs)

    l = Dense(2 * INPUT_LENGTH)(l)
    l = Activation('relu')(l)

    l = Dense(INPUT_LENGTH)(l)
    l = Activation('relu')(l)

    predictions = Dense(1)(l)
    return Model(inputs=inputs, outputs=predictions)


def loss_fn(y_true, y_pred):
    """ Modified MAE that weights correct wrong sign heavily """
    return K.mean(K.abs(y_pred - y_true) + K.maximum(0., -SIGN_PENALTY * tf.sign(y_pred) * tf.sign(y_true)), axis=-1)


def sign_accuracy(y_true, y_pred):
    return 1 - K.maximum(0., -1 * tf.sign(y_pred) * tf.sign(y_true))


def main(argv):
    if TRAIN_EXISTING:
        model = load_model('model-btc.h5', custom_objects={
                           'loss_fn': loss_fn, 'sign_accuracy': sign_accuracy})
    else:

        model = conv_net()
        model.compile(optimizer=Adam(), loss=loss_fn, metrics=['mse', 'mae', sign_accuracy])

    print "Loading data..."
    datadir = argv[1]

    x_train = []
    y_train = []
    for datafile in os.listdir(datadir):
        datafile = os.path.join(datadir, datafile)
        print "Loading file", datafile
        # Load historical price data
        data = np.genfromtxt(datafile, delimiter=',')
        print "Data loaded"

        for i in range(len(data) - INPUT_LENGTH - 1):
            x_train.append(data[i:i + INPUT_LENGTH, :])

            y_train.append(data[i + INPUT_LENGTH + 1, 3] - data[i + INPUT_LENGTH, 3])
        print "Data sliced into x & y"

    # OHLCV
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # split testing data

    # cut off first 1000 so that indicators that don't start at candle 0 are continuous
    x_train = x_train[1000:]
    y_train = y_train[1000:]
    x_test = x_train[-TESTING_SIZE:]
    x_train = x_train[:-TESTING_SIZE]
    y_test = y_train[-TESTING_SIZE:]
    y_train = y_train[:-TESTING_SIZE]

    print "Processed data:", len(x_train)

    model.fit(x_train, y_train, batch_size=8192, epochs=1, validation_split=0.1)
    model.save('model-btc.h5')

    print model.evaluate(x_test, y_test)
    print y_test[0]
    print model.predict(np.array([x_test[0]]))


def combine(data, size):
    if size == 1:
        return data
    ret = []
    for i in range(0, len(data), size):
        u = data[i:i + size]
        ret.append(
            [u[0, 0], u[0, 1], max(u[:, 2]), min(u[:, 3]), u[-1, 4], sum(u[:, 5]), sum(u[:, 6]), u[-1, 7]])

    return np.array(ret)

if __name__ == '__main__':
    main(sys.argv)
