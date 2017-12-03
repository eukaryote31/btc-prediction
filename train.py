import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution1D, LeakyReLU, MaxPooling1D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend
from keras import optimizers

import indicators

print "Loading data..."
datafile = 'bitcoin-historical-price/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20-truncated.csv'

# Load historical price data
data = np.genfromtxt(datafile, delimiter=',')
data = data[:1500000]
print "Data loaded"


# Number of candles to consider in input
INPUT_LENGTH = 60 * 1

TESTING_SIZE = 1000
TRAIN_EXISTING = False

x_train = []
y_train = []

indicators = [indicators.ema(12), indicators.ema(26), indicators.ema(65), indicators.ema(200),
              indicators.rsi(14)]
ind_data = []
for ind in indicators:
    ind_data.append(ind(data))

ind_data_sw = np.swapaxes(ind_data, 0, 1)
print "Indicators generated"

for i in range(len(data) - INPUT_LENGTH - 1):
    x_train.append(
        np.append(data[i:i + INPUT_LENGTH, 1:6], ind_data_sw[i:i + INPUT_LENGTH, :], axis=1))

    y_train.append(data[i + INPUT_LENGTH, 1:5])
print "Data sliced into x & y"

# cut off first 1000 so that indicators that don't start at candle 0 are continuous
x_train = x_train[1000:]
y_train = y_train[1000:]

# OHLCV
x_train = np.array(x_train)
y_train = np.array(y_train)

# split testing data
x_test = x_train[-TESTING_SIZE:]
x_train = x_train[:-TESTING_SIZE]
y_test = y_train[-TESTING_SIZE:]
y_train = y_train[:-TESTING_SIZE]

print "Processed data:", len(x_train)

if TRAIN_EXISTING:
    model = load_model('model-btc.h5')
else:
    model = Sequential()
    model.add(Convolution1D(filters=64, kernel_size=3, padding='causal',
                            input_shape=(INPUT_LENGTH, 5 + len(indicators))))
    model.add(LeakyReLU(0.1))
    model.add(Convolution1D(filters=64, kernel_size=3, padding='causal'))
    model.add(LeakyReLU(0.1))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(4))

model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=2048, epochs=3)
model.save('model-btc.h5')

print model.evaluate(x_test, y_test)
print y_test[0]
print model.predict(np.array([x_test[0]]))
