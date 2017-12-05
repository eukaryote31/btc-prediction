import numpy as np
import sys

import indicators

CANDLE_SIZE = 15


def main():
    print "Loadng from txt"
    data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=2, usecols=[1, 2, 3, 4, 6])

    print "Merging candles"
    data = combine(data, CANDLE_SIZE)

    print "Applying indicators"
    inds = [indicators.ema(12), indicators.ema(26), indicators.ema(65), indicators.ema(200),
            indicators.rsi(14), indicators.accdistdelt()]

    ind_data = []
    for ind in inds:
        ind_data.append(ind(data))

    print "Swapping axes"
    ind_data_sw = np.swapaxes(ind_data, 0, 1)

    data = np.append(data, ind_data_sw, axis=1)
    data = data[500:]

    print "Saving"
    np.savetxt('data.csv', data, delimiter=',')


def combine(data, size):
    if size == 1:
        return data
    ret = []
    for i in range(0, len(data), size):
        u = data[i:i + size]
        ret.append((u[0, 0], max(u[:, 1]), min(u[:, 2]), u[-1, 3], sum(u[:, 4])))

    return np.array(ret)


if __name__ == '__main__':
    main()
