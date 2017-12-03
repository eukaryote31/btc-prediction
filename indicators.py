import numpy as np


def ema(length):
    def ema_fn(data):
        return _ema(data[:, 4], length)

    return ema_fn


def rsi(length):
    def rsi_fn(data):
        u = []
        d = []

        prev = 0
        for n in data[:, 4]:
            diff = n - prev
            if diff > 0:
                u.append(diff)
                d.append(0)
            else:
                u.append(0)
                d.append(-diff)

            prev = n

        u = _ema(u, length)[length:]
        d = _ema(d, length)[length:]

        rs = np.float64(u) / d

        return np.append([None for i in range(length)], (100 - (100 / (1 + rs))))

    return rsi_fn


def _ema(arr, length):
    av = [None for i in range(length)]
    for i in range(length, len(arr)):
        if av[-1] is None:
            av.append(np.average(arr[i - length:i]))
        else:
            wmult = 2 / (np.float64(length) + 1)
            av.append(wmult * arr[i] + (1 - wmult) * av[-1])

    return np.array(av)
