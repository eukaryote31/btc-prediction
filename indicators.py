import numpy as np

def ema(length):
    def ema_fn(data):
        av = [None for i in range(length)]

        for i in range(length, len(data)):
            if av[-1] is None:
                av.append(np.average(data[i - length:i, 4]))
            else:
                wmult = 2 / (length + 1)
                av.append(wmult * data[i, 4] + (1 - wmult) * av[-1])

        return np.array(av)

    return ema_fn
