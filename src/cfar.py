import numpy as np

# https://github.com/tsaith/radar/blob/master/radar/signal/cfar.py


def detect_peaks(x, num_train, num_guard, rate_fa):
    """
    Detect peaks with CFAR algorithm.

    num_train: Number of training cells.
    num_guard: Number of guard cells.
    rate_fa: False alarm rate.
    """
    num_cells = x.size
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half

    alpha = num_train * (rate_fa ** (-1 / num_train) - 1)  # threshold factor

    has_peak = np.zeros(num_cells, dtype=bool)
    peak_at = []
    for i in range(num_side, num_cells - num_side):
        if i != i - num_side + np.argmax(x[i - num_side : i + num_side + 1]):
            continue

        sum1 = np.sum(x[i - num_side : i + num_side + 1])
        sum2 = np.sum(x[i - num_guard_half : i + num_guard_half + 1])
        p_noise = (sum1 - sum2) / num_train
        threshold = alpha * p_noise

        if x[i] > threshold:
            has_peak[i] = True
            peak_at.append(i)

    peak_at = np.array(peak_at, dtype=int)

    return peak_at
