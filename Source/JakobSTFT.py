from typing import Tuple

import numpy as np


class Windows:
    @staticmethod
    def rectangle(list_, order):
        q = np.zeros(len(list_))
        for n in range(len(list_)):
            if 0 <= list_[n] <= order:
                q[n] = 1
            else:
                q[n] = 0
        return q

    @staticmethod
    def barlett(list_, order):  # triangular
        q = np.zeros(len(list_))
        for n in range(len(list_)):
            if 0 <= list_[n] <= order / 2:
                q[n] = 2 * list_[n] / order
            elif order / 2 < list_[n] <= order:
                q[n] = 2 - 2 * list_[n] / order
            else:
                q[n] = 0
        return q

    @staticmethod
    def hann(list_, order):
        q = np.zeros(len(list_))
        for n in range(len(list_)):
            if 0 <= list_[n] <= order:
                q[n] = 0.5 - 0.5 * np.cos((2 * np.pi * list_[n]) / order)
            else:
                q[n] = 0
        return q

    @staticmethod
    def hamming(list_, order):
        q = np.zeros(len(list_))
        for n in range(len(list_)):
            if 0 <= list_[n] <= order:
                q[n] = 0.54 - 0.46 * np.cos((2 * np.pi * list_[n]) / order)
            else:
                q[n] = 0
        return q

    @staticmethod
    def blackman(list_, order):
        q = np.zeros(len(list_))
        for n in range(len(list_)):
            if 0 <= list_[n] <= order:
                q[n] = 0.42 - 0.5 * np.cos((2 * np.pi * list_[n]) / order) \
                       + 0.08 * np.cos((4 * np.pi * list_[n]) / order)
            else:
                q[n] = 0
        return q


class STFT:
    windows = {
        "rectangle": Windows.rectangle,
        "barlett": Windows.barlett,
        "hann": Windows.hann,
        "hamming": Windows.hamming,
        "blackman": Windows.blackman
    }

    @classmethod
    def stft(
            cls, signal: np.ndarray, M: int, n: int, window_func: str = "hamming", fs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param signal: obvious
        :param M: window length
        :param n: n_step overlap
        :param window_func: str dictating window function
        :param fs: sample frequency
        :return: time-axis, freq-axis, stft
        """
        N = len(signal)
        w = cls.windows[window_func](np.arange(0, M), M)

        XS = []
        for i in range(0, N - M, n):
            XS.append(np.fft.rfft(w * signal[i:i + M]))

        if n <= M:
            XS.append(np.fft.rfft(w * signal[int(len(signal) - M):]))

        XS = np.transpose(XS)

        f = np.linspace(0, fs / 2, len(XS))
        t = np.linspace(0, N / fs, len(XS[0]))

        return t, f, XS
