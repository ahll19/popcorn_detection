import audio2numpy as a2n
import matplotlib.pyplot as plt
import os
import numpy as np

from Source.JakobSTFT import STFT


class DataHandler:
    def __init__(self, directory_path: str):
        all_files = os.listdir(directory_path)
        sound_files = list()

        for file_name in all_files:
            if file_name.endswith("mp3") or file_name.endswith("wav"):
                sound_files.append(file_name)

        data = {}

        for file_name in sound_files:
            data[file_name] = {}

            tmp_arr, fs = a2n.audio_from_file(directory_path + file_name)
            data[file_name]["left channel"] = tmp_arr[:, 0]
            data[file_name]["right channel"] = tmp_arr[:, 1]
            data[file_name]["freq"] = fs

        self.data = data

    def plot_spectrogram(self, file_name: str, channel: str, frequency_resolution: float = 20, overlap: float = 0.25):
        M = int(self.data[file_name]["freq"] / (2 * frequency_resolution))
        n = int(M * overlap)

        t, f, sx = STFT.stft(
            signal=self.data[file_name][channel],
            window_func="hamming",
            M=M,
            n=n,
            fs=self.data[file_name]["freq"]
        )

        print(f"Frequency resolution is {f[1] - f[0]:.2f} Hz")

        plt.pcolormesh(t, f, np.abs(sx), vmin=np.min(np.abs(sx)), vmax=np.max(np.abs(sx)), cmap="jet")
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
