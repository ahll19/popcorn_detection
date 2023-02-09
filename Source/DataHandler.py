import audio2numpy as a2n
import matplotlib.pyplot as plt
import os
import numpy as np
import pydub

from scipy.signal import butter

from Source.JakobSTFT import STFT, Windows


class DataHandler:
    def __init__(self, directory_path: str):
        self.dir_path = directory_path
        all_files = os.listdir(directory_path)
        sound_files = list()

        for file_name in all_files:
            if file_name.endswith("mp3") or file_name.endswith("wav"):
                sound_files.append(file_name)

        data = {}

        for file_name in sound_files:
            data[file_name] = {}

            tmp_arr, fs = a2n.audio_from_file(directory_path + file_name)

            try:
                data[file_name]["left channel"] = tmp_arr[:, 0]
                data[file_name]["right channel"] = tmp_arr[:, 1]
            except IndexError:
                data[file_name]["left channel"] = tmp_arr
                data[file_name]["right channel"] = tmp_arr
            else:
                raise Exception("Something went wrong with the audio file: " + file_name + "")

            data[file_name]["freq"] = fs

        self.data = data

    def filter_data(self, file_name: str, low_cut: float, high_cut: float):
        if low_cut < 0:
            raise Exception("Low cut frequency must be positive")

        if high_cut > self.data[file_name]["freq"] / 2:
            raise Exception("High cut frequency must be less than half the sampling frequency")

        if low_cut > high_cut:
            raise Exception("Low cut frequency must be less than high cut frequency")

        b, a = butter(4, [low_cut, high_cut], btype="bandpass", fs=self.data[file_name]["freq"], output="ba")

        lc = "left channel"
        rc = "right channel"
        self.data[file_name]["filtered " + lc] = np.convolve(self.data[file_name][lc], b, mode="same")
        self.data[file_name]["filtered " + rc] = np.convolve(self.data[file_name][rc], b, mode="same")

    def plot_spectrogram(
            self,
            file_name: str,
            channel: str,
            frequency_resolution: float,
            overlap: float = 0.2,
            filtered: bool = False
    ):
        M = int(self.data[file_name]["freq"] / (2 * frequency_resolution))
        n = int(M * overlap)

        if filtered:
            channel = "filtered " + channel

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

    def write_filtered_to_mp3(self, file_name):
        channels = 2
        lc = self.data[file_name]["filtered left channel"]
        rc = self.data[file_name]["filtered right channel"]

        l_range = np.max(lc) - np.min(lc)
        r_range = np.max(rc) - np.min(rc)
        rc = (rc - np.min(rc)) / r_range
        lc = (lc - np.min(lc)) / l_range

        x = np.asarray([lc, rc])
        y = np.int16(x * 2 ** 15)
        song = pydub.AudioSegment(
            y.tobytes(),
            frame_rate=self.data[file_name]["freq"],
            sample_width=2,
            channels=channels
        )
        song.export(self.dir_path + "Filtered/" + file_name, format="mp3", bitrate="320k")
