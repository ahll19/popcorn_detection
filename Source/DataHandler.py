import audio2numpy as a2n
import matplotlib.pyplot as plt
import os
import numpy as np
import pydub

from scipy.signal import butter, sosfilt

from Source.JakobSTFT import STFT, Windows


class DataHandler:
    def __init__(self, directory_path: str):
        # TODO: create an attribute that stores file_names that the DataHandler is aware of
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
            except:
                raise Exception("Something went wrong with the audio file: " + file_name + "")

            data[file_name]["freq"] = fs

        self.data = data

    @staticmethod
    def _normalize_data_to_range(data: np.ndarray, min_val: float, max_val: float):
        data_range = np.max(data) - np.min(data)
        data = (data - np.min(data)) / data_range
        data = data * (max_val - min_val) + min_val

        return data

    def filter_data(self, file_name: str, low_cut: float = None, high_cut: float = None):
        # TODO: fix bandpass filter
        """
        Specifying only low_cut will create a lowpass filter,
        and specifying only high_cut will create a highpass filter.
        Specifying both will create a bandpass filter.

        :param file_name:
        :param low_cut:
        :param high_cut:
        :return:
        """
        sos = None

        if low_cut is not None:
            if low_cut < 0:
                raise Exception("Low cut frequency must be positive")

        if high_cut is not None:
            if high_cut > self.data[file_name]["freq"] / 2:
                raise Exception("High cut frequency must be less than half the sampling frequency")

        if low_cut is not None and high_cut is not None:
            if low_cut > high_cut:
                raise Exception("Low cut frequency must be less than high cut frequency")

        if low_cut is None and high_cut is None:
            raise Exception("Must specify at least one cut frequency")

        if low_cut is None and high_cut is not None:
            sos = butter(4, high_cut, btype="highpass", fs=self.data[file_name]["freq"], output="sos")

        if low_cut is not None and high_cut is None:
            sos = butter(4, low_cut, btype="lowpass", fs=self.data[file_name]["freq"], output="sos")

        if low_cut is not None and high_cut is not None:
            sos = butter(4, [low_cut, high_cut], btype="bandpass", fs=self.data[file_name]["freq"], output="sos")

        lc = "left channel"
        rc = "right channel"
        lc_filtered = sosfilt(sos, self.data[file_name][lc])
        rc_filtered = sosfilt(sos, self.data[file_name][rc])

        self.data[file_name]["filtered " + lc] = self._normalize_data_to_range(lc_filtered, -1, 1)
        self.data[file_name]["filtered " + rc] = self._normalize_data_to_range(rc_filtered, -1, 1)

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

        x = np.asarray([
            self._normalize_data_to_range(lc, -1, 1),
            self._normalize_data_to_range(rc, -1, 1)
        ])
        y = np.int16(x * 2 ** 15)
        song = pydub.AudioSegment(
            y.tobytes(),
            frame_rate=self.data[file_name]["freq"],
            sample_width=2,
            channels=channels
        )
        song.export(self.dir_path + "Filtered/" + file_name, format="mp3", bitrate="320k")
