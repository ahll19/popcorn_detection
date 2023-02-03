import pandas as pd
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from typing import Union
import pandas as pd
import numpy as np
import audio2numpy as a2n
import os


class DataHandler:
    def __init__(self, directory_path: str):
        all_files = os.listdir(directory_path)
        sound_files = list()

        for file_name in all_files:
            if file_name.endswith("mp3") or file_name.endswith("wav"):
                sound_files.append(file_name)

        data = {}

        for i, file_name in enumerate(sound_files):
            data[i] = {}

            tmp = a2n.audio_from_file(directory_path + file_name)
            data[i]["data"] = tmp[0]
            data[i]["freq"] = tmp[1]

        self.sounds = data

    def test(self):
        # TODO: fix this
        f, t, sxx = spectrogram(self.sounds[0]["data"], self.sounds[0]["freq"])
        plt.pcolormesh(t, f, sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

        print("uha")


if __name__ == "__main__":
    obj = DataHandler("/mnt/c/Users/ander/Documents/Git/popcorn_detection/Sounds/Microwave_no_popcorn/")
    obj.test()
