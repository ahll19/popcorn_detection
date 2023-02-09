import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Source.DataHandler import DataHandler
from Source.JakobSTFT import STFT


if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use("TkAgg")

    obj = DataHandler(
        directory_path="/home/anders/Git/popcorn_detection/Sounds/Microwave_popcorn/"
    )

    # obj.plot_spectrogram("popcorn-69750.mp3", "left channel", frequency_resolution=5)
    obj.filter_data("popcorn-69750.mp3", 50, 6000)
    # obj.plot_spectrogram("popcorn-69750.mp3", "left channel", frequency_resolution=5, filtered=True)
    obj.write_filtered_to_mp3("popcorn-69750.mp3")

    print("tihi")
