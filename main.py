import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Source.DataHandler import DataHandler
from Source.JakobSTFT import STFT


if __name__ == "__main__":
    # TODO: Bandpass filter based on spectrogram
    obj = DataHandler(
        directory_path="/home/aau/Git/popcorn_detection/Sounds/Microwave_popcorn/"
    )

    obj.plot_spectrogram("popcorn.mp3", "left channel")

    print("tihi")
