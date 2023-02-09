import numpy as np
import matplotlib.pyplot as plt
from Source.JakobSTFT import STFT
from Source.DataHandler import DataHandler

if __name__=='__main__':
    frequency_resolution = 5
    overlap = 0.2
    file_name = "popcorn-69750.mp3"

    data = DataHandler(
        directory_path="Sounds/Microwave_popcorn/Filtered/"
    ).data

    M = int(data[file_name]["freq"] / (2 * frequency_resolution))

    t, f, Sxx = STFT.stft(
        data[file_name]["left channel"], 
        M,
        int(M * overlap),
        fs=data[file_name]["freq"]
        )

    plt.pcolormesh(t, f, np.abs(Sxx), vmin=np.min(np.abs(Sxx)), vmax=np.max(np.abs(Sxx)), cmap="jet")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    