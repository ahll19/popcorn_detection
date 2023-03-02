import numpy as np
import matplotlib.pyplot as plt
from Source.JakobSTFT import STFT
from Source.DataHandler import DataHandler
from scipy.signal import find_peaks
from scipy.stats import poisson

def popcorn_timer(data: np.array, peaks: np.array, sps=44100, seconds=1):
        '''
        This function takes in the data and the peaks and returns the 
        probability of a pop and the number of pops in a given second(s).

        It should be used to determine when the popcorn is done.

        If the data has length sps*5 then it iterates over 5 seconds
        and will for each second calculate the probability of a pop and
        the number of pops in that second, given that seconds=1.

        :param data: the data from the audio file
        :param peaks: the peaks from the data
        :param sps: samples per second
        :param seconds: number of seconds that the function iterates over,
                        i.e. the step size in seconds.
        :return probs: list of probabilities of a pop in a given second(s)
        :return counts: list of number of pops in a given second(s)
        '''
        # for loop iterating over x seconds of data and using poi(k=peaks, mu=running average)
        # this is used to determine when the probability of a pop is too low then we stop the loop
        # and return the number of pops
        probs = []
        counts = []
        step = sps*seconds # 1 second with 44100 samples per second (standard for mp3)
        for i, second in enumerate(np.arange(0, len(data), step)):
            # calculate probability of a pop
            lower_peak = peaks[second<peaks]
            peaks_in_sec = lower_peak[lower_peak<second+step]
            counts.append(len(peaks_in_sec))
            
            probs.append(poisson.cdf(k=len(peaks_in_sec), mu=5))

            # if probability is too low then stop the loop
        
        return probs, counts

if __name__=='__main__':
    frequency_resolution = 5
    overlap = 0.2
    file_name = "popcorn-69750.mp3"

    obj = DataHandler(
        directory_path="Sounds/Microwave_popcorn/"
    )

    obj.filter_data(file_name, low_cut=2500)

    data = obj.data[file_name]['left channel']

    peaks, _ = find_peaks(data, prominence=1.5)
    plt.plot(data)
    plt.plot(peaks, data[peaks], "x")
    plt.show()

    

    probs, counts = popcorn_timer(data, peaks, seconds=1)
    plt.plot(probs)
    plt.show()

    plt.plot(counts)
    plt.show()
    '''
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
    '''