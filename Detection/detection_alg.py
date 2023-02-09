import numpy as np
import matplotlib.pyplot as plt

class Popcorn:

    @staticmethod
    def pop_detection(t, f, XS):
        '''
        :param t: time
        :param f: frequency
        :param XS: stft
        :return pop: list of tuples (time, frequency)
        '''
        pop = []
        for i in range(len(t)):
            for j in range(len(f)):
                if XS[j][i] > 0.5:
                    pop.append((t[i], f[j]))
        
        return pop

