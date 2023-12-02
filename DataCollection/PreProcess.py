import numpy as np
from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
import warnings

def highpass(hpdata, cutoff = 20, hpfs = 1000):
    b, a = butter(1, cutoff, "hp", fs=hpfs, analog=False, output="ba")
    return filtfilt(b, a, hpdata, axis=0, padlen=2)

def preprocess_data(lsl_data):
    # Set parameters
    Fs = 1000
    num_ch = 4
    
    #labels = []

    # Check sampling frequency
    actual_fs = 1 / np.mean(np.diff(lsl_data[:, 0]))
    #if np.abs(actual_fs - Fs) > 50:
        #warning("Actual Fs and Fs are quite different. Please check sampling frequency.")

    # Filter data
    filtered_lsl_data = np.zeros_like(lsl_data)
    filtered_lsl_data[:, 0] = lsl_data[:, 0]
    for ch in range(num_ch):
        filtered_lsl_data[:, 1+ch] = highpass(lsl_data[:, ch+1], 5, Fs)

    return filtered_lsl_data #returns time stamps and high pass filtered data for each channel
    # format of time stamps, rock, paper, scissors