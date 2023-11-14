import numpy as np
from scipy.signal import highpass
from scipy import warning

def preprocess_data(lsl_data, marker_data):
    # Set parameters
    Fs = 1000
    num_ch = 4
    epoched_data = np.empty((0, num_ch, 1400))  # Assuming 1400 as the number of timepoints in a trial
    labels = []

    # Check sampling frequency
    actual_fs = 1 / np.mean(np.diff(lsl_data[:, 0]))
    if np.abs(actual_fs - Fs) > 50:
        warning("Actual Fs and Fs are quite different. Please check sampling frequency.")

    # Filter data
    filtered_lsl_data = np.zeros_like(lsl_data)
    filtered_lsl_data[:, 0] = lsl_data[:, 0]
    for ch in range(num_ch):
        filtered_lsl_data[:, 1+ch] = highpass(lsl_data[:, ch+1], 5, Fs)

    # Epoch data
    epoched_data, labels = epoch_from_markers_to_labels(filtered_lsl_data, marker_data, 1400)

    return epoched_data, labels


def epoch_from_markers_to_labels(filtered_lsl_data, marker_data, epoch_length):
    # Assuming you have a function epochFromMarkersToLabels defined elsewhere
    # that performs the same epoching operation as in your MATLAB code
    # and returns ch x timepoints x trials and a list of labels
    epoched_data = np.empty((0, filtered_lsl_data.shape[1] - 1, epoch_length))
    labels = []

    # Your epoching logic here...

    return epoched_data, labels
