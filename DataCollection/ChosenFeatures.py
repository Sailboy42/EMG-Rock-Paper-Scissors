import numpy as np
from statsmodels.tsa.ar_model import AutoReg

# Assuming the data shape is: channels x timepoints x trials
# The following feature functions are designed to process this data

def variance(data):
    # Variance calculated over timepoints for each channel
    return np.var(data, axis=1).T  # Transpose to make it trials x channels

def mean_absolute_value(data):
    # Mean Absolute Value calculated over timepoints for each channel
    return np.mean(np.abs(data), axis=1).T

def root_mean_square(data):
    # Root Mean Square calculated over timepoints for each channel
    return np.sqrt(np.mean(np.square(data), axis=1)).T

def zero_crossings(data):
    # Zero Crossing calculated over timepoints for each channel
    zc = np.sum(np.abs(np.diff(np.sign(data), axis=1)) == 2, axis=1)
    return zc.T

def waveform_length(data):
    # Waveform Length calculated over timepoints for each channel
    wl = np.sum(np.abs(np.diff(data, axis=1)), axis=1)
    return wl.T

def mean_frequency(data):
    # Mean Frequency calculated over timepoints for each channel
    sp = np.fft.fft(data, axis=1)
    freq = np.fft.fftfreq(data.shape[1])
    p = np.abs(sp)**2
    mean_freq = np.sum(p * freq.reshape((1, -1)), axis=1) / np.sum(p, axis=1)
    return mean_freq.T

def autoregressive_coefficients(data):
    # Autoregressive Coefficients calculated over timepoints for each channel
    num_channels, num_trials = data.shape
    ar_coeffs = np.zeros((num_channels))

    for i in range(num_channels):
    # for j in range(num_trials):
        model = AutoReg(data[i, :], lags=1)
        model_fitted = model.fit()
        ar_coeffs[i] = model_fitted.params[1]

    return ar_coeffs

def extract_features(data):
    # Extract and combine all features
    var = variance(data)
    mav = mean_absolute_value(data)
    rms = root_mean_square(data)
    zc = zero_crossings(data)
    wl = waveform_length(data)
    mf = mean_frequency(data)
    ar = autoregressive_coefficients(data)

    # Concatenate all features horizontally
    print(f"var shape {var.shape} {mav.shape} {rms.shape} {zc.shape} {wl.shape} {mf.shape} {ar.shape}")
    combined_features = np.concatenate((var, mav, rms, zc, wl, mf, ar), axis=0)


    return combined_features
