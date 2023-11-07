import numpy as np
from statsmodels.tsa.ar_model import AutoReg


def variance(data):
    """
    Variance

    data: 3D matrix with dimensions: channels x timepoints x trials

    Returns: 2D matrix with dimensions: trials x channels
    """
    return np.var(data, axis=1)


def mean_absolute_value(data):
    """
    Mean Absolute Value

    data: 3D matrix with dimensions: channels x timepoints x trials

    Returns: 2D matrix with dimensions: trials x channels
    """
    return np.mean(np.abs(data), axis=1)


def root_mean_square(data):
    """
    Root Mean Square

    data: 3D matrix with dimensions: channels x timepoints x trials

    Returns: 2D matrix with dimensions: trials x channels
    """
    return np.sqrt(np.mean(np.square(data), axis=1))


def zero_crossings(data):
    """
    Zero Crossing

    data: 3D matrix with dimensions: channels x timepoints x trials

    Returns: 2D matrix with dimensions: trials x channels
    """
    return np.sum(np.abs(np.diff(np.sign(data), axis=1)) == 2, axis=1)


def waveform_length(data):
    """
    Waveform Length

    data: 3D matrix with dimensions: channels x timepoints x trials

    Returns: 2D matrix with dimensions: trials x channels
    """
    return np.sum(np.abs(np.diff(data, axis=1)), axis=1)


def mean_frequency(data):
    """
    Mean Frequency

    data: 3D matrix with dimensions: channels x timepoints x trials

    Returns: 2D matrix with dimensions: trials x channels
    """
    # power spectral density
    sp = np.fft.fft(data, axis=1)

    # extract the frequencies
    freq = np.fft.fftfreq(data.shape[1])
    freq = freq.reshape((1, -1, 1))

    # power of frequency = abs(spectrum)^2
    p = np.abs(sp)**2

    # return the weighted average of frequencies where
    # the weights are the power at each frequency
    mean_freq = np.sum(p * freq, axis=1) / np.sum(p, axis=1)

    return mean_freq.reshape(mean_freq.shape[0], mean_freq.shape[1])


def autoregressive_coefficients(data):
    """
    Autoregressive Coefficients

    data: 3D matrix with dimensions: channels x timepoints x trials

    Returns: 2D matrix with dimensions: trials x channels
    """
    num_channels, _, num_trials = data.shape
    ar_coeffs = np.zeros((num_channels, num_trials))

    for i in range(num_channels):
        for j in range(num_trials):
            model = AutoReg(data[i, j], lags=1)
            model_fitted = model.fit()
            ar_coeffs[i, j] = model_fitted.params[1]

    return ar_coeffs


def extract_features(data):
    var = variance(data)
    mav = mean_absolute_value(data)
    rms = root_mean_square(data)
    zc = zero_crossings(data)
    wl = waveform_length(data)
    mf = mean_frequency(data)
    ar = autoregressive_coefficients(data)

    return {
        'var': var,
        'mav': mav,
        'rms': rms,
        'zc': zc,
        'wl': wl,
        'mf': mf,
        'ar': ar
    }