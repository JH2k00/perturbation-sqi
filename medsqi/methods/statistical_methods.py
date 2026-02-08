import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import periodogram
from math import floor, ceil

def calculate_ksqi(signal:np.ndarray) -> np.ndarray:
    """
    Calculates the kSQI:
    The kurtosis of the signal.

    Parameters
    ----------
    signal : ndarray
        N-Dimensional numpy array containing the signals. The last dimension should correspond to the sample dimension.

    Returns
    -------
    ksqi : ndarray
        The kSQI calculated over the last dimension.

    References
    ----------
    .. [1] Clifford, G. D., Lopez, D., Li, Q., & Rezek, I. (2011, September). 
        Signal quality indices and data fusion for determining acceptability of electrocardiograms collected in noisy 
        ambulatory environments. In 2011 Computing in Cardiology (pp. 285-288). IEEE.
    """
    return kurtosis(signal, fisher=False, axis=-1)

def calculate_ssqi(signal:np.ndarray) -> np.ndarray:
    """
    Calculates the sSQI:
    The skewness of the signal.

    Parameters
    ----------
    signal : ndarray
        N-Dimensional numpy array containing the signals. The last dimension should correspond to the sample dimension.

    Returns
    -------
    ssqi : ndarray
        The sSQI calculated over the last dimension.

    References
    ----------
    .. [1] Clifford, G. D., Lopez, D., Li, Q., & Rezek, I. (2011, September). 
        Signal quality indices and data fusion for determining acceptability of electrocardiograms collected in noisy 
        ambulatory environments. In 2011 Computing in Cardiology (pp. 285-288). IEEE.
    """
    return skew(signal, axis=-1)

def calculate_psqi(signal:np.ndarray, fs:float, eps:float=1) -> np.ndarray:
    """
    Calculates the pSQI:
    The percentage of the signal x which appears to be a flat line (|dx/dt| < eps).

    Parameters
    ----------
    signal : ndarray
        N-Dimensional numpy array containing the signals in mV. The last dimension should correspond to the sample dimension.
    
    fs : float
        The sampling frequency of the signal in Hz.    

    eps : float
        The flat line threshold in mV.

    Returns
    -------
    psqi : ndarray
        The pSQI calculated over the last dimension.

    References
    ----------
    .. [1] Clifford, G. D., Lopez, D., Li, Q., & Rezek, I. (2011, September). 
        Signal quality indices and data fusion for determining acceptability of electrocardiograms collected in noisy 
        ambulatory environments. In 2011 Computing in Cardiology (pp. 285-288). IEEE.
    """
    dx_abs = np.abs(np.diff(signal, axis=-1)*fs) # Approximating dx/dt by (x_{i+1} - x_{i}) / (t_{i+1} - t{i}) = (x_{i+1} - x_{i}) * fs
    return (dx_abs < eps).astype(int).sum(axis=-1) / dx_abs.shape[-1]

def calculate_fsqi(signal:np.ndarray, fs:float) -> np.ndarray:
    """
    Calculates the fSQI:
    The ratio of power in the (5, 20) Hz band to the total power.

    Parameters
    ----------
    signal : ndarray
        N-Dimensional numpy array containing the signals. The last dimension should correspond to the sample dimension.
    
    fs : float
        The signal's sampling frequency in Hz.

    Returns
    -------
    fsqi : ndarray
        The fSQI calculated over the last dimension.

    References
    ----------
    .. [1] Clifford, G. D., Lopez, D., Li, Q., & Rezek, I. (2011, September). 
        Signal quality indices and data fusion for determining acceptability of electrocardiograms collected in noisy 
        ambulatory environments. In 2011 Computing in Cardiology (pp. 285-288). IEEE.
    """

    f, Pxx = periodogram(signal, fs, scaling="spectrum", detrend=False, axis=-1)
    power_5_14 = np.sum(Pxx[..., np.logical_and(f >= 5, f <= 20)], axis=-1, keepdims=True)
    total_power = np.sum(Pxx, axis=-1, keepdims=True)
    ratio =  power_5_14 / total_power
    ratio[total_power <= 0] = 1
    return ratio.squeeze(-1)

def calculate_spsqi(signal:np.ndarray) -> np.ndarray:
    """
    Calculates the spSQI:
    Defined as (w_2)^2 / (w_0 * w_4) where w_n is the nth-order spectral moment.

    Parameters
    ----------
    signal : ndarray
        N-Dimensional numpy array containing the signals. The last dimension should correspond to the sample dimension.

    Returns
    -------
    spsqi : ndarray
        The spSQI calculated over the last dimension.

    References
    ----------
    .. [1] Nemati, S., Malhotra, A., & Clifford, G. D. (2010). 
        Data fusion for improved respiration rate estimation. 
        EURASIP journal on advances in signal processing, 2010(1), 926305.
    """

    dx = np.diff(signal, axis=-1)
    ddx = np.diff(dx, axis=-1)

    return np.square(np.sum(np.square(dx), axis=-1)) / (np.sum(np.square(signal), axis=-1) * np.sum(np.square(ddx), axis=-1))


def calculate_hr_sqi_feats(signal:np.ndarray, beat_indices:np.ndarray, fs:float) -> np.ndarray:
    """
    Calculates the features used in the reliable heart rate SQI [1].
    
    Parameters
    ----------
    signal : ndarray
        1-D numpy array containing the signals.

    beat_indices : ndarray
        1-D numpy array containing the indices of the beats.

    fs : float
        The sampling frequency of the signal in Hz.

    Returns
    -------
    hr : float
        The heart rate in Hz.

    max_rr : float
        The longest RR interval in seconds.

    min_rr : float
        The shortest RR interval in seconds.

    template_corr : float
        The mean template matching correlation as defined in [1].

    References
    ----------
    .. [1] Orphanidou, C., Bonnici, T., Charlton, P., Clifton, D., Vallance, D., & Tarassenko, L. (2014). 
        Signal-quality indices for the electrocardiogram and photoplethysmogram: Derivation and applications 
        to wireless monitoring. IEEE journal of biomedical and health informatics, 19(3), 832-838.
    """
    
    assert (signal.ndim == 1) and (beat_indices.ndim == 1), "signal and beat_times must be 1D."

    if(beat_indices.shape[0] <= 1):
        return 0, 0, 0, 0 # Need at least two beats to calculate rr intervals.

    rr_intervals = np.diff(beat_indices)
    median_rr = np.median(rr_intervals).astype(int)
    hr = fs / median_rr # in Hz
    rr_intervals = rr_intervals / fs

    half_width = median_rr/2
    half_width_floor = floor(half_width)
    half_width_ceil = ceil(half_width)

    beats = []
    for peak in beat_indices:
        beat = signal[max(peak-half_width_floor, 0) : peak+half_width_ceil]
        if(len(beat) != median_rr): # Discard incomplete beats for now.
            continue
        beats.append(beat)
    if(len(beats) <= 1):
        template_corr = 1 # Need at least 2 beats to calculate a meaningful correlation
    else:
        beats = np.stack(beats, axis=0) # (n_complete_beats, median_rr)
        template = np.mean(beats, axis=0) # (median_rr, )

        b_norm = beats - np.mean(beats, axis=-1, keepdims=True)
        t_norm = template - np.mean(template)
        
        template_corr = np.mean(np.dot(b_norm, t_norm) / np.clip(np.sqrt(np.sum(b_norm**2, axis=-1) * np.sum(t_norm**2)), a_min=1e-24, a_max=None))   
    
    return hr, np.max(rr_intervals), np.min(rr_intervals), template_corr