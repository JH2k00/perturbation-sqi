import os
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from math import floor, ceil

# Find the records in a folder and its subfolders.
def find_records(folder, file_extension='.hea'):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == file_extension:
                record = os.path.relpath(os.path.join(root, file), folder)[:-len(file_extension)]
                records.add(os.path.join(folder, record))
    records = sorted(records)
    return records

# ECG-FM Utilities: Adapted from fairseq-signals repository
def resample(feats:np.ndarray, curr_sample_rate:float, desired_sample_rate:float) -> np.ndarray:
    """
    Resample an ECG using linear interpolation.
    """
    if curr_sample_rate == desired_sample_rate:
        return feats

    desired_sample_size = int(
        feats.shape[-1] * (desired_sample_rate / curr_sample_rate)
    )

    x = np.linspace(0, desired_sample_size - 1, feats.shape[-1])

    return interp1d(x, feats, kind='linear')(np.arange(desired_sample_size))

def preprocess_ecg_for_ECGFounder(data:np.ndarray, fs:float) -> np.ndarray:
    # Replace invalid values with 0 for now.
    data[~np.isfinite(data)] = 0

    sos = signal.butter(N=6, Wn=[1, 30], btype="bandpass", output="sos", fs=fs)
    data = signal.sosfiltfilt(sos, data, axis=-1)

    data = resample(data, fs, 500) # Resample to 500 Hz

    data = (data - np.mean(data, axis=(-1, -2), keepdims=True)) / (np.std(data, axis=(-1, -2), keepdims=True) + 1e-8)
    return data

def normalize_for_ECGFounder(data:np.ndarray) -> np.ndarray:
    # Replace invalid values with 0 for now.
    data[~np.isfinite(data)] = 0

    data = (data - np.mean(data, axis=(-1, -2), keepdims=True)) / (np.std(data, axis=(-1, -2), keepdims=True) + 1e-8)
    return data

def preprocess_ppg_for_SiamAF(data:np.ndarray, fs:float) -> np.ndarray:
    # Replace invalid values with 0 for now.
    data[~np.isfinite(data)] = 0
    if(fs > 80):
        sos = signal.butter(N=6, Wn=[0.67, 40], btype="bandpass", output="sos", fs=fs)
    else:
        sos = signal.butter(N=6, Wn=0.67, btype="highpass", output="sos", fs=fs) # There are no frequencies over 40 Hz anyway
    data = signal.sosfiltfilt(sos, data, axis=-1)
    data = resample(data, fs, 80) # Resample to 80 Hz

    data_min, data_max = np.min(data, axis=-1, keepdims=True), np.max(data, axis=-1, keepdims=True)
    data = (data - data_min) / (data_max - data_min + 1e-8)
    return data

def symmetric_zero_pad(signal:np.ndarray, segment_size:int) -> np.ndarray:
    signal_size = signal.shape[-1]
    assert signal_size <= segment_size, f"Signal size ({signal_size}) is greater than segment size ({segment_size})"

    padding = segment_size - signal_size
    if(padding > 0):
        padding_list = [(0, 0)] * signal.ndim
        padding_list[-1] = (floor(padding/2), ceil(padding/2)) # Only pad the last dimension
        signal = np.pad(signal, padding_list, mode='constant', constant_values=0)
    return signal