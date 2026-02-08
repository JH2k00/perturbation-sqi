import numpy as np
from typing import Tuple
from medsqi.datasets.utils import preprocess_ppg_for_SiamAF, symmetric_zero_pad
import os
from scipy.io import loadmat

fs = 80

def load_ppg_arrhythmia(data_path:str, window_size:int=2400) -> Tuple[np.ndarray]:
    """
    Loads in the PPG Arrhythmia dataset used for atrial fibrillation classification from PPG signals.
    The PPG is divided into segments of size window_size with zero-padding if necessary.

    Parameters
    ----------
    data_path : str
        Path to the PPG Arrhythmia dataset folder with the mat files.
    
    window_size : int
        Number of samples in a window.

    Returns
    -------
    ppgs : ndarray
        3-Dimensional ndarray of the shape (n_segments, 1, window_size) containing the PPG segments.
    
    labels: ndarray
        1-Dimensional containing the atrial fibrillation labels corresponding PPG segments.

    References
    ----------
    .. [1] Liu, Z., Zhou, B., Jiang, Z., Chen, X., Li, Y., Tang, M., & Miao, F. (2022). 
        Multiclass arrhythmia detection and classification from photoplethysmography 
        signals using a deep convolutional neural network. Journal of the American Heart Association, 11(7), e023555.
    """
    ppgs = []
    labels_list = []
    orig_freq_window_size = int(window_size*100/fs) # We want to resample to fs after windowing, so before windowing we need this window size.
    for filename in os.listdir(data_path):
        if(not filename.endswith(".mat")):
            continue
        data_dict = loadmat(os.path.join(data_path, filename)) # Original sampling frequency is 100 Hz
        ppg = data_dict["ppgseg"].flatten() # Reshape the ppg back to a continuous recording instead of 10s segments
        labels = (data_dict["labels"][:, 0] == 5).astype(int) # Has AF or not, one label per 10s segment
        labels = np.repeat(labels, 1000) # Upsample the labels to one per sample instead of one per 10s segment.

        segment_points = list(range(orig_freq_window_size, ppg.shape[0], orig_freq_window_size))
        ppg = np.split(ppg, segment_points, axis=-1)
        ppg[-1] = symmetric_zero_pad(ppg[-1], orig_freq_window_size) # Zero-pad the last possibly incomplete segment
        ppg = np.stack(ppg, axis=0)[:, np.newaxis, :]
        ppg = preprocess_ppg_for_SiamAF(ppg, fs=100)

        # Segment labels as well and group to one label per segment through the or operator (A segment is classified as AF, if at least 1 sample has AF)
        labels = np.split(labels, segment_points, axis=-1)
        labels[-1] = symmetric_zero_pad(labels[-1], orig_freq_window_size) # Zero-pad the last possibly incomplete segment
        labels = np.any(np.stack(labels, axis=0), axis=-1).astype(int)

        ppgs.append(ppg)
        labels_list.append(labels)

    return np.concatenate(ppgs, axis=0), np.concatenate(labels_list, axis=0)