import numpy as np
from typing import Tuple
from medsqi.datasets.utils import preprocess_ppg_for_SiamAF, symmetric_zero_pad

fs = 80

def load_deepbeat(data_path:str, window_size:int=2400) -> Tuple[np.ndarray]:
    """
    Loads in the Deepbeat dataset used for atrial fibrillation classification from PPG signals.
    The PPG is divided into segments of size window_size with zero-padding if necessary.

    Parameters
    ----------
    data_path : str
        Path to the Deepbeat dataset npz file.
    
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
    .. [1] Torres-Soto, J., & Ashley, E. A. (2020). 
        Multi-task deep learning for cardiac rhythm detection in wearable devices. 
        NPJ digital medicine, 3(1), 116.
    """
    data = np.load(data_path, allow_pickle=True)
    labels = data["rhythm"][:, 1]

    signals = np.transpose(data['signal'], (0, 2, 1)) # (n_segments, 1, n_samples)
    signals = preprocess_ppg_for_SiamAF(signals, fs=32) # (n_segments, 1, n_resampled_samples)
    signals = symmetric_zero_pad(signals, window_size)
 
    # Min-max normalize each segment independently at the end
    signals_min, signals_max = np.min(signals, axis=-1, keepdims=True), np.max(signals, axis=-1, keepdims=True)
    signals = (signals - signals_min) / (signals_max - signals_min + 1e-8)

    return signals, labels