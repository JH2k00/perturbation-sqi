import numpy as np
from typing import Tuple
import os
import wfdb

fs = 500

def load_ptt_ppg(data_path:str, window_size:int=5000) -> Tuple[np.ndarray|Tuple[np.ndarray]]:
    """
    Loads in the PTT PPG dataset used for R-peak estimation.
    The ECG is divided into segments of size window_size.

    Parameters
    ----------
    data_path : str
        Path to the PTT PPG dataset folder.
    
    window_size : int
        Number of samples in a window.

    Returns
    -------
    ecgs : ndarray
        2-Dimensional ndarray of the shape (n_segments, window_size) containing the ECG segments.
    
    annotations: Tuple[ndarray]
        Tuple containing the peak positions arrays for the corresponding ECG segments.

    References
    ----------
    .. [1] Mehrgardt, P., Khushi, M., Poon, S., & Withana, A. (2022). 
        Pulse Transit Time PPG Dataset (version 1.1.0). PhysioNet. 
        RRID:SCR_007345. https://doi.org/10.13026/jpan-6n92
    .. [2] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit,
         and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
    """
    ecgs = []
    annotations = []
    for file in os.listdir(data_path):
        if(not file.endswith(".hea")):
            continue
        filename = os.path.join(data_path, file.replace(".hea", ""))
        rec = wfdb.rdrecord(filename)
        ecg_lead_I = rec.p_signal[:, 0]
        n_samples = ecg_lead_I.shape[-1]
        rpeaks_anno = wfdb.rdann(filename, extension="atr").sample

        segment_points = list(range(window_size, n_samples, window_size))
        if(n_samples < window_size):
            print(f"Skipping record {filename} as it only contains {n_samples} samples.")
            continue
        ecg_lead_I = np.split(ecg_lead_I, segment_points)
        ecg_lead_I = ecg_lead_I[:-1] # Discard the last incomplete segment
        segment_points.insert(0, 0)
        #segment_points.append(n_samples)
        for i, ecg in enumerate(ecg_lead_I):
            anno = rpeaks_anno[np.logical_and(rpeaks_anno >= segment_points[i], rpeaks_anno < segment_points[i+1])] - segment_points[i]
            ecgs.append(ecg)
            annotations.append(anno)
    return np.stack(ecgs, axis=0), annotations