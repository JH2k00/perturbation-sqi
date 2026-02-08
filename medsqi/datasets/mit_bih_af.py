import numpy as np
from typing import Tuple
import wfdb
import os
from medsqi.datasets.utils import preprocess_ecg_for_ECGFounder, symmetric_zero_pad

fs = 500

def load_mit_bih_af(data_path:str, window_size:int=5000) -> Tuple[np.ndarray]:
    """
    Loads in the MIT-BIH AF dataset used for atrial fibrillation classification.
    The ECG is divided into segments of size window_size.

    Parameters
    ----------
    data_path : str
        Path to the MIT-BIH AF dataset folder.
    
    window_size : int
        Number of samples in a window.

    Returns
    -------
    ecgs : ndarray
        3-Dimensional ndarray of the shape (n_segments, 1, window_size) containing the ECG segments.
    
    labels: ndarray
        1-Dimensional containing the atrial fibrillation labels corresponding ECG segments.

    References
    ----------
    .. [1] Moody GB, Mark RG. A new method for detecting atrial fibrillation using R-R intervals. Computers in Cardiology. 10:227-230 (1983).
    .. [2] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit,
         and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
    """
    ecgs = []
    labels = []
    for file in os.listdir(data_path):
        if(not file.endswith(".dat")):
            continue
        filename = os.path.join(data_path, file.replace(".dat", ""))
        rec = wfdb.rdrecord(filename)
        ecg = rec.p_signal.T[[0], :]
        ecg = preprocess_ecg_for_ECGFounder(ecg, rec.fs)
        n_samples = ecg.shape[-1]

        anno = wfdb.rdann(filename, extension="atr")
        anno_resampled = list(np.round(anno.sample*500/rec.fs).astype(int)) # resample the annotations as well
        anno_resampled[0] = 0
        anno_resampled.append(n_samples)

        segment_points = list(range(window_size, n_samples, window_size))
        ecg = np.split(ecg, segment_points, axis=-1)
        ecg[-1] = symmetric_zero_pad(ecg[-1], window_size) # Zero-pad the last possibly incomplete segment

        ecg = np.stack(ecg, axis=0) # Shape: n_segments x 1 x window_size
        ecgs.append(ecg)

        # Create labels from annotations
        segment_points.insert(0, 0)
        segment_points.append(n_samples)
        cur_anno_idx = 0
        for i in range(len(segment_points)-1):
            start_idx = segment_points[i]
            end_idx = segment_points[i+1]
            if(start_idx >= anno_resampled[cur_anno_idx+1]): # The start_idx has surpassed the current end idx of the label
                cur_anno_idx += 1
            
            label_1 = "AFIB" in anno.aux_note[cur_anno_idx] # Label where the start_idx is located
            label_2 = "AFIB" in anno.aux_note[min(cur_anno_idx + int(end_idx > anno_resampled[cur_anno_idx+1]), len(anno.aux_note)-1)] # Label where the end_idx is located
            label = label_1 or label_2
            labels.append(int(label))
    
    # Standardize each segment independently at the end
    ecgs = np.concatenate(ecgs, axis=0)
    ecgs = (ecgs - np.mean(ecgs, axis=(-1, -2), keepdims=True)) / (np.std(ecgs, axis=(-1, -2), keepdims=True) + 1e-8)
    
    return ecgs, np.stack(labels, axis=0)
