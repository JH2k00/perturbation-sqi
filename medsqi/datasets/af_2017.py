import numpy as np
import pandas as pd
from typing import Tuple
import wfdb
import os
from medsqi.datasets.utils import preprocess_ecg_for_ECGFounder, symmetric_zero_pad

fs = 500

def load_af_2017(data_path:str, csv_ref_path:str, window_size:int=5000) -> Tuple[np.ndarray]:
    """
    Loads in the PhysioNet/Computing in Cardiology Challenge 2017 dataset used for atrial fibrillation classification.
    The ECG is divided into segments of size window_size.

    Parameters
    ----------
    data_path : str
        Path to the PhysioNet/Computing in Cardiology Challenge 2017 dataset folder.

    csv_ref_path : str
        Path to the csv file containing the filenames and annotations.
    
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
    .. [1] Clifford, G. D., Liu, C., Moody, B., Lehman, L. W. H., Silva, I., Li, Q., ... & Mark, R. G. (2017, September). 
        AF classification from a short single lead ECG recording: The PhysioNet/computing in cardiology challenge 2017. 
        In 2017 computing in cardiology (CinC) (pp. 1-4). IEEE.
    .. [2] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit,
         and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
    """
    annotation_to_label = {"N": 0, "A": 1, "O": 2, "~": 3}
    ecgs = []
    labels = []

    references_df = pd.read_csv(csv_ref_path)
    for idx in range(len(references_df)):
        filename, annotation = references_df.iloc[idx]
        rec = wfdb.rdrecord(os.path.join(data_path, filename))
        ecg = rec.p_signal.T
        ecg = preprocess_ecg_for_ECGFounder(ecg, rec.fs)
        n_samples = ecg.shape[-1]

        segment_points = list(range(window_size, n_samples, window_size))
        ecg = np.split(ecg, segment_points, axis=-1)
        ecg[-1] = symmetric_zero_pad(ecg[-1], window_size) # Zero-pad the last possibly incomplete segment

        ecg = np.stack(ecg, axis=0) # Shape: n_segments x 1 x window_size
        ecgs.append(ecg)
        labels.extend([annotation_to_label[annotation]]*ecg.shape[0])
    
    # Standardize each segment independently at the end
    ecgs = np.concatenate(ecgs, axis=0)
    ecgs = (ecgs - np.mean(ecgs, axis=(-1, -2), keepdims=True)) / (np.std(ecgs, axis=(-1, -2), keepdims=True) + 1e-8)

    return ecgs, np.stack(labels, axis=0)