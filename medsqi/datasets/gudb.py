"""
(C) 2018-2020 Bernd Porr <bernd.porr@glasgow.ac.uk>
(C) Luis Howell <2123374H@student.gla.ac.uk>

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

API for the data which loads, filters and exports
the ECG data.
"""

# Taken from https://github.com/berndporr/ECG-GUDB and then modified.

import numpy as np
import scipy.signal as signal
import io
import requests
from typing import Tuple

fs = 250

def load_gudb(data_path:str, window_size:int=2500) -> Tuple[np.ndarray|Tuple[np.ndarray]]:
    """
    Loads in the GUDb dataset used for R-peak estimation.
    Only the chest strap ECG is extracted and divided into segments of size window_size.
    Subject number 11 is skipped due to containing false labels.

    Parameters
    ----------
    data_path : str
        Path to the GUDb experiment_data folder.
    
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
    .. [1] Howell, L., & Porr, B. (2018). High precision ECG Database with annotated R peaks, 
        recorded and filmed under realistic conditions.
    """
    ecgs = []
    annotations = []
    for exp in GUDb.experiments:
        for subject_number in range(GUDb.total_subjects):
            if(subject_number == 11):
                continue # Subject number 11 has wrong labels
            ecg_class = GUDb(subject_number, exp, url=data_path)
            rpeaks_anno = ecg_class.anno_cs
            if(not ecg_class.anno_cs_exists):
                continue
            chest_strap_ecg = ecg_class.cs_V2_V1 # Taking the chest strap ECG only
            n_samples = chest_strap_ecg.shape[-1]
            segment_points = list(range(window_size, n_samples, window_size))
            if(n_samples < window_size):
                print(f"Skipping subject {subject_number} for experiment {exp} as it only contains {n_samples} samples.")
                continue
            chest_strap_ecg = np.split(chest_strap_ecg, segment_points)
            chest_strap_ecg = chest_strap_ecg[:-1] # Discard the last incomplete segment
            segment_points.insert(0, 0)
            for i, ecg in enumerate(chest_strap_ecg):
                anno = rpeaks_anno[np.logical_and(rpeaks_anno >= segment_points[i], rpeaks_anno < segment_points[i+1])] - segment_points[i]
                ecgs.append(ecg)
                annotations.append(anno)
    return np.stack(ecgs, axis=0), annotations

# Class which loads the dataset
class GUDb:

    experiments = ["sitting","maths","walking","hand_bike","jogging"]
    fs=250
    total_subjects = 25

    def loadDataFromURL(self,url):
        if ("http:" in url) or ("https:" in url):
            s=requests.get(url).content
            c=np.loadtxt(io.StringIO(s.decode('utf-8')))
            return c
        else:
            return np.loadtxt(url)
    
    def __init__(self,_subj,_experiment,url = "https://berndporr.github.io/ECG-GUDB/experiment_data"):
        """Specify the subject number and the experiment. Optional parameter url: different url or local path."""
        self.subj = _subj
        self.experiment = _experiment
        self.subjdir = url+"/"+("subject_%02d" % _subj)+"/"
        self.expdir = self.subjdir+self.experiment+"/"

        self.data=self.loadDataFromURL(self.expdir+"ECG.tsv")
        try:
            self.anno_cs=self.loadDataFromURL(self.expdir+"annotation_cs.tsv").astype(int)
            self.anno_cs_exists=True 
        except:
            self.anno_cs=False
            self.anno_cs_exists=False           
        try:
            self.anno_cables=self.loadDataFromURL(self.expdir+"annotation_cables.tsv").astype(int)
            self.anno_cables_exists=True 
        except:
            self.anno_cables=False
            self.anno_cables_exists=False   

        self.cs_V2_V1 = self.data[:, 0]
        self.einthoven_II = self.data[:, 1]
        self.einthoven_III = self.data[:, 2]
        self.einthoven_I = self.einthoven_II - self.einthoven_III
        self.acc_x = self.data[:, 3]
        self.acc_y = self.data[:, 4]
        self.acc_z = self.data[:, 5]

        self.T=1/self.fs
        self.t = np.linspace(0, self.T*len(self.cs_V2_V1), len(self.cs_V2_V1))


    def filter_data(self):
        """Filters the ECG data with a highpass at 0.1Hz and a bandstop around 50Hz (+/-2 Hz)"""

        b_dc, a_dc = signal.butter(4, (0.1/self.fs*2), btype='highpass')
        b_50, a_50 = signal.butter(4, [(48/self.fs*2),(52/self.fs*2)], btype='stop')

        self.cs_V2_V1_filt = signal.lfilter(b_dc, a_dc, self.cs_V2_V1)
        self.cs_V2_V1_filt = signal.lfilter(b_50, a_50, self.cs_V2_V1_filt)

        self.einthoven_II_filt = signal.lfilter(b_dc, a_dc, self.einthoven_II)
        self.einthoven_II_filt = signal.lfilter(b_50, a_50, self.einthoven_II_filt)

        self.einthoven_III_filt = signal.lfilter(b_dc, a_dc, self.einthoven_III)
        self.einthoven_III_filt = signal.lfilter(b_50, a_50, self.einthoven_III_filt)

        self.einthoven_I_filt = self.einthoven_II_filt-self.einthoven_III_filt

        return