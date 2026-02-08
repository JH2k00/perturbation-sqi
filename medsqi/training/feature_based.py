from medsqi.methods.statistical_methods import calculate_fsqi, calculate_ssqi, calculate_ksqi, calculate_spsqi, calculate_hr_sqi_feats
from medsqi.methods.matching_methods import calculate_bsqi
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import neurokit2 as nk
import xgboost
from typing import Dict, Any
from vg_beat_detectors.fast_nvg import FastNVG
from contextlib import redirect_stdout

models:Dict[str, BaseEstimator] = {
    "xgboost_reg": xgboost.XGBRegressor,
    "xgboost_clf": xgboost.XGBClassifier,
    "randomforest_reg": RandomForestRegressor,
    "randomforest_clf": RandomForestClassifier,
    "svr": SVR,
    "svc": SVC
}

class NullWriter:
    def write(self, arg): pass
    def flush(self): pass
null_writer = NullWriter()

def get_features(signals:np.ndarray, fs:float, config:Dict[str, Any]) -> np.ndarray:
    feats = np.stack([calculate_fsqi(signals, fs=fs), calculate_ssqi(signals), calculate_ksqi(signals), calculate_spsqi(signals)], axis=-1)
    hr_sqi_feats = []
    if(config["signal"].lower() == "ecg"):
        nvg = FastNVG(sampling_frequency=fs)
        def peak_detector_1(x):
            with redirect_stdout(null_writer): # Suppress the warnings from FastNVG
                return nvg.find_peaks(x)
        peak_detector_2 = lambda x: nk.ecg_peaks(nk.ecg_clean(x), sampling_rate=fs, method="neurokit")[1]["ECG_R_Peaks"]
    elif(config["signal"].lower() == "ppg"): # The PPG peak detection algorithms seem to be pretty slow
        peak_detector_1 = lambda x: nk.ppg_peaks(nk.ppg_clean(x), sampling_rate=fs, method="elgendi")[1]["PPG_Peaks"]
        peak_detector_2 = lambda x: nk.ppg_peaks(nk.ppg_clean(x), sampling_rate=fs, method="charlton")[1]["PPG_Peaks"]
    else:
        raise ValueError("signal in config is not one of [ecg, ppg], but the following" + config["signal"].lower())
    for i in range(signals.shape[0]):
        peaks_1 = peak_detector_1(signals[i, :])
        peaks_2 = peak_detector_2(signals[i, :])
        hr_sqi_feats.append([*calculate_hr_sqi_feats(signals[i, :], peaks_1, fs), calculate_bsqi(peaks_1/fs, peaks_2/fs)])

    feats = np.concatenate([feats, np.stack(hr_sqi_feats, axis=0)], axis=-1)
    return feats

def train_feature_model(signals:np.ndarray, metrics:np.ndarray, fs:float, config:Dict[str, Any]):
    model_name = config["model"]["model_name"]
    model:BaseEstimator = models[model_name](**config["model"]["hyperparameters"])
    model = GridSearchCV(estimator=model, **config["cv"])

    features = get_features(signals, fs, config)
    model.fit(features, metrics)
    return model