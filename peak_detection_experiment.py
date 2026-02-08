import medsqi.datasets.gudb as gudb
import medsqi.datasets.ptt_ppg as ptt_ppg
from medsqi.evaluation.utils import f1_with_tolerance, get_best_margin
from medsqi.training.feature_based import train_feature_model, get_features
from medsqi.training.deep_learning_based import train_ecgfounder_model, run_ecgfounder_inference
from medsqi.methods.perturbation_methods import PerturbationSQI
from medsqi.common import setup_logger, load_config
import numpy as np
import joblib
from scipy.stats import spearmanr
from typing import Dict, Any, Tuple
from vg_beat_detectors.fast_nvg import FastNVG
import argparse
import torch
import random
import os
import shutil
import logging
import h5py
from functools import partial

logger = logging.getLogger("medsqi")

def get_metrics(ecgs:np.ndarray, annotations:Tuple[np.ndarray], fs:float, config:Dict[str, Any]) -> np.ndarray:
    nvg = FastNVG(sampling_frequency=fs)
    metrics = []
    for i in range(ecgs.shape[0]):
        peaks = nvg.find_peaks(ecgs[i, :]) / fs
        metrics.append(f1_with_tolerance(annotations[i]/fs, peaks, tol=config["f1_tol"]))
    metrics = np.stack(metrics, axis=0)
    return metrics

def run_experiment(config:Dict[str, Any]):
    # Set seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # Load the PTT PPG dataset and calculate the metrics
    ecgs, annotations = ptt_ppg.load_ptt_ppg(config["ptt_ppg_path"])
    fs = ptt_ppg.fs
    metrics = get_metrics(ecgs, annotations, fs, config)
    logger.info("Loaded PTT PPG dataset and calculated the metrics.")
    
    # Train and save a feature model with cross validation to predict the metrics from SQI features
    model = train_feature_model(ecgs, metrics, fs, config)
    model_save_path = os.path.join(config["output_folder"], "models")
    os.makedirs(model_save_path, exist_ok=True)
    joblib.dump(model, filename=os.path.join(model_save_path, "model.pkl"))
    logger.info("Trained and saved the SQI feature based model.")

    # Finetune and save an ECGFounder model with early stopping to predict the metrics from the raw signals
    finetuned_ecgfounder = train_ecgfounder_model(ecgs, metrics, fs, config)
    torch.save({"state_dict": finetuned_ecgfounder.state_dict()}, os.path.join(model_save_path, "finetuned_ecgfounder.pt"))
    logger.info("Trained and saved the DL based model.")

    # Evaluate on GUDb
    ecgs, annotations = gudb.load_gudb(config["gudb_path"])
    fs = gudb.fs
    metrics = get_metrics(ecgs, annotations, fs, config)
    logger.info("Loaded GUDb dataset and calculated the metrics.")

    features = get_features(ecgs, fs, config)
    preds = model.predict(features)
    if(not config["model"]["classification"]):
        preds = np.clip(preds, *config["model"]["output_clip"])
    logger.info("Evaluated the SQI feature model on GUDb")

    # Evaluate the DL model
    preds_dl = run_ecgfounder_inference(finetuned_ecgfounder, ecgs, fs, config)
    logger.info("Evaluated the DL model on GUDb")

    # Evaluate the Perturbation SQI on GUDb
    psqi = PerturbationSQI(algorithm=lambda x: FastNVG(sampling_frequency=fs).find_peaks(x)/fs,
                           metric=partial(f1_with_tolerance, tol=config["f1_tol"]),
                           fs=fs, **config["psqi"])
    preds_psqi = psqi.predict(ecgs)
    logger.info("Evaluated the perturbation SQI on GUDb")

    # Save preds, metrics, and correlation coefficients
    results_save_path = os.path.join(config["output_folder"], "results")
    os.makedirs(results_save_path, exist_ok=True)
    h5_file_path = os.path.join(results_save_path, "results.h5")
    with h5py.File(h5_file_path, "w") as f:
        dset_features = f.create_dataset("feature_model_preds", data=preds)
        dset_features.attrs['spearman_corr'] = spearmanr(preds, metrics)[0]
        dset_features.attrs['pearson_corr'] = np.corrcoef(preds, metrics)[1, 0]
        best_margin, best_thres_features = get_best_margin(metrics, preds)
        dset_features.attrs['best_margin'] = best_margin
        dset_features.attrs['best_thres'] = best_thres_features

        dset_dl = f.create_dataset("dl_model_preds", data=preds_dl)
        dset_dl.attrs['spearman_corr'] = spearmanr(preds_dl, metrics)[0]
        dset_dl.attrs['pearson_corr'] = np.corrcoef(preds_dl, metrics)[1, 0]
        best_margin, best_thres_dl = get_best_margin(metrics, preds_dl)
        dset_dl.attrs['best_margin'] = best_margin
        dset_dl.attrs['best_thres'] = best_thres_dl

        dset_psqi = f.create_dataset("psqi_preds", data=preds_psqi)
        dset_psqi.attrs['spearman_corr'] = spearmanr(preds_psqi, metrics)[0]
        dset_psqi.attrs['pearson_corr'] = np.corrcoef(preds_psqi, metrics)[1, 0]
        best_margin, best_thres_psqi = get_best_margin(metrics, preds_psqi)
        dset_psqi.attrs['best_margin'] = best_margin
        dset_psqi.attrs['best_thres'] = best_thres_psqi

        f.create_dataset("metrics", data=metrics)
    logger.info(f"Saved metrics and preds in the following h5 file: {h5_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the peak detection SQI experiment.')
    parser.add_argument('--config_path', help='Path to the config file', type=str, required=True)

    args = parser.parse_args()
    config = load_config(args.config_path)

    # Copy config to the output folder and setup logger before running the experiment
    os.makedirs(config["output_folder"], exist_ok=True)
    shutil.copy(args.config_path, config["output_folder"])
    setup_logger(os.path.join(config["output_folder"], "experiment_logs.log"))

    # Run experiment
    run_experiment(config)
