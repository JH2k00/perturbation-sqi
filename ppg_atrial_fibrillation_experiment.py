import medsqi.datasets.deepbeat as deepbeat
import medsqi.datasets.ppg_arrhythmia as ppg_arrhythmia
from medsqi.evaluation.utils import accuracy, get_best_margin
from medsqi.training.feature_based import train_feature_model, get_features
from medsqi.training.deep_learning_based import train_ecgfounder_model, run_ecgfounder_inference
from medsqi.methods.perturbation_methods import PerturbationSQI
from medsqi.training.resnet1d import Res34SimSiam
from medsqi.common import setup_logger, load_config
import numpy as np
import joblib
from scipy.stats import spearmanr
from typing import Dict, Any
import argparse
import torch
import random
import os
import shutil
import logging
import h5py
from collections import OrderedDict

logger = logging.getLogger("medsqi")


def run_experiment(config:Dict[str, Any]):
    # Set seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # Set up SiamAF
    device = torch.device(config["device"])

    state_dict = torch.load(config["siamaf_ckpt_path"], weights_only=True) 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] #remove 'module'
        new_state_dict[name] = v

    state_dict = new_state_dict
    siamAF:torch.nn.Module = Res34SimSiam(512, 128, single_source_mode=True).cuda()
    siamAF.load_state_dict(state_dict)
    siamAF.eval()

    def predict_atrial_fibrillation(x:np.ndarray, batch_size:int=config["siamaf_batch_size"]):
        if(x.ndim == 2):
            x = x[None, ...]
        
        preds = torch.zeros(x.shape[0])
        with torch.inference_mode():
            for k in range(0, x.shape[0], batch_size):
                cur_x = torch.from_numpy(x[k:k+batch_size, ...]).float().to(device)
                preds[k:k+batch_size] = siamAF(cur_x, None)[-1].argmax(dim=-1)
        return preds.cpu().numpy()

    # Load the PPG Arrhythmia dataset and calculate the metrics
    ppgs, labels = ppg_arrhythmia.load_ppg_arrhythmia(config["ppg_arrhythmia_path"])
    fs = ppg_arrhythmia.fs
    metrics = accuracy(labels, predict_atrial_fibrillation(ppgs))
    logger.info("Loaded PPG Arrhythmia dataset and calculated the metrics.")

    # Train and save a model with cross validation to predict the metrics from SQI features
    model = train_feature_model(ppgs.squeeze(1), metrics, fs, config)
    model_save_path = os.path.join(config["output_folder"], "models")
    os.makedirs(model_save_path, exist_ok=True)
    joblib.dump(model, filename=os.path.join(model_save_path, "model.pkl"))
    logger.info("Trained and saved the SQI feature based model.")

    # Finetune and save an ECGFounder model with early stopping to predict the metrics from the raw signals
    finetuned_ecgfounder = train_ecgfounder_model(ppgs.squeeze(1), metrics, fs, config)
    torch.save({"state_dict": finetuned_ecgfounder.state_dict()}, os.path.join(model_save_path, "finetuned_ecgfounder.pt"))
    logger.info("Trained and saved the DL based model.")

    # Evaluate on Deepbeat
    ppgs, labels = deepbeat.load_deepbeat(config["deepbeat_path"])
    fs = deepbeat.fs
    metrics = accuracy(labels, predict_atrial_fibrillation(ppgs))
    logger.info("Loaded the Deepbeat dataset and calculated the metrics.")

    # Evaluate the DL model
    preds_dl = run_ecgfounder_inference(finetuned_ecgfounder, ppgs.squeeze(1), fs, config)
    logger.info("Evaluated the DL model on MIT-BIH AF")

    features = get_features(ppgs.squeeze(1), fs, config)
    preds = model.predict(features)
    if(not config["model"]["classification"]):
        preds = np.clip(preds, *config["model"]["output_clip"])
    logger.info("Evaluated the model on Deepbeat")

    # Evaluate the Perturbation SQI on GUDb
    psqi = PerturbationSQI(algorithm=predict_atrial_fibrillation,
                           metric=accuracy,
                           fs=fs, **config["psqi"])
    preds_psqi = psqi.predict(ppgs)
    logger.info("Evaluated the perturbation SQI on Deepbeat")

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
    parser = argparse.ArgumentParser(description='Run the PPG atrial fibrillation SQI experiment.')
    parser.add_argument('--config_path', help='Path to the config file', type=str, required=True)

    args = parser.parse_args()
    config = load_config(args.config_path)

    # Copy config to the output folder and setup logger before running the experiment
    os.makedirs(config["output_folder"], exist_ok=True)
    shutil.copy(args.config_path, config["output_folder"])
    setup_logger(os.path.join(config["output_folder"], "experiment_logs.log"))

    # Run experiment
    run_experiment(config)
