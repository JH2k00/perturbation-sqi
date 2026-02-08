import medsqi.datasets.mit_bih_af as mit_bih_af
import medsqi.datasets.af_2017 as af_2017
from medsqi.evaluation.utils import accuracy, get_best_margin
from medsqi.training.feature_based import train_feature_model, get_features
from medsqi.training.deep_learning_based import train_ecgfounder_model, run_ecgfounder_inference
from medsqi.methods.perturbation_methods import PerturbationSQI
from medsqi.training.net1d import Net1D
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

logger = logging.getLogger("medsqi")


def run_experiment(config:Dict[str, Any]):
    # Set seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # Set up ECGFounder
    device = torch.device(config["device"])

    ecg_founder = Net1D(
        in_channels=1, 
        base_filters=64, #32 64
        ratio=1, 
        filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
        m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
        kernel_size=16, 
        stride=2, 
        groups_width=16,
        verbose=False, 
        use_bn=False,
        use_do=False,
        return_features=False,
        n_classes=150)
    
    checkpoint = torch.load(config["ecgfounder"]["ckpt_path"], weights_only=False) 
    state_dict = checkpoint['state_dict']
    ecg_founder.load_state_dict(state_dict)

    ecg_founder.eval()
    ecg_founder.to(device)

    def predict_atrial_fibrillation(x:np.ndarray, batch_size:int=config["ecgfounder"]["batch_size"]):
        if(x.ndim == 2):
            x = x[None, ...]
        
        preds = torch.zeros(x.shape[0])
        with torch.inference_mode():
            for k in range(0, x.shape[0], batch_size):
                cur_x = torch.from_numpy(x[k:k+batch_size, ...]).float().to(device)
                preds[k:k+batch_size] = ecg_founder(cur_x)[..., 5].sigmoid().round().int()
        return preds.cpu().numpy()

    # Load the AF 2017 dataset and calculate the metrics
    ecgs, labels = af_2017.load_af_2017(config["af_2017_path"], config["af_2017_csv_ref_path"])
    fs = af_2017.fs
    metrics = accuracy(labels, predict_atrial_fibrillation(ecgs))
    logger.info("Loaded AF 2017 dataset and calculated the metrics.")

    # Train and save a model with cross validation to predict the metrics from SQI features
    model = train_feature_model(ecgs.squeeze(1), metrics, fs, config)
    model_save_path = os.path.join(config["output_folder"], "models")
    os.makedirs(model_save_path, exist_ok=True)
    joblib.dump(model, filename=os.path.join(model_save_path, "model.pkl"))
    logger.info("Trained and saved the SQI feature based model.")

    # Finetune and save an ECGFounder model with early stopping to predict the metrics from the raw signals
    finetuned_ecgfounder = train_ecgfounder_model(ecgs.squeeze(1), metrics, fs, config)
    torch.save({"state_dict": finetuned_ecgfounder.state_dict()}, os.path.join(model_save_path, "finetuned_ecgfounder.pt"))
    logger.info("Trained and saved the DL based model.")

    # Evaluate on MIT-BIH AF
    ecgs, labels = mit_bih_af.load_mit_bih_af(config["mit_bih_af_path"])
    fs = mit_bih_af.fs
    metrics = accuracy(labels, predict_atrial_fibrillation(ecgs))
    logger.info("Loaded MIT-BIH dataset and calculated the metrics.")

    features = get_features(ecgs.squeeze(1), fs, config)
    preds = model.predict(features)
    if(not config["model"]["classification"]):
        preds = np.clip(preds, *config["model"]["output_clip"])
    logger.info("Evaluated the model on MIT-BIH AF")

    # Evaluate the DL model
    preds_dl = run_ecgfounder_inference(finetuned_ecgfounder, ecgs.squeeze(1), fs, config)
    logger.info("Evaluated the DL model on MIT-BIH AF")

    # Evaluate the Perturbation SQI on GUDb
    psqi = PerturbationSQI(algorithm=predict_atrial_fibrillation,
                           metric=accuracy,
                           fs=fs, **config["psqi"])
    preds_psqi = psqi.predict(ecgs)
    logger.info("Evaluated the perturbation SQI on MIT-BIH AF")

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
    parser = argparse.ArgumentParser(description='Run the ECG atrial fibrillation SQI experiment.')
    parser.add_argument('--config_path', help='Path to the config file', type=str, required=True)

    args = parser.parse_args()
    config = load_config(args.config_path)

    # Copy config to the output folder and setup logger before running the experiment
    os.makedirs(config["output_folder"], exist_ok=True)
    shutil.copy(args.config_path, config["output_folder"])
    setup_logger(os.path.join(config["output_folder"], "experiment_logs.log"))

    # Run experiment
    run_experiment(config)
