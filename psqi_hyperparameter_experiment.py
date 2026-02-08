import medsqi.datasets.gudb as gudb
import medsqi.datasets.mit_bih_af as mit_bih_af
import medsqi.datasets.deepbeat as deepbeat
from medsqi.evaluation.utils import f1_with_tolerance, get_best_margin, accuracy
from medsqi.training.net1d import Net1D
from medsqi.training.resnet1d import Res34SimSiam
from medsqi.methods.perturbation_methods import PerturbationSQI
from medsqi.common import setup_logger, load_config
import numpy as np
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
from collections import OrderedDict

logger = logging.getLogger("medsqi")

def get_metrics_gudb(ecgs:np.ndarray, annotations:Tuple[np.ndarray], fs:float, config:Dict[str, Any]) -> np.ndarray:
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

    # Create a folder to save preds, metrics, and correlation coefficients
    results_save_path = os.path.join(config["output_folder"], "results")
    os.makedirs(results_save_path, exist_ok=True)

    global_snr_list = config["psqi"].pop("global_snr")
    ######################################################################### Peak detection experiment #########################################################################

    # Load in GUDb
    ecgs, annotations = gudb.load_gudb(config["gudb_path"])
    fs = gudb.fs
    metrics = get_metrics_gudb(ecgs, annotations, fs, config)
    logger.info("Loaded GUDb dataset and calculated the metrics.")

    # Evaluate the Perturbation SQI on GUDb with a hyperparameter sweep
    h5_file_path = os.path.join(results_save_path, "results_gudb.h5")
    best_margins_gudb = []
    with h5py.File(h5_file_path, "w") as f:
        f.create_dataset("metrics", data=metrics)
        for global_snr in global_snr_list:
            psqi = PerturbationSQI(algorithm=lambda x: FastNVG(sampling_frequency=fs).find_peaks(x)/fs,
                                metric=partial(f1_with_tolerance, tol=config["f1_tol"]), global_snr=global_snr,
                                fs=fs, **config["psqi"])
            preds = psqi.predict(ecgs)

            dset_psqi = f.create_dataset(f"psqi_preds_snr_{global_snr}", data=preds)
            dset_psqi.attrs['spearman_corr'] = spearmanr(preds, metrics)[0]
            dset_psqi.attrs['pearson_corr'] = np.corrcoef(preds, metrics)[1, 0]
            best_margin, best_thres_psqi = get_best_margin(metrics, preds)
            dset_psqi.attrs['best_margin'] = best_margin
            dset_psqi.attrs['best_thres'] = best_thres_psqi

            best_margins_gudb.append(best_margin)

    logger.info("Evaluated the perturbation SQI on GUDb")
    logger.info(f"Saved metrics and preds on GUDb in the following h5 file: {h5_file_path}")

    ######################################################################### ECG AF experiment #################################################################################

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

    # Evaluate on MIT-BIH AF
    ecgs, labels = mit_bih_af.load_mit_bih_af(config["mit_bih_af_path"])
    fs = mit_bih_af.fs
    metrics = accuracy(labels, predict_atrial_fibrillation(ecgs))
    logger.info("Loaded MIT-BIH dataset and calculated the metrics.")

    # Evaluate the Perturbation SQI on MIT-BIH AF with a hyperparameter sweep
    h5_file_path = os.path.join(results_save_path, "results_mit_bih_af.h5")
    best_margins_mit_bih_af = []
    with h5py.File(h5_file_path, "w") as f:
        f.create_dataset("metrics", data=metrics)
        for global_snr in global_snr_list:
            psqi = PerturbationSQI(algorithm=predict_atrial_fibrillation,
                                metric=accuracy, global_snr=global_snr,
                                fs=fs, **config["psqi"])
            preds = psqi.predict(ecgs)

            dset_psqi = f.create_dataset(f"psqi_preds_snr_{global_snr}", data=preds)
            dset_psqi.attrs['spearman_corr'] = spearmanr(preds, metrics)[0]
            dset_psqi.attrs['pearson_corr'] = np.corrcoef(preds, metrics)[1, 0]
            best_margin, best_thres_psqi = get_best_margin(metrics, preds)
            dset_psqi.attrs['best_margin'] = best_margin
            dset_psqi.attrs['best_thres'] = best_thres_psqi

            best_margins_mit_bih_af.append(best_margin)

    logger.info("Evaluated the perturbation SQI on MIT-BIH AF")
    logger.info(f"Saved metrics and preds on MIT-BIH AF in the following h5 file: {h5_file_path}")

    del ecg_founder

    ######################################################################### PPG AF experiment #################################################################################
    
    # Set up SiamAF
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
    
    # Evaluate on Deepbeat
    ppgs, labels = deepbeat.load_deepbeat(config["deepbeat_path"])
    fs = deepbeat.fs
    metrics = accuracy(labels, predict_atrial_fibrillation(ppgs))
    logger.info("Loaded the Deepbeat dataset and calculated the metrics.")

    # Evaluate the Perturbation SQI on Deepbeat with a hyperparameter sweep
    h5_file_path = os.path.join(results_save_path, "results_deepbeat.h5")
    best_margins_deepbeat = []
    with h5py.File(h5_file_path, "w") as f:
        f.create_dataset("metrics", data=metrics)
        for global_snr in global_snr_list:
            psqi = PerturbationSQI(algorithm=predict_atrial_fibrillation,
                                metric=accuracy, global_snr=global_snr,
                                fs=fs, **config["psqi"])
            preds = psqi.predict(ppgs)

            dset_psqi = f.create_dataset(f"psqi_preds_snr_{global_snr}", data=preds)
            dset_psqi.attrs['spearman_corr'] = spearmanr(preds, metrics)[0]
            dset_psqi.attrs['pearson_corr'] = np.corrcoef(preds, metrics)[1, 0]
            best_margin, best_thres_psqi = get_best_margin(metrics, preds)
            dset_psqi.attrs['best_margin'] = best_margin
            dset_psqi.attrs['best_thres'] = best_thres_psqi

            best_margins_deepbeat.append(best_margin)

    logger.info("Evaluated the perturbation SQI on Deepbeat")
    logger.info(f"Saved metrics and preds on Deepbeat in the following h5 file: {h5_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the perturbation SQI hyperparameter experiment.')
    parser.add_argument('--config_path', help='Path to the config file', type=str, required=True)

    args = parser.parse_args()
    config = load_config(args.config_path)

    # Copy config to the output folder and setup logger before running the experiment
    local_snr = config["psqi"]["local_snr"]
    config["output_folder"] = os.path.join(config["output_folder"], f"local_snr_{local_snr:.2f}") # Create an output folder based on the local_snr value
    os.makedirs(config["output_folder"], exist_ok=True)
    shutil.copy(args.config_path, config["output_folder"])
    setup_logger(os.path.join(config["output_folder"], "experiment_logs.log"))

    # Run experiment
    run_experiment(config)
