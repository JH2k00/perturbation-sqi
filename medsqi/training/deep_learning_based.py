import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from medsqi.training.net1d import Net1D
from medsqi.datasets.utils import normalize_for_ECGFounder
import torch
import copy

def run_ecgfounder_inference(ecgfounder:Net1D, signals:np.ndarray, fs:float, config:Dict[str, Any]) -> np.ndarray:
    signals_processed = normalize_for_ECGFounder(signals[:, np.newaxis, :])
    preds_dl = torch.zeros(signals_processed.shape[0])
    ecgfounder.eval()
    with torch.inference_mode():
        for k in range(0, signals_processed.shape[0], config["ecgfounder"]["batch_size"]):
            cur_x = torch.from_numpy(signals_processed[k:k+config["ecgfounder"]["batch_size"], ...]).float().to(config["device"])
            output = ecgfounder(cur_x)
            if(config["ecgfounder"]["classification"]):
                output = output.argmax(dim=-1)
            else:
                output = torch.clamp(output, *config["ecgfounder"]["output_clip"])
            preds_dl[k:k+config["ecgfounder"]["batch_size"]] = output.squeeze(-1)
    return preds_dl.cpu().numpy()

def train_ecgfounder_model(signals:np.ndarray, metrics:np.ndarray, fs:float, config:Dict[str, Any]) -> Net1D:
    signals_processed = normalize_for_ECGFounder(signals[:, np.newaxis, :])
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
        n_classes=config["ecgfounder"]["n_classes"] if config["ecgfounder"]["classification"] else 1)
    
    checkpoint = torch.load(config["ecgfounder"]["ckpt_path"], weights_only=False) 
    state_dict = checkpoint['state_dict']
    state_dict.pop('dense.weight')
    state_dict.pop('dense.bias')

    ecg_founder.load_state_dict(state_dict, strict=False)
    ecg_founder.to(device)

    X_train, X_val, y_train, y_val = train_test_split(signals_processed, metrics, **config["ecgfounder"]["train_test_split"])

    if(config["ecgfounder"]["classification"]):
        y_train = torch.from_numpy(y_train).long()
        y_val = torch.from_numpy(y_val).long()
    else:
        y_train = torch.from_numpy(y_train).float().unsqueeze(-1)
        y_val = torch.from_numpy(y_val).float().unsqueeze(-1)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["ecgfounder"]["batch_size"], shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(), y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["ecgfounder"]["batch_size"], shuffle=True)

    optimizer = torch.optim.AdamW(params=ecg_founder.parameters(), **config["ecgfounder"]["optimizer"])
    loss_fn = torch.nn.CrossEntropyLoss() if config["ecgfounder"]["classification"] else torch.nn.MSELoss()
    
    best_state_dict = None
    best_loss = np.inf
    n_patient_epochs = 0
    for i in range(config["ecgfounder"]["epochs"]):
        # Training
        ecg_founder.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            preds = ecg_founder(x)
            if not config["ecgfounder"]["classification"]:
                preds = torch.clamp(preds, *config["ecgfounder"]["output_clip"])
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
        
        # Validating
        with torch.no_grad():
            ecg_founder.eval()
            total_loss = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = ecg_founder(x)
                if not config["ecgfounder"]["classification"]:
                    preds = torch.clamp(preds, *config["ecgfounder"]["output_clip"])
                total_loss += loss_fn(preds, y).item()*x.shape[0] # Transform mean back to sum
            if(total_loss < best_loss):
                best_state_dict = copy.deepcopy(ecg_founder.state_dict())
                best_loss = total_loss
                n_patient_epochs = 0
            else:
                n_patient_epochs += 1

        # Early stopping
        if(n_patient_epochs >= config["ecgfounder"]["patience"]):
            break
    
    ecg_founder.load_state_dict(best_state_dict)
    ecg_founder.eval()
    return ecg_founder