import numpy as np

def f1_with_tolerance(label:np.ndarray, preds:np.ndarray, tol:float=0.02) -> float:
    """
    Calculates the F1 score for peak detection by considering peaks within a certain tolerance as a match.

    Parameters
    ----------
    label : ndarray
        1D numpy array containing the true position of the peaks in seconds. 

    preds : ndarray
        1D numpy array containing the position of the peaks detected by the algorithm in seconds. 
    
    tol : float
        The absolute tolerance in seconds for two beats to be considered matched.

    Returns
    -------
    F1 : float
        The F1 score.
    """ 
    used_preds = set()
    tp = 0
    for a in label:
        for i, b in enumerate(preds):
            if i in used_preds:
                continue
            if abs(a - b) <= tol:
                tp += 1
                used_preds.add(i)
                break
    
    fp = len(preds) - tp
    fn = len(label) - tp
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-24)
    
    return f1

def accuracy(y:np.ndarray, y_hat:np.ndarray):
    acc = y == y_hat.round()
    if(acc.size == 1):
        return acc[0]
    return acc


def get_best_margin(y:np.ndarray, y_hat:np.ndarray, min_n_samples:int=5):
    N = len(y)
    idx = np.argsort(y_hat)
    cumsum_sorted = np.cumsum(y[idx])
    total = cumsum_sorted[-1]
    best, best_thres = -np.inf, -np.inf
    idx = list(idx)

    unique_thres = set()
    unique_thres.add(y_hat[idx[0]])

    for k in range(2, N):
        if k-1 < min_n_samples or (N - k + 1) < min_n_samples: # Avoid outliers dominating the calculation
            continue
        thres = y_hat[idx[k-1]]
        if(thres in unique_thres): # Only consider the first instance of a threshold.
            continue
        mean_low = cumsum_sorted[k - 2] / (k - 1) # The lower mean includes all samples strictly smaller than the threshold
        mean_high = (total - cumsum_sorted[k - 2]) / (N - k + 1) # The upper mean includes all samples bigger or equal to the threshold
        diff = mean_high - mean_low
        unique_thres.add(thres)

        if(diff > best):
            best = diff # Track the highest possible margin between sample groups based on a threshold separation.
            best_thres = y_hat[idx[k-1]]

    return best, best_thres