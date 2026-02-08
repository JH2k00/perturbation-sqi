import numpy as np

def calculate_bsqi(beats_alg1:np.ndarray, beats_alg2:np.ndarray, tol:float=0.150) -> float:
    """
    Calculates the Beat SQI (bSQI):
    The ratio of beats detected synchronously by two different algorithms.

    Parameters
    ----------
    beats_alg1 : ndarray
        1D numpy array containing the position of the beats detected by the first algorithm in seconds. 

    beats_alg2 : ndarray
        1D numpy array containing the position of the beats detected by the second algorithm in seconds. 
    
    tol : float
        The absolute tolerance in seconds for two beats to be considered matched.

    Returns
    -------
    bsqi : float
        The beat SQI.

    References
    ----------
    .. [1] Clifford, G. D., Lopez, D., Li, Q., & Rezek, I. (2011, September). 
        Signal quality indices and data fusion for determining acceptability of electrocardiograms collected in noisy 
        ambulatory environments. In 2011 Computing in Cardiology (pp. 285-288). IEEE.
    """

    assert (beats_alg1.ndim == 1) and (beats_alg2.ndim == 1), "beats_alg1 and beats_alg2 must be 1D."
    if (beats_alg1.shape[0] <= 0) and (beats_alg2.shape[0] <= 0):
        print("WARNING: Both of the algorithms detected no beats. Returning 1.0")
        return 1.0

    matched = 0
    used_bt2 = set()
    for bt1 in beats_alg1:
        for i, bt2 in enumerate(beats_alg2):
            if i in used_bt2:
                continue
            if abs(bt1 - bt2) <= tol:
                matched += 1
                used_bt2.add(i)           
    n_all = beats_alg1.shape[0] + beats_alg2.shape[0] - matched
    return matched / n_all