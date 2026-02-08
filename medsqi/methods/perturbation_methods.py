import numpy as np
from typing import Callable, Any, Tuple
from scipy.signal import butter, sosfiltfilt
import cma
from tqdm import tqdm
from math import sqrt

class PerturbationSQI():
    """
    Implementation of the task- and metric-specific perturbation signal quality index (pSQI) [1].
    Given a fixed algorithm and metric, the pSQI calculates the worst-case value of the metric 
    under an additive, colored Gaussian noise peturbation with a lower-bounded signal-to-noise ratio.
    The pSQI has been empirically shown to correlate with the algorithm's performance on the input 
    signal, as evaluated by the metric [1]. 

    Parameters
    ----------
    algorithm : Callable[[ndarray], Any]
        The algorithm for the task at hand. Should accept a 1D ndarray as input and may return any output.
    
    metric : Callable[[Any, Any], float]
        The metric for the task at hand. Should accept two algorithm outputs (prediction, ground_truth) as input and return a float, where higher values indicate better performance.
    
    fs : float
        The signal's sampling frequency in Hz.
    
    f_highpass : float
        The highpass cutoff frequency in Hz. The highpass is used to filter the input signal before estimating its signal power.

    global_snr : float
        The global SNR in dB. Is used to scale the noise power.

    local_snr : float
        The local SNR in dB. Is used to clip the sample-wise noise amplitude.

    maxiter : int
        The maximum number of iterations for CMA-ES.

    popsize : int
        The population size of CMA-ES

    random_state : int
        The random seed.

    References
    ----------
    .. [1] Haidamous, J., & Hoog Antink, C. (2026). 
        Task and Metric Specific Signal Quality Indices for Medical Time-Series. Submitted.
    """
    def __init__(self, 
                 *,
                 algorithm:Callable[[np.ndarray], Any], 
                 metric:Callable[[Any, Any], float],
                 fs:float, # in Hz
                 f_highpass:float=0.5, # in Hz
                 global_snr:float=25, # in dB
                 local_snr:float=10, # in dB
                 maxiter:int=2,
                 popsize:int=5,
                 random_state:int|None=None): 
        self.fs = fs
        self.random_state = random_state
        self.maxiter = maxiter
        self.metric = metric
        self.popsize = popsize
        self.algorithm = algorithm
        self.global_snr = global_snr
        self.f_highpass = f_highpass
        self.max_rel_deviation = 1/sqrt(10**(local_snr/10)) # From the formula : SNR_{dB} = 10log_{10}(P_signal/P_noise)
        self.rng = np.random.default_rng(seed=self.random_state)
        self._context = None
        self.f_low = 0.1
        self.delta = self.fs/2-0.2
    
    def eval_one_suggestion(self, params):
        signal, y_hat, noise, signal_power, max_amplitude = self._context
        f_low, delta = params
        f_high = min(self.fs/2-0.1, delta+f_low)

        x_perturbed = self.add_perturbation_vectorized(signal, f_low, f_high, self.global_snr, noise, signal_power, max_amplitude)
        metrics_perturbed = float(self.metric(y_hat, self.algorithm(x_perturbed)))
        return metrics_perturbed

    def predict(self, X:np.ndarray|Tuple[np.ndarray]):
        res = []
        # Create highpass filter to remove baseline wander from the signal for the power estimation.
        sos_hp = butter(N=6, Wn=self.f_highpass, btype="highpass", output="sos", fs=self.fs) 
        for signal in tqdm(X):
            y_hat = self.algorithm(signal)
            noise = self.rng.standard_normal(signal.shape) # Sample white gaussian noise from N(0,1)

            signal_hp = sosfiltfilt(sos_hp, signal, axis=-1)
            signal_power = np.square(signal_hp).mean(axis=-1)
            max_amplitude = np.abs(signal_hp)*self.max_rel_deviation

            self._context = (signal, y_hat, noise, signal_power, max_amplitude)

            x, es = cma.fmin2(objective_function=self.eval_one_suggestion, 
                              x0=[self.f_low, self.delta], 
                              sigma0=self.fs/6, 
                              options={"bounds": [[0.1, 0.1], [self.fs/2-0.2, self.fs/2-0.2]], "verb_disp": 0, 
                                       "verb_log": 0, "maxiter": self.maxiter, "popsize": self.popsize,
                                       "seed": np.nan, "randn": lambda x, y: self.rng.standard_normal((x, y))})
            res.append(es.best.f)
            self.f_low, self.delta = es.result.xbest
        self._context = None
        return np.array(res)
    
    def add_perturbation_vectorized(self, 
                                    x:np.ndarray, 
                                    f_low:float, 
                                    f_high:float, 
                                    snr:float,
                                    noise:np.ndarray,
                                    signal_power:np.ndarray,
                                    max_amplitude:np.ndarray) -> np.ndarray:
        desired_noise_power = signal_power / 10**(snr/10) # From the formula : SNR_{dB} = 10log_{10}(P_signal/P_noise)

        # Create a bandpass filter and filter the white noise to create colored noise which has a zero periodogram outside of the frequency region of interest.
        sos = butter(N=6, Wn=[f_low, f_high], btype="bandpass", output="sos", fs=self.fs)
        noise_filt = sosfiltfilt(sos, noise, axis=-1)

        cur_noise_power = np.square(noise_filt).mean(axis=-1)
        noise_filt *= np.sqrt(desired_noise_power/cur_noise_power) # Multiply by the sqrt of power ratios to get the desired SNR
        noise_filt = np.clip(noise_filt, -max_amplitude, max_amplitude) # Clip the noise power to a max for each sample

        return x + noise_filt

    def set_fs(self, fs:float):
        self.fs = fs