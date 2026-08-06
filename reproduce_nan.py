import numpy as np
import sys
import os

# Add current dir and src to path
sys.path.append(os.getcwd())

from src.likelihoods import total_log_posterior, log_likelihood_CC, log_likelihood_SN
from src.base_model import CosmologyModel

class NanModel(CosmologyModel):
    def __init__(self):
        super().__init__("NanModel", ["p"], [(0, 1)])
    def H(self, z, p):
        return np.array([np.nan, 1.0, 1.0])

def test_nan_leak():
    model = NanModel()
    data_CC = {"z": np.array([0.1, 0.2, 0.3]), "Hz": np.array([70, 75, 80]), "sigma_Hz": np.array([1, 1, 1])}
    data_SN = {"z": np.array([0.1, 0.2, 0.3]), "mu": np.array([35, 36, 37]), "sigma_mu": np.array([1, 1, 1])}
    
    datasets = [
        (log_likelihood_CC, data_CC),
        (log_likelihood_SN, data_SN)
    ]
    
    params = [0.5]
    
    print("Testing total_log_posterior with NaN-producing model...")
    result = total_log_posterior(params, model, datasets)
    print(f"Result: {result}")
    
    if np.isnan(result):
        print("FAIL: log_posterior returned NaN!")
    else:
        print("SUCCESS: log_posterior caught the NaN and returned finite/inf.")

if __name__ == "__main__":
    test_nan_leak()
