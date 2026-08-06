#!/usr/bin/env python3
# benchmark.py — Python MCMC benchmark measuring Effective Sample Size (ESS) per second

import os
import sys
import time
import numpy as np
import emcee

# Add current dir and src to path
sys.path.append(os.getcwd())

from src.data_utils import load_CC, load_PPSH0ES
from src.models import LCDMModel, wCDMModel, CPLModel
from src.likelihoods import log_likelihood_CC, log_likelihood_SN, total_log_posterior

def main():
    cc_path = "Cosmic_chronometers_data.tex"
    sn_path = "Pantheon+SH0ES.dat.txt"
    
    if not os.path.exists(cc_path) or not os.path.exists(sn_path):
        print("Error: Data files not found. Run from the repository root.")
        sys.exit(1)
        
    data_CC = load_CC(cc_path)
    data_SN = load_PPSH0ES(sn_path)
    datasets = [
        (log_likelihood_CC, data_CC),
        (log_likelihood_SN, data_SN)
    ]
    
    model_name = sys.argv[1].upper() if len(sys.argv) > 1 else "LCDM"
    if model_name == "LCDM":
        model = LCDMModel()
        initial_pos = np.array([70.0, 0.3])
    elif model_name == "WCDM":
        model = wCDMModel()
        initial_pos = np.array([70.0, 0.3, -0.8])
    elif model_name == "CPL":
        model = CPLModel()
        initial_pos = np.array([70.0, 0.3, -0.8, -0.1])
    else:
        print(f"Unknown model: {model_name}. Use LCDM, wCDM, or CPL.")
        sys.exit(1)
        
    nwalkers = 40
    nsteps = 5000
    burn_in = 1000
    ndim = len(model.param_names)
    
    # Start walkers in a small ball
    pos = initial_pos + 1e-3 * np.random.randn(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, total_log_posterior, args=(model, datasets))
    
    print("Running Python emcee benchmark...")
    start_time = time.time()
    sampler.run_mcmc(pos, nsteps, progress=True)
    end_time = time.time()
    
    total_time = end_time - start_time
    
    # Compute Autocorrelation Time and ESS
    try:
        # Autocorrelation time (integrated autocorrelation time tau)
        tau = sampler.get_autocorr_time(discard=burn_in)
        # Total samples after burn-in = nwalkers * (nsteps - burn_in)
        total_samples = nwalkers * (nsteps - burn_in)
        ess = total_samples / tau
    except Exception as e:
        print(f"\nAutocorrelation time estimation failed: {e}")
        print("The chain might be too short or has not converged enough.")
        ess = np.array([np.nan] * ndim)
        tau = np.array([np.nan] * ndim)
        
    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    
    print("\n==================================================")
    print(f"🐍 Python emcee Benchmark Results ({model_name} Model)")
    print("==================================================")
    print(f"Total Wall Time:      {total_time:.2f} seconds")
    print(f"Total Evaluations:    {nwalkers * nsteps} likelihood calls")
    print("--------------------------------------------------")
    for i, name in enumerate(model.param_names):
        p_samples = flat_samples[:, i]
        med = np.percentile(p_samples, 50)
        print(f"{name:10} Median:      {med:.4f}")
        print(f"{name:10} Autocorr:    {tau[i]:.1f} steps")
        print(f"{name:10} ESS:         {ess[i]:.1f} samples")
        print(f"{name:10} ESS/sec:     {ess[i]/total_time:.3f} samples/sec")
        print("--------------------------------------------------")
    print("==================================================")

if __name__ == "__main__":
    main()
