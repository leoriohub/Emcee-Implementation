import os
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from .data_utils import load_CC, load_PPSH0ES
from . import models
from .base_model import CosmologyModel
from .likelihoods import log_likelihood_CC, log_likelihood_SN, total_log_posterior as _total_log_posterior

def save_mcmc_results(save_path, model, flat_samples, results):
    """Saves MCMC results and samples to a .npz file."""
    # Convert results to a simpler format for saving
    res_keys = list(results.keys())
    # results[key] is (median, minus, plus)
    res_values = np.array([results[k] for k in res_keys])
    
    np.savez(save_path, 
             samples=flat_samples, 
             model_name=model.name,
             param_names=model.param_names,
             res_keys=res_keys,
             res_values=res_values)
    print(f"Results saved to {save_path}")

def load_mcmc_results(load_path):
    """Loads MCMC results and samples from a .npz file."""
    data = np.load(load_path)
    samples = data['samples']
    param_names = [str(n) for n in data['param_names']]
    res_keys = [str(n) for n in data['res_keys']]
    res_values = data['res_values']
    
    results = {res_keys[i]: tuple(res_values[i]) for i in range(len(res_keys))}
    return samples, param_names, results

def run_mcmc(model, data_cc_path, data_sn_path, nwalkers=40, nsteps=5000, burn_in_fraction=0.2, save_path=None):
    """
    Runs MCMC for a given model and datasets.
    
    Args:
        model: An instance of CosmologyModel or a string key for built-in models.
        data_cc_path: Path to CC data.
        data_sn_path: Path to SN data.
    """
    # 1. Initialize Model
    if isinstance(model, str):
        # Dynamically find the model class in src.models
        available_models = {name[:-5].upper(): cls for name, cls in models.__dict__.items() 
                           if isinstance(cls, type) and issubclass(cls, CosmologyModel) and cls is not CosmologyModel}
        if model.upper() not in available_models:
            raise ValueError(f"Unknown model name: {model}. Available: {list(available_models.keys())}")
        model = available_models[model.upper()]()
    
    if not isinstance(model, CosmologyModel):
        raise TypeError("model must be an instance of CosmologyModel or a valid model name string.")
    print(f"--- Running MCMC for {model.name} ---")
    
    # 2. Load Data
    data_CC = load_CC(data_cc_path)
    data_SN = load_PPSH0ES(data_sn_path)
    datasets = [
        (log_likelihood_CC, data_CC),
        (log_likelihood_SN, data_SN)
    ]
    
    # 3. Setup emcee
    ndim = len(model.param_names)
    # Start walkers in a small ball around a central point in the prior
    initial_pos = np.array([np.mean(b) for b in model.param_bounds])
    pos = initial_pos + 1e-3 * np.random.randn(nwalkers, ndim)
    
    # REDUNDANT SAFETY WRAPPER: Ensure NO NaN reaches emcee
    def safe_posterior(p, m, d):
        try:
            res = _total_log_posterior(p, m, d)
            if not np.isfinite(res):
                return -1e300
            return float(res)
        except:
            return -1e300

    sampler = emcee.EnsembleSampler(nwalkers, ndim, safe_posterior, args=(model, datasets))
    
    # 4. Run
    print(f"Starting {nsteps} steps with {nwalkers} walkers...")
    sampler.run_mcmc(pos, nsteps, progress=True)
    print("MCMC Finished!")
    
    # 5. Process Results
    burn_in = int(nsteps * burn_in_fraction)
    flat_samples = sampler.get_chain(discard=burn_in, flat=True)
    
    results = {}
    for i, name in enumerate(model.param_names):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        results[name] = (mcmc[1], q[0], q[1])
        print(f"{name} = {mcmc[1]:.4f} (-{q[0]:.4f}, +{q[1]:.4f})")
        
    if save_path:
        save_mcmc_results(save_path, model, flat_samples, results)
        
    return model, sampler, flat_samples, results

def generate_comparison_table(results_list, names_list):
    """
    Creates a Markdown table summarizing parameters from multiple models.
    results_list: list of tuples (samples, param_names, results_dict)
    names_list: list of model names
    """
    table = "| Model | Parameter | Median | -Quantile | +Quantile |\n"
    table += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for (samples, param_names, res_dict), name in zip(results_list, names_list):
        for p_name in param_names:
            mid, low, high = res_dict[p_name]
            table += f"| {name} | {p_name} | {mid:.4f} | {low:.4f} | {high:.4f} |\n"
            
    return table

if __name__ == "__main__":
    from .plotting import plot_cosmo_corner
    # Example usage
    cc_path = "Cosmic_chronometers_data.tex"
    sn_path = "Pantheon+SH0ES.dat.txt"
    
    # Run LCDM by default
    model, sampler, samples, summary = run_mcmc("LCDM", cc_path, sn_path)
    
    # Use the generalized plotting utility
    plot_cosmo_corner(samples, labels=model.param_names)
