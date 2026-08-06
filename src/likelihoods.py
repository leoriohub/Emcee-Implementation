import numpy as np

def log_likelihood_CC(model, params, data_CC):
    """Log-likelihood for Cosmic Chronometers data."""
    z = data_CC["z"]
    Hz_data = data_CC["Hz"]
    sigma_Hz = data_CC["sigma_Hz"]
    
    Hz_model = model.H(z, *params)
    chi2 = np.sum(((Hz_data - Hz_model) / sigma_Hz)**2)
    return -0.5 * chi2

def log_likelihood_SN(model, params, data_SN):
    """Log-likelihood for Supernova data."""
    z = data_SN["z"]
    mu_data = data_SN["mu"]
    sigma_mu = data_SN["sigma_mu"]
    
    mu_model = model.mu(z, *params)
    chi2 = np.sum(((mu_data - mu_model) / sigma_mu)**2)
    return -0.5 * chi2

def total_log_posterior(params, model, datasets):
    """
    Generic posterior function.
    datasets: list of (likelihood_func, data) tuples
    """
    # Defensive check: ensure params are finite
    if not np.all(np.isfinite(params)):
        return -np.inf

    lp = model.log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
        
    log_lik = 0.0
    try:
        for lik_func, data in datasets:
            val = lik_func(model, params, data)
            if not np.isfinite(val):
                return -np.inf
            log_lik += val
        
        result = lp + log_lik
        # Final absolute check
        if not np.isfinite(result):
            return -np.inf
            
        return result
    except Exception:
        return -np.inf
