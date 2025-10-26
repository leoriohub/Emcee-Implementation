import numpy as np
import scipy.integrate

c = 299792.458

def H_LCDM(z, H0, Omega_m):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

def dl_LCDM(z_array, H0, Omega_m, npoints=5000):
    z_array = np.atleast_1d(z_array)
    z_max = z_array.max()
    if z_max == 0:
        return np.zeros_like(z_array)
    z_grid = np.linspace(0, z_max, npoints)
    Hz_grid = H_LCDM(z_grid, H0, Omega_m)
    
    integral_grid = scipy.integrate.cumulative_trapezoid(c / Hz_grid, z_grid, initial=0)
    
    D_M = np.interp(z_array, z_grid, integral_grid)
    dL = (1 + z_array) * D_M
    dL[dL < 1e-9] = 1e-9
    return dL


def mu_LCDM(z, H0, Omega_m):
    dL = dl_LCDM(z, H0, Omega_m)
    return 5 * np.log10(dL) + 25

def log_prior(params):
    H0, Omega_m = params
    if 40 < H0 < 100 and 0.0 < Omega_m < 1.0:
        return 0.0
    return -np.inf

def log_likelihood_CC(params, data_CC):
    H0, Omega_m = params
    model = H_LCDM(data_CC["z"], H0, Omega_m)
    chi2 = np.sum(((data_CC["Hz"] - model) / data_CC["sigma_Hz"])**2)
    return -0.5 * chi2

def log_likelihood_SN(params, data_SN):
    H0, Omega_m = params
    model = mu_LCDM(data_SN["z"], H0, Omega_m)
    chi2 = np.sum(((data_SN["mu"] - model) / data_SN["sigma_mu"])**2)
    return -0.5 * chi2

# --- Función Principal (la que usará emcee) ---
def log_posterior_lcdm(params, data_CC, data_SN):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
        
    try:
        ll_total = log_likelihood_CC(params, data_CC) + log_likelihood_SN(params, data_SN)
        return lp + ll_total
    except Exception:
        return -np.inf