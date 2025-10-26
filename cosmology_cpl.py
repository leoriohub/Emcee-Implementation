import numpy as np
import scipy.integrate

c = 299792.458 


def H_cpl(z, H0, Omega_m, w0, wa):
    om_m = Omega_m * (1.0 + z)**3
    z_ratio = z / (1.0 + z)
    om_de = (1.0 - Omega_m) * (1.0 + z)**(3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z_ratio)
    return H0 * np.sqrt(om_m + om_de)

def dl_cpl(z_array, H0, Omega_m, w0, wa, npoints=5000):
    z_array = np.atleast_1d(z_array)
    z_max = z_array.max()
    if z_max == 0:
        return np.zeros_like(z_array)
    z_grid = np.linspace(0, z_max, npoints)
    Hz_grid = H_cpl(z_grid, H0, Omega_m, w0, wa)
    
    integral_grid = scipy.integrate.cumulative_trapezoid(c / Hz_grid, z_grid, initial=0)
    
    D_M = np.interp(z_array, z_grid, integral_grid)
    dL = (1 + z_array) * D_M
    dL[dL < 1e-9] = 1e-9
    return dL

def mu_cpl(z, H0, Omega_m, w0, wa):
    dL = dl_cpl(z, H0, Omega_m, w0, wa)
    return 5.0 * np.log10(dL) + 25.0


def log_prior(params):
    H0, Omega_m, w0, wa = params
    if (40 < H0 < 100 and 0.0 < Omega_m < 1.0 and 
        -2.0 < w0 < 0.0 and -3.0 < wa < 3.0):
        return 0.0
    return -np.inf

def log_likelihood_CC(params, data_CC):
    H0, Omega_m, w0, wa = params
    model = H_cpl(data_CC["z"], H0, Omega_m, w0, wa)
    chi2 = np.sum(((data_CC["Hz"] - model) / data_CC["sigma_Hz"])**2)
    return -0.5 * chi2

def log_likelihood_SN(params, data_SN):
    H0, Omega_m, w0, wa = params
    model = mu_cpl(data_SN["z"], H0, Omega_m, w0, wa)
    chi2 = np.sum(((data_SN["mu"] - model) / data_SN["sigma_mu"])**2)
    return -0.5 * chi2

def log_posterior_cpl(params, data_CC, data_SN):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
        
    try:
        ll_total = log_likelihood_CC(params, data_CC) + log_likelihood_SN(params, data_SN)
        return lp + ll_total
    except Exception:
        return -np.inf