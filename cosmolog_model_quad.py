from scipy.integrate import quad
import scipy.integrate # Para tu 'cumulative_trapezoid'
import numpy as np


c = 299792.458  

def H_LCDM(z, H0, Omega_m):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

def integrando_dl(z, H0, Omega_m):
    return 1.0 / H_LCDM(z, H0, Omega_m)

def dl_LCDM(z_array, H0, Omega_m):

    z_array = np.atleast_1d(z_array)
    
    dl_list = []
    
    # Iteramos sobre CADA redshift en el array, puesto que quad no puede ser vectorizado
    for z_val in z_array:
        if z_val < 1e-9:
            dl_list.append(0.0)
            continue
            
        integral_result, _ = quad(integrando_dl, 0, z_val, args=(H0, Omega_m))
        
        D_M = c * integral_result
        
        dL = (1 + z_val) * D_M
        dl_list.append(dL)
    
    dL_array = np.array(dl_list)
    dL_array[dL_array < 1e-9] = 1e-9 # Evitar log(0)
    return dL_array


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
def log_posterior(params, data_CC, data_SN):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
        
    try:
        ll_total = log_likelihood_CC(params, data_CC) + log_likelihood_SN(params, data_SN)
        return lp + ll_total
    except Exception:
        return -np.inf