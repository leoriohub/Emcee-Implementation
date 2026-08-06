import numpy as np
import scipy.integrate

# Speed of light in km/s
C_LIGHT = 299792.458

def integrate_distance(h_func, z_array, params, npoints=5000):
    """
    Computes the luminosity distance dL(z) for a given expansion rate h_func.
    
    Args:
        h_func: Callable H(z, *params)
        z_array: Array of redshifts
        params: Tuple of model parameters
        npoints: Number of integration points
        
    Returns:
        dL: Luminosity distance array in Mpc
    """
    z_array = np.atleast_1d(z_array)
    z_max = z_array.max()
    if z_max == 0:
        return np.zeros_like(z_array)
    
    z_grid = np.linspace(0, z_max, npoints)
    Hz_grid = h_func(z_grid, *params)
    
    # Check for invalid physical values (H(z) <= 0 results in NaN/Inf)
    if np.any(Hz_grid <= 0) or np.any(~np.isfinite(Hz_grid)):
        return np.full_like(z_array, np.nan)
    
    # DM(z) = c * integral(0 to z) dz' / H(z')
    with np.errstate(divide='ignore', invalid='ignore'):
        integral_grid = scipy.integrate.cumulative_trapezoid(C_LIGHT / Hz_grid, z_grid, initial=0)
    
    # Interpolate to find values at specific redshifts
    D_M = np.interp(z_array, z_grid, integral_grid)
    
    # dL(z) = (1 + z) * DM(z)
    dL = (1 + z_array) * D_M
    
    # Numerical stability
    dL[dL < 1e-9] = 1e-9
    return dL

def distance_modulus(dL):
    """Computes the distance modulus mu = 5 log10(dL) + 25."""
    return 5.0 * np.log10(dL) + 25.0
