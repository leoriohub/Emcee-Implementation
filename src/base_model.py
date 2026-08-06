import numpy as np
from .physics_utils import integrate_distance, distance_modulus

class CosmologyModel:
    """
    Base class for cosmological models.
    Subclasses must implement the H(z) method.
    """
    def __init__(self, name, param_names, param_bounds):
        self.name = name
        self.param_names = param_names
        self.param_bounds = param_bounds  # List of (min, max) tuples

    def H(self, z, *params):
        """Expansion rate H(z). Must be implemented by subclasses."""
        raise NotImplementedError

    def dL(self, z, *params, npoints=5000):
        """Luminosity distance dL(z)."""
        return integrate_distance(self.H, z, params, npoints=npoints)

    def mu(self, z, *params, npoints=5000):
        """Distance modulus mu(z)."""
        dL_vals = self.dL(z, *params, npoints=npoints)
        return distance_modulus(dL_vals)

    def log_prior(self, params):
        """Checks if parameters are within bounds."""
        if len(params) != len(self.param_bounds):
            return -np.inf
            
        for val, (low, high) in zip(params, self.param_bounds):
            if not (low < val < high):
                return -np.inf
        return 0.0
