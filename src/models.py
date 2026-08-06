import numpy as np
from .base_model import CosmologyModel

class LCDMModel(CosmologyModel):
    def __init__(self):
        super().__init__(
            name="LCDM",
            param_names=[r"$H_0$", r"$\Omega_m$"],
            param_bounds=[(40, 100), (0.0, 1.0)]
        )

    def H(self, z, H0, Omega_m):
        return H0 * np.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))

class wCDMModel(CosmologyModel):
    def __init__(self):
        super().__init__(
            name="wCDM",
            param_names=[r"$H_0$", r"$\Omega_m$", r"$w$"],
            param_bounds=[(40, 100), (0.0, 1.0), (-2.0, 0.0)]
        )

    def H(self, z, H0, Omega_m, w):
        om_m = Omega_m * (1.0 + z)**3
        om_de = (1.0 - Omega_m) * (1.0 + z)**(3.0 * (1.0 + w))
        return H0 * np.sqrt(om_m + om_de)

class CPLModel(CosmologyModel):
    def __init__(self):
        super().__init__(
            name="CPL",
            param_names=[r"$H_0$", r"$\Omega_m$", r"$w_0$", r"$w_a$"],
            param_bounds=[(40, 100), (0.0, 1.0), (-2.0, 0.0), (-3.0, 3.0)]
        )

    def H(self, z, H0, Omega_m, w0, wa):
        om_m = Omega_m * (1.0 + z)**3
        z_ratio = z / (1.0 + z)
        om_de = (1.0 - Omega_m) * (1.0 + z)**(3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z_ratio)
        return H0 * np.sqrt(om_m + om_de)

class DGPModel(CosmologyModel):
    """Dvali-Gabadadze-Porrati (DGP) Brane-world model."""
    def __init__(self):
        super().__init__(
            name="DGP Brane-world",
            param_names=[r"$H_0$", r"$\Omega_m$"],
            param_bounds=[(40, 100), (0.0, 1.0)]
        )

    def H(self, z, H0, Omega_m):
        # E(z) = 0.5 * (1 - Omega_m) + sqrt(Omega_m * (1+z)^3 + (1-Omega_m)^2 / 4)
        term = Omega_m * (1 + z)**3 + (1 - Omega_m)**2 / 4.0
        Ez = 0.5 * (1 - Omega_m) + np.sqrt(np.maximum(term, 0))
        return H0 * Ez

class JBPModel(CosmologyModel):
    """Jassal-Bagla-Padmanabhan (JBP) parameterization."""
    def __init__(self):
        super().__init__(
            name="JBP w(z)",
            param_names=[r"$H_0$", r"$\Omega_m$", r"$w_0$", r"$w_a$"],
            param_bounds=[(40, 100), (0.1, 0.5), (-2.0, 0.0), (-5.0, 5.0)]
        )

    def H(self, z, H0, Omega_m, w0, wa):
        # w(z) = w0 + wa * z / (1+z)^2
        # Density evolution f(z) = (1+z)^(3(1+w0)) * exp(1.5 * wa * z^2 / (1+z)^2)
        exp_term = 1.5 * wa * z**2 / (1 + z)**2
        f_z = (1 + z)**(3 * (1 + w0)) * np.exp(exp_term)
        Ez_sq = Omega_m * (1 + z)**3 + (1 - Omega_m) * f_z
        return H0 * np.sqrt(np.maximum(Ez_sq, 0))

class GCGModel(CosmologyModel):
    """Generalized Chaplygin Gas (GCG) model."""
    def __init__(self):
        super().__init__(
            name="Generalized Chaplygin Gas",
            param_names=[r"$H_0$", r"$A_s$", r"$\alpha$"],
            param_bounds=[(40, 100), (0.0, 1.0), (-1.0, 2.0)]
        )

    def H(self, z, H0, As, alpha):
        # E(z) = [As + (1 - As) * (1+z)^(3(1+alpha))]^(1 / (2(1+alpha)))
        # Note: This is an effective one-component model
        term = As + (1 - As) * (1 + z)**(3 * (1 + alpha))
        Ez = np.power(np.maximum(term, 1e-9), 1.0 / (2.0 * (1.0 + alpha)))
        return H0 * Ez

class IDEModel(CosmologyModel):
    """Simple Interacting Dark Energy (IDE) model."""
    def __init__(self):
        super().__init__(
            name="Interacting DE",
            param_names=[r"$H_0$", r"$\Omega_m$", r"$\xi$"],
            param_bounds=[(40, 100), (0.01, 0.99), (-0.5, 0.5)]
        )

    def H(self, z, H0, Omega_m, xi):
        # rho_m ~ (1+z)^(3+xi)
        Ez_sq = Omega_m * (1 + z)**(3 + xi) + (1 - Omega_m)
        return H0 * np.sqrt(np.maximum(Ez_sq, 0))

class LogWModel(CosmologyModel):
    """Logarithmic w(z) parameterization."""
    def __init__(self):
        super().__init__(
            name="Log-w Parameterization",
            param_names=[r"$H_0$", r"$\Omega_m$", r"$w_0$", r"$w_1$"],
            param_bounds=[(40, 100), (0.1, 0.5), (-2.0, 0.0), (-2.0, 2.0)]
        )

    def H(self, z, H0, Omega_m, w0, w1):
        # w(z) = w0 + w1 * ln(1+z)
        # f(z) = (1+z)^(3(1+w0)) * exp(1.5 * w1 * (ln(1+z))^2)
        lnz = np.log1p(z)
        f_z = (1 + z)**(3 * (1 + w0)) * np.exp(1.5 * w1 * lnz**2)
        Ez_sq = Omega_m * (1 + z)**3 + (1 - Omega_m) * f_z
        return H0 * np.sqrt(np.maximum(Ez_sq, 0))

class StarobinskyModel(CosmologyModel):
    """
    Late-time f(R) Starobinsky model (phenomenological approximation).
    In late-time f(R), the expansion history follows a w(z) that is 
    distinct from LCDM but stays close to it.
    Equation: E(z)^2 = Omega_m(1+z)^3 + (1-Omega_m)(1 + alpha*z^2)
    """
    def __init__(self):
        super().__init__(
            name="Starobinsky f(R)",
            param_names=[r"$H_0$", r"$\Omega_m$", r"$\alpha$"],
            param_bounds=[(40, 100), (0.1, 0.5), (-1.0, 1.0)]
        )
    def H(self, z, H0, Omega_m, alpha):
        # A common phenomenological late-time expansion for f(R)
        Ez_sq = Omega_m * (1 + z)**3 + (1 - Omega_m) * (1 + alpha * z**2)
        return H0 * np.sqrt(np.maximum(Ez_sq, 0))

class BransDickeModel(CosmologyModel):
    """
    Brans-Dicke (Scalar-Tensor) model.
    Modifies the Hubble rate with an extra term related to the coupling parameter omega.
    Equation: E(z)^2 = Omega_m(1+z)^3 * (1+z)^(-epsilon) + (1-Omega_m)
    where epsilon ~ 1/omega_BD
    """
    def __init__(self):
        super().__init__(
            name="Brans-Dicke MG",
            param_names=[r"$H_0$", r"$\Omega_m$", r"$\epsilon$"],
            param_bounds=[(40, 100), (0.1, 0.5), (-0.5, 0.5)]
        )

    def H(self, z, H0, Omega_m, epsilon):
        # epsilon parameterization for Brans-Dicke expansion history
        om_m = Omega_m * (1 + z)**(3 - epsilon)
        om_de = (1 - Omega_m)
        Ez_sq = om_m + om_de
        return H0 * np.sqrt(np.maximum(Ez_sq, 0))
