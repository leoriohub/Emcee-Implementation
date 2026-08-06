# Tutorial: How to Add a New Cosmological Model

This guide explains how to extend the generalized framework with your own cosmological models. The architecture is designed so you only need to focus on the physics ($H(z)$), while the framework handles the data, integration, likelihoods, and MCMC execution.

## Step 1: Create Your Model Class

All models must inherit from the `CosmologyModel` base class found in `src/base_model.py`.

```python
import numpy as np
from src.base_model import CosmologyModel

class MyNewModel(CosmologyModel):
    def __init__(self):
        super().__init__(
            name="My New Physics Model",
            param_names=[r"H_0", r"p_1", r"p_2"],  # Use LaTeX labels if you like
            param_bounds=[(40, 100), (-5, 5), (0, 1)] # Define prior ranges
        )
```

## Step 2: Implement the Hubble Parameter $H(z)$

Override the `H` method. **Crucial:** Always use safeguards (like `np.where` or `np.nan`) to handle unphysical parameter regions where the expansion rate might become imaginary or negative.

```python
    def H(self, z, H0, p1, p2):
        # Example expansion rate: H(z) = H0 * sqrt(1 + p1*z + p2*z^3)
        term = 1 + p1 * z + p2 * z**3
        
        # Safeguard: If term <= 0, return NaN. 
        # The likelihood engine will automatically catch this and return -inf.
        return np.where(term > 0, H0 * np.sqrt(term), np.nan)
```

## Step 3: Use Your Model

You can now use your model in two ways:

### Option A: Direct Injection (Recommended for testing)
Pass an instance of your class directly to `run_mcmc`.

```python
from src.run_inference import run_mcmc

model_instance = MyNewModel()
model, sampler, samples, results = run_mcmc(model_instance, "cc_data.tex", "sn_data.txt")
```

### Option B: Permanent Integration
If you add your class to `src/models.py`, the pipeline will **automatically discover it**. You can then run it by name:

```python
# In src/models.py:
class MyNewModel(CosmologyModel):
    ...

# Then anywhere else:
run_mcmc("MYNEW", "cc_data.tex", "sn_data.txt")  # Name is derived from class name (lowercase/uppercase works)
```

## Best Practices
1. **Vectorization**: Ensure your `H(z)` function can handle `z` being a NumPy array (which it will be!). Use `np.sqrt`, `np.exp`, etc., instead of the `math` library.
2. **Safeguards**: Never let your function return a complex number or a negative $H(z)$. Return `np.nan` instead.
3. **Labels**: Use LaTeX in `param_names` (e.g., `r"\Omega_m"`) to get beautiful labels in your corner plots.
