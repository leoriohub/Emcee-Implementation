# Adding a cosmological model

Python models live in [`src/models.py`](src/models.py) and inherit from [`CosmologyModel`](src/base_model.py). A model supplies its parameter labels, prior bounds, and vectorized Hubble function `H(z)`.

## Define the model

Keep the parameter order consistent across `param_names`, `param_bounds`, and the arguments to `H`.

```python
import numpy as np
from src.base_model import CosmologyModel

class MyNewModel(CosmologyModel):
    def __init__(self):
        super().__init__(
            name="My New Model",
            param_names=[r"H_0", r"p_1", r"p_2"],
            param_bounds=[(40, 100), (-5, 5), (0, 1)],
        )

    def H(self, z, H0, p1, p2):
        term = 1 + p1 * z + p2 * z**2
        return np.where(term > 0, H0 * np.sqrt(np.maximum(term, 0)), np.nan)
```

`param_bounds` defines the uniform prior. `H` must work with NumPy arrays; return non-finite values for unphysical parameter combinations so the posterior rejects them.

## Try the model directly

Run this from the repository root. The file names below are the data files in this repository.

```python
from src.run_inference import run_mcmc

model, sampler, samples, results = run_mcmc(
    MyNewModel(),
    "Cosmic_chronometers_data.tex",
    "Pantheon+SH0ES.dat.txt",
)
```

Passing an instance is useful while developing. The sampler settings and posterior flow are described in [the Python MCMC guide](mcmc_implementation.md).

## Register it for lookup by name

To call `run_mcmc("MYNEW", ...)`, define `MyNewModel` in `src/models.py`. The runner discovers subclasses in that module and derives the lookup key by removing the `Model` suffix and uppercasing the class name.

```python
model, sampler, samples, results = run_mcmc(
    "MYNEW",
    "Cosmic_chronometers_data.tex",
    "Pantheon+SH0ES.dat.txt",
)
```

Use unique class names and check that the bounds and equations describe the same parameter order. The [`demo_new_model.ipynb`](demo_new_model.ipynb) notebook shows a model added to the shared pipeline.
