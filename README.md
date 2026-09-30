# Cosmological parameter estimation

MCMC fits of cosmological models to Cosmic Chronometer and Pantheon+SH0ES data. The repository has a Python implementation using `emcee` and a Julia implementation using Turing.

## Run the Python sampler

From the repository root, install the Python dependencies:

```sh
python -m pip install numpy scipy matplotlib emcee corner tqdm
```

Then run a model and plot its posterior samples:

```python
from src.run_inference import run_mcmc
from src.plotting import plot_cosmo_corner

model, sampler, samples, results = run_mcmc(
    "LCDM",
    "Cosmic_chronometers_data.tex",
    "Pantheon+SH0ES.dat.txt",
    nwalkers=40,
    nsteps=5000,
)

plot_cosmo_corner(samples, labels=model.param_names)
```

`run_mcmc` accepts a model name such as `"LCDM"` or a model instance. For an instance, import a class such as `wCDMModel` from `src.models` and pass `wCDMModel()`. It returns the model, sampler, flattened post-burn-in samples, and parameter summaries.

## Run the Julia sampler

Install its dependencies, then run the `LCDM` model for 1,000 samples using the full Pantheon+ covariance matrix:

```sh
julia julia/setup.jl
julia julia/run_mcmc.jl LCDM 1000 true
```

The last argument controls covariance use; pass `false` for diagonal errors only.

## Data and implementation

The data files are in the repository root:

- `Cosmic_chronometers_data.tex` — Cosmic Chronometer measurements.
- `Pantheon+SH0ES.dat.txt` — Pantheon+SH0ES supernova measurements.
- `Pantheon+SH0ES_STAT+SYS.cov.txt` — Pantheon+SH0ES covariance matrix.

### MCMC code

- `src/run_inference.py` — Python `emcee` sampler and inference flow.
- `src/likelihoods.py` — Cosmic Chronometer and supernova likelihoods, combined with model priors.
- `src/models.py` and `src/base_model.py` — cosmological models, parameter bounds, and priors.
- `src/data_utils.py` — Python data loaders.
- `src/plotting.py` — corner plots.
- `julia/run_mcmc.jl` and `julia/src/` — Julia Turing/NUTS sampler and supporting code.

The Python supernova loader uses diagonal errors. The Julia sampler uses the full covariance matrix when its last argument is `true`.

## Guides and notebooks

- [Python MCMC implementation](mcmc_implementation.md) — posterior construction, sampling flow, burn-in, and convergence caveats.
- [Adding a cosmological model](tutorial_add_new_model.md) — implementing `H(z)` and using or registering a model.
- [Original `emcee` walkthrough](emcee_implementation.ipynb) — model equations and a standalone sampler example.
- [Custom-model demo](demo_new_model.ipynb) — adds a model to the shared Python pipeline.
- [Model-comparison demo](demo_comparison.ipynb) — saves, reloads, and compares MCMC results.
- [Advanced-analysis notebook](advanced_analysis.ipynb) — comparison tables and extension-parameter plots.