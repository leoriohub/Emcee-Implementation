# Cosmological Parameter Estimation using MCMC

This repository contains a generalized, modular framework for estimating cosmological parameters using MCMC with Cosmic Chronometer and Supernova data.

## Overview

This project provides a flexible architecture for fitting cosmological models ($\Lambda CDM$, $\omega CDM$, $CPL$) to observational data. 

The code is organized in the `src/` directory for better maintainability and scalability:
- **`src/physics_utils.py`**: Shared physics constants and vectorized distance integration engines.
- **`src/base_model.py`**: Base `CosmologyModel` class providing inheritance for expansion models.
- **`src/models.py`**: Implementation of specific models (LCDM, wCDM, CPL).
- **`src/likelihoods.py`**: Generic likelihood engine for different cosmological probes.
- **`src/data_utils.py`**: Utilities for ingestion of CC and SN datasets.
- **`src/plotting.py`**: Custom, publication-ready corner plot utilities.
- **`src/run_inference.py`**: A unified pipeline for running MCMC with dynamic model discovery.

## Usage

### Running an Inference
The easiest way to run an MCMC simulation is using the unified pipeline:

```python
from src.run_inference import run_mcmc

# Run any model by name (e.g., "LCDM", "WCDM", "CPL")
model, sampler, samples, summary = run_mcmc("LCDM", "cc_data.tex", "sn_data.txt")

# Or pass a specific model instance directly
from src.models import wCDMModel
run_mcmc(wCDMModel(), "cc_data.tex", "sn_data.txt")
```

### Visualizing Results
The `plot_cosmo_corner` utility provides consistent styling across all models:

```python
from src.plotting import plot_cosmo_corner
plot_cosmo_corner(samples, labels=model.param_names)
```

## Legacy Notebooks
The files `notebook1_soluciones.ipynb` and `notebook2_soluciones.ipynb` contain the problems associated with notebooks 1 and 2 proposed by Prof. Susana Landau. To review the original MCMC implementation, please refer to the `emcee_implementation.ipynb` notebook.