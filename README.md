# Cosmological Parameter Estimation using MCMC

This repository contains Python code and a Jupyter Notebook for constraining cosmological parameters using Markov Chain Monte Carlo (MCMC) methods with data from Cosmic Chronometers (CC) and Type Ia Supernovae (SNe Ia).

## Overview

The primary goal of this project is to estimate key parameters of different cosmological models by comparing theoretical predictions to observational data. It uses the `emcee` library to perform MCMC sampling of the parameter space and the `corner` library for visualizing the results.

## Key Features

* **Cosmological Models:** Implements the Flat $\Lambda$CDM, Flat $w$CDM, and Flat CPL models.
* **Data Combination:** Uses likelihood functions combining $H(z)$ data from Cosmic Chronometers and $\mu(z)$ data from the Pantheon+SH0ES Supernova dataset.
* **MCMC Sampling:** Employs the `emcee` affine-invariant MCMC ensemble sampler.
* **Visualization:** Generates corner plots showing 1D marginalized distributions and 2D joint contours using a dedicated plotting function.
* **Modular Structure:** Model definitions and likelihoods are separated into Python scripts (`.py`), while the main analysis workflow is contained within a Jupyter Notebook (`.ipynb`).

## Models Implemented

The analysis assumes a **spatially flat** universe for all models.

1.  **Flat $\Lambda$CDM (`cosmology_lcdm.py`)**
    * Parameters: $H_0$, $\Omega_m$
    * Equation of State: $w = -1$ (fixed)
2.  **Flat $w$CDM (`cosmology_wcdm.py`)**
    * Parameters: $H_0$, $\Omega_m$, $w$
    * Equation of State: $w = \text{constant}$ (free parameter)
3.  **Flat CPL (`cosmology_cpl.py`)**
    * Parameters: $H_0$, $\Omega_m$, $w_0$, $w_a$
    * Equation of State: $w(z) = w_0 + w_a \frac{z}{1+z}$ (evolving)

## Data Used

* **Cosmic Chronometers (CC):** $H(z)$ measurements derived from passively evolving galaxies. Loaded from `Cosmic_chronometers_data.tex`.
* **Type Ia Supernovae (SNe Ia):** Distance moduli ($\mu$) from the Pantheon+SH0ES compilation. Loaded from `Pantheon+SH0ES.dat.txt`.
    * **Note:** This analysis uses only the diagonal errors (`MU_SH0ES_ERR_DIAG`) for the SNe likelihood and does not incorporate the systematic covariance matrix.

## Dependencies

The code relies on the following standard Python libraries:

* `numpy`
* `scipy`
* `matplotlib`
* `emcee`
* `corner`
* `multiprocess` (optional, used for parallel MCMC execution if enabled in the notebook)

You can typically install these using pip:
```bash
pip install numpy scipy matplotlib emcee corner multiprocess