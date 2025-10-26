
# Cosmological Parameter Estimation using MCMC

This repository contains Python code and Jupyter Notebooks for estimating cosmological parameters using MCMC with Cosmic Chronometer and Supernova data.

## Overview

This project fits different cosmological models ($\Lambda$CDM, $w$CDM, CPL) to observational data using the `emcee` MCMC sampler and visualizes the results with `corner` plots.

The files `notebook1_soluciones.ipynb` and `notebook2_soluciones.ipynb` contain the problems associated with notebooks 1 and 2. To review the MCMC implementation, please refer to the `emcee_implementation.ipynb` notebook. The `cosmology*.py` and `cornerplot.py` files contain more details.

## Dependencies

* `numpy`
* `scipy`
* `matplotlib`
* `emcee`
* `corner`

## Usage

Run the cells within the `emcee_implementaiton.ipynb` notebook to load data, select a model, run the MCMC, and plot the parameter constraints.