
# Cosmological Parameter Estimation using MCMC

This repository contains Python code and Jupyter Notebooks for estimating cosmological parameters using MCMC with Cosmic Chronometer and Supernova data.

## Overview

This project fits different cosmological models ($\Lambda CDM$, $\omega CDM$, $CPL$) to observational data using the `emcee` MCMC sampler and visualizes the results with `corner` plots.



## Solutions to problems proposed.

My Solutions to problems from class 1 and class 2 are inside the file `Soluciones Problemas clase 1-2.pdf`.

The files `notebook1_soluciones.ipynb` and `notebook2_soluciones.ipynb` contain the problems associated with notebooks 1 and 2 proposed by Prof. Susana Landau. To review the MCMC implementation, please refer to the `emcee_implementation.ipynb` notebook. The `cosmology*.py` and `cornerplot.py` files contain more details.

## Dependencies

* `numpy`
* `scipy`
* `matplotlib`
* `emcee`
* `corner`

## Usage

Run the cells within the `emcee_implementaiton.ipynb` notebook to load data, select a model, run the MCMC, and plot the parameter constraints.