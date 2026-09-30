# Python MCMC implementation

The Python entry point is `run_mcmc` in [`src/run_inference.py`](src/run_inference.py). The runnable example and dependency installation are in the [README](README.md#run-the-python-sampler).

## Sampling flow

1. `run_mcmc` accepts a model key such as `"LCDM"` or a `CosmologyModel` instance. String keys resolve to classes in [`src/models.py`](src/models.py).
2. [`src/data_utils.py`](src/data_utils.py) loads the Cosmic Chronometer table and Pantheon+SH0ES data. The supernova loader reads `zHD`, `MU_SH0ES`, and the diagonal uncertainty `MU_SH0ES_ERR_DIAG`.
3. [`src/likelihoods.py`](src/likelihoods.py) computes the Cosmic Chronometer and supernova log likelihoods. `total_log_posterior` adds them to the model's bounded prior from [`src/base_model.py`](src/base_model.py).
4. `run_mcmc` initializes walkers near the center of each parameter bound, creates an `emcee.EnsembleSampler`, and runs the requested number of steps.
5. It discards `int(nsteps * burn_in_fraction)` steps, flattens the remaining chain, and reports the 16th, 50th, and 84th percentiles for each parameter. The default burn-in fraction is `0.2`.

The function returns `(model, sampler, flat_samples, results)`. Pass `save_path="results.npz"` to save the samples and summary. [`src/plotting.py`](src/plotting.py) contains the corner-plot helpers.

## Limits to keep in mind

- The Python supernova likelihood uses the diagonal errors; it does not use the full covariance matrix.
- The runner reports percentile summaries but does not check chain convergence. Inspect the returned `sampler` before treating a run as converged.
- Non-finite posterior values and exceptions are rejected by the safe posterior wrapper in `run_mcmc`.

The Julia implementation is separate: [`julia/run_mcmc.jl`](julia/run_mcmc.jl) uses Turing and can load the full Pantheon+ covariance matrix. See the run commands in the [README](README.md#run-the-julia-sampler).
