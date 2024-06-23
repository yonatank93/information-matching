# MCMC_utils

This folder contains modules that are used in the MCMC simulation.
In the order they are used, here are short descriptions of the modules:

* prior.py - Contains functions to compute several prior distributions, such as uniform,
  normal, and Jeffreys prior.
* mcmc_save.py - Contains functions to save MCMC simulation results, whether the MCMC
  simulation is done using `emcee` or `ptemcee`.
* equilibration.py - Contains functions to estimate the burn-in time of the MCMC sampling
  using the marginal standard error rule (MSER).
* autocorrelation.py - Contains functions to estimate the autocorrelation integrated time
  of the MCMC chains. The calculation is done using `emcee.autocorr` module.
* convergence.py - Contains functions to assess the convergence of MCMC sampling by
  computing the multivariate potential scale reduction factor (PSRF).
