# Sampling utilization functions

This folder contains some utilization functions motivated by analysis in MCMC sampling
to estimate equilibration time, compute autocorrelation length, and check for sampling
convergence.

These utilization functions can be used in AL framework via information-matching. For
example, if we don't have a good guess of the initial parameters, we can randomly choose
the initial parameters. However, we might not want to carry out the information on these
random parameters. So, we can have some burn-in or equilibration period at the beginning
of the AL loop.

* equilibration.py - Contains functions to estimate the burn-in time of the MCMC sampling
  using the marginal standard error rule (MSER).
* autocorrelation.py - Contains functions to estimate the autocorrelation integrated time
  of the MCMC chains. The calculation is done using `emcee.autocorr` module.
* convergence.py - Contains functions to assess the convergence of MCMC sampling by
  computing the multivariate potential scale reduction factor (PSRF).
