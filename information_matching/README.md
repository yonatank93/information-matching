# Modules

This folder contains the modules that are needed in the indicator configuration
calculation.
In the order they are used, here are short descriptions of the modules:

* fim - This folder contains several modules to do numerical derivative for FIM
  calculation. We provide wrappers to Python's `numdifftools` and Julia's `NumDiffTools`,
  as well as a simple finite difference derivative methods.
* convex_optimization.py - Contains a class to do the convex optimization part in the
  indicator configuration calculation, starting from constructing the problem, solving it,
  and interpreting the result.
* leastsq.py - Contains functions to do potential training by minimizing the weighted
  least-squares cost function and to do post-processing on the optimization results.
  Essentially, this module wraps over `scipy.optimize.minimize` and `scipy.optimize.least_squares`.
  Additionally, the wrapper can also handle `geodesicLM`, which requires additional setup.
* termination.py - Contains functions to check whether the iterative information-matching
  active learning calculation has converged.
* summary.py - Contains a class to collect the information throughout the calculation and
  write summary results. This is an optional module to help recording the results in the
  active learning iteration.
* utils.py - Contains some utility functions and variables, such as to set a directory
  and copy the configuration data (via `shutil`), and some tolerances values.
* parallel.py - Contains classes and functions for parallelization.

Additional package:
* mcmc_utils - Contains some utilities that are usually used in the MCMC sampling, such
  as to burn-in estimation, autocorrelation length estimation, and convergence assessment.
  These analysis can be used in the information-matching active learning, e.g., to
  estimate the warm-up or burn-in period.
