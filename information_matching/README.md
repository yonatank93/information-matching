# Modules

This folder contains the modules that are needed in the indicator configuration
calculation.
In the order they are used, here are short descriptions of the modules:

* fim.py - Contains classes to do Jacobian and FIM calculations. There are currently 2
  classes, one uses Python's numdifftools while the other used Julia's numdifftools.
* convex_optimization.py - Contains a class to do the convex optimization part in the
  indicator configuration calculation, starting from constructing the problem, solving it,
  and interpreting the result.
* leastsq.py - Contains functions to do potential training by minimizing the weighted
  least-squares cost function and to do post-processing on the optimization results.
* termination.py - Contains functions to check whether the indicator configuration
  calculation has converged.
* summary.py - Contains a class to collect the information throughout the calculation and
  write summary results.
* utils.py - Contains some utility functions and variables.
* parallel.py - Contains classes and functions for parallelization. (I don't think this
  module is used by any script currently.)

Other packages in this directory:
* models
* mcmc_utils
