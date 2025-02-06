# Modules

This folder contains the modules that are needed in the indicator configuration
calculation.
In the order they are used, here are short descriptions of the modules:

* convex_optimization.py - Contains a class to do the convex optimization part in the
  indicator configuration calculation, starting from constructing the problem, solving it,
  and interpreting the result.
* leastsq.py - Contains functions to do potential training by minimizing the weighted
  least-squares cost function and to do post-processing on the optimization results.
  Essentially, this module wraps over `scipy.optimize.minimize` and
  `scipy.optimize.least_squares`. Additionally, the wrapper can also handle `geodesicLM`,
  which requires additional setup.
* precondition.py - Contains a function to precondition the FIM. The function doesn't
  change the FIM, instead it calculates multiplicative scaling factor(s) and convert the
  FIM input into a dictionary format that can be directly used by
  `information_matching.ConvexOpt`.
* transform.py - Contains several parameter transformation modules.

Additional package:
* fim - Contains several modules to do numerical derivative for FIM calculation. We
  provide wrappers to Python's `numdifftools` and Julia's `NumDiffTools`, as well as a
  simple finite difference derivative methods.
* utils - Contains some utility functions.
* sampling_utils - Contains some utilities to analyze samples, motivated by analysis in
  MCMC.
