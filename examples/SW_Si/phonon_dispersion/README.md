This folder contains the scripts to do indicator configuration calculation for the
following case:

* Model: Stillinger-Weber
* System: diamond silicon
* Training configurations: Configurations with varying lattice constants, unperturbed and
  perturbed
* Training quantities: energy for the unperturbed configurations, atomic forces for the
  perturbed configurations
* Target QoI: phonon dispersion curve with the lowest energy
* Target precission: 10% of predicted values, computed using parameters stored in OpenKIM,
  or a uniform error bars
  
The indicator configuration calculation is done by tuning the weights in the weighted
least-squares cost function so that the Fisher Information matrix (FIM) of the training
set is at least as large as the FIM of the target QoI. The weights are found by solving a
convex, semi-definite problem. Then, the weights are used to train the potential to find
a new optimal parameters. This process is then repeated until the weights converge.

In order to ensure that we get more and more information over the iteration, the weights
used in the training in step i are combinations between the optimal weights from the
convex optimization and the weights from the previous step. That is,

The target accuracy is obtained by evaluating the energy vs lattice constant
model at KIM's parameters, and take 10% of the predicted values. Suppose that the
configurations and weights of step 1 is `{"config1": w1a, "config2": w2a}`, where w1a and
w2a are the weight values. In the convex optimization of step 2, we get the weights
`{"config1": w1b, "config3": w3b}`. Then, the weights and configurations to use in the
model training step of step 2 is `{"config1": max([w1a, w1b]), "config2": w2a, "config3": w3b}`.
Note that in this way, we won't get the minimal set of the configurations, but that is not
the objective of the calculation. We just want a smaller subset of the training
configurations that contain information needed to make predictions of the target QoI.

Content of this folder:

* phonondispersion_data.pkl - It contains the predictions of the phonon dispersion curves,
  evaluated using KIM's parameters. It also contains 10% of the predicted values as the
  error bars.
* phonondispersion_data_uniform.pkl - Similar to phonondispersion)data.pkl, but the target
  error bars are set to be uniform for each branch, and the values are set to be 10% of
  the maximum energy for each branch.
* indicatorconfig_main.py - This file is the main file to run the indicator configuration
  calculation. The end result of this calculation is a set of reduced configurations with
  the corresponding optimal weights.
* indicatorconfig_linapprox_uq.py - Following the indicator configuration calculation,
  we propagate the uncertainty from the data to the parameters. In this script, the
  calculation is done using linearized model mapping from the data to the parameters.
  Having found the uncertainty of the parameters, we then propagate it to the target QoI
  to get the uncertainty of the target predictions. This is done by using linearized
  model mapping from the parameters to the target QoI. (We can also do Monte Carlo-type
  calculation.)
* parameter_variation.py - This script compare the parameter values across different
  iterations and uses MSER to estimate the burn-in period, which in this case is
  considered as the period in which the parameters are not settled around some values.
  
The calculation should be done by first running indicatorconfig_main.py, followed by
indicatorconfig_linapprox_uq.py.

Tips: We can initially set `maxsteps` and `warmup` in `indicatorconfig_main.py` to be the
same large numbers, then use `parameter_variation.py` to infer the burnin period. Then, we
can set `warmup` to this burnin value and rerun the indicator configuration calculation
again.
