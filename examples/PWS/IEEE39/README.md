# PWS IEEE 39-bus network

This folder contains the scripts and notebook to do information-matching calculation for
the IEEE 39-bus network:

* Model: IEEE 39-bus network
* Candidate configurations: 39 buses in the network, with voltage phasor measurements on
  each bus
* Target QoI: Full observability of the network, i.e., non-singular FIM


## Content

* energycurve_data.npz - It contains the predictions of the energy change vs lattice
  compression, evaluated using KIM's parameters. It also contains 10% of the predicted
  values as the error bars.
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

* models - A module that contains the model to compute the voltage phasor observations.
* compute_fims_IEEE39.py - A script to compute the FIM for each bus.
* fim_matching_IEEE39.ipynb - A notebook to find the optimal PMU placements for observing
  the entire network.
* fim_matching_IEEE39_singular_Area1.ipynb - A notebook to find the optimal PMU placements
  for area 1, as described in https://matpower.org/docs/ref/matpower5.0/menu5.0.html .
* fim_matching_IEEE39_singular_Area2.ipynb - A notebook to find the optimal PMU placements
  for area 2, as described in https://matpower.org/docs/ref/matpower5.0/menu5.0.html .
* fim_matching_IEEE39_singular_Area3.ipynb - A notebook to find the optimal PMU placements
  for area 3, as described in https://matpower.org/docs/ref/matpower5.0/menu5.0.html .


## Guide

1. First, compute the FIMs for all buses, by running `compute_fims_IEEE39.py`.
2. Then, find the optimal PMU placements to observe the entire network. This is done using
   the Jupyter notebook `fim_matching_IEEE39.ipynb`.
3. Finally, find the optimal PMU placements for each area using the notebooks
   `fim_matching_IEEE39_singular_Area1/Area2/Area3.ipynb`
