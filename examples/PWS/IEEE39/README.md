# PWS IEEE 39-bus network

This folder contains the scripts and notebook to do information-matching calculation for
the IEEE 39-bus network:

* Model: IEEE 39-bus network
* Candidate configurations: 39 buses in the network, with voltage phasor measurements on
  each bus
* Target QoI: Full observability of the network, i.e., non-singular FIM


## Content

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
