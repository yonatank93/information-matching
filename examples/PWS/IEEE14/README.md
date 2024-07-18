# PWS IEEE 14-bus network

This folder contains the scripts and notebook to do information-matching calculation for
the IEEE 14-bus network:

* Model: IEEE 14-bus network
* Candidate configurations: 14 buses in the network, with voltage phasor measurements on
  each bus
* Target QoI: Full observability of the network, i.e., non-singular FIM


## Content

* models - A module that contains the model to compute the voltage phasor observations.
* compute_fims_IEEE14.py - A script to compute the FIM for each bus.
* fim_matching_IEEE14.ipynb - A notebook to find the optimal PMU placements for observing
  the entire network.
* fim_matching_IEEE14_singular_AreaA.ipynb - A notebook to find the optimal PMU placements
  for area A, as described in https://ieeexplore.ieee.org/abstract/document/8586586 .
* fim_matching_IEEE14_singular_AreaB.ipynb - A notebook to find the optimal PMU placements
  for area B, as described in https://ieeexplore.ieee.org/abstract/document/8586586 .


## Guide

1. First, compute the FIMs for all buses, by running `compute_fims_IEEE14.py`.
2. Then, find the optimal PMU placements to observe the entire network. This is done using
   the Jupyter notebook `fim_matching_IEEE14.ipynb`.
3. Finally, find the optimal PMU placements for each area using the notebooks
   `fim_matching_IEEE14_singular_AreaA/AreaB.ipynb`
