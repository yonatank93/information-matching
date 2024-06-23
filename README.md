# Indicator Configuration

This repository contains the scripts that we use to find the indicator configurations,
i.e., the informative training data, via information-matching method. This method can be
used in optimal experimental design and active learning.



# Get started

Please read this [file](https://github.com/yonatank93/indicator_configuration/blob/main/indicator_config.pdf).
It contains the introduction of the method and the detail of the calculation process.



# Installation

## Dependencies
Run `./setup.sh` to install the dependencies and the path to the environment.
**Note:** Currently, user needs to install VASP, KIM API, and LAMMPS with KIM package and
Python module manually. The `setup.sh` file doesn't handle this process yet.



# Content

* src directory - Contains Python modules for the information-matching calculation.
  * convex_optimization.py - The main module that solves the convex optimization problem
	in information-matching method to fnd the optimal weights and data points.
  * src/fim - Contains modules to compute the FIM of a model. We provide numerical
	derivative calculation via regular finite difference, Python's `numdifftools`, or
	Julia's `NumDiffTools`.
  * src/mcmc_utils - Contains modules that are used in calculations related to MCMC
	simulation, e.g., estimating the burn-in, autocorrelation, and assessing sample 
	convergence.
	
* theory - Contains the explanations about the information-matching method.
  * indicator_config.pdf - Gives a mathematical explanation about the information-matching
	method and the theory behind this method.
  * explore_positive_definite_matrix.ipynb - A notebook that shows the geometrical
	representation of the positive semidefinite constraint in the information-matching
	method.

* examples - This is where the scripts to run indicator configuration calculations and to
  do post-processing for each example case are stored.
  * PWS - This folder contains application of the information-matching method to solve the
	optimal sensor placement problem in power systems.
  * ORCA - This folder contains applications of the information-matching method in
	underwater acoustic problem, where the model used is a normal-mode model called ORCA.
	The applications include source localization and ocean environment inference.
  * SW_Si - This folder contains the test cases that use Stillinger-Weber potential
	for silicon. The candidate configurations consist of diamond silicon configurations
	with various lattice parameter. There are various cases with different target QoI and
	different choice of target error barrs used in these applications.
  * SW_MoS2 - This folder contains the calculation to find the indicator configurations
	via information-matching method to train a SW potential for molybdenum disulfide.



# How to cite

We are working on publishing the paper about this method.



# Contact

Yonatan Kurniawan </br>
email: [kurniawanyo@outlook.com](mailto:kurniawanyo@outlook.com)
