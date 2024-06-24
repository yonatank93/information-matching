# Models SW MoS_2

This folder contains the models that use SW potential for molybdenum disulfide to compute
some quantities.

* models.py - Contains classes that can be use to compute the atomic forces for training
  and energy vs lattice parameter curve for the target properties.
  
Other modules:
* EnergyLatconstMoS2 - This module contains the longer script to compute the target
  properties. The class in models.py can be thought as wrapper classes for this module.
