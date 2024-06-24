# KIM Model

This directory contains the computational routines to compute the target material
properties for silicon in diamond structure. These routines are used in model_target.py.

Content:
* base.py - Base model class to deal with basic, fundamental operations, such as creating
  calculator and updating parameters.
* latticeconstant.py - Contains a routine to compute the equilibrium lattice constant and
  cohesive energy of cubic crystal.
* EnergyLatconstDiamondSi - This module can be used to compute the energy as a function
  of lattice parameter for silicon in diamond structure.
* ElasticConstantsCubic - This module can be used to compute the lattice and elastic
  constants for cubic crystal. This module is a modification of the similarly named test
  in OpenKIM.
* PhononDispersionCurveCubic - This module can be used to compute the phonon dispersion
  curve for cubic crystal. The calculation follows the example given in ASE documentation.
