# Models SW Si

This folder contains the models that use SW potential for silicon to compute some
quantities.

* model_train.py - Contains classes that can be use to compute the configuration energy
  and atomic forces. These classes can be used in the training process.
* model_target.py - Contains classes that compute the target properties for silicon in
  diamond structure.
  
  
Other modules:
* kimmodel - This module contains the longer script to compute the target properties. The
  classes in model_target.py can be thought as wrapper classes for the submodules inside
  this directory.
