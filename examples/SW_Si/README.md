# SW_Si

This folder contains several examples of information-matching calculation to find the
indicator configuration for Stillinger-Weber potential for silicon.

Prior to running any examples presented here, please extract the dataset first, which can
be done by executing

``` bash
$ tar -xzvf sw_si_training_dataset.tar.gz
```


## Additional requirements

Install additional requirements by executing

``` bash
$ pip install -r requirements.txt
```

Additional external requirements:
* KIM-API to access the interatomic potential which is archived in OpenKIM. To use the
  Refer to https://openkim.org/doc/usage/obtaining-models/ on how to install KIM-API.
  After KIM-API is installed, then install the required potential:
  ``` bash
  $ kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_006
  ```
* LAMMPS with Python interface and KIM package.


## Content

* sw_si_training_dataset - Contains all candidate configurations for diamond
  silicon system with various lattice length. For each lattice constant, there is a
  perfect diamond lattice configuration as well as some perturbed diamond configurattions.
  This dataset needs to be downloaded, e.g., by executing `python download_dataset.py`.
* models - A module that contains all the models needed to do the indicator configuration
  calculation for SW Si cases.
* alat_small_range - The target QoI is the energy change vs lattice compression for
  compression $\Delta a$ from -0.5 to 0.5 angstrom.
* alat_medium_range - The target QoI is the energy change vs lattice compression for
  compression $\Delta a$ from -1.0 to 1.0 angstrom.
* alat_wide_range - The target QoI is the energy change vs lattice compression for
  compression $\Delta a$ from -1.0 to 2.0 angstrom.
* lattice_elastic_constants - The target QoIs are the lattice and elastic constants of
  diamond silicon.
* phonon_dispersion - The target QoI is the lowest phonon dispersion energy band for
  diamond silicon.
