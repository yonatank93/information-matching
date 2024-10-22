# SW MoS$_2$

This folder contains the files and scripts to find the indicator configurations via
information-matching calculation using Stillinger-Weber potential for molybdenum disulfide.

Thanks to Mingjian Wen for providing the potential, dataset, and QoI data. His related
work can be found at https://doi.org/10.1063/1.5007842 . Prior to running information-
matching calculation, user needs to extract the training dataset, e.g., by executing

``` bash
$ tar -xzvf sw_mos2_training_dataset.tar.gz
```

Detail of the calculation:

* Model: Stillinger-Weber
* System: Monolayer molybdenum disulfide
* Training configurations: Configurations from snapshots of AIMD at 750 K
* Training quantities: atomic forces
* Target QoI: energy change vs lattice compression curve, with da = (-0.5, 0.5) Angstrom
* Target precission: There are several options, specified by the "error_str"
  * If "mingjian", use data predicted by Mingjian's model and the target error is 10% of
    the predictions.
  * If "yonatan", use data predicted by Yonatan's model and target error is 10% of the
    predictions.
  * If "ips", use data predicted by Mingjian's model and target error is the standard
    deviation obtained using different IPs, as presented in Mingjian's paper.



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
  $ kim-api-collections-management install user SW_MX2_WenShirodkarPlechac_2017_MoS__MO_201919462778_001
  $ kim-api-collections-management install user SW_MX2_KurniawanPetrieWilliams_2021_MoS__MO_677328661525_000
  ```
* LAMMPS with Python interface and KIM package.



## Content

* models - A module that contains all the models we need, including a model to compute the
  training quantities and a model to compute the target predictions.
* original_configs - Contains the training dataset from Mingjian, which we set as
  candidate configurations.
* data - A directory that contains all the target error bars for the target predictions.
  The suffix of the files inside follow the "error_str" convention.
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



## Guide

1. First, download the candidate dataset to use for this example. This can be done by
   executing the following commands

	``` Python
	from information_matching.utils import download_dataset, avail_dataset

	# Print all available precomputed dataset
	print(avail_dataset)

	# Download the MoS2 candidate training atomic configurations
	download_dataset("sw_mos2_training_dataset")
	```

2. To find the indicator configurations, run `indicatorconfig_main.py`.

3. After that, run `indicatorconfig_linapprox.py` to get the uncertainty of the
   parameters and the QoI using a linear approximation based on the FIM and Gaussian
   distribution.

**Tips:** We can initially set `maxsteps` and `warmup` in `indicatorconfig_main.py` to be
the same large numbers, then use `parameter_variation.py` to infer the burnin period.
Then, we can set `warmup` to this burnin value and rerun the indicator configuration
calculation again.
