# Underwater acoustic - ORCA

This directory contains the applications of information-matching via the FIM in underwater
acoustic using a normal-mode model called ORCA.

Thanks to Traci Nielsen for giving me access to the `uwlib` library to use ORCA.
Additionally, thanks to Michael Mortenson for developing the `fimpack` to help computing
the Jacobian of the ORCA model and for helping me to setup this library.


## Disclaimer

The calculations for the underwater acoustic cases require proprietary `uwlib` and
`fimpack` libraries developed at BYU that implement the ORCA model. However, Yonatan has
tried his best to pre-compute some necessary data and modify the codes so that other user
who doesn't have access to these libraries can still run the information-matching
calculations using the pre-computed data.


## Additional requirements

The calculation in these examples are done in Jupyter notebook. You can install Jupyter
using `pip install` or running

``` bash
$ pip install -r requirements.txt
```


## Content

* data - This folder contains the necessary data that are pre-computed to run the
  information-matching calculation. If there are any missing data, please contact the
  developer.
* *.ipynb - Jupyter notebooks that run the main calculations. See the Gude section for
  more details about what each notebook is doing.


## Guide

0. Although this example uses a proprietary library, we have precomputed the necessary
   data to reproduce the results. The data can be downloaded and extracted using an
   internal function, for example by executing
   ```python
   from information_matching.utils import download_dataset, avail_dataset

   # Print all available precomputed dataset
   print(avail_dataset)

   # Download dataset for underwater acoustic example
   download_dataset(["transmission_loss", "fim_environment", "fim_source"])
   ```
1. First, open and execute `00_generate_transmission_loss_data.ipynb` to extract the
   transmission loss data. These data are needed to plot the results of the informamtion-
   matching calculation. Without this data and the plots, the information-matching results
   might have less meaning.
2. Then, open and execute `01a_compute_fims_environment.ipynb` and `01b_compute_fims_source.ipynb`
   to compute the FIMs, where the derivatives are taken with respect to the environmental
   parameters and the source loctions, respectively.
   User is **required** to execute these notebooks to get the FIMs used in the information-
   matching calculations
3. Finally, user can run the information-matching calculation for this model. There are
   several cases provided here.
   a. `02_fim_matching_environment.ipynb` finds the optimal hydrophone locations to
      precisely infer the environmental parameters, including the basement/half-space
	  parameters. The calculation uses the transmission loss from both sound sources.
	  i) `02a_fim_matching_environment_top_source.ipynb` does similar calculation, but
		 it only uses the transmission loss data from the top source.
	  ii) `02ai_fim_matching_environment_top_source_singular.ipynb` also only uses the
		  transmission loss data from the top source, and we ignore the basement
		  parameters. That is, we set their target precision to infinity.
	  iii) `02b_fim_matching_environment_bottom_source.ipynb` includes the inference of
		   the basement parameters, and we only use the transmission loss data from the
		   bottom sound source.
   b. `03_fim_matching_source_localization.ipynb` finds the optimal hydrophon locations to
      precisely localize the sound sources. In this calculation, we treat the
	  environmental parameters as constants.
   c. `04_fim_matching_source_environment.ipynb` finds the optimal hydrophon locations to
      precisely localize the sound sources, but we still treat all environmental
	  parameters as tunable parameters. However, their target precisions are set to
	  infinity because we just want to localize the sources. At the end, we will also get
	  the uncertainty of the inferred environmental parameters.
	  i) `04a_fim_matching_source_environment_nobasement.ipynb` excludes the basement
		 parameters in the formulation. That is, the basement parameters are treated as
		 constants.
