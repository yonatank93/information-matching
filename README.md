# Information-matching

Information-matching is a Python package that provides modules and functions for
performing information-matching calculations. These calculations are designed to identify
informative training data, making the package particularly useful for Optimal Experimental
Design (OED) and Active Learning (AL) tasks.



## Installation

<!-- ### Using pip -->

<!-- ``` bash -->
<!-- pip install information-matching -->
<!-- ``` -->

### From source

``` bash
git clone https://github.com/yonatank93/information-matching.git
cd information-matching
python -m pip install .              # For a minimal installation
python -m pip install ".[examples]"  # For running example notebooks and scripts
python -m pip install -e ".[dev]"    # For development, testing, and examples
```



## Usage

A typical workflow to use information-matching method to find the indicator configurations
and the corresponding weights is as follows:
1. Define the models to compute the training and target quantities
2. Define the candidate configurations or data
3. Compute the FIM for each candidate configuration and the FI of the target quantities
4. Solve a convex optimization problem to match the FIMs
5. Propagate the uncertainties to the target quantities

We provide toy examples that illustrate the common workflow to use information-matching
method in
[OED](https://github.com/yonatank93/information-matching/tree/main/examples/Toy_example/weather_oed.ipynb)
and
[AL](https://github.com/yonatank93/information-matching/tree/main/examples/Toy_example/weather_al.ipynb).



	
## Examples

We provide several examples of the application of information-matching in OED and AL
problems in several different scientific fields. Use these examples to reproduce the
results in the corresponding paper.
**Note:** These examples have additional dependencies. Please read the README file for
those examples.
* Toy_example - 
  This folder contains notebooks that show general workflows for using
  information-matching in OED and AL problems. Use these examples as a tutorial on how
  to use this package.
* PWS -
  This folder contains application of the information-matching method to solve the
  optimal sensor placement problem in power systems.
* ORCA -
  This folder contains applications of the information-matching method in
  underwater acoustic problem, where the model used is a normal-mode model called ORCA.
  The applications include source localization and ocean environment inference.
* SW_Si -
  This folder contains the test cases that use Stillinger-Weber potential
  for silicon. The candidate configurations consist of diamond silicon configurations
  with various lattice parameter. There are various cases with different target QoI and
  different choice of target error barrs used in these applications.
* SW_MoS2 -
  This folder contains the calculation to find the indicator configurations
  via information-matching method to train a SW potential for molybdenum disulfide.




## How to cite

If you use this package, please cite our accompanying paper:

```
@article{information_matching,
  title		= {An information-matching approach to optimal experimental design and active learning}, 
  volume	= {128},
  rights	= {All rights reserved},
  ISSN		= {0003-6951},
  DOI		= {10.1063/5.0296026},
  number	= {6},
  journal	= {Applied Physics Letters},
  author	= {Kurniawan, Yonatan and Neilsen, Tracianne B. and Francis, Benjamin L. and Stankovic, Alex M. and Wen, Mingjian and Nikiforov, Ilia and Tadmor, Ellad B. and Bulatov, Vasily V. and Lordi, Vincenzo and Transtrum, Mark K.},
  year		= {2026},
  month		= feb,
  pages		= {064104}
}
```




## Contact

Yonatan Kurniawan </br>
email: [kurniawanyo@outlook.com](mailto:kurniawanyo@outlook.com)
