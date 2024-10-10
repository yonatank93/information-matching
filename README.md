# Information-matching

Information-matching is a Python package that provides modules and functions for
performing information-matching calculations. These calculations are designed to identify
informative training data, making the package particularly useful for Optimal Experimental
Design (OED) and Active Learning (AL) tasks.



## Installation

### Using pip

``` bash
pip install information-matching
```

### From source

``` bash
git clone https://github.com/yonatank93/information-matching.git
cd information-matching
pip install -e .
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

We are working on publishing a paper about this method.




## Contact

Yonatan Kurniawan </br>
email: [kurniawanyo@outlook.com](mailto:kurniawanyo@outlook.com)
