# Power Systems

Thanks to Ben Francis for providing the codes in Julia for the 14- and 39-bus systems. His
codes are stored in {IEEE14/IEEE39}/models/{IEEE14/IEEE39}_SEModel.jl

The content of this directory:
* IEEE14 - Contains the scripts and notebooks to do information calculation using IEEE
  14-bus system.
* IEEE39 - Contains the scripts and notebooks to do information calculation using IEEE
  39-bus system.
* install_julia_requirements.jl - A script to install required packages in Julia.


## Additional requirements

First, the calculation in these examples are done in Jupyter notebook. You can install
Jupyter using `pip install` or running

``` bash
$ pip install -r requirements.txt
```

Additionally, the original model is written in Julia language, and some calculations
in these examples require some packages in Julia, including PyCall to access the code
from Python. After installing Julia, run the following command in the current directory to
install the packages in Julia:

``` bash
$ julia install_julia_requirements.jl
$ pip install julia
```
