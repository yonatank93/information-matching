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

## Installing MISDP solver

For the power system examples, we modify the information-matching formulation and added an
additional constraint that the weights can only take binary values of 0 or 1. Essentially,
this means that we should place a PMU on a bus or not. We can then use the result from
this modified problem as an initial guess in the original formulation to get decimal
weights, which shows the target accuracy for the measurement.

The modified convex problem is considered as a mixed-integer semidefinite programming
(MISDP) problem. A solver for this type of problem is available in Matlab using SCIP-SDP
library. Here, we gives an instruction to install this library in Debian-based machine,
assuming Matlab is installed.

1. Installing [SCIP](https://scipopt.org/) (Solving Constrained Integer Programs)
   ```bash
   $ wget https://www.scipopt.org/download.php?fname=scipoptsuite-8.0.4.tgz  # Assuming to use ver8.0.4
   $ tar -xzvf scipoptsuite-8.0.4.tgz
   $ cd scipoptsuite-8.0.4
   $ mkdir build && cd build
   $ cmake .. -DAUTOBUILD=on -DCMAKE_INSTALL_PREFIX=${HOME}/local  # Assuming install dir is in $HOME/local
   $ make
   $ make check  # Run test and check installation
   $ make install
   ```

2. Installing [Mosek](https://www.mosek.com/) (as a SDP solver)
   ```bash
   $ wget https://download.mosek.com/stable/9.2.47/mosektoolslinux64x86.tar.bz2
   $ tar -xvf mosektoolslinux64x86.tar.bz2
   ```
   **Notes:**
   Mosek requries a license to run. One can request a free personal academic [license](https://www.mosek.com/products/academic-licenses/)
   which is valid for 365 days. The license is contained in a file called `mosek.lic`, and
   the license needs to be saved in `$HOME/mosek/mosek.lic`.

3. Installing [SCIP-SDP](https://www.opt.tu-darmstadt.de/scipsdp/)
   ```bash
   $ wget https://www.opt.tu-darmstadt.de/scipsdp/downloads/scipsdp-4.3.0.tgz
   $ tar -xzvf scipsdp-4.3.0.tgz
   $ cd scipsdp-4.3.0
   $ mkdir build && cd build
   $ cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/local \  # install dir
     -DSCIP_DIR=<path to scipoptsuite>/scip/build \
     -DSDPS=msk \  # use Mosek to solve sdp problem
     -DMOSEK_DIR=<path to mosek>/9.2/tools/platform/linux64x86  # path to Mosek solver
   $ make
   $ make install
   ```
  
4. Installing [MatlabSCIPInterface](https://github.com/scipopt/MatlabSCIPInterface)
   ```bash
   $ export SCIPDIR=${HOME}/local  # Exporting SCIP install directory
   $ git clone https://github.com/scipopt/MatlabSCIPInterface.git
   $ cd MatlabSCIPInterface
   $ matlabSCIPInterface_install  # Install SCIP matlab interface
   $ matlabSCIPSDPInterface_install  # Install SCIP-SDP matlab interface
   ```
