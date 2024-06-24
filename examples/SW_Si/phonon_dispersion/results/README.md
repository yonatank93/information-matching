`equilibration` contains the calculation results with 10 maximum iterations and >10
burn-in steps. The purpose is to estimate the number of burn-in steps.

`measurement` constains the calculation result where the number of burn-in steps is set to
be whatever number estimated previously (using results in `equilibration`).

`uniform` contains the results of calculation done similar to those in `measurement`, but
with uniform target error bars.
