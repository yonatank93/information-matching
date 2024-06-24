Folder `scale{i}` contains the results of indicator configuration calculation
where the target error bars are set to be `i*10%` of the predictions evaluated
using KIM's parameters.

`scale{i}_equilibration` contains the calculation results using scale i, with 10 maximum
iterations and >10 burn-in steps. The purpose is to estimate the number of burn-in steps.
