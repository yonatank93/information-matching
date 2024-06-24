`scale{i}_equilibration` contains the calculation results using scale i, with 10 maximum
iterations and >10 burn-in steps. The purpose is to estimate the number of burn-in steps.

The target error bars in `scale1` is 10% of the prediction at KIM's parameters.

`scale1a` is similar to `scale1`, but the initial parameter values are set to be zeros.
The number of burn-in steps is also different (7 steps instead of 3) and was obtained with
similar way as in `scale1_equilibration`.

The target error bars in `scale2` is 20% of the prediction at KIM's parameters.

The target error bars in `scale3` is 30% of the prediction at KIM's parameters.

The target error bars in `scale4` is 40% of the prediction at KIM's parameters.

The target error bars in `scale5` is 50% of the prediction at KIM's parameters.

The target error bars in `uniform` is a uniform values across da values, where the values
are set to be the maximum number of target error bars used in `scale1`.

`compare_errors.ipynb` is a Jupyter notebook that can be used to compare the results in
`scale1` to `scale5`. That is, this notebook can be used to explore the effect of the
indicator configuration calculation results as we scale the target error bars uniformly.
