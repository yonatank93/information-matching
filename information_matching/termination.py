"""This script contains utility functions to check termination conditions:
* check if the run has converged.
"""

import numpy as np


def check_convergence(weights_1, weights_2, tol=1e-8):
    """Compare the configurations and weights. The calculation converges if the set of
    weights are the same.
    """
    # Compare the configuration
    # If the 2 configuration sets are the same, set same=True, refering to
    # possibility of convergence, and move to the next step, i.e., comparing
    # the weight values.
    same_configs = list(weights_1) == list(weights_2)

    # Compare the weights
    # Only do the comparison if the configurations are the same, denoted by
    # same=True from the previous comparison.
    if same_configs:
        # If the weights are different, change same back to False.
        step1_weights = _get_weight_values(weights_1)
        step2_weights = _get_weight_values(weights_2)
        same_weights = np.allclose(step1_weights, step2_weights, rtol=tol, atol=tol)
    else:
        same_weights = False

    return same_weights


def _get_weight_values(weights_dict):
    """Retrieve the weight values for all configurations in ``weights_dict``."""
    weight_values = []
    for value in weights_dict.values():
        weight_values.append(value)
    return weight_values
