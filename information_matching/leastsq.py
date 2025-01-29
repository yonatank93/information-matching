"""This module contains functions and classes that are needed for model training.
For the least-squares optimization, we will use geodesiclm package. We also
have a class weights that set the weights for KLIFF model from the weights of
the reduced configurations.
"""

import numpy as np
import scipy.optimize

try:
    from geodesiclm import geodesiclm

    geodesicLM_avail = True
except ImportError:
    geodesicLM_avail = False


# Available methods
leastsq_methods = ["trf", "dogbox", "lm"]
minimize_methods = [
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-const",
    "dogled",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
]


def leastsq(func, x0, method, **kwargs):
    """This is the main function that solves the least squares problem. It uses either
    geodesicLM algorithm or scipy.optimize.least_squares function. geodesicLM is
    efficient, but requires the number of predictions to be at least as many as the number
    of parameters. Otherwise, use, e.g., TRF algorithm implemented in scipy.

    Parameters
    ----------
    func: callable ``func(x)``
        The function used in the minimization. If using method compatible with
        ``geodesic_lm`` or ``scipy.optimize.least_squares``, this should be the residuals.
        For method compatible with ``scipy.optimize.minimize``, this should be the cost.
    x0: np.ndarray (nparams,)
        Initial parameter guess.
    kwargs: dict
        Keyword arguments for the solver.

    Returns
    -------
    np.ndarray
        Optimal parameters.
    dict
        Result of the optimization.
    """
    if method == "geodesiclm":
        if geodesicLM_avail:
            return geodesiclm(func, x0, full_output=True, **kwargs)
        else:
            raise ImportError("Please install Geodesic-LM or choose different method")
    elif method in leastsq_methods:
        opt_result = scipy.optimize.least_squares(func, x0, **kwargs)
        return convert_leastsq_result_format(opt_result)
    elif method in minimize_methods:
        opt_result = scipy.optimize.minimize(func, x0, **kwargs)
        return convert_minimize_result_format(opt_result)
    else:
        raise ValueError("Method not available")


# GLM Keys: "converged", "iters", "msg", "fvec", "fjac"
# least_squares Keys: "x", "cost", "fun", "jac", "grad", "optimality",
# "active_mask", "nfev", "njev", "status", "message", "success"
# Connection: "status" -> "converged", ["nfev", "njev", 0, 0] -> "iters",
# "message" -> "msg", "fun" -> "fvec", "jac" -> "fjac"


def convert_leastsq_result_format(opt_result):
    """Convert the output of ``scipy.optimize.least_squares`` to be in the same format as
    the output of ``geodesicLM``.

    Parameters
    ----------
    opt_result
        Output of scipy.optimize.least_squares.

    Returns
    -------
    np.ndarray
        Optimal parameters.
    dict
        Result of the optimization.
    """
    x = opt_result.x
    info_dict = {
        "converged": opt_result.status,
        "iters": [opt_result.nfev, opt_result.njev, 0, 0],
        "msg": opt_result.message,
        "fvec": opt_result.fun,
        "fjac": opt_result.jac,
    }
    return x, info_dict


def convert_minimize_result_format(opt_result):
    """Convert the output of ``scipy.optimize.minimize`` to be in the same format as the
    output of ``geodesicLM``.

    Parameters
    ----------
    opt_result
        Output of scipy.optimize.least_squares.

    Returns
    -------
    np.ndarray
        Optimal parameters.
    dict
        Result of the optimization.
    """
    x = opt_result.x
    info_dict = {
        "converged": opt_result.status,
        "iters": [opt_result.nfev, opt_result.njev, 0, 0],
        "msg": opt_result.message,
        "fvec": opt_result.fun,
        "fjac": opt_result.jac,
    }
    return x, info_dict


def compare_opt_results(opt_results):
    """Compare the optimization results from different starting points and return the
    result with the lowest cost. ``opt_results`` is a dictionary, where each item is the
    full output of ``geodesiclm``.

    Parameters
    ----------
    opt_results: dict
        A dictionary containing the results of multiple optimizations, e.g., from using
        different initial guess.

    Returns
    -------
    np.ndarray
        Best optimal parameters.
    float
        Best cost.
    dict
        Result of the best optimization.
    """
    # Comparing the cost and get the point with the lowest cost
    # Instantiate an array to store the values to compare. The first column
    # contains the key and the second column contain the minimum cost found.
    min_cost_list = np.empty((0, 2))
    for key in opt_results:
        opt_result = opt_results[key]
        res = opt_result[1]["fvec"]  # residuals
        if isinstance(res, float):  # This result comes from scipy.optimize.minimize
            cost = res
        else:  # This result comes from methods that use residual
            cost = np.sum(res**2) / 2
        min_cost_list = np.vstack((min_cost_list, [key, cost]))

    # Find the best result
    idx = np.nanargmin(min_cost_list[:, 1])  # Index that we want
    min_key = int(min_cost_list[idx, 0])
    opt_result = opt_results[min_key]
    opt_cost = min_cost_list[idx, 1]
    new_bestfit = opt_result[0]
    new_bestfit = opt_result[0]

    return new_bestfit, opt_cost, opt_result
