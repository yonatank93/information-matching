import numpy as np
from scipy.stats import multivariate_normal


def logprior_uniform(x, bounds):
    """Compute the logarithm of a uniform distribution as the log-prior.

    Parameters
    ----------
    x : np.ndarray
        Parameters to evaluate
    bounds : np.ndarray (nparams, 2,)
        Lower (left column) and upper (right column) bounds of the
        uniform prior.

    Returns
    -------
    float
        Log-prior using uniform prior distribution.
    """
    bounds = np.array(bounds)
    l_bound = bounds[:, 0]
    u_bound = bounds[:, 1]

    if all(np.less(x, u_bound)) and all(np.greater(x, l_bound)):
        ret = 0.0
    else:
        ret = -np.inf
    return ret


def logprior_normal(x, mean, cov):
    """Logarithm of a multivariate normal distribution.

    Parameters
    ----------
    x : np.ndarray
        Parameters to evaluate
    mean : np.ndarray
        Vector of the mean of the multivariate normal distribution.
    cov : np.ndarray (nparams, nparams,)
        Covariance matrix.

    Returns
    -------
    float
        Log-prior value.
    """
    pdf = multivariate_normal.pdf(x, mean=mean, cov=cov)
    logpdf = np.log(pdf)
    return logpdf


def logprior_jeffreys(self, x, jacobian):
    """Log-prior using Jeffreys prior.

    Parameters
    ----------
    x : np.ndarray
        Parameters to evaluate
    jacobian : callable jacobian(x)
        A function to compute the Jacobian matrix of the model

    Returns
    -------
    float
        Log-prior value.
    """
    Jac = jacobian(x)
    sigs = np.linalg.svd(Jac, compute_uv=False)
    logp = np.sum(np.log(sigs))
    return logp
