import numpy as np
import pickle


def ptemcee_save(sampler, filename=None, **dump_kwargs):
    """Put the informations from ``emcee.ptsampler.PTSampler`` as a dictionary
    and save it as a PKL file.

    Parameters
    ----------
    sampler: object
        ``emcee.ptsampler.PTSampler`` object.
    filename: str (optional)
        Path and name of file to export the dictionary.
    **dump_kwargs: dict (optional)
        Other keyword arguments for ``pickle.dump``.

    Returns
    -------
    results: dict
        Dictionary containing PTMCMC results.

    Note
    ----
    The meaning of the dictionary keys:
    * chain - Position of walkers at every iteration, stored as an array with
    shape (ntemps, nwalkers, nsteps, nparams).
    * acceptance_fraction - Fraction of the accepted proposal move.
    * adaptation_lag - This quantity is like a decay rate, used to calculate
    kappa in eq. 12 in `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.
    * adaptation_time - This quantity is like a decay time, used to calculate
    kappa in eq. 12 in `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.
    * beta_history - List of betas for each iteration.
    * betas - List of initial betas.
    * loglikelihood - Values of log likelihood for each iteration.
    * logprobability - Values of log probability, which is product of the
    likelihood and prior, for each iteration.
    * nprop - Number of proposal move made, which is usually equal to number of
    step or iteration.
    * nprop_accepted - Number of accepted proposed move, which is equal to
    :math:`nprop * acceptance_fraction`.
    * nswap - Number of attemps to swap chains from adjacent temperature.
    * nswap_accepted - Number of accepted temperature swap.
    * tswap_acceptance_fraction - Acceptance fraction of temperature swap,
    which can be calculated as ratio of nswap_accepted and nswap.
    """
    results = dict(
        chain=sampler.chain,
        acceptance_fraction=sampler.acceptance_fraction,
        adaptation_lag=sampler.adaptation_lag,
        adaptation_time=sampler.adaptation_time,
        beta_history=sampler.beta_history,
        betas=sampler.betas,
        loglikelihood=sampler.loglikelihood,
        logprobability=sampler.logprobability,
        nprop=sampler.nprop,
        nprop_accepted=sampler.nprop_accepted,
        nswap=sampler.nswap,
        nswap_accepted=sampler.nswap_accepted,
        tswap_acceptance_fraction=sampler.tswap_acceptance_fraction,
    )

    if filename:
        with open(filename, "wb") as f:
            pickle.dump(results, f, **dump_kwargs)

    return results


def emcee_save(sampler, filename=None, **dump_kwargs):
    """Put the informations from ``emcee.EnsembleSampler`` as a dictionary and
    save it as a PKL file.

    Parameters
    ----------
    sampler: object
        ``emcee.ptsampler.PTSampler`` object.
    filename: str (optional)
        Path and name of file to export the dictionary.
    **dump_kwargs: dict (optional)
        Other keyword arguments for ``pickle.dump``.

    Returns
    -------
    results: dict
        Dictionary containing PTMCMC results.

    Note
    ----
    The meaning of the dictionary keys:
    * chain - Position of walkers at every iteration, stored as an array with
    shape (nwalkers, nsteps, nparams).
    * acceptance_fraction - Fraction of the accepted proposal move.
    * logprobability - Values of log probability, which is product of the
    likelihood and prior, for each iteration.
    which can be calculated as ratio of nswap_accepted and nswap.
    """
    results = dict(
        chain=sampler.chain,
        acceptance_fraction=sampler.acceptance_fraction,
        logprobability=sampler.lnprobability,
    )
    if filename:

        with open(filename, "wb") as f:
            pickle.dump(results, f, **dump_kwargs)

    return results
