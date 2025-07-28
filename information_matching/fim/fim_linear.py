from multiprocessing import Pool
import numpy as np

from .fim_base import FIMBase


class FIM_linear(FIMBase):
    """A class to compute the Jacobian and the FIM of a linear model without access to
    the design matrix.

    In theory, the Jacobian for a linear model is the design matrix itself. In practice,
    however, we may not have access to the design matrix, but we can overcome this by
    evaluating the model at the basis vectors of the parameter space to extract the
    design matrix. This class implements this approach.

    Parameters
    ----------
    model: callable ``model(x, **kwargs)``
        A function that we will evaluate the derivative of.
    idx_list: list of int (optional)
        A list that contains the indices of the parameters to compute the Jacobian for.
        If not provided, the Jacobian will be computed for all parameters.
    pool: object with a `map` method (optional)
        An object with map method for parallelization, e.g., ``multiprocessing.Pool``
        or ``concurrent.futures.ThreadPoolExecutor``. If not provided, the Jacobian will
        be computed in serial.
    """

    def __init__(self, model, idx_list=None, pool=None):
        super().__init__(model, None, None)
        self._idx_list = idx_list
        self._pool = pool
        if pool is None:
            self.map_fn = map
        else:
            self.map_fn = pool.map

    def Jacobian(self, x, *args, **kwargs):
        """Compute the Jacobian of the model, evaluated at parameter ``x``.
        Parameter ``x`` should be written in the parameterization that the model
        uses.

        Parameters
        ----------
        x: np.ndarray (nparams,)
            Parameter values in which the Jacobian is evaluated. It should be
            written in the parameterization that the model uses.
        args, kwargs:
            Additional positional and keyword arguments for the model.

        Returns
        -------
        np.ndarray (npred, nparams)
        """
        # Generate the parameters to evaluate
        nparams = len(x)
        V = np.eye(nparams)
        if self._idx_list is not None:
            V = V[:, self._idx_list]

        # Generate the design matrix
        jac = list(self.map_fn(self._model_wrapper, V.T))
        return np.array(jac).T
