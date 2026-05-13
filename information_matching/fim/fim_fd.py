import numpy as np

from .fim_base import FIMBase
from .finitediff import FiniteDifference, avail_method


class FIM_fd(FIMBase):
    """A class to compute the Jacobian and the FIM of a model using finite difference

    Parameters
    ----------
    model: callable ``model(x, **kwargs)``
        A function that we will evaluate the derivative of.
    transform: TransformBase, optional
        A transformation class instance with ``transform(x)`` and
        ``inverse_transform(x)`` methods for transforming the parameters in the
        Jacobian/FIM calculation.
    method: str
        A string that indicates the finite difference method to use in the derivative
        estimation, the available methods are: "FD", "FD2", "FD3", "FD4", "CD", "CD4".
    h: float or list (nparams,)
        Step size to use in the finite difference derivative.
    pool: object with a `map` method (optional)
        An object with map method for parallelization, e.g., ``multiprocessing.Pool``
        or ``concurrent.futures.ThreadPoolExecutor``. If not provided, the Jacobian will
        be computed in serial.
    """

    def __init__(self, model, transform=None, method="CD", h=0.1, pool=None):
        if method.upper() not in avail_method:
            raise ValueError(
                f"Method {method} is not available. Please choose from {avail_method}."
            )

        super().__init__(model, transform)
        self._method = method.upper()
        self._h = h
        self._pool = pool
        if pool is None:
            self.map_fn = map
        else:
            self.map_fn = pool.map

    def _model_args_wrapper(self, *args, **kwargs):
        """A wrapper function that inserts the keyword arguments to the model."""

        def model_eval(x):
            return self._model_wrapper(x, *args, **kwargs)

        return model_eval

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
        # Model to compute the derivative of
        fn = self._model_args_wrapper(*args, **kwargs)
        # Apply parameter transformation
        params = self.transform.transform(x)
        nparams = len(params)

        # Formatting h, we prefer to have a list of h values for each parameter, which
        # allows us to use different step size for each parameter.
        if isinstance(self._h, (float, int)):
            h = np.repeat(self._h, nparams)
        elif isinstance(self._h, (list, np.ndarray)):
            assert len(h) == nparams, "Please specify one step size for each parameter."
            h = self._h

        # Instantiate FiniteDifference class
        finitediff = FiniteDifference(params, h, method=self._method)
        # Generate perturbed parameters set that we use in derivative estimation
        params_set = finitediff.generate_params_set()
        # Iterate over this parameter set and evaluate the model
        results_list = list(self.map_fn(fn, params_set.values()))
        # Convert the results list to dictionary that can be input to
        # finitediff.estimate_derivative
        predictions_set = {key: preds for key, preds in zip(params_set, results_list)}
        # Estimate the derivative
        Jacobian = finitediff.estimate_derivative(predictions_set)

        return Jacobian
