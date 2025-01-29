from multiprocessing import Pool
import numpy as np

from .fim_base import FIMBase
from .finitediff import FiniteDifference, avail_method


class FIM_fd(FIMBase):
    """A class to compute the Jacobian and the FIM of a model using finite difference

    Parameters
    ----------
    model: callable ``model(x, **kwargs)``
        A function that we will evaluate the derivative of.
    transform: callable ``transform(x)``
        A function to perform transformation from the parameterization of the
        model to what ever parameterization we want to use.
    inverse_transform: callable ``inverse_transform(x)``
        This is the inverse of transformation function above.
    method: str
        A string that indicates the finite difference method to use in the derivative
        estimation, the available methods are: "FD", "FD2", "FD3", "FD4", "CD", "CD4".
    h: float or list (nparams,)
        Step size to use in the finite difference derivative.
    nprocs: int
        Number of parallel processes to use in the Jacobian computation.
    """

    def __init__(
        self, model, transform=None, inverse_transform=None, method="FD", h=0.1, nprocs=1
    ):
        super().__init__(model, transform, inverse_transform)
        self._method = method
        self._h = h
        self._nprocs = nprocs

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
        params = self.transform(x)
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
        if self._nprocs == 1:  # Just in case if the function is not picklable
            results_list = [fn(p) for p in params_set.values()]
        else:
            with Pool(self._nprocs) as p:
                results_list = p.map(fn, params_set.values())
        # Convert the results list to dictionary that can be input to
        # finitediff.estimate_derivative
        predictions_set = {key: preds for key, preds in zip(params_set, results_list)}
        # Estimate the derivative
        Jacobian = finitediff.estimate_derivative(predictions_set)

        return Jacobian
