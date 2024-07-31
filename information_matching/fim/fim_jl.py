"""Note: To use this module, user needs to install Julia with NumDiffTools package
(https://github.com/mktranstrum/NumDiffTools.jl) installed. Then, user also needs to
install PyCall to run the calculation from Python.
"""

import julia
from julia import NumDiffTools

from .fim_base import FIMBase


class FIM_jl(FIMBase):
    """A class to compute the Jacobian and the FIM of a model using julia.

    Parameters
    ----------
    model: callable ``model(x, **kwargs)``
        A function that we will evaluate the derivative of.
    transform: callable ``transform(x)``
        A function to perform transformation from the parameterization of the
        model to what ever parameterization we want to use.
    inverse_transform: callable ``inverse_transform(x)``
        This is the inverse of transformation function above.
    kwargs: dict
        Additional keyword arguments for ``Numdifftools.jacobian``.
    """

    def __init__(self, model, transform=None, inverse_transform=None, **kwargs):
        super().__init__(model, transform, inverse_transform)
        self._jac_kwargs = kwargs

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
        fn = self._model_args_wrapper(*args, **kwargs)
        params = self.transform(x)
        return NumDiffTools.jacobian(fn, params, **self._jac_kwargs)
