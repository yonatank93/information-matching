"""Note: To use this module, user needs to install Julia with NumDiffTools package
(https://github.com/mktranstrum/NumDiffTools.jl) installed. Then, user also needs to
install PyCall to run the calculation from Python.
"""

import julia
from julia import Base, NumDiffTools


def default_transform(x):
    return x


class FIM_jl:
    """A class to compute the jacobian and the FIM of a model using julia.

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
        Additional keyword arguments for the model.
    """

    def __init__(self, model, transform=None, inverse_transform=None, **kwargs):
        self.model = model
        self.fn = self._model_wrapper(kwargs)

        # Get the parameter transformation
        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform

        if inverse_transform is None:
            self.inverse_transform = default_transform
        else:
            self.inverse_transform = inverse_transform

    def _model_wrapper(self, kwargs):
        """A wrapper function that inserts the keyword arguments to the model."""

        def model_eval(x):
            return self.model(x, **kwargs)

        return model_eval

    def compute_jacobian(self, x, **kwargs):
        """Compute the jacobian of the model, evaluated at parameter ``x``.
        Parameter ``x`` should be written in the parameterization that the model
        uses.

        Parameters
        ----------
        x: np.ndarray (nparams,)
            Parameter values in which the Jacobian is evaluated. It should be
            written in the parameterization that the model uses.
        kwargs: dict
            Additional keyword arguments for the function to compute the jacobian.

        Returns
        -------
        np.ndarray (npred, nparams)
        """
        params = self.transform(x)
        return NumDiffTools.jacobian(self.fn, params, **kwargs)

    def compute_FIM(self, x, **kwargs):
        """Compute the FIM.

        Parameters
        ----------
        x: np.ndarray (nparams,)
            Parameter values in which the FIM is evaluated. It should be
            written in the parameterization that the model uses.
        kwargs: dict
            Additional keyword arguments for the function to compute the jacobian.

        Returns
        -------
        np.ndarray (nparams, nparams)
        """

        Jac = self.compute_jacobian(x, **kwargs)
        return Jac.T @ Jac
