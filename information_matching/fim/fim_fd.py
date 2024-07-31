import numpy as np

from .fim_base import FIMBase


# Forward Difference Formulas
def FD(f, x, v, h):
    return (f(x + h * v) - f(x)) / h


def FD2(f, x, v, h):
    return (-f(x + 2 * h * v) + 4 * f(x + h * v) - 3 * f(x)) / (2 * h)


def FD3(f, x, v, h):
    return (f(x + 4 * h * v) - 12 * f(x + 2 * h * v) + 32 * f(x + h * v) - 21 * f(x)) / (
        12 * h
    )


def FD4(f, x, v, h):
    return (
        -f(x + 8 * h * v)
        + 28 * f(x + 4 * h * v)
        - 224 * f(x + 2 * h * v)
        + 512 * f(x + h * v)
        - 315 * f(x)
    ) / (168 * h)


# Center Difference Formulas
def CD(f, x, v, h):
    return (f(x + h * v) - f(x - h * v)) / (2 * h)


def CD4(f, x, v, h):
    return (
        -f(x + 2 * h * v) + 8 * f(x + h * v) - 8 * f(x - h * v) + f(x - 2 * h * v)
    ) / (12 * h)


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
    deriv_fn: callable ``deriv_fn(f, x, v, h)``
        A function to do finite difference derivative where f is the function to take the
        derivative, x is the vector of parameters to evaluate the derivative, v is the
        direction to take the derivative, and h is the step size.
    h: float or list (nparams,)
        Step size to use in the finite difference derivative.
    """

    def __init__(self, model, transform=None, inverse_transform=None, deriv_fn=FD, h=0.1):
        super().__init__(model, transform, inverse_transform)
        self._deriv_fn = deriv_fn
        self._h = h

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

        # Now, create a list of v vectors. This should just be the column of an identity
        # matrix, since we want to perturb the parameters one at a time for each column
        # of the Jacobian.
        vs = np.eye(nparams)

        # Compute the Jacobian
        def jac_column_wrapper(ii):
            return self._deriv_fn(fn, x, vs[ii], h[ii])

        # Note: There is a possiblility to parallelize this part in the future
        Jacobian_T = np.array(list(map(jac_column_wrapper, range(nparams))))
        return Jacobian_T.T
