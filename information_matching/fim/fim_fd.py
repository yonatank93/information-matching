import numpy as np


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


def default_transform(x):
    return x


class FIM_fd:
    """A class to compute the jacobian and the FIM of a model using finite difference

    Parameters
    ----------
    model: callable ``model(x, **kwargs)``
        A function that we will evaluate the derivative of.
    deriv_fn: callable ``deriv_fn(f, x, v, h)``
        A function to do finite difference derivative where f is the function to take the
        derivative, x is the vector of parameters to evaluate the derivative, v is the
        direction to take the derivative, and h is the step size.
    transform: callable ``transform(x)``
        A function to perform transformation from the parameterization of the
        model to what ever parameterization we want to use.
    inverse_transform: callable ``inverse_transform(x)``
        This is the inverse of transformation function above.
    kwargs: dict
        Additional keyword arguments for the model.
    """

    def __init__(
        self, model, deriv_fn=FD, transform=None, inverse_transform=None, **kwargs
    ):
        self.model = model
        self.fn = self._model_wrapper(kwargs)
        self.deriv_fn = deriv_fn

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

    def compute_jacobian(self, x, h=0.1):
        """Compute the jacobian of the model, evaluated at parameter ``x``.
        Parameter ``x`` should be written in the parameterization that the model
        uses.

        Parameters
        ----------
        x: np.ndarray (nparams,)
            Parameter values in which the Jacobian is evaluated. It should be
            written in the parameterization that the model uses.
        h: float or list (nparams,)
            Step size to use in the finite difference derivative.

        Returns
        -------
        np.ndarray (npred, nparams)
        """
        # Apply parameter transformation
        params = self.transform(x)
        nparams = len(params)
        # Formatting h, we prefer to have a list of h values for each parameter, which
        # allows us to use different step size for each parameter.
        if isinstance(h, (float, int)):
            h = np.repeat(h, nparams)
        elif isinstance(h, (list, np.ndarray)):
            assert len(h) == nparams, "Specify one step size for each parameter."

        # Now, create a list of v vectors. This should just be the column of an identity
        # matrix, since we want to perturb the parameters one at a time for each column
        # of the Jacobian.
        vs = np.eye(nparams)

        # Compute the Jacobian
        def jac_column_wrapper(ii):
            return self._compute_jacobian_one_column(x, vs[ii], h[ii])

        # Note: There is a possiblility to parallelize this part in the future
        jacobian_T = np.array(list(map(jac_column_wrapper, range(nparams))))
        return jacobian_T.T

    def _compute_jacobian_one_column(self, x, v, h):
        """Compute one column of the Jacobian matrix."""
        return self.deriv_fn(self.fn, x, v, h)

    def compute_FIM(self, x, h=0.1):
        """Compute the FIM.

        Parameters
        ----------
        x: np.ndarray (nparams,)
            Parameter values in which the FIM is evaluated. It should be
            written in the parameterization that the model uses.
        h: float or list (nparams,)
            Step size to use in the finite difference derivative.

        Returns
        -------
        np.ndarray (nparams, nparams)
        """
        Jac = self.compute_jacobian(x, h)
        return Jac.T @ Jac
