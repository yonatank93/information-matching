import numdifftools as nd


def default_transform(x):
    return x


class FIM_nd:
    """A class to compute the jacobian and the FIM of a model using numdifftools.

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
        Additional keyword arguments for ``numdifftools.Jacobian``.
    """

    def __init__(self, model, transform=None, inverse_transform=None, **kwargs):
        self.model = model
        self.jac_func = nd.Jacobian(self.model, method="forward", **kwargs)

        # Get the parameter transformation
        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform

        if inverse_transform is None:
            self.inverse_transform = default_transform
        else:
            self.inverse_transform = inverse_transform

    def compute_jacobian(self, x, *args, **kwargs):
        """Compute the jacobian of the model, evaluated at parameter ``x``.
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
        params = self.transform(x)
        return self.jac_func(params, *args, **kwargs)

    def compute_FIM(self, x, *args, **kwargs):
        """Compute the FIM.

        Parameters
        ----------
        x: np.ndarray (nparams,)
            Parameter values in which the FIM is evaluated. It should be
            written in the parameterization that the model uses.
        args, kwargs:
            Additional positional and keyword arguments for the model.

        Returns
        -------
        np.ndarray (nparams, nparams)
        """
        Jac = self.compute_jacobian(x, *args, **kwargs)
        return Jac.T @ Jac

    def __call__(self, x, *args, **kwargs):
        return self.compute_FIM(x, *args, **kwargs)
