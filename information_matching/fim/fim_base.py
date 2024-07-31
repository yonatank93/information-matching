from abc import abstractmethod


def default_transform(x):
    return x


class FIMBase:
    """Abstract base class for the FIM modules."""

    def __init__(self, model, transform=None, inverse_transform=None, **kwargs):
        """Instantiate FIM class

        Parameters
        ----------
        model: callable ``model(x, **kwargs)``
            A function that we will evaluate the derivative of.
        transform: callable ``transform(x)``
            A function to transform the parameters from the model parameterization to the
            parameterization that we want to use to differentiate the model.
        inverse_transform: callable ``inverse_transform(x)``
            This is the inverse of transformation function above.
        kwargs: dict
            Additional keyword arguments for the function to evaluate the Jacobian, e.g.
            the step size.

        """
        self.model = model

        # Get the parameter transformation
        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform

        if inverse_transform is None:
            self.inverse_transform = default_transform
        else:
            self.inverse_transform = inverse_transform

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def FIM(self, x, *args, **kwargs):
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
        Jac = self.Jacobian(x, *args, **kwargs)
        return Jac.T @ Jac

    @abstractmethod
    def __call__(self, x, *args, **kwargs):
        return self.FIM(x, *args, **kwargs)

    def _model_wrapper(self, x, *args, **kwargs):
        """This is the function that we feed into the function that computes the Jacobian.

        Parameters
        ----------
        x: np.ndarray (nparams,)
            Parameter values to evaluate the derivative. This should be the transformed
            parameters.
        """
        # Transform the parameters
        params_orig = self.inverse_transform(x)
        # Evaluate model
        return self.model(params_orig, *args, **kwargs)
