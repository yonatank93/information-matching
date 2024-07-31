import numdifftools as nd

from .fim_base import FIMBase


class FIM_nd(FIMBase):
    """A class to compute the Jacobian and the FIM of a model using numdifftools.

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
        super().__init__(model, transform, inverse_transform)
        self.jac_func = nd.Jacobian(self._model_wrapper, method="forward", **kwargs)

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
        params = self.transform(x)
        return self.jac_func(params, *args, **kwargs)
