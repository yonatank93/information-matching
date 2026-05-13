import numdifftools as nd

from .fim_base import FIMBase


class FIM_nd(FIMBase):
    """A class to compute the Jacobian and the FIM of a model using numdifftools.

    Parameters
    ----------
    model: callable ``model(x, **kwargs)``
        A function that we will evaluate the derivative of.
    transform: TransformBase, optional
        A transformation class instance with ``transform(x)`` and
        ``inverse_transform(x)`` methods for transforming the parameters in the
        Jacobian/FIM calculation.
    kwargs: dict
        Additional keyword arguments for ``numdifftools.Jacobian``.
    """

    def __init__(self, model, transform=None, **kwargs):
        super().__init__(model, transform)
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
        params = self.transform.transform(x)
        return self.jac_func(params, *args, **kwargs)
