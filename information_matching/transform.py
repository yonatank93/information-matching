"""Collection of parameter transformations."""

from abc import ABC, abstractmethod
import numpy as np


class TransformBase(ABC):
    """Abstract base class for parameter transformations."""

    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs

    @abstractmethod
    def transform(self, x):
        """Transform parameters from the original parameterization to the transformed
        parameterization to work with.

        Parameters
        ----------
        x: np.ndarray
            Parameter values in the original parameterization.

        Returns
        -------
        np.ndarray
            Parameter values in the transformed parameterization.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, x):
        """Transform parameters from the transformed parameterization to the original
        parameterization to work with.

        Parameters
        ----------
        x: np.ndarray
            Parameter values in the transformed parameterization.

        Returns
        -------
        np.ndarray
            Parameter values in the original parameterization.
        """
        raise NotImplementedError

    @property
    def jsonable_kwargs(self):
        """Convert the keyword arguments used when initializing the class to a JSON
        serializable dictionary.

        This is mainly used if other code need to save the transformation information
        as a metadata to a JSON file.
        """
        kwargs = self._kwargs.copy()
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray):
                kwargs[key] = val.tolist()
        return kwargs

    def __call__(self, x):
        return self.transform(x)


class AffineTransform(TransformBase):
    """An affine transformation class :math:`y = A.x + b`.

    Parameters
    ----------
    A: float or np.ndarray
        2d array or a float.

    b: float or np.ndarray
        1d array or a float.

    Ainv: None or float or np.ndarray
        Inverse of A. If None is given, inverse will be computed by taking reciprocal
        of a if a is a float, otherwise using `np.linalg.pinv(a)`.
    """

    def __init__(self, a=1.0, b=0.0, ainv=None):
        super().__init__(a=a, b=b, ainv=ainv)
        self.a = a
        if isinstance(self.a, (float, int)):
            # Use * multiplication operator
            self._mult_fn = self._scalar_mult
        elif isinstance(self.a, np.ndarray):
            # Use @ multiplication operator
            self._mult_fn = self._matrix_mult

        self.b = b

        if ainv is None:
            if isinstance(a, (float, int)):
                self.ainv = 1 / float(a)
            elif isinstance(a, np.ndarray):
                self.ainv = np.linalg.pinv(a)
        else:
            self.ainv = ainv

        super().__init__(a=self.a, b=self.b, ainv=self.ainv)

    def transform(self, x):
        """Perform parameter transformation to the transformed space.

        Parameters
        ----------
        x: np.ndarray
            Parameter values to transform.

        Returns
        -------
        np.ndarray
            Parameter values in affine transformed space.
        """
        return self._mult_fn(self.a, x) + self.b

    def inverse_transform(self, x):
        """Invert the transformation back to the original space.

        Parameters
        ----------
        x: np.ndarray
            Parameter values to inverse transform.

        Returns
        -------
        np.ndarray
            Parameter values in the original parameterization.
        """
        return self._mult_fn(self.ainv, (x - self.b))

    @staticmethod
    def _scalar_mult(a, b):
        """Multiplication function when the arguments are scalar."""
        return a * b

    @staticmethod
    def _matrix_mult(a, b):
        """Multiplication function when the arguments are vectors or matrices."""
        return a @ b


class LogTransform(TransformBase):
    """An logarithmic transformation base e class :math:`y = log(|x|)`.

    This transformation preserves the sign, and the sign needs to be specified as an
    input argument.

    Parameters
    ----------
    sign: np.ndarray
        Sign of the parameter values, which is used to invert the transformation. Make
        sure that the element values are either 1 or -1 only.
    """

    def __init__(self, sign):
        self.sign = sign
        super().__init__(sign=self.sign)

    def transform(self, x):
        """Perform parameter transformation to the transformed space.

        Parameters
        ----------
        x: np.ndarray
            Parameter values to transform.

        Returns
        -------
        np.ndarray
            Parameter values in logarithmically transformed space.
        """
        return np.log(np.abs(x))

    def inverse_transform(self, x):
        """Invert the transformation back to the original space.

        Parameters
        ----------
        x: np.ndarray
            Parameter values to inverse transform.

        Returns
        -------
        np.ndarray
            Parameter values in the original parameterization.
        """
        return self.sign * np.exp(x)


avail_transform = {"affine": AffineTransform, "log": LogTransform}
