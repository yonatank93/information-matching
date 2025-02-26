"""Collection of parameter transformations."""

from abc import ABC, abstractmethod
import copy
import numpy as np


def func_wrapper(func, transform):
    """Wrapper function that evaluates the model in the transformed parameter space.
    Basically, this function just applies inverse transformation and then calls the
    model function.

    An example use case for this function wrapper is if we want to compute the FIM by
    taking derivative in the transformed space. This function can help with the parameter
    conversion, and the returned function can be directly used with the modules in
    `information_matching.fim`.

    Parameters
    ----------
    func: callable
        The model function to evaluate.
    transform: TransformBase
        The transformation class instance to use.

    Returns
    -------
    func_orig: callable
        The model function that evaluates the model in the transformed parameter space.

    Notes
    -----
    The parameter input to the returned function must be in the transformed parameter
    space.
    """

    def func_orig(x, *args, **kwargs):
        xorig = transform.inverse_transform(x)
        return func(xorig, *args, **kwargs)

    return func_orig


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
    """An affine transformation class :math:`y = A(x-x0) + b`. This form of
    transformation separates shifts in input space and output space.

    Notes
    -----
    The class can also be used to subsample the parameters, i.e, if you want to vary only
    a subset of parameters and keep the rest fixed. In this case, we can set the
    transformation matrix A to be a diagonal matrix with 1's and 0's. The shift in the
    input space x0 can be used in the inverse transformation to add the fixed parameters
    back.

    Parameters
    ----------
    A: float or np.ndarray
        Transformation matrix. If a float is given, the transformation is a scalar
        multiplication. If a 1d array is given, the transformation is a matrix
        multiplication.

    x0: float or np.ndarray
        Shift in the input space. If a float is given, the shift is a scalar addition.
        If a 1d array is given, the shift is a vector addition.

    b: float or np.ndarray
        Shift in the output space. If a float is given, the shift is a scalar addition.
        If a 1d array is given, the shift is a vector addition.

    Ainv: None or float or np.ndarray
        Inverse of A. If None is given, inverse will be computed by taking reciprocal
        of a if a is a float, otherwise using `np.linalg.pinv(a)`.
    """

    def __init__(self, a=1.0, x0=0.0, b=0.0, ainv=None):
        super().__init__(a=a, b=b, ainv=ainv)
        self.a = a
        if isinstance(self.a, (float, int)):
            # Use * multiplication operator
            self._mult_fn = self._scalar_mult
        elif isinstance(self.a, np.ndarray):
            # Use @ multiplication operator
            self._mult_fn = self._matrix_mult

        self.x0 = x0
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
        return self._mult_fn(self.a, (x - self.x0)) + self.b

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
        return self._mult_fn(self.ainv, (x - self.b)) + self.x0

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


class AtanhTransform(TransformBase):
    """An inverse hyperbolic tangent transformation class
    :math:`y = atanh((x - b) / a)`.

    This transformation is useful when the parameter values are bounded between two
    values, where the midpoint is at :math:`b` and the half-range is :math:`a`.

    Parameters
    ----------
    low: float
        Lower bound of the parameter values.
    high: float
        Upper bound of the parameter values.
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.a = (high - low) / 2  # Half-range
        self.b = (high + low) / 2  # Midpoint
        super().__init__(low=self.low, high=self.high)

    def transform(self, x):
        """Perform parameter transformation to the transformed space.

        Parameters
        ----------
        x: np.ndarray
            Parameter values to transform.

        Returns
        -------
        np.ndarray
            Parameter values in arctanh transformed space.
        """
        return np.arctanh((x - self.b) / self.a)

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
        return self.a * np.tanh(x) + self.b


class SplitTransform(TransformBase):
    """A class that combines multiple transformations by applying each transformation
    to a subset of the parameters.

    Parameters
    ----------
    transform_list: list
        List of transformation class instances to combine.

    param_idx: list
        A nested list of indices to specify which parameters to apply the
        transformations. For example, the first transformation is applied to the list
        of parameters specified by `param_idx[0]`, the second transformation is applied
        to the list of parameters specified by `param_idx[1]`, and so on.
    """

    def __init__(self, transform_list, param_idx):
        self.transform_list = transform_list
        self.param_idx = param_idx
        super().__init__(transform_list=transform_list, param_idx=param_idx)
        # Stringify keyword arguments for JSON serialization
        transform_str = []
        for inst in transform_list:
            for key in avail_transform:
                if isinstance(inst, avail_transform[key]):
                    name = key
                    break
            inst_kwargs = inst.jsonable_kwargs
            transform_str.append({"transform_type": name, "transform_args": inst_kwargs})
        self._kwargs = {
            "transform_list": transform_str,
            "param_idx": [list(p) for p in param_idx],
        }

    def transform(self, x):
        """Perform parameter transformation to the transformed space.

        Parameters
        ----------
        x: np.ndarray
            Parameter values to transform.

        Returns
        -------
        np.ndarray
            Parameter values in transformed space.
        """
        params = copy.deepcopy(x)
        for transform, idx in zip(self.transform_list, self.param_idx):
            params[idx] = transform.transform(x[idx])
        return params

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
        params = copy.deepcopy(x)
        for transform, idx in zip(self.transform_list, self.param_idx):
            params[idx] = transform.inverse_transform(x[idx])
        return params


class ComposedTransform(TransformBase):
    """A class that combines multiple transformations by composing the transformations.
    That is, given a list of transformations, the first transformation in the list is
    applied to the input parameters, then the second transformation is applied to the
    output of the first transformation, and so on.

    Notes
    -----
    The order in which the transformation is applied follows the same order as the a
    processing pipeline. That is, the first transformation is applied to the input
    parameters, then the second transformation is applied to the output of the first
    transformation, and so on.

    Parameters
    ----------
    transform_list: list
        List of transformation class instances to combine.
    """

    def __init__(self, transform_list):
        self.transform_list = transform_list
        super().__init__(transform_list=transform_list)
        # Stringify keyword arguments for JSON serialization
        self._kwargs = {"transform_list": []}
        for inst in transform_list:
            for key in avail_transform:
                if isinstance(inst, avail_transform[key]):
                    name = key
                    break
            inst_kwargs = inst.jsonable_kwargs
            self._kwargs["transform_list"].append(
                {"transform_type": name, "transform_args": inst_kwargs}
            )

    def transform(self, x):
        """Perform parameter transformation to the transformed space.

        Parameters
        ----------
        x: np.ndarray
            Parameter values to transform.

        Returns
        -------
        np.ndarray
            Parameter values in transformed space.
        """
        params = copy.deepcopy(x)
        for transform in self.transform_list:
            params = transform.transform(params)
        return params

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
        params = copy.deepcopy(x)
        # We need to invert the order of the transformation for the inverse
        # transformation
        for transform in self.transform_list[::-1]:
            params = transform.inverse_transform(params)
        return params


avail_transform = {
    "AffineTransform": AffineTransform,
    "LogTransform": LogTransform,
    "AtanhTransform": AtanhTransform,
    "SplitTransform": SplitTransform,
    "ComposedTransform": ComposedTransform,
}


def transform_builder(transform_type, transform_args):
    """Construct a transformation class instance from the given string type and a
    dictionary arguments.

    Parameters
    ----------
    transform_type: str
        Type of transformation to construct. Check the available transformations by
        looking at the keys of the `avail_transform` dictionary

    transform_args: dict
        Keyword arguments to pass to the transformation class.

    Returns
    -------
    transform: TransformBase
        Transformation class instance.
    """
    if transform_type not in avail_transform:
        raise ValueError(f"Unknown transformation type: {transform_type}")
    elif transform_type in ["SplitTransform"]:
        return avail_transform[transform_type](
            transform_list=[
                transform_builder(inst["transform_type"], inst["transform_args"])
                for inst in transform_args["transform_list"]
            ],
            param_idx=transform_args["param_idx"],
        )
    elif transform_type == "ComposedTransform":
        return avail_transform[transform_type](
            transform_list=[
                transform_builder(inst["transform_type"], inst["transform_args"])
                for inst in transform_args["transform_list"]
            ]
        )
    else:
        return avail_transform[transform_type](**transform_args)
