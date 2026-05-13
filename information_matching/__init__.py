try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0+unknown"

from . import convex_optimization
from . import leastsq
from . import fim
from . import precondition
from . import transform
from . import utils

from .convex_optimization import ConvexOpt

__all__ = ["ConvexOpt"]
