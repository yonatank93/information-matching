import math
import numpy as np

# Parameters for Production
FIRE_LOG = "fire.log"
FIRE_MAX_STEPS_VFERV = 50  #
FIRE_MAX_STEPS_VFM = 500  #
FIRE_UNCERT_STEPS = 20
FIRE_TOL = 1e-3  # absolute
FMIN_FTOL_VFERV = 1e-6  #
FMIN_FTOL_VFM = 1e-10  #
FMIN_XTOL = 1e-10  # relative
VFE_TOL = 1e-5  # absolute
MAX_LOOPS = 20
CELL_SIZE_MIN = 3
CELL_SIZE_MAX = 5
COLLAPSE_CRITERIA_VOLUME = 0.1
COLLAPSE_CRITERIA_ENERGY = 0.1
DYNAMIC_CELL_SIZE = True  # Increase Cell Size According to lattice structure
MDMIN_TOL = 1e-3  # absolute
MDMIN_MAX_STEPS = 200
NEB_POINTS = 20
UNCERT_STEPS = 20
EPS_VFERV = 1e-3  #
EPS_VFM = 1e-10  #

# Parameters for Debugging
# FIRE_MAX_STEPS = 200
# FIRE_TOL = 1e-3 # absolute #This is for VFERV. For VFM, 1e-2
# FMIN_FTOL = 1e-3 # relative
# FMIN_XTOL = 1e-5 # relative

# Extrapolation Parameters for VFERV
FITS_CNT_VFERV = [2, 3, 3, 3, 3]  # Number of data points used for each fitting
FITS_ORDERS_VFERV = [
    [0, 3],
    [0, 3],
    [0, 3, 4],
    [0, 3, 5],
    [0, 3, 6],
]  # Number of orders included in each fitting
# Fit Results Used (Corresponding to the above)

# Extrapolation Parameters for VFERV
FITS_CNT_VFM = [2, 3]  # Number of data points used for each fitting
FITS_ORDERS_VFM = [
    [0, 3],
    [0, 3],
]  # Number of orders included in each fitting

FITS_VFE_VALUE = 0  # Vacancy Formation Energy
FITS_VFE_UNCERT_VFERV = [1, 2]  #
FITS_VFE_UNCERT_VFM = [1]  #
FITS_VRV_VALUE = 0  # Vacancy Relaxation Volume
FITS_VRV_UNCERT = [1, 2]
FITS_VME_VALUE = 0  # Vacancy Migration Energy
FITS_VME_UNCERT = [1]

SQRT3 = math.sqrt(3.0)
PERFECT_CA = math.sqrt(8.0 / 3.0)
HCP_CUBIC_CELL = np.array([1.0, SQRT3, 1.0])
HCP_CUBIC_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5 * SQRT3, 0.0],
        [0.5, 0.5 / 3.0 * SQRT3, 0.5],
        [0.0, (0.5 + 0.5 / 3.0) * SQRT3, 0.5],
    ]
)
HCP_CUBIC_NATOMS = 4

# Strings for Output
KEY_SOURCE_VALUE = "source-value"
KEY_SOURCE_UNIT = "source-unit"
KEY_SOURCE_UNCERT = "source-std-uncert-value"
UNIT_ENERGY = "eV"
UNIT_LENGTH = "angstrom"
UNIT_ANGLE = "degree"
UNIT_PRESSURE = "GPa"
UNIT_TEMPERATURE = "K"
UNIT_VOLUME = UNIT_LENGTH + "^3"
SPACE_GROUPS = {
    "fcc": "Fm-3m",
    "bcc": "Im-3m",
    "sc": "Pm-3m",
    "diamond": "Fd-3m",
    "hcp": "P63/mmc",
}
WYCKOFF_CODES = {
    "fcc": ["4a"],
    "bcc": ["2a"],
    "sc": ["1a"],
    "diamond": ["8a"],
    "hcp": ["2d"],
}
WYCKOFF_SITES = {
    "fcc": [[0.0, 0.0, 0.0]],
    "bcc": [[0.0, 0.0, 0.0]],
    "sc": [[0.0, 0.0, 0.0]],
    "diamond": [[0.0, 0.0, 0.0]],
    "hcp": [[2.0 / 3.0, 1.0 / 3.0, 0.25]],
}
