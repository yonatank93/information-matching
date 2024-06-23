"""This script contains utility functions and class:
* Set file and directory that creates all the necessary directories if not
  exists.
* Copy function to copy the indicator configurations.
"""


from pathlib import Path
import shutil

import numpy as np

PARENT_DIR = Path(__file__).absolute().parents[1]
EXAMPLES_DIR = PARENT_DIR / "examples"

# Tolerances in the calculation
eps = np.finfo(float).eps
tol = eps ** 0.5


def set_directory(path):
    """Set the directory and create it if it doesn't exist."""
    if not path.exists():
        path.mkdir(parents=True)

    return path


def set_file(filepath):
    """Set the path of a file and create the parent directories if they don't
    exist.
    """
    parent = filepath.parent
    if not parent.exists():
        parent.mkdir(parents=True)
    return filepath


def copy_configurations(configs_weights_dict, source_dir, target_dir):
    """Copy the configuration files deduced from the weights dictionary to
    the target directory.

    Parameters
    ----------
    configs_weights_dict: dict
        A dictionary containing the weights of the reduced configurations.
    source_dif: str or pathlib.Path
        Source directory that contains the original configuration.
    target_dir: str or pathlib.Path
        Target directory to copy the files into.
    kinf: str "energy" or "forces"
    """
    for config_id in configs_weights_dict:
        path = Path(source_dir) / config_id
        shutil.copy(path, target_dir)
