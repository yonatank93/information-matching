"""This script contains utility functions and class:
* Set file and directory that creates all the necessary directories if not
  exists.
* Copy function to copy the indicator configurations.
"""


from pathlib import Path
import shutil
import requests
import tarfile

import numpy as np

PARENT_DIR = Path(__file__).absolute().parents[1]
EXAMPLES_DIR = PARENT_DIR / "examples"

# Tolerances in the calculation
eps = np.finfo(float).eps
tol = eps**0.5


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


def download_dataset(dataset_info):
    """A function to download and extract the dataset.

    Parameters
    ----------
    dataset_info : dict
        A dictionary with the following keys:
        - dataset_path: Path
            The path to the directory where the dataset will be extracted.
        - url: str
            The URL from which the dataset will be downloaded.
        - tar_path: Path
            The target path to the tar.gz file that will be downloaded.
    """
    # Get the values from the dictionary
    dataset_path = Path(dataset_info["dataset_path"])
    url = dataset_info["url"]
    tar_path = Path(dataset_info["tar_path"])

    # Check if the dataset is already downloaded and extracted
    if not dataset_path.exists():
        # Check if the tar.gz file is already downloaded
        if not tar_path.exists():
            # Download the dataset
            print(f"Downloading dataset from {url}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(tar_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        # Extract the dataset
        print(f"Extracting dataset to {dataset_path}")
        tf = tarfile.open(tar_path, "r:gz")
        tf.extractall(dataset_path.parent)
        tf.close()
    else:
        print(f"Dataset already exists at {dataset_path}")
