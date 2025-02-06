"""This script contains utility functions and class:
* Set file and directory that creates all the necessary directories if not
  exists.
* Copy function to copy the indicator configurations.
* Download function to download the dataset used in the examples.
"""

from pathlib import Path
import shutil
import wget
import tarfile

REPO_DIR = Path(__file__).absolute().parents[1]
EXAMPLES_DIR = REPO_DIR / "examples"
DATASET_DIR = EXAMPLES_DIR / "dataset"

# Information that will be used to extract and place the dataset in the correct folder
dataset_url = "https://figshare.com/ndownloader/files/49910310"
dataset_info = {
    # Underwater acoustic ORCA dataset
    "transmission_loss": {
        "tarfile": DATASET_DIR / "transmission_loss.tar.gz",
        "target": EXAMPLES_DIR / "ORCA/data/transmission_loss",
    },
    "fim_environment": {
        "tarfile": DATASET_DIR / "fim_environment.tar.gz",
        "target": EXAMPLES_DIR / "ORCA/data/FIMs/environment",
    },
    "fim_source": {
        "tarfile": DATASET_DIR / "fim_source.tar.gz",
        "target": EXAMPLES_DIR / "ORCA/data/FIMs/source",
    },
    # Material science SW potentials dataset
    "sw_si_training_dataset": {
        "tarfile": DATASET_DIR / "sw_si_training_dataset.tar.gz",
        "target": EXAMPLES_DIR / "SW_Si/sw_si_training_dataset",
    },
    "sw_mos2_training_dataset": {
        "tarfile": DATASET_DIR / "sw_mos2_training_dataset.tar.gz",
        "target": EXAMPLES_DIR / "SW_MoS2/sw_mos2_training_dataset",
    },
}
avail_dataset = list(dataset_info)  # Available dataset names


def set_directory(path):
    """Set the directory and create it if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_file(filepath):
    """Set the path of a file and create the parent directories if they don't
    exist.
    """
    parent = filepath.parent
    parent.mkdir(parents=True, exist_ok=True)
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


def download_dataset(dataset_name):
    """A function to download and extract the dataset.

    Parameters
    ----------
    dataset_name : str or list or "all"
        The name of the dataset to download. See ``avail_dataset`` for available datasets.
    """
    if isinstance(dataset_name, str):
        if dataset_name == "all":
            dataset_name = avail_dataset
        else:
            assert (
                dataset_name in avail_dataset
            ), f"Available datasets are {avail_dataset}"
            dataset_name = [dataset_name]
    for name in dataset_name:
        _download_one_dataset(name)


def _download_one_dataset(dataset_name):
    """Download one dataset."""
    # First, check if the specific dataset is already extracted
    data_info_dict = dataset_info[dataset_name]
    if not data_info_dict["target"].exists():
        # If not, check if the entire dataset is already downloaded and extracted
        if not DATASET_DIR.exists():
            # Check if the tar.gz file is already downloaded
            dataset_tarfile = (
                EXAMPLES_DIR / "information_matching_examples_dataset.tar.gz"
            )
            if not dataset_tarfile.exists():
                # Download the dataset
                print(f"Downloading dataset from {dataset_url}")
                wget.download(dataset_url, str(dataset_tarfile))

            # Extract the dataset
            print(f"Extracting dataset to {DATASET_DIR}")
            tf = tarfile.open(dataset_tarfile, "r:gz")
            tf.extractall(DATASET_DIR.parent)
            tf.close()

        # Extract the dataset
        print(f"Extracting dataset to {data_info_dict['target']}")
        tf = tarfile.open(data_info_dict["tarfile"], "r:gz")
        tf.extractall(data_info_dict["target"].parent)
        tf.close()

    print(f"{dataset_name} dataset is ready at {data_info_dict['target']}")
