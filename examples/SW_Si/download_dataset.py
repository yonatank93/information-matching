"""Run this script to download the dataset for the SW Si example."""

from pathlib import Path
from information_matching.utils import download_dataset


FILE_PATH = Path(__file__).resolve().parent

# Dictionary with the dataset information
dataset_info = {
    "dataset_path": FILE_PATH / "sw_si_training_dataset",
    "url": "https://raw.githubusercontent.com/yonatank93/information-matching/"
    "main/examples/SW_Si/sw_si_training_dataset.tar.gz",
    "tar_path": FILE_PATH / "sw_si_training_dataset.tar.gz",
}

# Download the dataset
download_dataset(dataset_info)
