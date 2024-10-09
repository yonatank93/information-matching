"""Run this script to download the dataset for the underwater acoustic example."""

from pathlib import Path
from information_matching.utils import download_dataset


FILE_PATH = Path(__file__).resolve().parent
DATA_PATH = FILE_PATH / "data"


# Dictionary with the dataset information
dataset_dict = {
    "transmission_loss": {
        "dataset_path": DATA_PATH / "transmission_loss",
        "url": "https://media.githubusercontent.com/media/yonatank93/information-matching/"
        "main/examples/ORCA/data/transmission_loss.tar.gz",
        "tar_path": DATA_PATH / "transmission_loss.tar.gz",
    },
    "fim_environment": {
        "dataset_path": DATA_PATH / "FIMs" / "environment",
        "url": "https://raw.githubusercontent.com/yonatank93/information-matching/"
        "main/examples/ORCA/data/FIMs/fim_environment.tar.gz",
        "tar_path": DATA_PATH / "FIMs" / "fim_environment.tar.gz",
    },
    "fim_source": {
        "dataset_path": DATA_PATH / "FIMs" / "source",
        "url": "https://raw.githubusercontent.com/yonatank93/information-matching/"
        "main/examples/ORCA/data/FIMs/fim_source.tar.gz",
        "tar_path": DATA_PATH / "FIMs" / "fim_source.tar.gz",
    },
}

# Iterate over this dictionary and download the datasets
for dataset_info in dataset_dict.values():
    download_dataset(dataset_info)
