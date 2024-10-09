"""Run this script to download the dataset for the SW MoS2 example."""

from pathlib import Path
import requests
import tarfile


FILE_PATH = Path(__file__).resolve().parent

# Check if the dataset is already downloaded and extracted
example_name = "SW_Si"
dataset_path = FILE_PATH / "sw_si_training_dataset"

if not dataset_path.exists():
    # Check if the tar.gz file is already downloaded
    tar_path = dataset_path.with_suffix(".tar.gz")

    if not tar_path.exists():
        # Prepare the url
        url = (
            "https://raw.githubusercontent.com/yonatank93/information-matching/main/examples/"
            f"{example_name}/{dataset_path.name}.tar.gz"
        )

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
    tf.extractall(FILE_PATH)
