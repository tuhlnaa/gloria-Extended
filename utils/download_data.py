import kaggle
from pathlib import Path

def download_kaggle_data(identifier: str, destination_dir: str = './data', unzip: bool = True, is_competition: bool = False) -> str:
    """
    Downloads a Kaggle dataset or competition data using the Kaggle API and saves it to the specified directory.

    Parameters:
    - identifier (str): The Kaggle dataset or competition identifier. For datasets: 'username/dataset-name'. 
                        For competitions: competition name (e.g., 'titanic').
    - destination_dir (str): Directory where the data will be downloaded (default: './data').
    - unzip (bool): If True, the downloaded data will be unzipped (default: True).
    - is_competition (bool): Set to True if downloading a competition dataset (default: False).

    Returns:
    - str: Path to the downloaded or extracted folder.
    """
    # Ensure the destination directory exists
    destination_path = Path(destination_dir)
    destination_path.mkdir(parents=True, exist_ok=True)

    if is_competition:
        # Download competition data
        kaggle.api.competition_download_files(identifier, path=str(destination_path))
    else:
        # Download general dataset
        kaggle.api.dataset_download_files(identifier, path=str(destination_path), unzip=unzip)

    # Return the path where the data was saved or extracted
    if unzip:
        return str(destination_path)
    else:
        return str(destination_path / f"{identifier.split('/')[-1]}.zip")


if __name__ == "__main__":
    # General dataset identifier
    dataset_identifier = "ashery/chexpert"
    dataset_path = download_kaggle_data(dataset_identifier, destination_dir=r"./download")
    print(f"Dataset downloaded and saved to: {dataset_path}")