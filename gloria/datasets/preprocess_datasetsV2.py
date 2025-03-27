"""
Preprocessing module for medical image datasets used in the GLoRIA framework.

This module provides data preprocessing functionality for various medical imaging datasets:
- CheXpert
- RSNA Pneumonia 
- SIIM Pneumothorax

GLoRIA: A Multimodal Global-Local Representation Learning Framework 
for Label-efficient Medical Image Recognition
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gloria.constants import (
    # Pneumonia constants
    PNEUMONIA_DATA_DIR,
    PNEUMONIA_IMG_DIR,
    PNEUMONIA_ORIGINAL_TRAIN_CSV,
    PNEUMONIA_TRAIN_CSV,
    PNEUMONIA_VALID_CSV,
    PNEUMONIA_TEST_CSV,
    
    # Pneumothorax constants
    PNEUMOTHORAX_DATA_DIR,
    PNEUMOTHORAX_IMG_DIR,
    PNEUMOTHORAX_ORIGINAL_TRAIN_CSV,
    PNEUMOTHORAX_TRAIN_CSV,
    PNEUMOTHORAX_VALID_CSV,
    PNEUMOTHORAX_TEST_CSV,
    
    # CheXpert constants
    CHEXPERT_ORIGINAL_TRAIN_CSV,
    CHEXPERT_MASTER_CSV,
    CHEXPERT_TRAIN_CSV,
    CHEXPERT_VALID_CSV,
    CHEXPERT_5x200,
    CHEXPERT_PATH_COL,
    CHEXPERT_VALID_NUM,
    CHEXPERT_COMPETITION_TASKS,
)


def preprocess_pneumonia_data(test_fraction: float = 0.15) -> None:
    """
    Preprocess the RSNA Pneumonia dataset and save train/validation/test splits.
    
    Args:
        test_fraction: Fraction of data to use for testing and validation
                       (each will be this percentage of the total data)
    
    Raises:
        FileNotFoundError: If the required dataset files cannot be found
    """
    try:
        df = pd.read_csv(PNEUMONIA_ORIGINAL_TRAIN_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Please make sure the RSNA Pneumonia dataset is stored at {PNEUMONIA_DATA_DIR}"
        )

    # Create bounding boxes
    df["bbox"] = df.apply(_create_pneumonia_bbox, axis=1)

    # Aggregate multiple boxes
    df = df[["patientId", "bbox"]]
    df = df.groupby("patientId").agg(list)
    df = df.reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # Create labels
    df["Target"] = df["bbox"].apply(lambda x: 0 if x is None else 1)

    # Add path to images
    df["Path"] = df["patientId"].apply(lambda x: PNEUMONIA_IMG_DIR / f"{x}.dcm")

    # Split data
    train_df, test_val_df = train_test_split(df, test_size=test_fraction * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    _print_split_statistics(train_df, valid_df, test_df, "Target")

    # Save processed data
    train_df.to_csv(PNEUMONIA_TRAIN_CSV)
    valid_df.to_csv(PNEUMONIA_VALID_CSV)
    test_df.to_csv(PNEUMONIA_TEST_CSV)


def _print_split_statistics(
        train_df: pd.DataFrame, 
        valid_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        label_column: str
    ) -> None:
    """
    Print statistics about dataset splits.
    
    Args:
        train_df: Training data DataFrame
        valid_df: Validation data DataFrame
        test_df: Test data DataFrame
        label_column: Name of the column containing labels
    """
    print(f"Number of train samples: {len(train_df)}")
    print(train_df[label_column].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df[label_column].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df[label_column].value_counts())



def _create_pneumonia_bbox(row: pd.Series) -> Union[int, List[int]]:
    """
    Create bounding box coordinates from a row in the Pneumonia dataset.
    
    Args:
        row: DataFrame row containing bounding box information
        
    Returns:
        Either 0 (no bounding box) or a list of [x1, y1, x2, y2] coordinates
    """
    if row["Target"] == 0:
        return 0
    else:
        x1 = row["x"]
        y1 = row["y"]
        x2 = x1 + row["width"]
        y2 = y1 + row["height"]
        return [x1, y1, x2, y2]


def preprocess_pneumothorax_data(test_fraction: float = 0.15) -> None:
    """
    Preprocess the SIIM Pneumothorax dataset and save train/validation/test splits.
    
    Args:
        test_fraction: Fraction of data to use for testing and validation
                       (each will be this percentage of the total data)
    """
    try:
        df = pd.read_csv(PNEUMOTHORAX_ORIGINAL_TRAIN_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Please make sure the SIIM Pneumothorax dataset is stored at {PNEUMOTHORAX_DATA_DIR}"
        )

    # Get image paths
    img_paths = _get_pneumothorax_image_paths()

    # No encoded pixels mean healthy
    df["Label"] = df.apply(
        lambda x: 0.0 if x[" EncodedPixels"] == " -1" else 1.0, axis=1
    )
    df["Path"] = df["ImageId"].apply(lambda x: img_paths.get(x, None))

    # Filter out any rows with missing image paths
    df = df.dropna(subset=["Path"])

    # Split data
    train_df, test_val_df = train_test_split(df, test_size=test_fraction * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    _print_split_statistics(train_df, valid_df, test_df, "Label")

    # Save processed data
    train_df.to_csv(PNEUMOTHORAX_TRAIN_CSV)
    valid_df.to_csv(PNEUMOTHORAX_VALID_CSV)
    test_df.to_csv(PNEUMOTHORAX_TEST_CSV)


def _get_pneumothorax_image_paths() -> Dict[str, str]:
    """Find all DICOM files in the Pneumothorax image directory."""
    img_paths = {}
    for subdir, _, files in tqdm(os.walk(PNEUMOTHORAX_IMG_DIR), desc="Finding image files"):
        for file in files:
            if file.endswith(".dcm"):
                # Remove .dcm extension
                file_id = file[:-4]
                img_paths[file_id] = os.path.join(subdir, file)
    return img_paths



def preprocess_chexpert_data() -> None:
    """
    Preprocess the CheXpert dataset and save train/validation/test splits.
    
    Also creates a special 5x200 subset for specific tasks.
    
    Raises:
        FileNotFoundError: If the required dataset files cannot be found
    """
    try:
        df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Please make sure the CheXpert dataset is stored at the location specified in constants"
        )

    # Create the 5x200 special subset
    df_200 = _preprocess_chexpert_5x200_data()
    
    # Remove the 5x200 samples from the main dataset
    df = df[~df[CHEXPERT_PATH_COL].isin(df_200[CHEXPERT_PATH_COL])]
    
    # Create validation set
    valid_ids = np.random.choice(len(df), size=CHEXPERT_VALID_NUM, replace=False)
    valid_df = df.iloc[valid_ids]
    train_df = df.drop(valid_ids, errors="ignore")

    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of valid samples: {len(valid_df)}")
    print(f"Number of chexpert5x200 samples: {len(df_200)}")

    # Save processed data
    train_df.to_csv(CHEXPERT_TRAIN_CSV)
    valid_df.to_csv(CHEXPERT_VALID_CSV)
    df_200.to_csv(CHEXPERT_5x200)


def _preprocess_chexpert_5x200_data() -> pd.DataFrame:
    """
    Create a special subset of the CheXpert dataset with 200 samples for each of 5 tasks.
    
    Returns:
        DataFrame containing the 5x200 samples
    """
    df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
    df = df.fillna(0)
    df = df[df["Frontal/Lateral"] == "Frontal"]

    df_master = pd.read_csv(CHEXPERT_MASTER_CSV)
    df_master = df_master[["Path", "Report Impression"]]

    task_dfs = []
    for i, _ in enumerate(CHEXPERT_COMPETITION_TASKS):
        # Create a one-hot index for the current task
        index = np.zeros(14)
        index[i] = 1
        
        # Filter rows where the selected task is positive and others are zero
        df_task = df[
            (df["Atelectasis"] == index[0])
            & (df["Cardiomegaly"] == index[1])
            & (df["Consolidation"] == index[2])
            & (df["Edema"] == index[3])
            & (df["Pleural Effusion"] == index[4])
            & (df["Enlarged Cardiomediastinum"] == index[5])
            & (df["Lung Lesion"] == index[7])
            & (df["Lung Opacity"] == index[8])
            & (df["Pneumonia"] == index[9])
            & (df["Pneumothorax"] == index[10])
            & (df["Pleural Other"] == index[11])
            & (df["Fracture"] == index[12])
            & (df["Support Devices"] == index[13])
        ]
        # Sample 200 images for this task
        df_task = df_task.sample(n=200, random_state=i)
        task_dfs.append(df_task)
    
    # Combine all task datasets
    df_200 = pd.concat(task_dfs)

    # Get reports
    df_200 = pd.merge(df_200, df_master, how="left", left_on="Path", right_on="Path")

    return df_200


# Define mapping of dataset names to their preprocessing functions
DATASET_PROCESSORS: Dict[str, Callable] = {
    "chexpert": preprocess_chexpert_data,
    "pneumonia": preprocess_pneumonia_data,
    "pneumothorax": preprocess_pneumothorax_data,
}


def available_datasets() -> List[str]:
    """
    Returns the names of available datasets.
    
    Returns:
        List of dataset names that can be preprocessed
    """
    return list(DATASET_PROCESSORS.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess medical imaging datasets for GLoRIA")
    parser.add_argument("--dataset", type=str, help="Dataset type, one of [chexpert, pneumonia, pneumothorax]", required=True)
    args = parser.parse_args()

    dataset = args.dataset.lower()

    if dataset in DATASET_PROCESSORS:
        DATASET_PROCESSORS[dataset]()
    else:
        raise ValueError(
            f"Dataset '{dataset}' not found; available datasets = {available_datasets()}"
        )