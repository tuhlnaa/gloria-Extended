"""
GLoRIA: A Multimodal Global-Local Representation Learning Framework
for Label-efficient Medical Image Recognition

Dataset implementations for Pneumonia and Pneumothorax datasets.
"""
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import torchvision.transforms as T

from albumentations.pytorch import ToTensorV2
from albumentations import Normalize, Resize, Compose, Affine
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

from data.chexpert_dataset import ImageBaseDataset, build_transformation


@dataclass
class PneumoniaConfig:
    """Constants and configuration for the Pneumonia dataset."""
    PATH_COL: str = "patientId"
    TARGET_COL: str = "Target"


class PneumoniaImageDataset(ImageBaseDataset):
    """Dataset class for the RSNA Pneumonia dataset."""
    
    def __init__(
        self,
        config: object,
        split: str = "train",
        transform: Optional[object] = None,
    ):
        """
        Initialize the Pneumonia dataset.
        
        Args:
            config: Configuration object
            split: Data split ('train' or 'valid')
            transform: Torchvision transforms to apply
        """
        super().__init__(config, split, transform)
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.pneumonia_config = PneumoniaConfig()
        
        # Load the CSV file and handle train/validation split
        self.df = self._load_and_split_data(config, split)
        
        # If in detection phase, only keep positive samples
        if config.phase == "detection":
            self.df = self.df[self.df[self.pneumonia_config.TARGET_COL] == 1]
        
        # Apply data fraction sampling if specified in config
        if hasattr(config.dataset, 'frac') and config.dataset.frac != 1 and split == "train":
            orig_size = len(self.df)
            self.df = self.df.sample(frac=config.dataset.frac, random_state=42)
            print(f"Applied dataset fraction {config.dataset.frac}: reduced from {orig_size} to {len(self.df)} samples")


    def _load_and_split_data(self, config: object, split: str) -> pd.DataFrame:
        """Load data from CSV and split into train/valid sets since they're now in one file."""
        # Load the combined CSV file
        df = pd.read_csv(self.data_dir / config.train_csv)
        
        # Split the data based on a validation ratio (e.g., 80% train, 20% valid)
        if not hasattr(config, 'valid_ratio'):
            valid_ratio = 0.3  # Default validation ratio if not specified
        else:
            valid_ratio = config.valid_ratio
            
        # Create a deterministic split based on index
        n_samples = len(df)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        split_idx = int(n_samples * (1 - valid_ratio))
        
        if split == "train":
            split_indices = indices[:split_idx]
        else:  # valid
            split_indices = indices[split_idx:]
            
        return df.iloc[split_indices].reset_index(drop=True)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and label at the specified index."""
        row = self.df.iloc[index]
        
        # Get and process image
        img_path = row[self.pneumonia_config.PATH_COL]
        img_path = str(self.data_dir / "stage_2_train_images" / img_path)
        img = self.read_from_dicom(img_path)
        
        # Get label
        label = float(row[self.pneumonia_config.TARGET_COL])
        label = torch.tensor([label])
        
        return img, label


    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)
        
        
    def read_from_dicom(self, img_path: Union[str, Path]) -> Image.Image:
        """Read and preprocess a DICOM image."""
        dcm = pydicom.dcmread(img_path + ".dcm")
        x = dcm.pixel_array
        
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)
            
        # Resize and transform the image
        x = self._resize_img(x, self.config.dataset.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img


def get_pneumonia_dataloader(
        config,
        split: str = "train",
        transform: Optional[T.Compose] = None, 
    ) -> DataLoader:
    """
    Create a DataLoader for the pneumonia medical imaging dataset.
    
    Args:
        config: Configuration object containing dataset parameters
            Expected structure:
            - config.data_dir: Base directory containing pneumonia data
            - config.{split}_csv: Path to the specific split CSV file
            - config.model.batch_size: Batch size
            - config.dataset.num_workers: Number of workers
            - config.dataset.fraction: Optional fraction of data to use (for training)
        split: Dataset split ('train', 'valid', or 'test')
        transform: Optional custom transformation pipeline; if None, uses transforms built from config
        
    Returns:
        DataLoader for the pneumonia dataset
    """
    # Create transformation if not provided
    if transform is None:
        transform = build_transformation(config, split)
    
    # Create dataset
    dataset = PneumoniaImageDataset(
        config=config,
        split=split,
        transform=transform,
    )
    
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! No images found. Please check the paths and file formats.")
    
    # Create dataloader with parameters from config
    data_loader = DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        shuffle=(split == "train"),
        num_workers=config.dataset.num_workers,
        pin_memory=getattr(config.model, "pin_memory", False),
        drop_last=getattr(config.model, "drop_last", False) if split == "train" else True
    )
    print(f"DataLoader created successfully with {len(data_loader)} batches")
    
    return data_loader, dataset