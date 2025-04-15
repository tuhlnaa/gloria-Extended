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
    PATH_COL: str = "Path"
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
        
        self.pneumonia_config = PneumoniaConfig()
        
        # Load the CSV file and handle train/validation split
        self.df = self._load_and_split_data(config, split)
        
        # If in detection phase, only keep positive samples
        if config.phase == "detection":
            self.df = self.df[self.df[self.pneumonia_config.TARGET_COL] == 1]
        
        # Apply data fraction sampling if specified in config
        if hasattr(config.data, 'frac') and config.dataset.frac != 1 and split == "train":
            orig_size = len(self.df)
            self.df = self.df.sample(frac=config.dataset.frac, random_state=42)
            print(f"Applied dataset fraction {config.dataset.frac}: reduced from {orig_size} to {len(self.df)} samples")

    def _load_and_split_data(self, config: object, split: str) -> pd.DataFrame:
        """
        Load data from CSV and split into train/valid sets since they're now in one file.
        
        Args:
            config: Configuration object
            split: Data split ('train' or 'valid')
            
        Returns:
            DataFrame containing the appropriate split
        """
        # Load the combined CSV file
        df = pd.read_csv(config.train_csv)
        
        # Split the data based on a validation ratio (e.g., 80% train, 20% valid)
        if not hasattr(config, 'valid_ratio'):
            valid_ratio = 0.2  # Default validation ratio if not specified
        else:
            valid_ratio = config.valid_ratio
            
        # Create a deterministic split based on index
        n_samples = len(df)
        indices = np.arange(n_samples)
        np.random.seed(42)  # For reproducibility
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
        dcm = pydicom.read_file(img_path)
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


class PneumothoraxImageDataset(ImageBaseDataset):
    """Dataset class for the SIIM-ACR Pneumothorax dataset."""
    
    def __init__(
        self,
        config: object,
        split: str = "train",
        transform: Optional[object] = None,
    ):
        """Initialize the Pneumothorax dataset."""
        super().__init__(config, split, transform)

        self.config = config
            
        # Set up transformations for segmentation if needed
        if config.phase == "segmentation":
            transform = None
            self.seg_transform = self.get_transforms()
        else:
            self.transform = transform
            
        # Load the CSV file based on split
        self.df = self._load_data(config, split)

        # Process class labels (positive/negative samples)
        self.df["class"] = self.df["has_pneumo"].astype(bool)
        
        # Handle segmentation case - balance positive and negative samples
        if config.phase == "segmentation" and split == "train":
            self._balance_segmentation_samples()
            
        # Apply data fraction sampling if specified in config
        if hasattr(config.dataset, 'fraction') and config.dataset.fraction != 1 and split == "train":
            ids = self.df["ImageId"].unique()
            n_samples = int(len(ids) * config.dataset.fraction)
            series_selected = np.random.choice(ids, size=n_samples, replace=False)
            self.df = self.df[self.df["ImageId"].isin(series_selected)]
            
        # Create a list of unique image IDs
        self.imgids = self.df["ImageId"].unique().tolist()


    def _load_data(self, config: object, split: str) -> pd.DataFrame:
        """Load data from the appropriate CSV file based on split."""
        if split == "train":
            csv_path = Path(config.data_dir) / config.train_csv
        else:  # valid
            csv_path = Path(config.data_dir) / config.valid_csv
            
        df = pd.read_csv(csv_path)
        
        return df


    def _balance_segmentation_samples(self) -> None:
        """Balance positive and negative samples for segmentation tasks."""
        self.df_neg = self.df[self.df["class"] == False]
        self.df_pos = self.df[self.df["class"] == True]
        
        # Get count of unique positive samples
        n_pos = self.df_pos["ImageId"].nunique()
        
        # Select equal number of negative samples
        neg_series = self.df_neg["ImageId"].unique()

        neg_series_selected = np.random.choice(
            neg_series, size=n_pos, replace=False
        )
        
        # Filter negative samples and combine with positive samples
        self.df_neg = self.df_neg[
            self.df_neg["ImageId"].isin(neg_series_selected)
        ]
        self.df = pd.concat([self.df_pos, self.df_neg])


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and label/mask at the specified index."""
        imgid = self.imgids[index]
        imgid_df = self.df.groupby("ImageId").get_group(imgid)
        
        # Read the image
        filename = imgid_df.iloc[0]["new_filename"]
        img_path = Path(self.config.data_dir) / "png_images" / filename
        img = (self.read_image(img_path) / 255).astype(np.float32)
        
        # Handle segmentation or classification based on phase
        if self.config.phase == "segmentation":
            # For segmentation, load the corresponding mask
            mask_filename = filename  # Assuming mask has same name pattern
            mask_path = Path(self.config.data_dir) / "png_masks" / mask_filename
            mask = (self.read_mask(mask_path) / 255).astype(np.float32)

            # Apply transformations
            augmented = self.seg_transform(image=img, mask=mask)
            img = augmented["image"]
            label = augmented["mask"].squeeze()
        else:
            # Get classification label
            label = imgid_df.iloc[0]["has_pneumo"]
            label = torch.tensor([label])
            
            # Apply transformations for classification
            if self.transform:
                img = self.transform(img)
            
        return img, label


    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.imgids)
        

    def read_image(self, img_path: Union[str, Path]) -> np.ndarray:
        """Read and preprocess a PNG image."""
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        

    def read_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """Read mask image."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # Normalize to binary
        return mask
        

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        """Get transformations pipeline for image augmentation."""
        list_transforms = []

        # Add augmentations for training
        if self.split == "train":
            list_transforms.extend([
                Affine(
                    rotate=10,            # Rotate limit of 10 degrees
                    scale=0.1,            # Scale limit of 0.1
                    translate_percent=0,  # No translation
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                )
            ])

        # Apply normalization if specified
        if hasattr(self.config.transforms, 'norm') and self.config.transforms.norm is not None:
            if self.config.transforms.norm == "imagenet":
                list_transforms.append(Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ))
            elif self.config.transforms.norm == "half":
                list_transforms.append(Normalize(
                    mean=[0.5, 0.5, 0.5], 
                    std=[0.5, 0.5, 0.5]
                ))
            else:
                raise ValueError(f"Unsupported normalization method: {self.config.transforms.norm}")

        # Add standard preprocessing
        list_transforms.extend([
            Resize(self.config.dataset.image.imsize, self.config.dataset.image.imsize),
            ToTensorV2(),
        ])
        
        return Compose(list_transforms)


def get_pneumothorax_dataloader(
        config,
        split: str = "train",
        view_type: str = "Frontal",
        transform: Optional[T.Compose] = None, 
    ) -> DataLoader:
    """
    Create a DataLoader for the CheXpert medical imaging dataset.
    
    Args:
        config: Configuration object containing dataset parameters
            Expected structure:
            - config.data_dir: Base directory containing CheXpert data
            - config.{split}_csv: Path to the specific split CSV file
            - config.model.batch_size: Batch size
            - config.dataset.num_workers: Number of workers
            - config.dataset.fraction: Optional fraction of data to use (for training)
        split: Dataset split ('train', 'valid', or 'test')
        view_type: Type of X-ray view ('Frontal', 'Lateral', or 'All')
        transform: Optional custom transformation pipeline; if None, uses transforms built from config
        
    Returns:
        DataLoader for the CheXpert dataset
    """
    # Create transformation if not provided
    if transform is None:
        transform = build_transformation(config, split)
    
    # Create dataset
    dataset = PneumothoraxImageDataset(
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