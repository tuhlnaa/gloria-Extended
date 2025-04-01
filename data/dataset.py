"""
GLoRIA: A Multimodal Global-Local Representation Learning Framework
for Label-efficient Medical Image Recognition

Dataset and transformation utilities for medical image processing.
"""
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd

from PIL import Image
from enum import Enum
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

@dataclass
class CheXpertConfig:
    """Constants and configuration for the CheXpert dataset."""
    VIEW_COL: str = "Frontal/Lateral"
    PATH_COL: str = "Path"
   
    TASKS: List[str] = field(default_factory=lambda: [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Lesion", "Lung Opacity", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices"
    ])
   
    COMPETITION_TASKS: List[str] = field(default_factory=lambda: [
        "Atelectasis", "Cardiomegaly", "Consolidation",
        "Edema", "Pleural Effusion"
    ])
   
    # Based on original CheXpert paper
    UNCERTAIN_MAPPINGS: Dict[str, int] = field(default_factory=lambda: {
        "Atelectasis": 1,
        "Cardiomegaly": 0,
        "Consolidation": 0,
        "Edema": 1,
        "Pleural Effusion": 1,
    })


class ImageFormat(Enum):
    """Supported image formats for medical datasets."""
    JPG = "jpg"
    DICOM = "dicom"


class ImageBaseDataset(Dataset):
    """Base dataset class for medical image datasets."""
    
    def __init__(
            self,
            config: object,
            split: str = "train",
            transform: Optional[T.Compose] = None,
        ):
        """
        Initialize the base image dataset.
        
        Args:
            config: Configuration object containing dataset parameters
            split: Data split ('train', 'valid', or 'test')
            transform: Torchvision transforms to apply to images
        """
        self.config = config
        self.transform = transform
        self.split = split


    def __getitem__(self, index: int) -> Tuple:
        """Get item at specified index."""
        raise NotImplementedError


    def __len__(self) -> int:
        """Get dataset length."""
        raise NotImplementedError


    def read_image(self, img_path: Union[str, Path], format: ImageFormat = ImageFormat.JPG) -> Image.Image:
        """Read image from file path based on format."""
        if format == ImageFormat.JPG:
            return self._read_from_jpg(img_path)
        elif format == ImageFormat.DICOM:
            return self._read_from_dicom(img_path)
        else:
            raise ValueError(f"Unsupported image format: {format}")


    def _read_from_jpg(self, img_path: Union[str, Path]) -> Image.Image:
        """Read and preprocess a JPG image."""
        img_path = Path(img_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise FileNotFoundError(f"Image not found or couldn't be read: {img_path}")
            
        # Resize image to configured dimensions
        img = self._resize_img(img, self.config.dataset.image.imsize)
        pil_img = Image.fromarray(img).convert("RGB")
        
        if self.transform is not None:
            pil_img = self.transform(pil_img)
            
        return pil_img


    def _read_from_dicom(self, img_path: Union[str, Path]) -> Image.Image:
        """Read and preprocess a DICOM image."""
        raise NotImplementedError("DICOM reading not implemented yet")


    def _resize_img(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image while preserving aspect ratio and padding shorter dimension."""
        height, width = img.shape
        
        # Determine which dimension to scale
        if height > width:
            # Image is taller
            scale_factor = target_size / height
            new_height = target_size
            new_width = int(width * scale_factor)
        else:
            # Image is wider or square
            scale_factor = target_size / width
            new_width = target_size
            new_height = int(height * scale_factor)
            
        # Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        
        # Calculate padding
        pad_height = target_size - new_height
        pad_width = target_size - new_width
        
        # Calculate padding for each side
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        
        # Apply padding
        padded_img = np.pad(
            resized_img, [(top, bottom), (left, right)], 
            mode="constant", constant_values=0
        )
        
        return padded_img


class CheXpertImageDataset(ImageBaseDataset):
    """Dataset class for the CheXpert medical imaging dataset."""
    
    def __init__(
        self, 
        config: object, 
        split: str = "train", 
        transform: Optional[T.Compose] = None, 
        view_type: str = "Frontal"
    ):
        """
        Initialize the CheXpert dataset.
        
        Args:
            config: Configuration object
            split: Data split ('train', 'valid', or 'test')
            transform: Torchvision transforms to apply
            view_type: Type of X-ray view ('Frontal', 'Lateral', or 'All')
        """
        super().__init__(config, split, transform)
        
        self.chexpert_config = CheXpertConfig()
        
        #if not self.chexpert_config.DATA_DIR.exists():
        if not Path(config.data_dir).exists():
            raise RuntimeError(
                "CheXpert data path not found.\n"
                "Make sure to download data from:\n"
                "https://stanfordmlgroup.github.io/competitions/chexpert/"
                f" and update DATA_DIR in CheXpertConfig"
            )

        # Load the appropriate CSV file based on split
        csv_path = self._get_csv_path(config, split)
        self.df = pd.read_csv(csv_path)
        
        # Apply data fraction sampling if specified in config
        if hasattr(config.dataset, 'fraction') and config.dataset.fraction != 1 and split == "train":
            orig_size = len(self.df)
            self.df = self.df.sample(frac=config.dataset.fraction, random_state=42)
            print(f"Applied dataset fraction {config.dataset.fraction}: reduced from {orig_size} to {len(self.df)} samples")

        # Filter by view type if specified
        if view_type != "All":
            self.df = self.df[self.df[self.chexpert_config.VIEW_COL] == view_type]

        # Process image paths to be absolute
        self._process_image_paths(config)
        
        # Handle missing values and uncertain labels
        self._preprocess_labels()


    def _get_csv_path(self, config: object,  split: str) -> Path:
        """Get the CSV file path for the given split."""
        if split == "train":
            return Path(config.data_dir) / config.train_csv # self.chexpert_config.TRAIN_CSV
        elif split == "valid":
            return Path(config.data_dir) / config.valid_csv # self.chexpert_config.VALID_CSV
        else:  # test
            return Path(config.data_dir) / config.test_csv  # self.chexpert_config.TEST_CSV


    def _process_image_paths(self, config) -> None:
        """
        Process image paths to be absolute paths using pathlib.
        Converts relative paths in the dataset to absolute paths.
        """
        path_col = self.chexpert_config.PATH_COL
        
        def convert_to_absolute_path(relative_path: str) -> str:
            path_components = relative_path.split("/")[1:]
            relative_subpath = Path(*path_components)
            absolute_path = Path(config.data_dir) / relative_subpath
            return str(absolute_path)
            
        # Apply the conversion function to all paths in the dataframe
        self.df[path_col] = self.df[path_col].apply(convert_to_absolute_path)


    def _preprocess_labels(self) -> None:
        """Handle missing values and uncertain labels."""
        # Fill NaN values with 0
        self.df = self.df.fillna(0)
        
        # Replace uncertain labels (-1) with mappings from the paper
        uncertain_mask = {k: -1 for k in self.chexpert_config.COMPETITION_TASKS}
        self.df = self.df.replace(uncertain_mask, self.chexpert_config.UNCERTAIN_MAPPINGS)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and labels at the specified index."""
        row = self.df.iloc[index]
        
        # Get and process image
        img_path = row[self.chexpert_config.PATH_COL]
        img = self._read_from_jpg(img_path)
        
        # Get labels for competition tasks
        labels = torch.tensor(list(row[self.chexpert_config.COMPETITION_TASKS]))
        
        return img, labels


    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)


def build_transformation(config: object, split: str) -> T.Compose:
    """
    Build image transformation pipeline based on configuration.
    
    Args:
        config: Configuration object with transform parameters
        split: Data split ('train', 'valid', or 'test')
        
    Returns:
        Composed transformation pipeline
    """
    transforms = []
    
    # Apply data augmentation only for training
    if split == "train":
        # Random crop
        if hasattr(config.transforms, 'random_crop') and config.transforms.random_crop is not None:
            transforms.append(T.RandomCrop(config.transforms.random_crop.crop_size))
            
        # Random horizontal flip
        if hasattr(config.transforms, 'random_horizontal_flip') and config.transforms.random_horizontal_flip is not None:
            transforms.append(T.RandomHorizontalFlip(p=config.transforms.random_horizontal_flip))
            
        # Random affine transformation
        if hasattr(config.transforms, 'random_affine') and config.transforms.random_affine is not None:
            transforms.append(
                T.RandomAffine(
                    degrees=config.transforms.random_affine.degrees,
                    translate=list(config.transforms.random_affine.translate),
                    scale=list(config.transforms.random_affine.scale),
                )
            )
            
        # Color jitter
        if hasattr(config.transforms, 'color_jitter') and config.transforms.color_jitter is not None:
            transforms.append(
                T.ColorJitter(
                    brightness=list(config.transforms.color_jitter.brightness),
                    contrast=list(config.transforms.color_jitter.contrast),
                )
            )
    else:
        # For validation/test, use center crop instead of random crop
        if hasattr(config.transforms, 'random_crop') and config.transforms.random_crop is not None:
            transforms.append(T.CenterCrop(config.transforms.random_crop.crop_size))
    
    # Convert to tensor (required for all splits)
    transforms.append(T.ToTensor())
    
    # Apply normalization if specified
    if hasattr(config.transforms, 'norm') and config.transforms.norm is not None:
        if config.transforms.norm == "imagenet":
            transforms.append(T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ))
        elif config.transforms.norm == "half":
            transforms.append(T.Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]
            ))
        else:
            raise ValueError(f"Unsupported normalization method: {config.transforms.norm}")
    
    return T.Compose(transforms)


def get_chexpert_dataloader(
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
    dataset = CheXpertImageDataset(
        config=config,
        split=split,
        transform=transform,
        view_type=view_type
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
    
    return data_loader