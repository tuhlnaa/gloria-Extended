"""
GLoRIA: A Multimodal Global-Local Representation Learning Framework
for Label-efficient Medical Image Recognition

Dataset and transformation utilities for medical image processing.
"""
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass


from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

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
            cfg: object,
            split: str = "train",
            transform: Optional[T.Compose] = None,
        ):
        """
        Initialize the base image dataset.
        
        Args:
            cfg: Configuration object containing dataset parameters
            split: Data split ('train', 'valid', or 'test')
            transform: Torchvision transforms to apply to images
        """
        self.cfg = cfg
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
        img = self._resize_img(img, self.cfg.data.image.imsize)
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
        cfg: object, 
        split: str = "train", 
        transform: Optional[T.Compose] = None, 
        view_type: str = "Frontal"
    ):
        """
        Initialize the CheXpert dataset.
        
        Args:
            cfg: Configuration object
            split: Data split ('train', 'valid', or 'test')
            transform: Torchvision transforms to apply
            view_type: Type of X-ray view ('Frontal', 'Lateral', or 'All')
        """
        super().__init__(cfg, split, transform)
        
        self.chexpert_config = CheXpertConfig()
        
        #if not self.chexpert_config.DATA_DIR.exists():
        if not Path(cfg.path.data_dir).exists():
            raise RuntimeError(
                "CheXpert data path not found.\n"
                "Make sure to download data from:\n"
                "https://stanfordmlgroup.github.io/competitions/chexpert/"
                f" and update DATA_DIR in CheXpertConfig"
            )

        # Load the appropriate CSV file based on split
        csv_path = self._get_csv_path(cfg, split)
        self.df = pd.read_csv(csv_path)
        
        # Apply data fraction sampling if specified in config
        if hasattr(cfg.data, 'frac') and cfg.data.frac != 1 and split == "train":
            self.df = self.df.sample(frac=cfg.data.frac, random_state=42)

        # Filter by view type if specified
        if view_type != "All":
            self.df = self.df[self.df[self.chexpert_config.VIEW_COL] == view_type]

        # Process image paths to be absolute
        self._process_image_paths(cfg)
        
        # Handle missing values and uncertain labels
        self._preprocess_labels()


    def _get_csv_path(self, cfg: object,  split: str) -> Path:
        """Get the CSV file path for the given split."""
        if split == "train":
            return Path(cfg.path.data_dir) / cfg.path.train_csv # self.chexpert_config.TRAIN_CSV
        elif split == "valid":
            return Path(cfg.path.data_dir) / cfg.path.valid_csv # self.chexpert_config.VALID_CSV
        else:  # test
            return Path(cfg.path.data_dir) / cfg.path.test_csv  # self.chexpert_config.TEST_CSV


    def _process_image_paths(self, cfg) -> None:
        """
        Process image paths to be absolute paths using pathlib.
        Converts relative paths in the dataset to absolute paths.
        """
        path_col = self.chexpert_config.PATH_COL
        
        def convert_to_absolute_path(relative_path: str) -> str:
            path_components = relative_path.split("/")[1:]
            relative_subpath = Path(*path_components)
            absolute_path = Path(cfg.path.data_dir) / relative_subpath
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


def build_transformation(cfg: object, split: str) -> T.Compose:
    """
    Build image transformation pipeline based on configuration.
    
    Args:
        cfg: Configuration object with transform parameters
        split: Data split ('train', 'valid', or 'test')
        
    Returns:
        Composed transformation pipeline
    """
    transforms = []
    
    # Apply data augmentation only for training
    if split == "train":
        # Random crop
        if hasattr(cfg.transforms, 'random_crop') and cfg.transforms.random_crop is not None:
            transforms.append(T.RandomCrop(cfg.transforms.random_crop.crop_size))
            
        # Random horizontal flip
        if hasattr(cfg.transforms, 'random_horizontal_flip') and cfg.transforms.random_horizontal_flip is not None:
            transforms.append(T.RandomHorizontalFlip(p=cfg.transforms.random_horizontal_flip))
            
        # Random affine transformation
        if hasattr(cfg.transforms, 'random_affine') and cfg.transforms.random_affine is not None:
            transforms.append(
                T.RandomAffine(
                    degrees=cfg.transforms.random_affine.degrees,
                    translate=list(cfg.transforms.random_affine.translate),
                    scale=list(cfg.transforms.random_affine.scale),
                )
            )
            
        # Color jitter
        if hasattr(cfg.transforms, 'color_jitter') and cfg.transforms.color_jitter is not None:
            transforms.append(
                T.ColorJitter(
                    brightness=list(cfg.transforms.color_jitter.brightness),
                    contrast=list(cfg.transforms.color_jitter.contrast),
                )
            )
    else:
        # For validation/test, use center crop instead of random crop
        if hasattr(cfg.transforms, 'random_crop') and cfg.transforms.random_crop is not None:
            transforms.append(T.CenterCrop(cfg.transforms.random_crop.crop_size))
    
    # Convert to tensor (required for all splits)
    transforms.append(T.ToTensor())
    
    # Apply normalization if specified
    if hasattr(cfg.transforms, 'norm') and cfg.transforms.norm is not None:
        if cfg.transforms.norm == "imagenet":
            transforms.append(T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ))
        elif cfg.transforms.norm == "half":
            transforms.append(T.Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]
            ))
        else:
            raise ValueError(f"Unsupported normalization method: {cfg.transforms.norm}")
    
    return T.Compose(transforms)