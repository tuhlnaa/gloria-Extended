
import cv2
import torchvision.transforms as T

from albumentations.pytorch import ToTensorV2
from albumentations import Normalize, Resize, Compose, Affine
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from typing import Optional
from torch.utils.data import DataLoader


def get_cat_dataloader(
        config,
        split: str = "train",
        view_type: str = "Frontal",
        transform: Optional[T.Compose] = None, 
    ) -> DataLoader:
    """
    Create a DataLoader for the cat imaging dataset.
    
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
    if split == "train":
        dataset = SimpleOxfordPetDataset(config.data_dir, "train")

        # Create dataloader with parameters from config
        data_loader = DataLoader(
            dataset,
            batch_size=config.model.batch_size if split == "train" else 1,
            shuffle=(split == "train"),
            num_workers=config.dataset.num_workers,
            pin_memory=getattr(config.model, "pin_memory", False),
            drop_last=getattr(config.model, "drop_last", False) if split == "train" else True
        )
        print(f"DataLoader created successfully with {len(data_loader)} batches")

    elif split == "valid":
        valid_dataset = SimpleOxfordPetDataset(config.data_dir, "valid")

        # Create dataloader with parameters from config
        data_loader = DataLoader(
            dataset,
            batch_size=config.model.batch_size if split == "train" else 1,
            shuffle=(split == "train"),
            num_workers=config.dataset.num_workers,
            pin_memory=getattr(config.model, "pin_memory", False),
            drop_last=getattr(config.model, "drop_last", False) if split == "train" else True
        )
        print(f"DataLoader created successfully with {len(data_loader)} batches")

    return data_loader, dataset




def get_transforms(config):
    """Get transformations pipeline for image augmentation."""
    list_transforms = []

    # Add augmentations for training
    if config.split == "train":
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
    if hasattr(config.transforms, 'norm') and config.transforms.norm is not None:
        if config.transforms.norm == "imagenet":
            list_transforms.append(Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ))
        elif config.transforms.norm == "half":
            list_transforms.append(Normalize(
                mean=[0.5, 0.5, 0.5], 
                std=[0.5, 0.5, 0.5]
            ))
        elif config.transforms.norm == "zero_one":
            # Normalize to [0,1] range
            # For values in [0,255], dividing by 255 scales to [0,1]
            list_transforms.append(Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ))
        else:
            raise ValueError(f"Unsupported normalization method: {config.transforms.norm}")

    # Add standard preprocessing
    list_transforms.extend([
        Resize(config.dataset.image.imsize, config.dataset.image.imsize),
        ToTensorV2(),
    ])
    
    return Compose(list_transforms)

"""
Batch 1:
  Image shape: torch.Size([16, 3, 256, 256])
  Labels shape: torch.Size([16, 1, 256, 256])
  Memory format: True
  Device: cpu
  Data type: torch.uint8
  Labels type: torch.float32
  Data range: (tensor(255, dtype=torch.uint8), tensor(0, dtype=torch.uint8))
  Label range: (tensor(1.), tensor(0.))
Batch 2:
...
"""