import cv2
import torch
import numpy as np

from albumentations.pytorch import ToTensorV2
from albumentations import Normalize, Resize, Compose, Affine
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader


def collate_with_transforms(batch: List[Dict[str, Any]], transforms: Compose) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that applies transforms to each sample in a batch.
    
    Args:
        batch: List of samples from the dataset
        transforms: Albumentations transformation pipeline to apply
        
    Returns:
        Dictionary with batched and transformed images and masks
    """
    transformed_batch = []
    
    for item in batch:
        image = item["image"]
        mask = item["mask"]
        
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
            
        # If image is in CHW format (PyTorch), convert to HWC format (Albumentations)
        if image.shape[0] == 3 and len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
            
        # Handle mask format - it could be (1, H, W) which needs conversion
        if mask.shape[0] == 1 and len(mask.shape) == 3:
            # Option 1: Convert to (H, W) by removing the channel dimension
            mask = mask.squeeze(0)
            # Option 2: Convert to (H, W, 1) if needed
            # mask = np.transpose(mask, (1, 2, 0))
        
        # Apply transforms
        transformed = transforms(image=image, mask=mask)
        transformed_image = transformed["image"]  # This will be a tensor due to ToTensorV2
        transformed_mask = transformed["mask"]    # This will be a tensor due to ToTensorV2
        
        transformed_batch.append({
            "image": transformed_image,
            "mask": transformed_mask
        })
    
    # Stack the transformed tensors
    images = torch.stack([item["image"] for item in transformed_batch])
    masks = torch.stack([item["mask"] for item in transformed_batch])
    
    # return {"image": images, "mask": masks}
    return images, masks


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


def get_cat_dataloader(
        config,
        split: str = "train",
        view_type: str = "Frontal",
        transform: Optional[Compose] = None, 
    ) -> DataLoader:
    """
    Create a DataLoader for the cat imaging dataset with transforms applied in collate_fn.
    
    Args:
        config: Configuration object containing dataset parameters
        split: Dataset split ('train', 'valid', or 'test')
        view_type: Type of X-ray view ('Frontal', 'Lateral', or 'All')
        transform: Optional custom transformation pipeline; if None, uses transforms built from config
        
    Returns:
        DataLoader for the dataset with transforms applied in collate_fn
    """
    # Get the appropriate dataset
    if split == "train":
        dataset = SimpleOxfordPetDataset(config.data_dir, "train")
    elif split == "valid":
        dataset = SimpleOxfordPetDataset(config.data_dir, "valid")
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Set up transforms
    if transform is None:
        # Create a config object with the split for get_transforms
        transform_config = config
        transform_config.split = split
        transforms = get_transforms(transform_config)
    else:
        transforms = transform
    
    # Create a collate function with the transforms
    collate_fn = lambda batch: collate_with_transforms(batch, transforms)

    # Create dataloader with parameters from config and custom collate_fn
    data_loader = DataLoader(
        dataset,
        batch_size=config.model.batch_size if split == "train" else 1,
        shuffle=(split == "train"),
        num_workers=config.dataset.num_workers,
        pin_memory=getattr(config.model, "pin_memory", False),
        drop_last=getattr(config.model, "drop_last", False) if split == "train" else True,
        collate_fn=collate_fn  # Use our custom collate function
    )
    
    print(f"DataLoader created successfully with {len(data_loader)} batches")
    return data_loader, dataset
