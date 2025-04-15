import sys
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.pneumothorax_dataset import get_pneumothorax_dataloader
from utils.logging_utils import LoggingManager


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def main():
    set_seed()
    config = OmegaConf.load("./test/pneumothorax_segmentation_dataset_config.yaml")

    # Print configuration using the logging utility
    LoggingManager.print_config(config, "Configuration")

    # Create dataloader
    data_loader, _ = get_pneumothorax_dataloader(config, split="train")

    # Iterate through batches
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Image shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Memory format: {images.is_contiguous()}")
        print(f"  Device: {images.device}")
        print(f"  Data type: {images.dtype}")
        print(f"  Labels type: {labels.dtype}")
        print(f"  Data range: {images.max(), images.min()}")
        print(f"  Label range: {images.max(), images.min()}")
            
        # Only show first 2 batches
        if batch_idx >= 1:
            break


if __name__ == "__main__":
    main()

"""
╭──────────────────────────────────┬─────────────────────────────────────────────────────╮
│                        Parameter │ Value                                               │
├──────────────────────────────────┼─────────────────────────────────────────────────────┤
│                         data_dir │ './X-ray/SIIM-ACR-Pneumothorax'                     │
│                        train_csv │ 'stage_1_train_images.csv'                          │
│                        valid_csv │ 'stage_1_test_images.csv'                           │
│                            phase │ 'segmentation'                                      │
│                 model.batch_size │ 16                                                  │
│                     dataset.name │ 'chexpert'                                          │
│                 dataset.fraction │ 1.0                                                 │
│              dataset.num_workers │ 0                                                   │
│             dataset.image.imsize │ 256                                                 │
│ transforms.random_crop.crop_size │ 224                                                 │
╰──────────────────────────────────┴─────────────────────────────────────────────────────╯

DataLoader created successfully with 85 batches
Batch 1:
  Image shape: torch.Size([16, 3, 256, 256])
  Labels shape: torch.Size([16, 256, 256])
  Memory format: True
  Device: cpu
  Data type: torch.float32
  Labels type: torch.float32
  Data range: (tensor(1.), tensor(0.))
  Label range: (tensor(1.), tensor(0.))

Batch 2:
  Image shape: torch.Size([16, 3, 256, 256])
  Labels shape: torch.Size([16, 256, 256])
  Memory format: True
  Device: cpu
  Data type: torch.float32
  Labels type: torch.float32
  Data range: (tensor(1.), tensor(0.))
  Label range: (tensor(1.), tensor(0.))

"""