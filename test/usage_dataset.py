import sys
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import get_chexpert_dataloader
from utils.logging_utils import LoggingManager


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def main():
    set_seed()
    config = OmegaConf.load("./test/usage_dataset_config.yaml")
    
    # Print configuration using the logging utility
    LoggingManager.print_config(config, "Configuration")

    # Create dataloader
    data_loader = get_chexpert_dataloader(config, split="train", view_type="Frontal")

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
        
        # Print unique labels for each category in the competition tasks
        print("  Label distributions:")
        for i, task in enumerate(["No Finding", "Cardiomegaly", "Edema", "Consolidation", "Atelectasis"]):
            print(f"    {task}: {labels[:, i].tolist()}")
            
        # Only show first 3 batches
        if batch_idx >= 2:
            break


if __name__ == "__main__":
    main()

"""
╭──────────────────────────────────┬───────────────────────────────────────────────────╮
│                        Parameter │ Value                                             │
├──────────────────────────────────┼───────────────────────────────────────────────────┤
│                     data.dataset │ 'chexpert'                                        │
│                data.image.imsize │ 256                                               │
│ transforms.random_crop.crop_size │ 224                                               │
│                 train.batch_size │ 16                                                │
│                train.num_workers │ 0                                                 │
│                    path.data_dir │ './X-ray/CheXpert-v1.0-small'                     │
│                   path.train_csv │ 'train.csv'                                       │
│                   path.valid_csv │ 'valid.csv'                                       │
│                    path.test_csv │ 'valid.csv'                                       │
╰──────────────────────────────────┴───────────────────────────────────────────────────╯

Batch 1:
  Image shape: torch.Size([16, 3, 224, 224])
  Labels shape: torch.Size([16, 5])
  Memory format: True
  Device: cpu
  Data type: torch.float32
  Labels type: torch.float64
  Data range: (tensor(1.), tensor(0.))
  Label distributions:
    No Finding: tensor([0., 1., 0., 0., 1.,...
    Cardiomegaly: tensor([0., 1., 0., 0., 1.,...
    Edema: tensor([0., 0., 0., 0., 0.,...
    Consolidation: tensor([0., 1., 1., 0.,...
    Atelectasis: tensor([0., 1., 0., 0.,...

Batch 2:
  Image shape: torch.Size([16, 3, 224, 224])
  Labels shape: torch.Size([16, 5])
  Memory format: True
  Device: cpu
  Data type: torch.float32
  Labels type: torch.float64
  Data range: (tensor(1.), tensor(0.))
  Label distributions: ...

Batch 3:
  Image shape: torch.Size([16, 3, 224, 224])
  Labels shape: torch.Size([16, 5])
  Memory format: True
  Device: cpu
  Data type: torch.float32
  Labels type: torch.float64
  Data range: (tensor(1.), tensor(0.))
  Label distributions: ...
"""