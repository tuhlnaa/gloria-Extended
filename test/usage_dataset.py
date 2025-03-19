import sys
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import CheXpertImageDataset, build_transformation


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
    
    # Create transform
    transform = build_transformation(config, "train")
    
    # Create dataset
    dataset = CheXpertImageDataset(config, split="train", transform=transform, view_type="Frontal")
    
    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=False,
        drop_last=True
    )
    
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