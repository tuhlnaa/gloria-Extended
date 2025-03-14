import sys
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
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


class SimpleConfig:
    """Simple configuration class to mimic the expected config structure."""
    
    def __init__(self):
        self.data = type('obj', (), {
            'dataset': 'chexpert',
            #'frac': 1.0,
            'image': type('obj', (), {'imsize': 256}) 
        })
        
        self.transforms = type('obj', (), {
            'random_crop': type('obj', (), {'crop_size': 224}),
            # 'random_horizontal_flip': 0.5,
            # 'random_affine': None,
            # 'color_jitter': None,
            # 'norm': 'imagenet'
        })
        
        self.train = type('obj', (), {
            'batch_size': 16,
            'num_workers': 0
        })

        self.path = type('obj', (), {
            'data_dir': r"D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0-small",
            "train_csv": "train.csv",
            "valid_csv": "valid.csv",
            "test_csv": "valid.csv"  # using validation set as test set (test set labels hidden)
        })


def main():
    set_seed()
    
    # Create a simple configuration
    cfg = SimpleConfig()
    
    # Create transform
    transform = build_transformation(cfg, "train")
    
    # Create dataset
    dataset = CheXpertImageDataset(cfg, split="train", transform=transform, view_type="Frontal")
    
    # Create dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
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