import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Union
from torch.utils.data import DataLoader

class BaseTrainer:
    """Base class for all trainers with common functionality."""
    
    def __init__(self, config: OmegaConf, train_loader: DataLoader):
        self.config = config
        self.learning_rate = config.lr_scheduler.learning_rate
        self.device = config.device.device
        self.train_loader = train_loader
        
        # Optimization components (initialized later)
        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.criterion = None
    
    def setup_optimization(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        from gloria import builder
        
        self.optimizer = builder.build_optimizer(
            self.config, 
            self.learning_rate, 
            self.model
        )
        self.scheduler = builder.build_scheduler(
            self.config, 
            self.optimizer, 
            self.train_loader
        )
