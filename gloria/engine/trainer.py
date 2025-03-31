"""
Training module for GLoRIA model and other classification models.

This module handles the training loop and related functionality for image classification
models, with special support for GLoRIA pre-trained models.
"""
import os
import torch
import torch.nn as nn

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from gloria import builder
from gloria.utils.metrics import ClassificationMetrics
from .validator import Validator


class Trainer:
    """Trainer class for image classification models."""

    def __init__(self, config):
        self.config = config
        self.learning_rate = config.lr_scheduler.learning_rate
        self.device = config.device.device
        self.datamodule = None
        
        # Initialize the model
        self.model = self._initialize_model().to(self.device)
        print(f"Using model: [{type(self.model).__name__}]")

        # Initialize loss function
        self.criterion = builder.build_loss(config)
        
        # Optimization components (initialized later)
        self.optimizer = None
        self.scheduler = None


    def _initialize_model(self) -> torch.nn.Module:
        """Initialize the model based on configuration."""
        return builder.build_image_model(self.config)


    def setup_optimization(self, datamodule=None) -> None:
        """Set up optimizer and learning rate scheduler."""
        self.datamodule = datamodule
        self.optimizer = builder.build_optimizer(
            self.config, 
            self.learning_rate, 
            self.model
        )
        self.scheduler = builder.build_scheduler(
            self.config, 
            self.optimizer, 
            self.datamodule
        )


    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()

        # Create metrics tracker
        metrics = ClassificationMetrics(split='train')
        metrics.reset()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch: {epoch}")):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()              # Reset gradients
            logits = self.model(images)             # Forward pass
            loss = self.criterion(logits, labels)   # Compute loss
            loss.backward()                         # Backward pass
            
            # Gradient clipping
            if hasattr(self.config.optimizer, 'clip_grad'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.optimizer.clip_grad
                )

            self.optimizer.step()  # Update parameters

            metrics.update(logits, labels, loss)

            # Print progress at intervals
            if batch_idx % self.config.get('log_interval', 10) == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}")
        
        return metrics.compute()


    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path where to save the checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler["scheduler"].state_dict() if self.scheduler else None,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file
            load_optimizer: Whether to load optimizer and scheduler states
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"Checkpoint loaded from {path}")


