
import torch
import torch.nn as nn

from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union
from torch.utils.data import DataLoader

from gloria import builder
from gloria.utils.metrics import ClassificationMetrics, GradientMonitor

class Trainer:
    """Trainer class for image classification models."""

    def __init__(self, config: OmegaConf, train_loader: DataLoader):
        self.config = config
        self.learning_rate = config.lr_scheduler.learning_rate
        self.device = config.device.device
        self.train_loader = train_loader
        
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


    def setup_optimization(self) -> None:
        """Set up optimizer and learning rate scheduler."""
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


    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()

        # Create metrics tracker
        metrics = ClassificationMetrics(split='train')
        metrics.reset()
        grad_monitor = GradientMonitor(split='train')
        grad_monitor.reset()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch: {epoch}")):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()              # Reset gradients
            logits = self.model(images)             # Forward pass
            loss = self.criterion(logits, labels)   # Compute loss
            loss.backward()                         # Backward pass

            # Monitor gradients before clipping
            _ = grad_monitor.update_before_clip(self.model)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.optimizer.clip_grad
            )

            self.optimizer.step()  # Update parameters

            if self.config.lr_scheduler.name == "LinearWarmupCosine":
                self.scheduler.step()

            # Monitor gradients after clipping
            _ = grad_monitor.update_after_clip(self.model)
            
            metrics.update(logits, labels, loss)

        # Compute metrics
        class_metrics = metrics.compute()
        grad_metrics = grad_monitor.compute()
        combined_metrics = {**class_metrics, **grad_metrics}

        return combined_metrics


    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Resume training from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epochs = checkpoint.get("epochs", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_metric = checkpoint.get("best_metrics", float("inf"))

        return start_epochs, best_val_metric, best_val_loss