import torch
import segmentation_models_pytorch as smp

from typing import Dict, Tuple
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from gloria import builder, gloria
from gloria.utils.metrics import SegmentationMetrics


class SegmentationTrainer:
    """Trainer class for image segmentation models."""

    def __init__(self, config: OmegaConf, train_loader):
        """Initialize the segmentation trainer."""
        self.config = config
        self.lr = config.lr_scheduler.learning_rate
        self.device = config.device.device
        self.train_loader = train_loader
        
        # Initialize the model
        self.model = self._initialize_model().to(self.device)
        print(f"Using model: [{type(self.model).__name__}]")

        # Initialize loss function
        self.criterion = builder.build_loss(config)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.setup_optimization()

    def _initialize_model(self) -> torch.nn.Module:
        """Initialize the segmentation model based on configuration."""
        if self.config.model.vision.model_name in gloria.available_models():
            return gloria.load_img_segmentation_model(
                self.config.model.vision.model_name,
                device=self.config.device
            )
        else:
            return smp.Unet("resnet50", encoder_weights=None, activation=None)


    def setup_optimization(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        self.optimizer = builder.build_optimizer(self.config, self.lr, self.model)
        self.scheduler = builder.build_scheduler(self.config, self.optimizer, self.train_loader)


    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        
        # Create metrics tracker
        metrics = SegmentationMetrics(split='train')
        metrics.reset()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch: {epoch}")):
            image, mask = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()           # Reset gradients
            logits = self.model(image)           # Forward pass
            logits = logits.squeeze()     
            loss = self.criterion(logits, mask)  # Compute loss
            loss.backward()                      # Backward pass

            # Gradient clipping if needed
            if self.config.optimizer.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.optimizer.clip_grad)

            self.optimizer.step()                # Update parameters
            
            # Update learning rate scheduler if needed
            if self.config.lr_scheduler.name == "LinearWarmupCosine":
                self.scheduler.step()
            
            metrics.update(logits, mask, loss)
        
        # Compute metrics
        return metrics.compute()


    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move batch data to the correct device."""
        x, y = batch
        return x.to(self.device), y.to(self.device)
    

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
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
