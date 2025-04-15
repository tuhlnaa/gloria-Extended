import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from typing import Dict, List, Tuple, Union
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from rich import print

from gloria import builder
from gloria.utils.metrics import GradientMonitor, SegmentationMetricsV2


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
        if self.config.model.transfer_checkpoint:
            return load_img_segmentation_model(
                self.config,
                name=self.config.model.vision.model_name
            )
        else:
            return smp.Unet("resnet50", encoder_weights=self.config.model.pretrained)


    def setup_optimization(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        self.optimizer = builder.build_optimizer(self.config, self.lr, self.model)
        self.scheduler = builder.build_scheduler(self.config, self.optimizer, self.train_loader)


    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        
        # Create metrics tracker
        metrics = SegmentationMetricsV2(split='train', num_classes = self.config.dataset.num_classes)
        metrics.reset()
        grad_monitor = GradientMonitor(split='train')
        grad_monitor.reset()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch: {epoch}")):
            image, mask = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()           # Reset gradients
            logits = self.model(image)           # Forward pass 
            loss = self.criterion(logits, mask)  # Compute loss
            loss.backward()                      # Backward pass

            # Monitor gradients before clipping
            _ = grad_monitor.update_before_clip(self.model)

            # Gradient clipping if needed
            if self.config.optimizer.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.optimizer.clip_grad)

            # Monitor gradients after clipping
            _ = grad_monitor.update_after_clip(self.model)

            self.optimizer.step()                # Update parameters
            
            # Update learning rate scheduler if needed
            if self.config.lr_scheduler.name == "LinearWarmupCosine":
                self.scheduler.step()
            
            metrics.update(logits, mask, loss)

        # Compute metrics
        main_metrics = metrics.compute()
        grad_metrics = grad_monitor.compute()
        combined_metrics = {**main_metrics, **grad_metrics}

        return combined_metrics


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



# Constants for model paths and dimensions
MODEL_CHECKPOINTS = {
    "gloria_resnet50": "./pretrained/chexpert_resnet50.ckpt",
    "gloria_resnet18": "./pretrained/chexpert_resnet18.ckpt",
}

SEGMENTATION_MODEL_CHECKPOINTS = {
    "gloria_resnet50": "./pretrained/chexpert_resnet50.ckpt",
}

FEATURE_DIMENSIONS = {
    "gloria_resnet50": 2048, 
    "gloria_resnet18": 2048
}

def available_segmentation_models() -> List[str]:
    """Returns the names of available GLoRIA segmentation models."""
    return list(SEGMENTATION_MODEL_CHECKPOINTS.keys())

def available_models() -> List[str]:
    """Returns the names of available GLoRIA models."""
    return list(MODEL_CHECKPOINTS.keys())


def load_img_segmentation_model(
        config: OmegaConf, 
        name: str = "resnet50", 
    ) -> nn.Module:
    """Load a GLoRIA pretrained segmentation model.

    Args:
        name: A model name
        device: The device to put the loaded model

    Returns:
        nn.Module: The GLoRIA pretrained image segmentation model
    """
    # Initialize segmentation model
    seg_model = smp.Unet(
        encoder_name=name, 
        encoder_weights=None, 
        activation=None
    )

    # Load and prepare encoder weights
    checkpoint = torch.load(config.model.transfer_checkpoint, map_location=config.device.device)

    encoder_state_dict = {}
    
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("gloria.img_encoder.model"):
            # Extract encoder part from key
            encoder_key = ".".join(key.split(".")[3:])
            encoder_state_dict[encoder_key] = value
        elif key.startswith("img_encoder.model"):
            encoder_key = ".".join(key.split(".")[2:])
            encoder_state_dict[encoder_key] = value

    # Remove FC layer weights as they're not needed for segmentation
    encoder_state_dict["fc.bias"] = None
    encoder_state_dict["fc.weight"] = None
    
    # Load weights into encoder
    seg_model.encoder.load_state_dict(encoder_state_dict)
    
    return seg_model


def _get_checkpoint_path(name: str, available_models_dict: Dict[str, str]) -> str:
    """Helper function to get checkpoint path from model name."""
    if name in available_models_dict:
        ckpt_path = available_models_dict[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        model_type = "segmentation " if available_models_dict == SEGMENTATION_MODEL_CHECKPOINTS else ""
        available = (available_segmentation_models() if available_models_dict == SEGMENTATION_MODEL_CHECKPOINTS 
                    else available_models())
        raise RuntimeError(
            f"Model {name} not found; available {model_type}models = {available}"
        )

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Model {name} not found.\n"
            "Make sure to download the pretrained weights from \n"
            "    https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh \n"
            "and copy it to the ./pretrained folder."
        )
    
    return ckpt_path