import os
import json
import torch
import numpy as np
import segmentation_models_pytorch as smp

from typing import Dict, List, Tuple
from omegaconf import OmegaConf

from gloria import builder, gloria


class SegmentationModel:
    """
    Pure PyTorch model for image segmentation tasks.
    
    Supports both GLoRIA pre-trained models and other image segmentation models.
    Handles training, validation, and testing with appropriate metrics tracking.
    """

    def __init__(self, config: OmegaConf, train_loader):
        """Initialize the segmentation model."""
        self.config = config
        self.lr = config.lr_scheduler.learning_rate
        self.device = config.device.device
        self.train_loader = train_loader
        
        # Initialize the appropriate model based on configuration
        self.model = self._initialize_model()
        self.model.to(self.device)
        print(f"Used [{type(self.model).__name__}]")

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
        epoch_loss = 0.0
        epoch_dice = 0.0
        all_logits = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            x, mask = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            logits = self.model(x)
            logits = logits.squeeze()
            loss = self.criterion(logits, mask)
            loss.backward()

            # Gradient clipping if needed
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.optimizer.clip_grad)

            self.optimizer.step()
            
            epoch_loss += loss.item()
            probs = torch.sigmoid(logits)
            dice = self._get_dice(probs, mask)
            epoch_dice += dice
            
            all_logits.append(logits.detach())
            all_labels.append(mask.detach())
        
        # Calculate epoch metrics
        metrics = self._compute_epoch_metrics(all_logits, all_labels, epoch_loss, epoch_dice, len(train_loader), "train")
            
        return metrics


    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model on the validation set."""
        self.model.eval()
        val_loss = 0.0
        val_dice = 0.0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = self._prepare_batch(batch)
                
                logits = self.model(x)
                logits = logits.squeeze()
                loss = self.criterion(logits, y)
                
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                dice = self._get_dice(probs, y)
                val_dice += dice
                
                all_logits.append(logits)
                all_labels.append(y)
        
        # Calculate validation metrics
        metrics = self._compute_epoch_metrics(all_logits, all_labels, val_loss, val_dice, len(val_loader), "val")
        return metrics


    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move batch data to the correct device."""
        x, y = batch
        return x.to(self.device), y.to(self.device)


    def _compute_epoch_metrics(
        self, 
        logits: List[torch.Tensor],
        labels: List[torch.Tensor],
        loss_sum: float,
        dice_sum: float,
        num_batches: int,
        split: str
    ) -> Dict[str, float]:
        """
        Compute metrics for an epoch.
        
        Args:
            logits: List of model logits from each batch
            labels: List of labels from each batch
            loss_sum: Sum of losses across batches
            dice_sum: Sum of dice scores across batches
            num_batches: Number of batches processed
            split: Current split ('train', 'val', or 'test')
            
        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        # Calculate mean metrics
        mean_loss = loss_sum / num_batches
        mean_dice = dice_sum / num_batches
        
        # Create metrics dictionary
        metrics = {
            f"{split}_loss": mean_loss,
            f"{split}_dice": mean_dice,
        }
        
        # Print metrics
        print(f"{split.capitalize()} metrics - Loss: {mean_loss:.4f}, Dice: {mean_dice:.4f}")
        
        return metrics


    def _get_dice(self, probability: torch.Tensor, truth: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculate Dice coefficient between probability and ground truth.
        
        Args:
            probability: Predicted probabilities
            truth: Ground truth segmentation masks
            threshold: Probability threshold for binary segmentation
            
        Returns:
            float: Mean Dice coefficient
        """
        batch_size = len(truth)
        
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        return torch.mean(dice).detach().item()
