"""
Validation and testing module for GLoRIA model and other classification models.

This module handles validation, testing, and metric computation for image
classification models, with special support for GLoRIA pre-trained models.
"""

import torch
import numpy as np

from tqdm.auto import tqdm
from typing import Dict, List
from torch.utils.data import DataLoader

from gloria.utils.metrics import ClassificationMetrics


class Validator:
    """
    Validator class for image classification models.
    
    Handles validation, testing, and metrics computation.
    """
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module, config):
        self.model = model
        self.criterion = loss_fn
        self.config = config
        self.device = config.device.device


    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model on the validation set."""
        self.model.eval()
        
        # Create metrics tracker
        metrics = ClassificationMetrics(split='val')
        metrics.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate"):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)            # Forward pass
                loss = self.criterion(logits, labels)  # Compute loss

                # Binary prediction for each class (1 if probability >= 0.5, else 0)
                probabilities = torch.sigmoid(logits)
                predicted_labels = (probabilities >= 0.5).float()

                metrics.update(logits, labels, loss)

        return metrics.compute()


    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test the model on the test set and save results."""
        self.model.eval()

        # Create metrics tracker
        metrics = ClassificationMetrics(split='test')
        metrics.reset()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test"):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)            # Forward pass
                loss = self.criterion(logits, labels)  # Compute loss
                
                # Binary prediction for each class (1 if probability >= 0.5, else 0)
                probabilities = torch.sigmoid(logits)
                predicted_labels = (probabilities >= 0.5).float()

                metrics.update(logits, labels, loss)

        return metrics.compute()
