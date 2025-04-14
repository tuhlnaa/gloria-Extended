"""
Validation and testing module for GLoRIA model and other classification models.

This module handles validation, testing, and metric computation for image
classification models, with special support for GLoRIA pre-trained models.
"""

from omegaconf import OmegaConf
import torch
import numpy as np

from tqdm.auto import tqdm
from typing import Any, Dict, List
from torch.utils.data import DataLoader

from gloria.utils.metrics import GloriaMetrics


class GloriaValidator:
    """
    Validator class for image classification models.
    
    Handles validation, testing, and metrics computation.
    """
    def __init__(self, config: OmegaConf, model: torch.nn.Module, loss_fn: torch.nn.Module):
        self.model = model
        self.criterion = loss_fn
        self.config = config
        self.device = config.device.device


    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model on the validation set."""
        self.model.eval().to(self.device)

        # Create metrics tracker
        metrics = GloriaMetrics(split='val')
        metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate"):
                batch = self._prepare_batch(batch)
                
                # Forward pass
                img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents = self.model(batch)

                # Compute loss
                loss_result = self.criterion(
                    img_emb_local=img_emb_local,
                    img_emb_global=img_emb_global,
                    text_emb_local=text_emb_local,
                    text_emb_global=text_emb_global,
                    sents=sents
                )
                    
                # Update metrics
                metrics.update(loss_result)
                
        # Compute and return metrics for the validation set
        computed_metrics = metrics.compute()
        
        return computed_metrics
    

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to the correct device."""
        prepared_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
                
        return prepared_batch

