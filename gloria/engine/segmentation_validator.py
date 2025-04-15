import torch

from typing import Dict, Tuple
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from gloria.utils.metrics import SegmentationMetricsV2

class SegmentationValidator:
    """Validator class for image segmentation models."""
    
    def __init__(self, config: OmegaConf, model: torch.nn.Module, loss_fn: torch.nn.Module):
        """Initialize the validator."""
        self.model = model
        self.criterion = loss_fn
        self.config = config
        self.device = config.device.device


    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model on the validation set."""
        self.model.eval()
        
        # Create metrics tracker
        metrics = SegmentationMetricsV2(split='val', num_classes = self.config.dataset.num_classes)
        metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate"):
                image, mask = self._prepare_batch(batch)
                logits = self.model(image)            # Forward pass
                loss = self.criterion(logits, mask)  # Compute loss
                
                metrics.update(logits, mask, loss)
        
        return metrics.compute()
    

    def test(self, test_loader) -> Dict[str, float]:
        """Test the model on the test set."""
        self.model.eval()
        
        # Create metrics tracker
        metrics = SegmentationMetricsV2(split='test')
        metrics.reset()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test"):
                x, y = self._prepare_batch(batch) 
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                metrics.update(logits, y, loss)
        
        return metrics.compute()


    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move batch data to the correct device."""
        x, y = batch
        return x.to(self.device), y.to(self.device)

