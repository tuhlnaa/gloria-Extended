
import os
import torch

from omegaconf import OmegaConf
from typing import Dict, Tuple, Any, List

from gloria import builder
#from .. import utils


class GLoRIAModel:
    """
    Pure PyTorch model for GLoRIA pretraining.
    
    GLoRIA (Global-Local Representation Learning Framework) is designed for 
    multimodal medical image recognition with label efficiency.
    """
    
    def __init__(self, config: OmegaConf) -> None:
        """Initialize the GLoRIA model."""
        self.config = config
        self.lr = config.lr_scheduler.learning_rate
        self.device = config.device.device
        self.datamodule = None

        # Initialize the appropriate model based on configuration
        self.model = self._initialize_model()
        self.model.to(self.device)
        print(f"Used [{type(self.model).__name__}]")

        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.setup_optimization()


    def _initialize_model(self) -> torch.nn.Module:
        """Initialize the GLoRIA model based on configuration."""
        return builder.build_gloria_model(self.config)
    

    def setup_optimization(self, datamodule=None) -> None:
        """
        Set up optimizer and learning rate scheduler.
        
        Args:
            datamodule: Optional data module for scheduler steps
        """
        self.datamodule = datamodule
        self.optimizer = builder.build_optimizer(self.config, self.lr, self.model)
        self.scheduler = builder.build_scheduler(self.config, self.optimizer, self.datamodule)


    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            
            # Forward pass and loss calculation
            img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents = self.model(batch)
            loss, attn_maps = self.model.compute_loss(
                img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents
            )
            
            # Backward pass
            loss.backward()

            # Gradient clipping if needed
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.optimizer.clip_grad)

            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Visualize attention maps periodically if configured
            update_interval = self.config.train.update_interval
            if update_interval is not None and batch_idx % update_interval == 0:
                imgs = batch["imgs"].cpu()
                self.model.plot_attn_maps(
                    attn_maps, imgs, sents, epoch, batch_idx
                )
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}")
        
        # Calculate epoch metrics
        metrics = {
            "train_loss": epoch_loss / len(train_loader)
        }
        
        print(f"Train metrics - Loss: {metrics['train_loss']:.4f}")
        
        return metrics


    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model on the validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dict[str, float]: Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Forward pass and loss calculation
                img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents = self.model(batch)
                loss, _ = self.model.compute_loss(
                    img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents
                )
                
                val_loss += loss.item()
        
        # Calculate validation metrics
        metrics = {
            "val_loss": val_loss / len(val_loader)
        }
        
        print(f"Validation metrics - Loss: {metrics['val_loss']:.4f}")
        
        return metrics
    

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to the correct device."""
        prepared_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
                
        return prepared_batch


    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, path)
    

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])