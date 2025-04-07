
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
    
    def __init__(self, config: OmegaConf, train_loader) -> None:
        """Initialize the GLoRIA model."""
        self.config = config
        self.lr = config.lr_scheduler.learning_rate
        self.device = config.device.device
        self.train_loader = train_loader

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
    

    def setup_optimization(self) -> None:
        """
        Set up optimizer and learning rate scheduler.
        
        Args:
            datamodule: Optional data module for scheduler steps
        """
        self.optimizer = builder.build_optimizer(self.config, self.lr, self.model)
        self.scheduler = builder.build_scheduler(self.config, self.optimizer, self.train_loader)


    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        
        # Initialize all metrics trackers
        metrics_sum = {
            "train_loss": 0.0,
            "train_global_loss": 0.0,
            "train_local_loss": 0.0,
            "train_local_loss_i2t": 0.0,
            "train_local_loss_t2i": 0.0,
            "train_global_loss_i2t": 0.0,
            "train_global_loss_t2i": 0.0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            batch = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            
            # Forward pass and loss calculation
            img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents = self.model(batch)
            loss_result = self.model.compute_loss(
                img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents
            )
            
            # Backward pass
            loss_result.total_loss.backward()

            # Gradient clipping if needed
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.optimizer.clip_grad)

            self.optimizer.step()
            self.scheduler.step()
            
            # Update all metrics
            metrics_sum["train_loss"] += loss_result.total_loss.item()
            metrics_sum["train_global_loss"] += loss_result.global_loss.item()
            metrics_sum["train_local_loss"] += loss_result.local_loss.item()
            metrics_sum["train_local_loss_i2t"] += loss_result.local_loss_image_to_text.item()
            metrics_sum["train_local_loss_t2i"] += loss_result.local_loss_text_to_image.item() 
            metrics_sum["train_global_loss_i2t"] += loss_result.global_loss_image_to_text.item()
            metrics_sum["train_global_loss_t2i"] += loss_result.global_loss_text_to_image.item()
            
            # Visualize attention maps periodically if configured
            visualization_interval = self.config.misc.visualization_interval
            if visualization_interval is not None and batch_idx % visualization_interval == 0:
                imgs = batch["imgs"].cpu()
                self.model.plot_attn_maps(
                    loss_result.attn_maps, imgs, sents, epoch, batch_idx
                )

            # Print progress
            if batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss_result.total_loss.item():.6f}")
        
        # Calculate epoch metrics
        num_batches = len(train_loader)
        metrics = {key: value / num_batches for key, value in metrics_sum.items()}
        
        # Print detailed metrics
        print(f"Train metrics - Total Loss: {metrics['train_loss']:.4f}, "
              f"Global Loss: {metrics['train_global_loss']:.4f}, "
              f"Local Loss: {metrics['train_local_loss']:.4f}")
        
        return metrics


    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model on the validation set."""
        self.model.eval()

        # Initialize all metrics trackers
        metrics_sum = {
            "val_loss": 0.0,
            "val_global_loss": 0.0,
            "val_local_loss": 0.0,
            "val_local_loss_i2t": 0.0,
            "val_local_loss_t2i": 0.0,
            "val_global_loss_i2t": 0.0,
            "val_global_loss_t2i": 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._prepare_batch(batch)
                
                # Forward pass and loss calculation
                img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents = self.model(batch)
                loss_result = self.model.compute_loss(
                    img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents
                )
                
                # Update all metrics
                metrics_sum["val_loss"] += loss_result.total_loss.item()
                metrics_sum["val_global_loss"] += loss_result.global_loss.item()
                metrics_sum["val_local_loss"] += loss_result.local_loss.item()
                metrics_sum["val_local_loss_i2t"] += loss_result.local_loss_image_to_text.item()
                metrics_sum["val_local_loss_t2i"] += loss_result.local_loss_text_to_image.item() 
                metrics_sum["val_global_loss_i2t"] += loss_result.global_loss_image_to_text.item()
                metrics_sum["val_global_loss_t2i"] += loss_result.global_loss_text_to_image.item()
                
        
        # Calculate epoch metrics
        num_batches = len(val_loader)
        metrics = {key: value / num_batches for key, value in metrics_sum.items()}
        
        # Print detailed metrics
        print(f"Validation metrics - Total Loss: {metrics['val_loss']:.4f}, "
              f"Global Loss: {metrics['val_global_loss']:.4f}, "
              f"Local Loss: {metrics['val_local_loss']:.4f}")
        
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