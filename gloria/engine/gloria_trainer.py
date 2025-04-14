import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Any, Dict, Union
from torch.utils.data import DataLoader
from rich import print

from gloria import builder
from gloria.engine.base_trainer import BaseTrainer
from gloria.utils.metrics import GloriaMetrics, GradientMonitor
from gloria.utils.utils import plot_attn_maps

class GloriaTrainer(BaseTrainer):
    """
    Pure PyTorch model for GLoRIA pretraining.
    
    GLoRIA (Global-Local Representation Learning Framework) is designed for 
    multimodal medical image recognition with label efficiency.
    """

    def __init__(self, config: OmegaConf, train_loader: DataLoader):
        super().__init__(config, train_loader)
        
        # Initialize the model
        self.model = self._initialize_model().to(self.device)
        print(f"Using model: [{type(self.model).__name__}]")

        # Initialize loss function
        self.criterion = builder.build_loss(config)


    def _initialize_model(self) -> torch.nn.Module:
        """Initialize the model based on configuration."""
        return builder.build_gloria_model(self.config)


    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        
        # Create metrics tracker
        metrics = GloriaMetrics(split='train')
        metrics.reset()
        grad_monitor = GradientMonitor(split='train')
        grad_monitor.reset()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch: {epoch}")):
            batch = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            
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
            
            # Backward pass
            loss_result.total_loss.backward()

            # Monitor gradients before clipping
            _ = grad_monitor.update_before_clip(self.model)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.optimizer.clip_grad
            )

            # Monitor gradients after clipping
            _ = grad_monitor.update_after_clip(self.model)

            self.optimizer.step()
            
            # Update learning rate scheduler if needed
            if self.config.lr_scheduler.name == "LinearWarmupCosine":
                self.scheduler.step()
            
            # Update metrics
            metrics.update(loss_result)
            
            # Visualize attention maps periodically if configured
            visualization_interval = self.config.misc.visualization_interval
            if visualization_interval is not None and batch_idx % visualization_interval == 0:
                imgs = batch["imgs"].cpu()
                plot_attn_maps(self.config, loss_result.attn_maps, imgs, sents, epoch, batch_idx)

        # Compute metrics
        class_metrics = metrics.compute()
        grad_metrics = grad_monitor.compute()
        combined_metrics = {**class_metrics, **grad_metrics}

        return combined_metrics


    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to the correct device."""
        prepared_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
                
        return prepared_batch


    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Resume training from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if hasattr(self.model, 'img_encoder') and hasattr(self.model, 'text_encoder'):
            print("[bold blue]Successfully created a complete model, transferring to current model...[/bold blue]")
        else:
            raise ValueError(f"Could not create a complete model with both encoders")
        
        model_state_dict = builder.normalize_model_state_dict(checkpoint)
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epochs = checkpoint.get("epochs", 0)
        best_val_metrics = checkpoint.get("best_metrics", float("inf"))

        val_loss = best_val_metrics["val_loss"]
        print(f"[bold blue]Loaded checkpoint from '{checkpoint_path}' at epoch {start_epochs} with loss {val_loss:.4f}.[/bold blue]")

        return start_epochs, best_val_metrics