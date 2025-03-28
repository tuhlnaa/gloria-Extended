import torch
from typing import Dict, Tuple, Any, List
from pytorch_lightning import LightningModule

from .. import builder
from .. import utils


class PretrainModel(LightningModule):
    """
    Lightning module for the GLoRIA model pretraining.
    
    GLoRIA (Global-Local Representation Learning Framework) is designed for 
    multimodal medical image recognition with label efficiency.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(self.config)
        self.gloria = builder.build_gloria_model(config)
        self.learning_rate = config.lightning.trainer.lr
        self.data_module = None


    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = builder.build_optimizer(self.config, self.learning_rate, self.gloria)
        scheduler = builder.build_scheduler(self.config, optimizer, self.data_module)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        loss, attn_maps, sents = self.shared_step(batch, "train")

        # Visualize attention maps periodically if configured
        update_interval = self.config.train.update_interval
        if update_interval is not None and batch_idx % update_interval == 0:
            imgs = batch["imgs"].cpu()
            self.gloria.plot_attn_maps(
                attn_maps, imgs, sents, self.current_epoch, batch_idx
            )
            
        return loss


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute validation step."""
        loss, _, _ = self.shared_step(batch, "val")
        return loss


    def shared_step(self, batch: Dict[str, torch.Tensor], split: str) -> Tuple[torch.Tensor, Any, List[str]]:
        """
        Shared step logic used by both training and validation steps.
        
        Args:
            batch: Dictionary containing batch data
            split: String indicating the current split ("train" or "val")
            
        Returns:
            Tuple containing (loss, attention_maps, sentences)
        """
        # Forward pass through GLoRIA model
        img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents = self.gloria(batch)
        
        # Calculate loss and get attention maps
        loss, attn_maps = self.gloria.compute_loss(
            img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents
        )

        # Log metrics
        log_step = split == "train"
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_step,
            logger=True,
            prog_bar=True,
        )

        return loss, attn_maps, sents