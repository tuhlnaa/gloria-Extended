import os
import json
import torch
import numpy as np

from pytorch_lightning.core import LightningModule
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Dict, List, Tuple

from .. import builder
from .. import gloria

class ClassificationModel(LightningModule):
    """
    PyTorch Lightning module for image classification tasks.
    
    Supports both GLoRIA pre-trained models and other image classification models.
    Handles training, validation, and testing with appropriate metrics tracking.
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.lr = config.lightning.trainer.lr
        self.dm = None
        
        # Initialize the appropriate model based on configuration
        self.model = self._initialize_model()
        
        # Initialize loss function
        self.loss = builder.build_loss(config)
        

    def _initialize_model(self):
        """
        Initialize the classification model based on configuration.
        
        Returns:
            torch.nn.Module: Initialized model
        """
        if self.cfg.model.vision.model_name in gloria.available_models():
            return gloria.load_img_classification_model(
                self.cfg.model.vision.model_name,
                num_classes=self.cfg.model.vision.num_targets,
                freeze_encoder=self.cfg.model.vision.freeze_cnn,
                device=self.cfg.device
            )
        else:
            return builder.build_image_model(self.cfg)


    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dict: Configuration for optimizer and scheduler
        """
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def training_step(self, batch, batch_idx):
        """Process a single training batch."""
        return self._shared_step(batch, "train")


    def validation_step(self, batch, batch_idx):
        """Process a single validation batch."""
        return self._shared_step(batch, "val")


    def test_step(self, batch, batch_idx):
        """Process a single test batch."""
        return self._shared_step(batch, "test")


    def training_epoch_end(self, outputs):
        """Process the end of a training epoch."""
        self._shared_epoch_end(outputs, "train")


    def validation_epoch_end(self, outputs):
        """Process the end of a validation epoch."""
        self._shared_epoch_end(outputs, "val")


    def test_epoch_end(self, outputs):
        """Process the end of a test epoch."""
        self._shared_epoch_end(outputs, "test")


    def _shared_step(self, batch, split: str) -> Dict:
        """
        Shared step function for training, validation, and testing.
        
        Args:
            batch: Input batch containing images and labels
            split: Current split ('train', 'val', or 'test')
            
        Returns:
            Dict: Dictionary containing loss, logits, and labels
        """
        x, y = batch
        logit = self.model(x)
        loss = self.loss(logit, y)

        # Log only on step during training, otherwise only on epoch
        log_on_step = split == "train"
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_on_step,
            logger=True,
            prog_bar=True,
        )

        return {"loss": loss, "logit": logit, "y": y}


    def _shared_epoch_end(self, step_outputs: List[Dict], split: str) -> None:
        """
        Shared epoch end function for computing metrics.
        
        Args:
            step_outputs: Outputs from each step in the epoch
            split: Current split ('train', 'val', or 'test')
        """
        # Concatenate all batch outputs
        logits = torch.cat([x["logit"] for x in step_outputs])
        labels = torch.cat([x["y"] for x in step_outputs])
        probabilities = torch.sigmoid(logits)

        # Convert to numpy arrays for sklearn metrics
        labels_np = labels.detach().cpu().numpy()
        probs_np = probabilities.detach().cpu().numpy()

        # Calculate metrics per class
        auroc_list, auprc_list = self._calculate_metrics(labels_np, probs_np)

        # Log mean metrics
        mean_auprc = np.mean(auprc_list)
        mean_auroc = np.mean(auroc_list)
        self.log(f"{split}_auroc", mean_auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_auprc", mean_auprc, on_epoch=True, logger=True, prog_bar=True)

        # Save test results
        if split == "test":
            self._save_test_results(mean_auroc, mean_auprc)
            

    def _calculate_metrics(self, labels: np.ndarray, probabilities: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Calculate AUROC and AUPRC for each class.
        
        Args:
            labels: Ground truth labels
            probabilities: Predicted probabilities
            
        Returns:
            Tuple[List[float], List[float]]: Lists of AUROC and AUPRC values per class
        """
        auroc_list, auprc_list = [], []
        
        for i in range(labels.shape[1]):
            class_labels = labels[:, i]
            class_probs = probabilities[:, i]

            if np.isnan(class_probs).any():
                auprc_list.append(0)
                auroc_list.append(0)
            else:
                auprc_list.append(average_precision_score(class_labels, class_probs))
                auroc_list.append(roc_auc_score(class_labels, class_probs))
                
        return auroc_list, auprc_list
            

    def _save_test_results(self, auroc: float, auprc: float) -> None:
        """Save test results to a file."""
        results_path = os.path.join(self.cfg.output_dir, "results.json")
        results = {"auroc": auroc, "auprc": auprc}
        
        with open(results_path, "w") as fp:
            json.dump(results, fp)