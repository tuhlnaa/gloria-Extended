import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Optional
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision


class ClassificationMetrics(nn.Module):
    """A PyTorch module for computing classification metrics using torchmetrics."""

    def __init__(self, num_classes: Optional[int] = None, split: str = 'val'):
        """
        Initialize the classification metrics tracker.

        Args:
            num_classes: Number of classes to track metrics for (can be inferred from data)
            split: The data split ('train', 'val', or 'test')
        """
        super().__init__()
        self.split = split
        self.num_classes = num_classes
        
        # Initialize metrics if num_classes is known
        if num_classes is not None:
            self.auroc_metric = MultilabelAUROC(num_labels=num_classes, average=None)
            self.auprc_metric = MultilabelAveragePrecision(num_labels=num_classes, average=None)
        else:
            self.auroc_metric = None
            self.auprc_metric = None
        
        # Reset to initialize states
        self.reset()


    def reset(self):
        """Reset all accumulated states for a new computation cycle."""
        self.logits_list = []
        self.labels_list = []
        self.loss_sum = 0.0
        self.num_batches = 0


    def update(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            loss: Optional[torch.Tensor] = None
        ) -> None:
        """Update states with predictions and targets from a new batch."""
        self.logits_list.append(preds.detach().clone())
        self.labels_list.append(targets.detach().clone())

        # Update loss tracking if provided
        if loss is not None:
            self.loss_sum += loss.item()
        
        self.num_batches += 1
        
        # If num_classes wasn't specified, infer it from the first batch and initialize metrics
        if self.num_classes is None and targets.ndim > 1:
            self.num_classes = targets.shape[1]
            self.auroc_metric = MultilabelAUROC(num_labels=self.num_classes, average=None)
            self.auprc_metric = MultilabelAveragePrecision(num_labels=self.num_classes, average=None)


    def compute(self) -> Dict[str, float]:
        """Compute final metrics from accumulated data."""
        if not self.logits_list or self.num_classes is None:
            print(f"Warning: No data accumulated or num_classes not set")
            return {f"{self.split}_loss": 0.0}
        
        # Concatenate all batch outputs
        logits_cat = torch.cat(self.logits_list)
        labels_cat = torch.cat(self.labels_list)
        probabilities = torch.sigmoid(logits_cat)
        
        # Calculate metrics
        labels_cat = labels_cat.to(torch.int32)
        auroc_list = self.auroc_metric(probabilities, labels_cat).cpu().tolist()
        auprc_list = self.auprc_metric(probabilities, labels_cat).cpu().tolist()

        # Calculate mean metrics, filtering out NaN values (torchmetrics may return NaN for classes with all same labels)
        mean_auprc = float(np.mean([x for x in auprc_list if not np.isnan(x) and x != 0.0]))
        mean_auroc = float(np.mean([x for x in auroc_list if not np.isnan(x) and x != 0.0]))
        mean_loss = self.loss_sum / self.num_batches if self.num_batches > 0 else 0.0
        
        # Create metrics dictionary
        metrics = {
            f"{self.split}_loss": mean_loss,
            f"{self.split}_mean_auroc": mean_auroc,
            f"{self.split}_mean_auprc": mean_auprc,
        }
        
        # Add per-class metrics
        for i, (auroc, auprc) in enumerate(zip(auroc_list, auprc_list)):
            metrics[f"{self.split}_auroc_class_{i}"] = float(auroc)
            metrics[f"{self.split}_auprc_class_{i}"] = float(auprc)
        
        # Print metrics summary
        print(f"{self.split.capitalize()} metrics - Loss: {mean_loss:.4f}, "
              f"AUROC: {mean_auroc:.4f}, AUPRC: {mean_auprc:.4f}")
        
        return metrics


    def forward(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            loss: Optional[torch.Tensor] = None, 
            compute_metrics: bool = False
        ) -> Optional[Dict[str, float]]:
        """
        Forward pass that updates metrics and optionally computes them.

        Args:
            preds: Model predictions/logits
            targets: Ground truth labels
            loss: Optional loss value for this batch
            compute_metrics: Whether to compute and return metrics after update

        Returns:
            Optional[Dict[str, float]]: Dictionary of metrics if compute_metrics is True, else None
        """
        self.update(preds, targets, loss)
        
        if compute_metrics:
            return self.compute()
        return None