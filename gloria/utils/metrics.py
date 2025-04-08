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
    

class GradientMonitor(nn.Module):
    """A PyTorch module for monitoring gradient norms during training."""

    def __init__(self, split: str = 'train', norm_type: int = 2):
        """
        Initialize the gradient monitor.

        Args:
            split: The data split (typically 'train')
            norm_type: Order of the norm (typically 2 for L2 norm)
        """
        super().__init__()
        self.split = split
        self.norm_type = norm_type
        self.reset()


    def reset(self):
        """Reset all accumulated states for a new computation cycle."""
        self.grad_norms_before = []
        self.grad_norms_after = []
        self.num_steps = 0


    def update_before_clip(self, model):
        """Record gradient norm before clipping."""
        norm = self.get_grad_norm(model, self.norm_type)
        self.grad_norms_before.append(norm)
        return norm


    def update_after_clip(self, model):
        """Record gradient norm after clipping."""
        norm = self.get_grad_norm(model, self.norm_type)
        self.grad_norms_after.append(norm)
        self.num_steps += 1
        return norm


    def get_grad_norm(self, model, norm_type=2):
        """Calculate the gradient norm of model parameters."""
        parameters = [p for p in model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type
        )
        return total_norm.item()


    def compute(self) -> Dict[str, float]:
        """Compute gradient statistics from accumulated data."""
        if not self.grad_norms_before or not self.grad_norms_after:
            print(f"Warning: No gradient data accumulated")
            return {}
        
        # Calculate statistics
        mean_before = float(np.mean(self.grad_norms_before))
        mean_after = float(np.mean(self.grad_norms_after))
        max_before = float(np.max(self.grad_norms_before))
        max_after = float(np.max(self.grad_norms_after))
        
        # Calculate mean clipping ratio
        clip_ratios = [after/before if before > 0 else 1.0 
                      for before, after in zip(self.grad_norms_before, self.grad_norms_after)]
        mean_clip_ratio = float(np.mean(clip_ratios))
        
        # Create metrics dictionary
        metrics = {
            f"{self.split}_grad_norm_before_mean": mean_before,
            f"{self.split}_grad_norm_after_mean": mean_after,
            f"{self.split}_grad_norm_before_max": max_before,
            f"{self.split}_grad_norm_after_max": max_after,
            f"{self.split}_grad_clip_ratio": mean_clip_ratio,
        }
        
        # Print summary
        print(f"Gradient stats - Before: {mean_before:.4f} (max: {max_before:.4f}), "
              f"After: {mean_after:.4f} (max: {max_after:.4f}), "
              f"Clip ratio: {mean_clip_ratio:.4f}")
        
        return metrics