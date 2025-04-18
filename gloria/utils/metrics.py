import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

from typing import Dict, Optional
from torchmetrics.segmentation import DiceScore
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


class GloriaMetrics(nn.Module):
    """A PyTorch module for computing GLoRIA metrics."""

    def __init__(self, split: str = 'val'):
        """
        Initialize the GLoRIA metrics tracker.

        Args:
            split: The data split ('train', 'val', or 'test')
        """
        super().__init__()
        self.split = split
        
        # Reset to initialize states
        self.reset()


    def reset(self):
        """Reset all accumulated states for a new computation cycle."""
        self.metrics_sum = {
            f"{self.split}_loss": 0.0,
            f"{self.split}_global_loss": 0.0,
            f"{self.split}_local_loss": 0.0,
            f"{self.split}_local_loss_i2t": 0.0,
            f"{self.split}_local_loss_t2i": 0.0,
            f"{self.split}_global_loss_i2t": 0.0,
            f"{self.split}_global_loss_t2i": 0.0
        }
        self.num_batches = 0


    def update(self, loss_result) -> None:
        """
        Update states with loss results from a new batch.
        
        Args:
            loss_result: The loss result object containing all relevant losses
        """
        self.metrics_sum[f"{self.split}_loss"] += loss_result.total_loss.item()
        self.metrics_sum[f"{self.split}_global_loss"] += loss_result.global_loss.item()
        self.metrics_sum[f"{self.split}_local_loss"] += loss_result.local_loss.item()
        self.metrics_sum[f"{self.split}_local_loss_i2t"] += loss_result.local_loss_image_to_text.item()
        self.metrics_sum[f"{self.split}_local_loss_t2i"] += loss_result.local_loss_text_to_image.item() 
        self.metrics_sum[f"{self.split}_global_loss_i2t"] += loss_result.global_loss_image_to_text.item()
        self.metrics_sum[f"{self.split}_global_loss_t2i"] += loss_result.global_loss_text_to_image.item()
        
        self.num_batches += 1


    def compute(self) -> Dict[str, float]:
        """Compute final metrics from accumulated data."""
        if self.num_batches == 0:
            print(f"Warning: No data accumulated")
            return {key: 0.0 for key in self.metrics_sum.keys()}
        
        # Calculate mean metrics
        metrics = {key: value / self.num_batches for key, value in self.metrics_sum.items()}
        
        # Print detailed metrics summary
        print(f"{self.split.capitalize()} metrics - "
              f"Total Loss: {metrics[f'{self.split}_loss']:.4f}, "
              f"Global Loss: {metrics[f'{self.split}_global_loss']:.4f}, "
              f"Local Loss: {metrics[f'{self.split}_local_loss']:.4f}")
        
        return metrics


    def forward(
            self, 
            loss_result,
            compute_metrics: bool = False
        ) -> Optional[Dict[str, float]]:
        """
        Forward pass that updates metrics and optionally computes them.

        Args:
            loss_result: The loss result object containing all relevant losses
            compute_metrics: Whether to compute and return metrics after update

        Returns:
            Optional[Dict[str, float]]: Dictionary of metrics if compute_metrics is True, else None
        """
        self.update(loss_result)
        
        if compute_metrics:
            return self.compute()
        return None
    

class SegmentationMetrics(torch.nn.Module):
    """A PyTorch module for computing segmentation metrics."""

    def __init__(self, split: str = 'val'):
        """
        Initialize the segmentation metrics tracker.
        
        Args:
            split: The data split ('train', 'val', or 'test')
        """
        super().__init__()
        self.split = split
        
        # Reset to initialize states
        self.reset()

    def reset(self):
        """Reset all accumulated states for a new computation cycle."""
        self.logits_list = []
        self.labels_list = []
        self.loss_sum = 0.0
        self.dice_sum = 0.0
        self.num_batches = 0

    def update(
            self, 
            logits: torch.Tensor, 
            targets: torch.Tensor, 
            loss: torch.Tensor
        ) -> None:
        """Update states with predictions and targets from a new batch."""
        self.logits_list.append(logits.detach().clone())
        self.labels_list.append(targets.detach().clone())
        self.loss_sum += loss.item()
        
        # Calculate Dice score
        probs = torch.sigmoid(logits)
        dice = self._get_dice(probs, targets)
        self.dice_sum += dice
        
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        """Compute final metrics from accumulated data."""
        if not self.logits_list:
            print(f"Warning: No data accumulated")
            return {f"{self.split}_loss": 0.0, f"{self.split}_dice": 0.0}
        
        # Calculate mean metrics
        mean_loss = self.loss_sum / self.num_batches
        mean_dice = self.dice_sum / self.num_batches
        
        # Create metrics dictionary
        metrics = {
            f"{self.split}_loss": mean_loss,
            f"{self.split}_dice": mean_dice,
        }
        
        # Print metrics
        print(f"{self.split.capitalize()} metrics - Loss: {mean_loss:.4f}, Dice: {mean_dice:.4f}")
        
        return metrics

    def _get_dice(self, probability: torch.Tensor, truth: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculate Dice coefficient between probability and ground truth.
        
        Args:
            probability: Predicted probabilities
            truth: Ground truth segmentation masks
            threshold: Probability threshold for binary segmentation
            
        Returns:
            float: Mean Dice coefficient
        """
        batch_size = len(truth)
        
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        return torch.mean(dice).detach().item()
    

class SegmentationMetricsV2(torch.nn.Module):
    """A PyTorch module for computing segmentation metrics."""

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
            self.dice_metric = DiceScore(
                num_classes, 
                input_format='index', 
                include_background=False
            )
        else:
            raise ValueError("num_classes is None")

        # Reset to initialize states
        self.reset()

    def reset(self):
        """Reset all accumulated states for a new computation cycle."""
        self.logits_list = []
        self.labels_list = []
        self.loss_sum = 0.0
        self.dice_sum = 0.0
        self.num_batches = 0

    def update(
            self, 
            logits: torch.Tensor, 
            targets: torch.Tensor, 
            loss: torch.Tensor
        ) -> None:
        """Update states with predictions and targets from a new batch."""
        self.logits_list.append(logits.detach().clone().cpu())
        self.labels_list.append(targets.detach().clone().cpu())

        # Update loss tracking if provided
        if loss is not None:
            self.loss_sum += loss.item()
        
        self.num_batches += 1


    def compute(self) -> Dict[str, float]:
        """Compute final metrics from accumulated data."""
        if not self.logits_list:
            print(f"Warning: No data accumulated")
            return {f"{self.split}_loss": 0.0, f"{self.split}_dice": 0.0}
        
        # Concatenate all batch outputs
        logits_cat = torch.cat(self.logits_list)
        labels_cat = torch.cat(self.labels_list)

        # Calculate metrics
        probabilities = torch.sigmoid(logits_cat)
        predicted_mask = (probabilities > 0.5).to(torch.int64)
        labels_cat = labels_cat.unsqueeze(1).to(torch.int64)

        mean_dice = self.dice_metric(predicted_mask, labels_cat).cpu().tolist()
        mean_loss = self.loss_sum / self.num_batches

        # Create metrics dictionary
        metrics = {
            f"{self.split}_loss": mean_loss,
            f"{self.split}_dice": mean_dice,
        }
        
        # Print metrics
        print(f"{self.split.capitalize()} metrics - Loss: {mean_loss:.4f}, Dice: {mean_dice:.4f}")
        
        return metrics


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

        # Count extreme values
        exploding_threshold = 10.0  # Consider gradients above this threshold as potentially exploding
        vanishing_threshold = 1e-3  # Consider gradients below this threshold as potentially vanishing
        
        exploding_count = sum(1 for norm in self.grad_norms_before if norm > exploding_threshold)
        vanishing_count = sum(1 for norm in self.grad_norms_before if 0 < norm < vanishing_threshold)
        
        exploding_ratio = exploding_count / len(self.grad_norms_before)
        vanishing_ratio = vanishing_count / len(self.grad_norms_before)

        # Create metrics dictionary
        metrics = {
            f"{self.split}_grad_norm_before_mean": mean_before,
            f"{self.split}_grad_norm_after_mean": mean_after,
            f"{self.split}_grad_norm_before_max": max_before,
            f"{self.split}_grad_norm_after_max": max_after,
            f"{self.split}_grad_clip_ratio": mean_clip_ratio,
            f"{self.split}_exploding_grad_ratio": exploding_ratio,
            f"{self.split}_vanishing_grad_ratio": vanishing_ratio,
        }
        
        # Alert the user about potential issues
        if exploding_ratio > 0.05:
            print(f"⚠️ WARNING: Potential exploding gradients detected ({exploding_ratio*100:.1f}% of batches)")
        if vanishing_ratio > 0.05:
            print(f"⚠️ WARNING: Potential vanishing gradients detected ({vanishing_ratio*100:.1f}% of batches)")
        
        return metrics