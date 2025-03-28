"""
Validation and testing module for GLoRIA model and other classification models.

This module handles validation, testing, and metric computation for image
classification models, with special support for GLoRIA pre-trained models.
"""

import os
import json
import torch
import numpy as np

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score


class Validator:
    """
    Validator class for image classification models.
    
    Handles validation, testing, and metrics computation.
    """
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module, config):
        self.model = model
        self.criterion = loss_fn
        self.config = config
        self.device = config.device.device


    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model on the validation set."""
        self.model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Training"):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)            # Forward pass
                loss = self.criterion(logits, labels)  # Compute loss
                
                val_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(labels)
        
        # Calculate validation metrics
        metrics = compute_classification_metrics(
            all_logits, 
            all_labels, 
            val_loss, 
            len(val_loader), 
            "val"
        )
        
        return metrics


    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Test the model on the test set and save results."""
        self.model.eval()
        test_loss = 0.0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Training"):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images)            # Forward pass
                loss = self.criterion(logits, labels)  # Compute loss
                
                test_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(labels)
        
        # Calculate test metrics
        metrics = compute_classification_metrics(
            all_logits, 
            all_labels, 
            test_loss, 
            len(test_loader), 
            "test"
        )
        
        # Save test results
        self._save_test_results(metrics["test_mean_auroc"], metrics["test_mean_auprc"])
        
        return metrics


    def _save_test_results(self, auroc: float, auprc: float) -> None:
        """
        Save test results to a JSON file.
        
        Args:
            auroc: Area Under ROC Curve score
            auprc: Area Under Precision-Recall Curve score
        """
        results_path = os.path.join(self.config.output_dir, "results.json")
        results = {
            "auroc": float(auroc),  # Convert to Python float for JSON serialization
            "auprc": float(auprc)   # Convert to Python float for JSON serialization
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        with open(results_path, "w") as fp:
            json.dump(results, fp, indent=4)
        
        print(f"Test results saved to {results_path}")


def compute_classification_metrics(
        logits: List[torch.Tensor],
        labels: List[torch.Tensor],
        loss_sum: float,
        num_batches: int,
        split: str
    ) -> Dict[str, float]:
    """
    Compute classification metrics from model outputs.
    
    Args:
        logits: List of model logits from each batch
        labels: List of labels from each batch
        loss_sum: Sum of losses across batches
        num_batches: Number of batches processed
        split: Current split ('train', 'val', or 'test')
        
    Returns:
        Dict[str, float]: Dictionary of computed metrics
    """
    # Concatenate all batch outputs
    logits_cat = torch.cat(logits)
    labels_cat = torch.cat(labels)
    probabilities = torch.sigmoid(logits_cat)
    
    # Convert to numpy arrays for sklearn metrics
    labels_np = labels_cat.cpu().numpy()
    probs_np = probabilities.cpu().numpy()
    
    # Calculate metrics per class
    auroc_list, auprc_list = _calculate_per_class_metrics(labels_np, probs_np)
    
    # Calculate mean metrics
    mean_auprc = float(np.mean(auprc_list))
    mean_auroc = float(np.mean(auroc_list))
    mean_loss = loss_sum / num_batches
    
    # Create metrics dictionary
    metrics = {
        f"{split}_loss": mean_loss,
        f"{split}_mean_auroc": mean_auroc,
        f"{split}_mean_auprc": mean_auprc,
    }
    
    # Add per-class metrics
    for i, (auroc, auprc) in enumerate(zip(auroc_list, auprc_list)):
        metrics[f"{split}_auroc_class_{i}"] = float(auroc)
        metrics[f"{split}_auprc_class_{i}"] = float(auprc)
    
    # Print metrics summary
    print(f"{split.capitalize()} metrics - Loss: {mean_loss:.4f}, "
          f"AUROC: {mean_auroc:.4f}, AUPRC: {mean_auprc:.4f}")
    
    return metrics


def _calculate_per_class_metrics(
        labels: np.ndarray, 
        probabilities: np.ndarray
    ) -> Tuple[List[float], List[float]]:
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
        
        # Skip calculation if there are NaN values or only one class present
        if np.isnan(class_probs).any() or len(np.unique(class_labels)) < 2:
            auprc_list.append(0.0)
            auroc_list.append(0.0)
        else:
            try:
                auprc = average_precision_score(class_labels, class_probs)
                auroc = roc_auc_score(class_labels, class_probs)
                auprc_list.append(auprc)
                auroc_list.append(auroc)
            except Exception as e:
                print(f"Error calculating metrics for class {i}: {e}")
                auprc_list.append(0.0)
                auroc_list.append(0.0)
                
    return auroc_list, auprc_list