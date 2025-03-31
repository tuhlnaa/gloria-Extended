import os
import json
import torch
import numpy as np

from omegaconf import OmegaConf
from typing import Dict, List, Tuple
from sklearn.metrics import average_precision_score, roc_auc_score

from gloria import builder, gloria


class ClassificationModel:
    """
    Pure PyTorch model for image classification tasks.
    
    Supports both GLoRIA pre-trained models and other image classification models.
    Handles training, validation, and testing with appropriate metrics tracking.
    """

    def __init__(self, config: OmegaConf):
        """Initialize the classification model."""
        self.config = config
        self.lr = config.lr_scheduler.learning_rate
        self.device = config.device.device
        self.datamodule = None
        
        # Initialize the appropriate model based on configuration
        self.model = self._initialize_model()
        self.model.to(self.device)
        print(f"Used [{type(self.model).__name__}]")

        # Initialize loss function
        self.loss_fn = builder.build_loss(config)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.setup_optimization()


    def _initialize_model(self) -> torch.nn.Module:
        """Initialize the classification model based on configuration."""
        return builder.build_image_model(self.config)
        # 🛠️
        # if self.config.model.vision.model_name in gloria.available_models():
        #     return gloria.load_img_classification_model(
        #         self.config.model.vision.model_name,
        #         num_classes=self.config.model.vision.num_targets,
        #         freeze_encoder=self.config.model.vision.freeze_cnn,
        #         device=self.device
        #     )
        # else:
        #     return builder.build_image_model(self.config)


    def setup_optimization(self, datamodule=None) -> None:
        """
        Set up optimizer and learning rate scheduler.
        
        Args:
            datamodule: Optional data module for scheduler steps
        """
        self.datamodule = datamodule
        self.optimizer = builder.build_optimizer(self.config, self.lr, self.model)
        self.scheduler = builder.build_scheduler(self.config, self.optimizer, self.datamodule) # 🛠️


    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        all_logits = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            x, y = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()

            # Gradient clipping could be added here if needed:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.optimizer.clip_grad)

            self.optimizer.step()
            
            epoch_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(y.detach())
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}")
                
        # Calculate epoch metrics
        metrics = self._compute_epoch_metrics(all_logits, all_labels, epoch_loss, len(train_loader), "train")

        # 🛠️
        # # Step LR scheduler if it's epoch-based
        # if self.scheduler is not None:
        #     self.scheduler.step()
            
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
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = self._prepare_batch(batch)
                
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                
                val_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(y)
        
        # Calculate validation metrics
        metrics = self._compute_epoch_metrics(all_logits, all_labels, val_loss, len(val_loader), "val")
        return metrics


    def test(self, test_loader) -> Dict[str, float]:
        """Test the model on the test set and save results."""
        self.model.eval()
        test_loss = 0.0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = self._prepare_batch(batch)
                
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                
                test_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(y)
        
        # Calculate test metrics
        metrics = self._compute_epoch_metrics(all_logits, all_labels, test_loss, len(test_loader), "test")
        
        # Save test results
        self._save_test_results(metrics["test_auroc"], metrics["test_auprc"])
        
        return metrics


    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move batch data to the correct device."""
        x, y = batch
        return x.to(self.device), y.to(self.device)


    def _compute_epoch_metrics(
        self, 
        logits: List[torch.Tensor],
        labels: List[torch.Tensor],
        loss_sum: float,
        num_batches: int,
        split: str
    ) -> Dict[str, float]:
        """
        Compute metrics for an epoch.
        
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
        auroc_list, auprc_list = self._calculate_metrics(labels_np, probs_np)
        
        # Calculate mean metrics
        mean_auprc = np.mean(auprc_list)
        mean_auroc = np.mean(auroc_list)
        mean_loss = loss_sum / num_batches
        
        # Create metrics dictionary
        metrics = {
            f"{split}_loss": mean_loss,
            f"{split}_mean_auroc": mean_auroc,
            f"{split}_mean_auprc": mean_auprc,
            **{f"{split}_auroc_class_{i}": auroc for i, auroc in enumerate(auroc_list)},
            **{f"{split}_auprc_class_{i}": auprc for i, auprc in enumerate(auprc_list)}
        }
        
        # Print metrics
        print(f"{split.capitalize()} metrics - Loss: {mean_loss:.4f}, AUROC: {mean_auroc:.4f}, AUPRC: {mean_auprc:.4f}")
        
        return metrics


    def _calculate_metrics(
            self, 
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
                auprc_list.append(0)
                auroc_list.append(0)
            else:
                auprc_list.append(average_precision_score(class_labels, class_probs))
                auroc_list.append(roc_auc_score(class_labels, class_probs))
                
        return auroc_list, auprc_list


    def _save_test_results(self, auroc: float, auprc: float) -> None:
        """Save test results to a file."""
        results_path = os.path.join(self.config.output_dir, "results.json")
        results = {"auroc": auroc, "auprc": auprc}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        with open(results_path, "w") as fp:
            json.dump(results, fp, indent=4)
    

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