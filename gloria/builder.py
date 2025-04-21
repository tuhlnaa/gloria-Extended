"""
GLoRIA: A Global-Local Representation Learning Framework for medical images.

This module provides factory functions for building GLoRIA models and related components.
It supports pretraining, classification, and segmentation phases.
"""
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp

from typing import Dict, Any, List, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR
)
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from torchvision.ops import sigmoid_focal_loss

from gloria.models.vision_model import GloriaImageClassifier
from gloria.utils.losses import CombinedBinaryLoss, GloriaLoss

from . import models
from . import loss


FEATURE_DIMENSIONS = {
    "resnet50": 2048, 
    "resnet18": 2048
}


def build_gloria_model(config: Dict[str, Any]):
    """Build a GLoRIA model from configuration."""
    return models.gloria_model.GLoRIA(config)


def build_gloria_from_checkpoint(checkpoint_path: str):
    """Build a GLoRIA model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["hyper_parameters"]
    checkpoint_state_dict = checkpoint["state_dict"]

    # Fix checkpoint keys by removing "gloria." prefix if present
    fixed_state_dict = {
        k.split("gloria.")[-1]: v for k, v in checkpoint_state_dict.items()
    }
    
    gloria_model = build_gloria_model(config)
    gloria_model.load_state_dict(fixed_state_dict)
    
    return gloria_model


def normalize_model_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalizes state dictionary keys for compatibility."""
    
    if "state_dict" in state_dict.keys():
        normalized_dict = {}
        for key, value in state_dict["state_dict"].items():
            # Remove 'gloria.' prefix if present
            new_key = key.split("gloria.")[-1]
            normalized_dict[new_key] = value
            
        # Remove problematic keys that might cause issues during loading
        if "text_encoder.model.embeddings.position_ids" in normalized_dict:
            del normalized_dict["text_encoder.model.embeddings.position_ids"]

        return normalized_dict
    else:
        return state_dict["model_state_dict"]
    

def build_image_model(config: Dict[str, Any]):
    """Build the appropriate image model based on the configuration phase."""
    phase = config.phase.lower()
    image_model_class = models.IMAGE_MODELS[phase]

    if config.model.transfer_checkpoint is not None:
        checkpoint = torch.load(config.model.transfer_checkpoint, map_location="cpu")
        model_state_dict = normalize_model_state_dict(checkpoint)

        gloria_model = build_gloria_model(config)
        gloria_model.load_state_dict(model_state_dict)

        # Load pretrained image encoder
        image_encoder = copy.deepcopy(gloria_model.img_encoder)
        del gloria_model  # Free up memory

        # Extract required parameters from config
        num_classes = config.model.vision.num_targets
        feature_dim = FEATURE_DIMENSIONS[config.model.vision.model_name]
        freeze_encoder = config.model.vision.freeze_cnn 

        # Instantiate GloriaImageClassifier with the correct parameters
        return GloriaImageClassifier(
            image_encoder=image_encoder,
            num_classes=num_classes,
            feature_dim=feature_dim,
            freeze_encoder=freeze_encoder
        )
    else:
        return image_model_class(config)


def build_image_modelV0(config: Dict[str, Any]):
    """Build the appropriate image model based on the configuration phase."""
    phase = config.phase.lower()
    image_model_class = models.IMAGE_MODELS[phase]
    return image_model_class(config) 

    
def build_text_model(config: Dict[str, Any]):
    """Build a BERT-based text encoder model."""
    return models.text_model.BertEncoder(config)


def build_optimizer(config: Dict[str, Any], lr: float, model: nn.Module) -> Optimizer:
    """Build an optimizer for the model based on configuration."""
    # Only include parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer_name = config.optimizer.name
    weight_decay = config.optimizer.weight_decay
    
    optimizers = {
        "SGD": lambda: torch.optim.SGD(
            trainable_params, 
            lr=lr, 
            momentum=config.momentum, 
            weight_decay=weight_decay
        ),
        "Adam": lambda: torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.5, 0.999),
        ),
        "AdamW": lambda: torch.optim.AdamW(
            trainable_params, 
            lr=lr, 
            weight_decay=weight_decay
        ),
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
    return optimizers[optimizer_name]()


def build_scheduler(
        config: Dict[str, Any], 
        optimizer: Optimizer, 
        train_loader
    ):
    """
    Build a learning rate scheduler based on configuration.
    
    Args:
        config: Configuration dictionary containing scheduler settings
        optimizer: PyTorch optimizer to schedule
    """
    scheduler_name = config.lr_scheduler.name
    
    # Calculate training steps with validation frequency
    num_training_steps_per_epoch = len(train_loader)
    total_steps = num_training_steps_per_epoch * config.lr_scheduler.epochs
    warmup_steps = int(total_steps * config.lr_scheduler.warmup_ratio)

    if scheduler_name == "warmup":
        def warmup_lr_lambda(epoch):
            if epoch <= 3:
                return 0.001 + epoch * 0.003
            if epoch >= 22:
                return 0.01 * (1 - epoch / 200.0) ** 0.9
            return 0.01
        
        scheduler = LambdaLR(optimizer, warmup_lr_lambda)
    elif scheduler_name == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
    elif scheduler_name == "LinearWarmupCosine":
        scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_steps,
        max_epochs=total_steps,
        warmup_start_lr=config.lr_scheduler.learning_rate * 0.1,
        eta_min=1e-6)
    elif scheduler_name == None:
        return None  # Early return if no scheduler specified
    else:
        raise ValueError(f"Unsupported loss function: {scheduler_name}")

    
    return scheduler


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Build a loss function based on configuration."""
    loss_type = config.criterion.name
    
    loss_functions = {
        "MixedLossV2": lambda: CombinedBinaryLoss(config),
        "DiceLossV2": lambda: smp.losses.DiceLoss(mode=config.criterion.label_mode),
        "GloriaLoss": lambda: GloriaLoss(config),
        "BCE": lambda: _create_bce_loss(config),
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unsupported loss function: {loss_type}")
        
    return loss_functions[loss_type]()


def _create_bce_loss(config: Dict[str, Any]) -> nn.BCEWithLogitsLoss:
    """Helper function to create BCE loss with optional class weights."""
    if hasattr(config.criterion, 'class_weights') and config.criterion.class_weights is not None:
        weight = torch.Tensor(config.criterion.class_weights)
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    else:
        return nn.BCEWithLogitsLoss()
