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
from gloria.utils.losses import GloriaLoss

from . import models
from . import lightning
from . import datasets
from . import loss


FEATURE_DIMENSIONS = {
    "resnet50": 2048, 
    "resnet18": 2048
}


class CombinedBinaryLoss(nn.Module):
    def __init__(self, config, dice_weight=0.5, focal_weight=0.5, bce_weight=0.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        """
        For binary medical image segmentation tasks, the combination of Dice and Focal Loss often provides the best results because:
        Dice Loss directly optimizes the evaluation metric
        Focal Loss addresses class imbalance while providing stable gradients
        Including BCE is less critical when using Focal Loss, as Focal Loss is an enhanced version of BCE designed specifically for class imbalance.
        """
        super().__init__()
        self.config = config
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # For binary segmentation
        self.dice = smp.losses.DiceLoss(mode=config.criterion.label_mode)
        
        # Only used if bce_weight > 0
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, y_pred, y_true):
        # For binary segmentation:
        # - y_pred should be shape (B, 1, H, W) or (B, H, W) with logits
        # - y_true should be shape (B, 1, H, W) or (B, H, W) with values 0 or 1

        # Calculate individual losses
        dice_loss = self.dice(y_pred, y_true)
        
        # Focal loss
        if self.config.criterion.label_mode == 'binary':
            y_pred = y_pred.squeeze(1)

        focal_loss = sigmoid_focal_loss(
            y_pred, y_true,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean"
        )
        
        # Calculate BCE only if weight > 0
        bce_loss = self.bce(y_pred, y_true) if self.bce_weight > 0 else 0
        
        # Combine losses with weights
        return (self.dice_weight * dice_loss + 
                self.focal_weight * focal_loss + 
                self.bce_weight * bce_loss)
    

def build_data_module(config: Dict[str, Any]):
    """Build the appropriate data module based on the configuration."""
    phase = config.phase.lower()
    
    if phase == "pretrain":
        data_module_class = datasets.DATA_MODULES["pretrain"]
    else:
        dataset_name = config.data.dataset.lower()
        data_module_class = datasets.DATA_MODULES[dataset_name]
    
    return data_module_class(config)


def build_lightning_model(config: Dict[str, Any], data_module: Optional[Any] = None):
    """Build the appropriate PyTorch Lightning module based on the configuration."""
    phase = config.phase.lower()
    lightning_module_class = lightning.LIGHTNING_MODULES[phase]
    lightning_module = lightning_module_class(config)
    
    if data_module is not None:
        lightning_module.dm = data_module
    
    return lightning_module


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
        #"DiceLoss": lambda: loss.segmentation_loss.DiceLoss(),
        "MixedLossV2": lambda: CombinedBinaryLoss(config),
        "DiceLossV2": lambda: smp.losses.DiceLoss(mode=config.criterion.label_mode),
        "FocalLoss": lambda: loss.segmentation_loss.FocalLoss(),
        "MixedLoss": lambda: loss.segmentation_loss.MixedLoss(alpha=config.criterion.alpha),
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


def build_transformation(config: Dict[str, Any], split: str) -> transforms.Compose:
    """
    Build image transformations based on configuration and dataset split.
    
    Args:
        config: Configuration dictionary containing transformation settings
        split: Dataset split ('train' or other)
        
    Returns:
        Composed PyTorch transformations
    """
    transform_list = []
    
    # Apply augmentations only for training
    if split == "train":
        transform_list.extend(_build_train_transforms(config))
    else:
        # For validation/test, only use center crop if random crop is defined
        if hasattr(config.transforms, 'random_crop') and config.transforms.random_crop is not None:
            transform_list.append(transforms.CenterCrop(config.transforms.random_crop.crop_size))
    
    # Common transformations for all splits
    transform_list.append(transforms.ToTensor())
    
    # Add normalization if specified
    if hasattr(config.transforms, 'norm') and config.transforms.norm is not None:
        transform_list.append(_get_normalization(config.transforms.norm))
    
    return transforms.Compose(transform_list)


def _build_train_transforms(config: Dict[str, Any]) -> List[nn.Module]:
    """Helper function to build training-specific transformations."""
    train_transforms = []
    
    if hasattr(config.transforms, 'random_crop') and config.transforms.random_crop is not None:
        train_transforms.append(
            transforms.RandomCrop(config.transforms.random_crop.crop_size)
        )
    
    if hasattr(config.transforms, 'random_horizontal_flip') and config.transforms.random_horizontal_flip is not None:
        train_transforms.append(
            transforms.RandomHorizontalFlip(p=config.transforms.random_horizontal_flip)
        )
    
    if hasattr(config.transforms, 'random_affine') and config.transforms.random_affine is not None:
        train_transforms.append(
            transforms.RandomAffine(
                degrees=config.transforms.random_affine.degrees,
                translate=list(config.transforms.random_affine.translate),
                scale=list(config.transforms.random_affine.scale),
            )
        )
    
    if hasattr(config.transforms, 'color_jitter') and config.transforms.color_jitter is not None:
        train_transforms.append(
            transforms.ColorJitter(
                brightness=list(config.transforms.color_jitter.bightness),
                contrast=list(config.transforms.color_jitter.contrast),
            )
        )
    
    return train_transforms


def _get_normalization(norm_type: str) -> transforms.Normalize:
    """Helper function to get the appropriate normalization transform."""
    if norm_type == "imagenet":
        return transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )
    elif norm_type == "half":
        return transforms.Normalize(
            mean=(0.5, 0.5, 0.5), 
            std=(0.5, 0.5, 0.5)
        )
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")


def print_model_structure(model, prefix='', max_params=10):
    """
    Print the structure of a PyTorch model with parameter shapes (limited output).
    
    Args:
        model: PyTorch model
        prefix: String prefix for nested components
        max_params: Maximum number of parameters to print
    """
    print(f"\n{'='*80}\nMODEL STRUCTURE ANALYSIS (TOP LEVEL)\n{'='*80}")
    
    # Get all parameters
    params = list(model.named_parameters())
    total_params = len(params)
    
    # Print limited number of parameters
    print(f"Total parameters: {total_params}")
    print(f"Showing first {min(max_params, total_params)} parameters:")
    
    for i, (name, param) in enumerate(params):
        if i < max_params:
            print(f"{prefix}{name}: {param.shape}")
        else:
            print(f"... and {total_params - max_params} more parameters")
            break
    
    # Print only top-level modules
    print(f"\nTop-level modules:")
    for name, _ in model.named_children():
        print(f"{prefix}Module: {name}")


def print_checkpoint_structure(checkpoint, max_keys=10):
    """
    Print the structure of a PyTorch checkpoint (limited output).
    
    Args:
        checkpoint: The loaded checkpoint dictionary
        max_keys: Maximum number of keys to print
    """
    print(f"\n{'='*80}\nCHECKPOINT STRUCTURE ANALYSIS\n{'='*80}")
    
    # Check if it's a state_dict directly or needs to be extracted
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Checkpoint contains 'state_dict' key")
    else:
        state_dict = checkpoint
        print("Checkpoint appears to be a direct state_dict")
    
    # Print limited number of keys
    total_keys = len(state_dict)
    print(f"Total keys in checkpoint: {total_keys}")
    print(f"Showing first {min(max_keys, total_keys)} keys:")
    
    for i, (key, value) in enumerate(state_dict.items()):
        if i < max_keys:
            if hasattr(value, 'shape'):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {type(value)}")
        else:
            print(f"... and {total_keys - max_keys} more keys")
            break
    
    # Look for potential model structure indicators
    prefixes = set()
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) > 0:
            prefixes.add(parts[0])
    
    print(f"\nTop-level components in checkpoint: {sorted(list(prefixes))}")


def compare_model_to_checkpoint(model, checkpoint, max_items=10):
    """
    Compare a model structure to a checkpoint to identify mismatches (limited output).
    
    Args:
        model: PyTorch model
        checkpoint: The loaded checkpoint dictionary
        max_items: Maximum number of items to print in each category
    """
    print(f"\n{'='*80}\nMODEL VS CHECKPOINT COMPARISON\n{'='*80}")
    
    # Extract state_dict if needed
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Get model state dict
    model_state_dict = model.state_dict()
    
    # Check for missing keys in checkpoint (keys in model but not in checkpoint)
    missing_in_checkpoint = set(model_state_dict.keys()) - set(state_dict.keys())
    if missing_in_checkpoint:
        print(f"\nKeys in model but missing in checkpoint ({len(missing_in_checkpoint)}):")
        for i, key in enumerate(sorted(missing_in_checkpoint)):
            if i < max_items:
                print(f"  - {key}: {model_state_dict[key].shape}")
            else:
                print(f"  - ... and {len(missing_in_checkpoint) - max_items} more")
                break
    
    # Check for extra keys in checkpoint (keys in checkpoint but not in model)
    extra_in_checkpoint = set(state_dict.keys()) - set(model_state_dict.keys())
    if extra_in_checkpoint:
        print(f"\nKeys in checkpoint but missing in model ({len(extra_in_checkpoint)}):")
        for i, key in enumerate(sorted(extra_in_checkpoint)):
            if i < max_items:
                if hasattr(state_dict[key], 'shape'):
                    print(f"  - {key}: {state_dict[key].shape}")
                else:
                    print(f"  - {key}: {type(state_dict[key])}")
            else:
                print(f"  - ... and {len(extra_in_checkpoint) - max_items} more")
                break
    
    # Check for shape mismatches
    common_keys = set(model_state_dict.keys()) & set(state_dict.keys())
    shape_mismatches = []
    for key in common_keys:
        if hasattr(state_dict[key], 'shape') and hasattr(model_state_dict[key], 'shape'):
            if state_dict[key].shape != model_state_dict[key].shape:
                shape_mismatches.append((key, state_dict[key].shape, model_state_dict[key].shape))
    
    if shape_mismatches:
        print(f"\nShape mismatches between model and checkpoint ({len(shape_mismatches)}):")
        for i, (key, checkpoint_shape, model_shape) in enumerate(shape_mismatches):
            if i < max_items:
                print(f"  - {key}: checkpoint {checkpoint_shape} vs model {model_shape}")
            else:
                print(f"  - ... and {len(shape_mismatches) - max_items} more")
                break
    
    # Summary
    print(f"\nSummary:")
    print(f"  - Total keys in model: {len(model_state_dict)}")
    print(f"  - Total keys in checkpoint: {len(state_dict)}")
    print(f"  - Common keys: {len(common_keys)}")
    print(f"  - Missing in checkpoint: {len(missing_in_checkpoint)}")
    print(f"  - Extra in checkpoint: {len(extra_in_checkpoint)}")
    print(f"  - Shape mismatches: {len(shape_mismatches)}")
    
    # Print key patterns that might be useful for debugging
    print("\nKey pattern analysis:")
    model_prefixes = analyze_key_patterns(model_state_dict.keys(), max_items=5)
    checkpoint_prefixes = analyze_key_patterns(state_dict.keys(), max_items=5)
    
    return len(missing_in_checkpoint) == 0 and len(shape_mismatches) == 0


def analyze_key_patterns(keys, max_items=5):
    """Analyze key patterns to find common prefixes and structure"""
    prefixes = {}
    for key in keys:
        parts = key.split('.')
        if len(parts) > 0:
            prefix = parts[0]
            if prefix in prefixes:
                prefixes[prefix] += 1
            else:
                prefixes[prefix] = 1
    
    print(f"  Top {min(max_items, len(prefixes))} key prefixes:")
    for i, (prefix, count) in enumerate(sorted(prefixes.items(), key=lambda x: x[1], reverse=True)):
        if i < max_items:
            print(f"    - '{prefix}': {count} keys")
        else:
            break
    
    return prefixes

"""
# Debug: Print checkpoint structure
print_checkpoint_structure(checkpoint)

# Debug: Print model structure to compare with checkpoint
print_model_structure(gloria_model)

# Debug: Detailed comparison between model and checkpoint
compare_model_to_checkpoint(gloria_model, model_state_dict)

"""