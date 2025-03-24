"""Main training script for RSNA trauma detection."""

import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp
from torchmetrics.segmentation import DiceScore

from pathlib import Path
from typing import Dict, Tuple
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from flash.core.optimizers import LinearWarmupCosineAnnealingLR

from configs.config import TrainingConfig, parse_args
from data.dataset import get_chexpert_dataloader, get_motion_segmentation_dataloader
# from models.losses import CombinedLoss
# from engine.trainer import Trainer
from utils.logging_utils import LoggingManager

# from engine.trainer import MultiHeadTrainer
# from utils.metrics import RSNAMetrics


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def create_criterion() -> Dict[str, nn.Module]:
    """Create loss functions for each head."""
    # Define loss function and optimizer
    # Choose between different loss functions
    loss_type = "combined"  # Options: "ce", "dice", "combined", "torchmetrics"
    
    if loss_type == "ce":
        # Standard Cross Entropy Loss
        criterion = nn.CrossEntropyLoss()

    elif loss_type == "dice":
        # Custom Dice Loss
        criterion = smp.losses.DiceLoss(mode='multiclass')

    # elif loss_type == "combined":
    #     # Combined CE and Dice Loss (weighted)
    #     criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
        
    else:
        ValueError("XXXXXXXXX")
    

    return criterion


def setup_training(config, args) -> Tuple[nn.Module, torch.device, Dict[str, DataLoader]]:

    # Initialize logger with appropriate configuration
    neptune_config = None
    clearml_config = None
    if args.logger == 'neptune':
        if not args.neptune_project or not args.neptune_api_token:
            raise ValueError("Neptune project and API token required when using Neptune logger")

        neptune_config = {
            'project': args.neptune_project,
            'api_token': args.neptune_api_token,
            'experiment_name': args.experiment_name,
            'run_id': args.neptune_run_id
        }

    elif args.logger == 'clearml':
        clearml_config = {
            'project': args.clearml_project,
            'task_name': args.experiment_name,
            'task_type': args.clearml_task_type,
            'reuse_last_task_id': args.clearml_task_id,
            "tags": args.clearml_tags,
        }

    logger = LoggingManager(
        output_dir=args.output_dir,
        logger_type=args.logger,
        neptune_config=neptune_config,
        clearml_config=clearml_config
    )

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup data loaders
    train_loader = get_chexpert_dataloader(config, split="train", view_type="Frontal")
    val_loader = get_chexpert_dataloader(config, split="valid", view_type="Frontal")

    # Initialize model with support for resuming
    model = smp.PSPNet(
        encoder_name="resnet34",        # Choose encoder
        encoder_weights="imagenet",     # Use pre-trained weights
        in_channels=3,                  # RGB input
        classes=args.num_classes,     # Number of classes (background + foreground)
    ).to(args.device)

    logger.log_model_summary(model)
    
    return model, {'train': train_loader, 'val': val_loader}, logger


def run_training_pipeline(config: TrainingConfig) -> dict:
    """Enhanced training function with advanced features."""
    # Set up training components
    model, dataloaders, logger = setup_training(config)
    quit()
    
    # Calculate training steps with validation frequency
    num_training_steps_per_epoch = len(dataloaders['train'])
    total_steps = num_training_steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    # Initialize Optimizer, Criterion, Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = create_criterion()
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_steps,
        max_epochs=total_steps,
        warmup_start_lr=config.learning_rate * 0.1,
        eta_min=1e-6)
    
    
    # Initialize metrics and trainer
    metrics = DiceScore(num_classes=config.num_classes, input_format='index')
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        output_dir=config.output_dir,
        metrics=metrics,
        logger=logger,
        amp_mode="no", #config.amp,
    )

    # Log hyperparameters
    hyperparameters = {
        'model_config': vars(config),
        'training_config': {
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': config.lr,
            'warmup_period': warmup_steps,
            'total_steps': total_steps,
            'device': config.device,
            'amp_mode': config.amp
        }
    }
    logger.log_hyperparameters(hyperparameters)


    # Resume if specified
    if config.resume:
        trainer.resume_from_checkpoint(config.resume)
        # Log the resumed checkpoint as an artifact
        logger.log_artifact(config.resume, "checkpoints/resumed_from")

    # Print training configuration
    LoggingManager.print_training_config(
        args=config,
        total_steps=total_steps,
        steps_per_epoch=num_training_steps_per_epoch,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        loss_functions=criterion
    )

    trainer.train(
        train_loader=dataloaders['train'],
        criterion=criterion,
        val_loader=dataloaders['val'],
        num_epochs=config.epochs
    )

    logger.close()  # Cleanup


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the training script."""
    parser = argparse.ArgumentParser(description="GLoRIA training and evaluation script")
    
    # Basic configuration
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--test", action="store_true", help="Run model evaluation")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the model checkpoint")
    
    # Experiment settings
    parser.add_argument("--random_seed", type=int, default=23, help="Random seed for reproducibility",)
    parser.add_argument("--train_pct", type=float, default=1.0, help="Percentage of training data to use")
    parser.add_argument("--splits", type=int, default=1, help="Number of training splits to use")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    
    # Add PyTorch Lightning Trainer arguments
    parser = Trainer.add_argparse_args(parser)
    
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = OmegaConf.load(args.config)
    config = parse_args()
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full config
    OmegaConf.save(config=OmegaConf.create(config), f=output_path / 'config.yaml')

    # Train model
    run_training_pipeline(args, config)

    # Cleanup logging
    pass


if __name__ == '__main__':
    main()