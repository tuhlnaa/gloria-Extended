"""
GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition
Main training and evaluation script.
"""
import os
import torch
import argparse
import datetime
import numpy as np

from pathlib import Path
from typing import Dict, List, Optional, Union
from dateutil import tz
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.trainer import Trainer

import gloria

# Enable deterministic behavior while keeping performance optimizations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


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


def setup_callbacks(config: Dict) -> List:
    """Set up training callbacks based on configuration."""
    callbacks = [LearningRateMonitor(logging_interval="step")]
    
    if "checkpoint_callback" in config.lightning:
        checkpoint_callback = ModelCheckpoint(**config.lightning.checkpoint_callback)
        callbacks.append(checkpoint_callback)
        
    if "early_stopping_callback" in config.lightning:
        early_stopping_callback = EarlyStopping(**config.lightning.early_stopping_callback)
        callbacks.append(early_stopping_callback)
        
    return callbacks


def setup_logger(config: Dict) -> Optional[pl_loggers.logger.Logger]:
    """Set up experiment logger based on configuration."""
    if "logger" not in config.lightning:
        return None
        
    logger_config = config.lightning.logger
    logger_type = logger_config.pop("logger_type")
    logger_class = getattr(pl_loggers, logger_type)
    
    # Set logger name with experiment details
    logger_config.name = f"{config.experiment_name}_{config.extension}"
    logger = logger_class(**logger_config)
    
    # Restore the logger_type field in the config
    config.lightning.logger.logger_type = logger_type
    
    return logger


def setup_directories(config: Dict) -> None:
    """Create necessary directories for experiment outputs."""
    paths = [
        config.lightning.logger.save_dir,
        config.lightning.checkpoint_callback.dirpath,
        config.output_dir,
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_config(config: Dict, output_path: str) -> None:
    """Save configuration to YAML file."""
    with open(output_path, "w") as fp:
        OmegaConf.save(config=config, f=fp.name)


def run_experiment(config: Dict, args: argparse.Namespace) -> None:
    """Run a single training/evaluation experiment."""
    # Initialize data module
    datamodule = gloria.builder.build_data_module(config)
    
    # Initialize model
    model = gloria.builder.build_lightning_model(config, datamodule)
    
    # Set up callbacks, logger and trainer
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    
    # Configure trainer
    config.lightning.trainer.val_check_interval = args.val_check_interval
    config.lightning.trainer.auto_lr_find = args.auto_lr_find
    trainer_args = argparse.Namespace(**config.lightning.trainer)
    
    trainer = Trainer.from_argparse_args(
        args=trainer_args,
        deterministic=True,
        callbacks=callbacks,
        logger=logger,
    )
    
    # Find optimal learning rate if enabled
    if trainer_args.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
        new_lr = lr_finder.suggestion()
        model.lr = new_lr
        print(f"{'='*40} Learning rate updated to {new_lr} {'='*40}")
    # breakpoint()
    
    # Train model
    if args.train:
        trainer.fit(model, datamodule)

    # Test model
    if args.test:
        checkpoint_callback = next(
            (cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), 
            None
        )
        ckpt_path = None
        
        if checkpoint_callback and args.train:
            ckpt_path = checkpoint_callback.best_model_path
        elif config.model.checkpoint:
            ckpt_path = config.model.checkpoint
            
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    # Save best checkpoint paths
    if "checkpoint_callback" in config.lightning:
        checkpoint_callback = next(
            (cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), 
            None
        )
        if checkpoint_callback:
            ckpt_paths = os.path.join(
                config.lightning.checkpoint_callback.dirpath, 
                "best_ckpts.yaml"
            )
            checkpoint_callback.to_yaml(filepath=ckpt_paths)


def main() -> None:
    args = parse_arguments()
    config = OmegaConf.load(args.config)
    
    # Update configuration with command-line arguments
    config.data.frac = args.train_pct
    
    # Update experiment name if needed
    if config.trial_name:
        config.experiment_name = f"{config.experiment_name}_{config.trial_name}"
    
    if args.train_pct < 1.0:
        config.experiment_name = f"{config.experiment_name}_{args.train_pct}"
    
    # Run experiments for each split
    for split_idx in range(args.splits):
        # Generate timestamp for this run
        timestamp = datetime.datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
        
        # Set random seed for this split
        split_seed = split_idx + 1
        seed_everything(split_seed)
        
        # Set extension name (split number or timestamp if only one split)
        config.extension = str(split_seed) if args.splits > 1 else timestamp
        
        # Set output directories
        config.output_dir = f"./data/output/{config.experiment_name}/{config.extension}"
        config.lightning.checkpoint_callback.dirpath = os.path.join(
            config.lightning.checkpoint_callback.dirpath,
            f"{config.experiment_name}/{config.extension}",
        )
        
        # Create directories and save config
        setup_directories(config)
        save_config(config, os.path.join(config.output_dir, "config.yaml"))
        
        # Run the experiment
        run_experiment(config, args)


if __name__ == "__main__":
    main()

"""
python run.py --config E:\Kai_2\CODE_Repository\gloria-Extended\configs\chexpert_classification_config.yaml --train
"""