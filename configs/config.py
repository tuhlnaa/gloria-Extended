"""
Based on: https://github.com/huggingface/pytorch-image-models/blob/main/train.py
"""
import yaml
import argparse

from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from dataclasses import dataclass, field

from utils.logging_utils import LoggingManager

@dataclass
class TrainingConfig:
    """Enhanced training configuration with advanced features."""
    # Dataset parameters
    data_dir: str
    num_workers: int = 0

    # Training parameters
    batch_size: int = 32
    
    epochs: int = 8
    early_stopping_patience: int = 7
    
    # Optimization parameters
    learning_rate: float = 1e-3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Model parameters
    pretrained: bool = True
    
    # Advanced training features
    amp: str = "no"
    validation_freq: int = 100
    log_freq: int = 100
    save_freq: int = 100
    resume: Optional[str] = None
    
    # Logging configuration
    logger: str = "tensorboard"
    neptune_project: Optional[str] = None
    neptune_api_token: Optional[str] = None
    experiment_name: Optional[str] = None
    neptune_run_id: Optional[str] = None

    output_dir: str = "output"

    # Additional parameters can be added here for future expansion
    
    def __post_init__(self):
        """Validate and convert paths to Path objects."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        

def parse_args() -> TrainingConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RSNA Trauma Detection model')
    parser.add_argument('--config', type=Path, help='YAML config file specifying default arguments')
    parser.add_argument('--output_dir', type=Path, required=True, help='Path to save model and results')

    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    parser.add_argument('--data_dir', type=Path, required=True, help='Path to the dataset directory')
    parser.add_argument('--mask_dir', type=Path, help='')
    parser.add_argument('--csv_path', type=Path, required=True, help='Path to the CSV file containing class ID to name mapping')

    group.add_argument('--selected_classes', nargs='*', default=None, help='List of class names to use (others will be treated as background)')
    group.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    group.add_argument('--split', type=float, default=0.85,  help='')
    group.add_argument('--image_size', type=int, default=512,  help='')
    group.add_argument('--num_classes', type=int, default=2,  help='')
    
    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    group.add_argument('--resume', type=str, help='Resume full model and optimizer state from checkpoint')

    # Optimizer parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--weight-decay', type=float, default=2e-5, help='')

    # Learning rate schedule parameters
    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    group.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer")

    # Device & distributed
    group = parser.add_argument_group('Device parameters')
    group.add_argument('--amp', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Use automatic mixed precision training. Options: no, fp16, bf16')
    group.add_argument('--device', default='cuda:0', type=str, help="Device (accelerator) to use.")

    # Logging parameters
    group = parser.add_argument_group('Logging parameters')
    group.add_argument('--logger', type=str, default='clearml', choices=['tensorboard', 'neptune', 'clearml'], help='Logger to use')
    group.add_argument('--neptune_project', type=str, help='Neptune project name')
    group.add_argument('--neptune_api_token', type=str, help='Neptune API token')
    group.add_argument('--neptune_run_id', type=str, help='Neptune run ID to resume (e.g., "CLS-123")')

    # ClearML logger parameters
    group.add_argument('--clearml_project', type=str, help='ClearML project name')
    group.add_argument('--clearml_task_id', type=str_or_bool, default='false', help='ClearML task ID to resume (string) or boolean flag')
    group.add_argument('--clearml_task_type', type=str, default='training', choices=['training', 'testing', 'inference', 'data_processing'], help='ClearML task type')
    group.add_argument('--clearml_tags', nargs='*', help='List of tags for ClearML task')

    # Common logger parameters
    group.add_argument('--experiment_name', type=str, help='Name for the experiment')

    # Misc
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    group.add_argument('--validation_freq', type=int, default=100, help='Frequency of validation (iterations)')
    group.add_argument('--log_freq', type=int, default=100, help='Frequency of logging metrics (iterations)')
    group.add_argument('--save_freq', type=int, default=100, help='Frequency of saving checkpoints (iterations)')
    group.add_argument('--patience', type=int, default=7, help='Early stopping patience')

    args = parser.parse_args()

    # Create config with priority: defaults < config file < command line args
    config = TrainingConfig(data_dir=args.data_dir)
    
    if args.config:
        yaml_config = load_yaml_config(args.config)
        config = OmegaConf.merge(
            OmegaConf.structured(config),
            OmegaConf.create(yaml_config)
        )
    
    # Update with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            setattr(config, key, value)
    
    # Print configuration using the logging utility
    LoggingManager.print_config(config, "Command Line Arguments")

    return config


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def str_or_bool(value):
    # Handle boolean string values
    if isinstance(value, str):
        if value.lower() in ('yes', 'true'):
            return True
        elif value.lower() in ('no', 'false'):
            return False
        else:
            return str(value)
    # If it's not a boolean string, return the string value itself
    return value

