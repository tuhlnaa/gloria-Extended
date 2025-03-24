"""
Based on: https://github.com/huggingface/pytorch-image-models/blob/main/train.py
"""
import yaml
import argparse

from pathlib import Path
from typing import Optional, Dict, Any
from omegaconf import OmegaConf

from utils.logging_utils import LoggingManager


def parse_args() -> Dict[str, Any]:
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
    group.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')

    # Optimizer parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay factor')
    
    # Learning rate schedule parameters
    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--epochs', type=int, default=8, help='Number of epochs to train')
    group.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for the optimizer")
    group.add_argument('--warmup_ratio', type=float, default=0.1, help="Ratio of warmup steps")
    group.add_argument('--patience', type=int, default=7, help='Early stopping patience')

    # Device & distributed
    group = parser.add_argument_group('Device parameters')
    group.add_argument('--amp', type=str, default='no', choices=['no', 'fp16', 'bf16'], help='Use automatic mixed precision training. Options: no, fp16, bf16')
    group.add_argument('--device', default='cuda:0', type=str, help="Device (accelerator) to use.")

    # Logging parameters
    group = parser.add_argument_group('Logging parameters')
    group.add_argument('--logger', type=str, default='tensorboard', choices=['tensorboard', 'neptune', 'clearml'], help='Logger to use')
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

    args = parser.parse_args()
    
    # Convert namespace to dictionary
    config = vars(args)
    
    # If config file is provided, load it and update the config
    if args.config:
        yaml_config = load_yaml_config(args.config)
        # Start with yaml config and override with command line args that are not None
        merged_config = {**yaml_config}
        for key, value in config.items():
            if value is not None and key != 'config':
                merged_config[key] = value
        config = merged_config
    
    # Validate and convert paths
    validate_paths(config)
    
    # Print configuration using the logging utility
    LoggingManager.print_config(config, "Configuration")

    return config


def validate_paths(config: Dict[str, Any]) -> None:
    """Validate and ensure that required paths exist."""
    if 'data_dir' in config and config['data_dir']:
        if not Path(config['data_dir']).exists():
            raise ValueError(f"Data directory {config['data_dir']} does not exist")
    
    # Ensure output_dir is a Path object
    if 'output_dir' in config and config['output_dir']:
        config['output_dir'] = Path(config['output_dir'])


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
