import torch
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from flash.core.optimizers import LinearWarmupCosineAnnealingLR

from configs.config import parse_args, save_config
from data.utils import dataset_factory
from data.dataset import get_chexpert_dataloader
from data.pretraining_datasetV2 import get_chexpert_multimodal_dataloader
from gloria.engine.trainer import Trainer
from gloria.engine.validator import Validator
from gloria.models import pytorch
from utils.logging_utils import LoggingManager


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


def setup_training(config: OmegaConf) -> Tuple[nn.Module, torch.device, Dict[str, DataLoader]]:

    # Initialize logger with appropriate configuration
    neptune_config = None
    clearml_config = None
    if config.logging.logger == 'neptune':
        if not config.logging.neptune.project or not config.logging.neptune.api_token:
            raise ValueError("Neptune project and API token required when using Neptune logger")

        neptune_config = {
            'project': config.logging.neptune.project,
            'api_token': config.logging.neptune.api_token,
            'experiment_name': config.logging.experiment_name,
            'run_id': config.logging.neptune.run_id
        }

    elif config.logging.logger == 'clearml':
        clearml_config = {
            'project': config.logging.clearml.project,
            'task_name': config.logging.experiment_name,
            'task_type': config.logging.clearml.task_type,
            'reuse_last_task_id': config.logging.clearml.task_id,
            "tags": config.logging.clearml.tags,
        }

    logger = LoggingManager(
        output_dir=config.output_dir,
        logger_type=config.logging.logger,
        neptune_config=neptune_config,
        clearml_config=clearml_config
    )

    # Set random seed for reproducibility
    set_seed(config.misc.seed)

    # Setup data loaders using the dataset factory
    train_loader, _ = dataset_factory.get_dataloader(config, split="train")
    val_loader, _ = dataset_factory.get_dataloader(config, split="valid")

    # Initialize model
    model_class = pytorch.PYTORCH_MODULES[config.phase.lower()]
    model = model_class(config)

    logger.log_model_summary(model)
    
    return model, {'train': train_loader, 'val': val_loader}, logger


# def run_training_pipeline(config) -> Dict[str, float]:
#     """Run the complete training pipeline."""

#     # Initialize data loaders and logger
#     model, dataloaders, logger = setup_training(config)

#     # # Calculate training steps with validation frequency
#     # num_training_steps_per_epoch = len(dataloaders['train'])
#     # total_steps = num_training_steps_per_epoch * config.epochs
#     # warmup_steps = int(total_steps * config.warmup_ratio)
    
#     # scheduler = LinearWarmupCosineAnnealingLR(
#     #     optimizer,
#     #     warmup_epochs=warmup_steps,
#     #     max_epochs=total_steps,
#     #     warmup_start_lr=config.learning_rate * 0.1,
#     #     eta_min=1e-6)
    
#     # # Resume if specified
#     # if config.resume:
#     #     trainer.resume_from_checkpoint(config.resume)
#     #     # Log the resumed checkpoint as an artifact
#     #     # logger.log_artifact(config.resume, "checkpoints/resumed_from")

#     # # Print training configuration
#     # LoggingManager.print_training_config(
#     #     args=config,
#     #     total_steps=total_steps,
#     #     steps_per_epoch=num_training_steps_per_epoch,
#     #     train_loader=dataloaders['train'],
#     #     val_loader=dataloaders['val'],
#     #     loss_functions=criterion
#     # )

#     # Set up training components
#     trainer = Trainer(config)
#     validator = Validator(trainer.model, trainer.criterion, config)
    
#     # Set up optimization components
#     trainer.setup_optimization(dataloaders.get('datamodule'))
    
#     # Training loop
#     best_val_metric = 0.0
#     for epoch in range(config.lr_scheduler.epochs):
#         # Train for one epoch
#         train_metrics = trainer.train_epoch(dataloaders['train'], epoch)
#         train_metrics.update({"learning_rate": trainer.optimizer.param_groups[0]['lr']})
        
#         # Log training metrics
#         logger.log_metrics(train_metrics, epoch)
        
#         # Run validation
#         val_metrics = validator.validate(dataloaders['val'])
#         logger.log_metrics(val_metrics, epoch)
        
#         # Step scheduler if needed
#         if hasattr(config.lr_scheduler, 'step_frequency') and epoch % config.lr_scheduler.step_frequency == 0:
#             if hasattr(trainer.scheduler, 'step'):
#                 trainer.scheduler["scheduler"].step(val_metrics.get("val_loss", None))
        
#         # Save checkpoint if improved
#         current_metric = val_metrics.get("val_mean_auroc", 0)
#         if current_metric > best_val_metric:
#             best_val_metric = current_metric
#             checkpoint_path = config.output_dir / "best_model.pth"
#             trainer.save_checkpoint(checkpoint_path)
            
#         # Save regular checkpoint
#         if (epoch + 1) % config.get('checkpoint_frequency', 5) == 0:
#             checkpoint_path = config.output_dir / f"checkpoint_epoch_{epoch+1}.pth"
#             trainer.save_checkpoint(checkpoint_path)
    
#     logger.close()  # Cleanup

#     # Run final evaluation on test set if available
#     final_metrics = {}
#     if 'test' in dataloaders:
#         test_metrics = validator.test(dataloaders['test'])
#         logger.log_metrics(test_metrics, config.lr_scheduler.epochs)
#         final_metrics.update(test_metrics)
    
#     return final_metrics


def run_training_pipeline(config: OmegaConf) -> dict:
    """Enhanced training function with advanced features."""
    # Set up training components
    model, dataloaders, logger = setup_training(config)

    for epoch in range(0, config.lr_scheduler.epochs):
        train_metrics = model.train_epoch(dataloaders['train'], epoch)
        train_metrics.update({"learning_rate": model.optimizer.param_groups[0]['lr']})

        if epoch % 3 == 0: 
            model.scheduler["scheduler"].step(train_metrics["train_loss"])

        # Log metrics
        logger.log_metrics(train_metrics, epoch)

        val_metrics = model.validate(dataloaders['val'])
        #val_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        logger.log_metrics(val_metrics, epoch)


def main():
    """Main function with enhanced features."""
    config = parse_args()

    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full config
    save_config(config, path=output_path / 'config.yaml')

    run_training_pipeline(config)


if __name__ == '__main__':
    main()

"""
python train.py --config configs\default_training_config.yaml
"""