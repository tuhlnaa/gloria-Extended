import torch
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Tuple
from torch.utils.data import DataLoader

from data.utils import dataset_factory
from configs.config import parse_args, save_config
from gloria.engine.trainer import Trainer
from gloria.engine.validator import Validator
from gloria.models import pytorch
from utils.checkpoint import CheckpointHandler
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

    # Upload a configuration file as an artifact
    logger.log_artifact(local_path= config.data_dir / "config.yaml")

    # Set random seed for reproducibility
    set_seed(config.misc.seed)

    # Setup data loaders using the dataset factory
    train_loader, _ = dataset_factory.get_dataloader(config, split="train")
    val_loader, _ = dataset_factory.get_dataloader(config, split="valid")

    # Initialize model
    model_class = pytorch.PYTORCH_MODULES[config.phase.lower()]
    model = model_class(config, train_loader)
    logger.log_model_summary(model)

    # Checkpoint handler
    checkpoint_handler = CheckpointHandler(
        save_dir=config.output_dir,
        filename_prefix="checkpoint",
        max_save_num=2,
        save_interval=config.misc.save_freq
    )

    return model, {'train': train_loader, 'val': val_loader}, logger, checkpoint_handler


# def run_training_pipeline(config: OmegaConf) -> Dict[str, float]:
#     """Run the complete training pipeline."""

#     # Initialize data loaders and logger
#     model, dataloaders, logger, checkpoint_handler = setup_training(config)

#     start_epochs = 0
#     patience_counter = 0

#     if config.misc.monitor_metric == "val_loss":
#         best_val_metric = float("inf")
#     else:
#         best_val_metric = float("-inf")

#     # Resume if specified
#     if config.model.resume:
#         start_epochs, best_val_metric, _ = trainer.resume_from_checkpoint(config.model.resume)
#         # Log the resumed checkpoint as an artifact
#         # logger.log_artifact(config.resume, "checkpoints/resumed_from")

#     # Set up training components
#     trainer = Trainer(config, dataloaders['train'])
#     validator = Validator(config, trainer.model, trainer.criterion)
#     trainer.setup_optimization()

#     # Print training configuration
#     LoggingManager.print_training_config(
#         args=config,
#         train_loader=dataloaders['train'],
#         val_loader=dataloaders['val'],
#         loss_functions=trainer.criterion
#     )

#     # Training loop
#     for epoch in range(start_epochs, config.lr_scheduler.epochs):
#         # Train for one epoch
#         train_metrics = trainer.train_epoch(dataloaders['train'], epoch)
#         train_metrics.update({"learning_rate": trainer.optimizer.param_groups[0]['lr']})
        
#         # Log training metrics
#         logger.log_metrics(train_metrics, epoch+1)
        
#         # Run validation
#         val_metrics = validator.validate(dataloaders['val'])
#         logger.log_metrics(val_metrics, epoch+1)
        
#         # Step scheduler if needed
#         if hasattr(config.lr_scheduler, 'step_frequency') and epoch % config.lr_scheduler.step_frequency == 0:
#             trainer.scheduler.step(val_metrics.get("val_loss", None))
        
#         # Early stopping check
#         if hasattr(config.lr_scheduler, 'patience'):
#             if (val_metrics[config.misc.monitor_metric] > best_val_metric and config.misc.monitor_metric != "val_loss") or (val_metrics[config.misc.monitor_metric] < best_val_metric and config.misc.monitor_metric == "val_loss"):
#                 patience_counter += 1
#                 if patience_counter >= config.lr_scheduler.patience:
#                     print("Early stopping triggered")
#                     break
#             else:
#                 patience_counter = 0

#         # Save best model
#         if (val_metrics[config.misc.monitor_metric] > best_val_metric and config.misc.monitor_metric != "val_loss") or (val_metrics[config.misc.monitor_metric] < best_val_metric and config.misc.monitor_metric == "val_loss"):
#             best_val_metric = val_metrics[config.misc.monitor_metric]
#             checkpoint_handler.save_checkpoint(
#                 epochs=epoch+1,
#                 model_state=trainer.model.state_dict(),
#                 optimizer_state=trainer.optimizer.state_dict(),
#                 scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler else None,
#                 metrics=val_metrics,
#                 is_best=True,
#             )

#         checkpoint_handler.save_checkpoint(
#             epochs=epoch+1,
#             model_state=trainer.model.state_dict(),
#             optimizer_state=trainer.optimizer.state_dict(),
#             scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler else None,
#             metrics=val_metrics,
#             is_best=False,
#         )

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
    model, dataloaders, logger, checkpoint_handler = setup_training(config)

    for epoch in range(0, config.lr_scheduler.epochs):
        train_metrics = model.train_epoch(dataloaders['train'], epoch)
        train_metrics.update({"learning_rate": model.optimizer.param_groups[0]['lr']})

        # Log metrics
        logger.log_metrics(train_metrics, epoch)

        # Run validation
        val_metrics = model.validate(dataloaders['val'])
        logger.log_metrics(val_metrics, epoch)

        # Step scheduler if needed
        if hasattr(config.lr_scheduler, 'step_frequency') and epoch % config.lr_scheduler.step_frequency == 0:
            model.scheduler.step(val_metrics.get("val_loss", None))
        

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
python train.py --config configs\default_config.yaml
"""