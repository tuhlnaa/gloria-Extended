import torch.nn as nn

from typing import Dict, Callable, Any
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from gloria.engine.classification_trainer import Trainer
from gloria.engine.classification_validator import Validator
from gloria.engine.gloria_trainer import GloriaTrainer
from gloria.engine.gloria_validator import GloriaValidator
from gloria.engine.segmentation_trainer import SegmentationTrainer
from gloria.engine.segmentation_validator import SegmentationValidator

class TrainerFactory:
    """
    Factory class for managing different trainer implementations.
    
    This class allows automatic switching between different trainer implementations
    based on the model name specified in the configuration.
    """
    
    def __init__(self):
        """Register trainer implementations with their respective model names"""
        self._trainers: Dict[str, Callable] = {
            "classification": Trainer,
            "segmentation": SegmentationTrainer,
            "gloria": GloriaTrainer,
            "gloria_classification": Trainer,
            # Register other trainer implementations here
        }
    
    def get_trainer(self, config: OmegaConf, train_loader: DataLoader, model_name: str = None, **kwargs) -> Any:
        """
        Get the appropriate trainer based on model name.
        
        Args:
            config: Configuration object
            train_loader: Training data loader
            model_name: Name of model to use (overrides config.model.name if provided)
            **kwargs: Additional arguments to pass to the trainer constructor
            
        Returns:
            An instance of the appropriate trainer
            
        Raises:
            ValueError: If the specified model/trainer is not registered
        """
        # Use model_name if provided, otherwise use from config
        model_to_use = model_name if model_name else config.model.name.lower()

        if model_to_use not in self._trainers:
            available_trainers = list(self._trainers.keys())
            raise ValueError(f"Trainer for model '{model_to_use}' not found. Available trainers: {available_trainers}")
        
        # Get the appropriate trainer class
        trainer_class = self._trainers[model_to_use]
        
        # Return an instance of the trainer
        return trainer_class(config, train_loader, **kwargs)
    
    def register_trainer(self, name: str, trainer_class: Callable) -> None:
        """
        Register a new trainer implementation.
        
        Args:
            name: Name for the trainer (typically matching model name)
            trainer_class: Trainer class to register
        """
        self._trainers[name] = trainer_class


class ValidatorFactory:
    """
    Factory class for managing different validator implementations.
    
    This class allows automatic switching between different validator implementations
    based on the model name specified in the configuration.
    """
    
    def __init__(self):
        """Register validator implementations with their respective model names"""
        self._validators: Dict[str, Callable] = {
            "classification": Validator,
            "segmentation": SegmentationValidator,
            "gloria": GloriaValidator,
            "gloria_classification": Validator,
            # Register other validator implementations here
        }
    
    def get_validator(self, config: OmegaConf, model: nn.Module, loss_fn: nn.Module, model_name: str = None, **kwargs) -> Any:
        """
        Get the appropriate validator based on model name.
        
        Args:
            config: Configuration object
            model: The model to validate
            loss_fn: Loss function to use during validation
            model_name: Name of model to use (overrides config.model.name if provided)
            **kwargs: Additional arguments to pass to the validator constructor
            
        Returns:
            An instance of the appropriate validator
            
        Raises:
            ValueError: If the specified model/validator is not registered
        """
        # Use model_name if provided, otherwise use from config
        model_to_use = model_name if model_name else config.model.name.lower()
        
        if model_to_use not in self._validators:
            available_validators = list(self._validators.keys())
            raise ValueError(f"Validator for model '{model_to_use}' not found. Available validators: {available_validators}")
        
        # Get the appropriate validator class
        validator_class = self._validators[model_to_use]
        
        # Return an instance of the validator
        return validator_class(config, model, loss_fn, **kwargs)
    
    def register_validator(self, name: str, validator_class: Callable) -> None:
        """
        Register a new validator implementation.
        
        Args:
            name: Name for the validator (typically matching model name)
            validator_class: Validator class to register
        """
        self._validators[name] = validator_class


# Create a singleton instance
trainer_factory = TrainerFactory()
validator_factory = ValidatorFactory()


