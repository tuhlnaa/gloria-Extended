
from typing import Dict, Callable, Tuple, Any
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from data.chexpert_dataset import get_chexpert_dataloader, get_chexpert_multimodal_dataloader
from data.pet_dataset import get_pet_dataloader
from data.pneumothorax_dataset import get_pneumothorax_dataloader


class DatasetFactory:
    """
    Factory class for managing dataset loaders.
    
    This class allows automatic switching between different dataset loaders
    based on the dataset name specified in the configuration.
    """
    
    def __init__(self):
        """Register dataset loaders with their respective names"""
        self._dataset_loaders: Dict[str, Callable] = {
            "chexpert": get_chexpert_dataloader,
            "chexpert_multimodal": get_chexpert_multimodal_dataloader,
            "pneumothorax": get_pneumothorax_dataloader,
            "pet": get_pet_dataloader,
            # Register other dataset loaders here
        }
    

    def get_dataloader(self, config: OmegaConf, dataset_name: str = None, **kwargs) -> Tuple[DataLoader, Any]:
        """
        Get the appropriate dataloader based on dataset name.
        
        Args:
            config: Configuration object
            dataset_name: Name of dataset to use (overrides config.dataset.dataset_name if provided)
            **kwargs: Additional arguments to pass to the dataloader function
            
        Returns:
            Tuple of (DataLoader, dataset_info)
            
        Raises:
            ValueError: If the specified dataset is not registered
        """
        # Use dataset_name if provided, otherwise use from config
        dataset_to_use = dataset_name if dataset_name else config.dataset.name
        
        if dataset_to_use not in self._dataset_loaders:
            available_datasets = list(self._dataset_loaders.keys())
            raise ValueError(f"Dataset '{dataset_to_use}' not found. Available datasets: {available_datasets}")
        
        # Get the appropriate dataloader function
        dataloader_fn = self._dataset_loaders[dataset_to_use]
        
        # Return the dataloader and additional info
        return dataloader_fn(config, **kwargs)
    
    
    def register_dataloader(self, name: str, loader_fn: Callable) -> None:
        """
        Register a new dataloader function.
        
        Args:
            name: Name for the dataloader
            loader_fn: Function that returns (dataloader, info)
        """
        self._dataset_loaders[name] = loader_fn


# Create a singleton instance
dataset_factory = DatasetFactory()