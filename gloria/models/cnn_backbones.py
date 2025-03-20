"""
CNN backbone models for feature extraction in the GLoRIA framework.

This module provides CNN architectures (ResNet, DenseNet, ResNeXt) that are used as 
feature extractors in the GLoRIA multimodal representation learning framework.
"""

from typing import Tuple, Optional, Literal, Dict, Callable
import torch.nn as nn
from torchvision import models as tv_models


ModelReturnType = Tuple[nn.Module, int, Optional[int]]
WeightsType = Literal['DEFAULT', None]

# Model registry for easier access
MODEL_REGISTRY: Dict[str, Callable[[WeightsType], ModelReturnType]] = {}


def register_model(name: str):
    """Decorator to register model builders in the MODEL_REGISTRY."""
    def decorator(func):
        MODEL_REGISTRY[name] = func
        return func
    return decorator


def get_backbone(name: str, weights: WeightsType = 'DEFAULT') -> ModelReturnType:
    """Get a backbone model by name.
    
    Args:
        name: Model name (e.g., 'resnet50', 'densenet121')
        weights: Weight initialization strategy ('DEFAULT' for pretrained)
        
    Returns:
        tuple: (model, feature_dimensions, projection_dimensions)
        
    Raises:
        ValueError: If the model name is not registered
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](weights)


def _create_feature_extractor(
        model_builder: Callable,
        weights: WeightsType,
        classifier_attr: str
    ) -> ModelReturnType:
    """Create a feature extractor from a torchvision model.
    
    Args:
        model_builder: Function that builds the base model
        weights: Weight initialization strategy ('DEFAULT' for pretrained)
        classifier_attr: Attribute name of the classifier layer ('fc' or 'classifier')
    
    Returns:
        tuple: (model, feature_dimensions, projection_dimensions)
    """
    model = model_builder(weights=weights)
    classifier = getattr(model, classifier_attr)
    feature_dims = classifier.in_features
    
    # Replace classifier with identity layer
    setattr(model, classifier_attr, Identity())
    
    # Return projection dims (used for ResNet models) or None
    projection_dims = 1024 if classifier_attr == 'fc' else None
    
    return model, feature_dims, projection_dims


class Identity(nn.Module):
    """Identity layer that passes input unchanged.
    
    Used to replace classification layers in pretrained models to extract features.
    """
    def forward(self, x):
        return x


@register_model('resnet_18')
def resnet_18(weights: WeightsType = 'DEFAULT') -> ModelReturnType:
    """ResNet-18 model for feature extraction."""
    return _create_feature_extractor(tv_models.resnet18, weights, 'fc')


@register_model('resnet_34')
def resnet_34(weights: WeightsType = 'DEFAULT') -> ModelReturnType:
    """ResNet-34 model for feature extraction."""
    return _create_feature_extractor(tv_models.resnet34, weights, 'fc')


@register_model('resnet_50')
def resnet_50(weights: WeightsType = 'DEFAULT') -> ModelReturnType:
    """ResNet-50 model for feature extraction."""
    return _create_feature_extractor(tv_models.resnet50, weights, 'fc')


@register_model('densenet_121')
def densenet_121(weights: WeightsType = 'DEFAULT') -> ModelReturnType:
    """DenseNet-121 model for feature extraction."""
    return _create_feature_extractor(tv_models.densenet121, weights, 'classifier')


@register_model('densenet_161')
def densenet_161(weights: WeightsType = 'DEFAULT') -> ModelReturnType:
    """DenseNet-161 model for feature extraction."""
    return _create_feature_extractor(tv_models.densenet161, weights, 'classifier')


@register_model('densenet_169')
def densenet_169(weights: WeightsType = 'DEFAULT') -> ModelReturnType:
    """DenseNet-169 model for feature extraction."""
    return _create_feature_extractor(tv_models.densenet169, weights, 'classifier')


@register_model('resnext_50')
def resnext_50(weights: WeightsType = 'DEFAULT') -> ModelReturnType:
    """ResNeXt-50 (32x4d) model for feature extraction."""
    return _create_feature_extractor(tv_models.resnext50_32x4d, weights, 'fc')


@register_model('resnext_101')
def resnext_101(weights: WeightsType = 'DEFAULT') -> ModelReturnType:
    """ResNeXt-101 (32x8d) model for feature extraction."""
    return _create_feature_extractor(tv_models.resnext101_32x8d, weights, 'fc')
