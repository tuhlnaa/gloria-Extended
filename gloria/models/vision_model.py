from typing import Tuple, Union
from numpy.lib.function_base import extract
import torch
import torch.nn as nn

from . import cnn_backbones
from omegaconf import OmegaConf


class ImageEncoder(nn.Module):
    """
    Image encoder for GLoRIA model that extracts both global and local features.
    
    This encoder utilizes CNN backbones to extract features and provides 
    embeddings for both global image context and local image regions.
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.output_dim = config.model.text.embedding_dim
        self.model_name = config.model.vision.model_name
        
        # Get backbone model
        model_function = getattr(cnn_backbones, self.model_name)
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=config.model.vision.pretrained
        )
        
        # Define embedding layers
        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # Freeze CNN if specified in config
        if config.model.vision.freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, x: torch.Tensor, get_local: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            get_local: If True, return both global and local features
            
        Returns:
            global_ft: Global features of shape (batch_size, feature_dim)
            local_ft: Local features of shape (batch_size, interm_feature_dim, h, w) if get_local=True
        """
        if "resnet" in self.model_name or "resnext" in self.model_name:
            global_ft, local_ft = self._resnet_forward(x)
        elif "densenet" in self.model_name:
            global_ft, local_ft = self._densenet_forward(x)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        if get_local:
            return global_ft, local_ft
        else:
            return global_ft


    def generate_embeddings(self, global_features: torch.Tensor, 
                           local_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate embeddings from extracted features.
        
        Args:
            global_features: Global features from CNN backbone
            local_features: Local features from CNN backbone
            
        Returns:
            global_emb: Global embeddings
            local_emb: Local embeddings
        """
        global_emb = self.global_embedder(global_features)
        local_emb = self.local_embedder(local_features)
        
        return global_emb, local_emb


    def _resnet_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ResNet-based models.
        
        Args:
            x: Input tensor
            
        Returns:
            global_features: Global features
            local_features: Local features for region-level analysis
        """
        # Resize input to expected dimensions
        x = nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)
        x= x[:10,]   # 🛠️
        # Extract features through backbone layers
        x = self.model.conv1(x)   # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)

        # Store intermediate features for local representation
        local_features = x
        
        x = self.model.layer4(x) # (batch_size, 512, 10, 10)
        
        # Global pooling and reshape
        x = self.pool(x)
        global_features = torch.flatten(x, 1)
        
        return global_features, local_features


    def _densenet_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DenseNet-based models.
        
        Args:
            x: Input tensor
            
        Returns:
            global_features: Global features
            local_features: Local features for region-level analysis
        """
        # To be implemented based on DenseNet architecture
        raise NotImplementedError("DenseNet forward pass not implemented yet")


    def init_weights(self):
        """Initialize trainable weights with uniform distribution."""
        init_range = 0.1
        
        # Use proper weight initialization for linear and conv layers
        if hasattr(self, 'global_embedder'):
            nn.init.uniform_(self.global_embedder.weight, -init_range, init_range)
            
        if hasattr(self, 'local_embedder'):
            nn.init.uniform_(self.local_embedder.weight, -init_range, init_range)



class PretrainedImageClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_cls: int,
        feature_dim: int,
        freeze_encoder: bool = True,
    ):
        super(PretrainedImageClassifier, self).__init__()
        self.img_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, num_cls)
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred


class ImageClassifier(nn.Module):
    def __init__(self, cfg, image_encoder=None):
        super(ImageClassifier, self).__init__()

        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.img_encoder, self.feature_dim, _ = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.classifier = nn.Linear(self.feature_dim, cfg.model.vision.num_targets)

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred
