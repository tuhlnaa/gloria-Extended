import torch
import torch.nn as nn
from rich import print
from typing import Optional, Tuple, Union

from gloria.models import cnn_backbones

class ImageEncoder(nn.Module):
    """
    Image encoder for GLoRIA model that extracts both global and local features.
    
    This encoder utilizes CNN backbones to extract features and provides 
    embeddings for both global image context and local image regions.
    
    Attributes:
        output_dim: Dimension of output embeddings
        model_name: Name of the CNN backbone model
        model: CNN backbone model for feature extraction
        feature_dim: Dimension of global features
        interm_feature_dim: Dimension of intermediate features
        global_embedder: Linear layer for global feature embedding
        local_embedder: Convolutional layer for local feature embedding
        pool: Global pooling layer
    """
    def __init__(self, config):
        super().__init__()
        
        self.output_dim = config.model.text.embedding_dim
        self.model_name = config.model.vision.model_name
        
        self.model, self.feature_dim, self.interm_feature_dim = cnn_backbones.get_backbone(
            name=self.model_name, 
            weights=config.model.vision.pretrained
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
            self._freeze_backbone()


    def _get_backbone(self, model_name: str, pretrained: bool = True) -> Tuple:
        """Initialize the backbone CNN model."""
        model_function = getattr(cnn_backbones, model_name)
        return model_function(weights=pretrained)
    

    def _freeze_backbone(self) -> None:
        """Freeze all parameters in the backbone CNN."""
        for param in self.model.parameters():
            param.requires_grad = False
        print("[bold blue]Training only the classifier head.[/bold blue]")


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
        """Forward pass for ResNet-based models."""
        # Resize input to expected dimensions
        x = nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)
        
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
        """Forward pass for DenseNet-based models."""
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


class ImageClassifier(nn.Module):
    """
    Generic image classifier implementing transfer learning with configurable backbone architectures.
    
    This class supports two common transfer learning approaches:
    1. Feature extraction: Uses a frozen pretrained backbone as a fixed feature extractor,
       training only the classifier head (when freeze_encoder=True)
    2. Fine-tuning: Trains both the backbone and classifier together, allowing the
       backbone to adapt to the new task (when freeze_encoder=False)
    
    Attributes:
        img_encoder: Backbone CNN for feature extraction (can be frozen or trainable)
        feature_dim: Dimension of extracted features from the backbone
        classifier: Linear classifier head that maps features to class predictions
    """
    def __init__(
            self, 
            config, 
            num_classes: Optional[int] = None, 
            pretrained: Optional[str] = None, 
            freeze_encoder: bool = False    
        ):
        """
        dropout_rate = 0.5 is a common starting point that provides a balance between regularization and maintaining network capacity
        Smaller datasets or more complex models might benefit from higher dropout rates (0.5-0.7)
        Larger datasets or simpler models might work better with lower dropout rates (0.2-0.4)
        This should ideally be tuned as a hyperparameter

        For ResNet50 (2048 features), hidden layers of 512-1024 are common (1/4 to 1/2)
        For large models, keeping more capacity (like 1/2 or 1/3 reduction) often works better
        For simpler tasks or smaller datasets, more aggressive reduction (1/8 or 1/16) can help prevent overfitting
        """
        super().__init__()

        if pretrained is None:
            pretrained = config.model.vision.pretrained

        if num_classes is None:
            num_classes = config.model.vision.num_targets

        hidden_dim = getattr(config.model.optimization, 'hidden_dim', 512)
        dropout_rate = getattr(config.model.optimization, 'dropout_rate', 0.5)

        self.img_encoder, self.feature_dim, _ = cnn_backbones.get_backbone(
            name=config.model.vision.model_name, 
            weights=pretrained
        )

        if config.model.optimization.use:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False
            print["[bold blue]Training only the classifier head.[/bold blue]"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier."""
        features = self.img_encoder(x)
        predictions = self.classifier(features)
        return predictions


class GloriaImageClassifier(ImageClassifier):
    """
    Image classifier that uses a pre-trained GLoRIA image encoder.
    
    This classifier leverages the representations learned by the GLoRIA image encoder
    for downstream classification tasks.
    """
    def __init__(
        self,
        image_encoder: ImageEncoder,
        num_classes: int,
        feature_dim: int,
        freeze_encoder: bool = True,
    ):
        # Skip parent's __init__ and go to grandparent (nn.Module)
        nn.Module.__init__(self)
        
        self.img_encoder = image_encoder
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False
            print("[bold blue]Training only the classifier head.[/bold blue]")
