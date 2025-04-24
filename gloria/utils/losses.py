"""
This module implements attention and loss functions for the GLoRIA framework.
Adapted from: https://github.com/mrlibw/ControlGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from torchvision.ops import sigmoid_focal_loss
from collections import namedtuple
from typing import Tuple, List, Literal


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class AttentionModule(nn.Module):
    """
    Compute attention between query and context tensors.
    """
    def __init__(self, temperature: float = 4.0):
        """
        Args:
            temperature: Scaling factor for attention weights
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Tensor of shape (batch_size, embedding_dim, query_length)
            context: Tensor of shape (batch_size, embedding_dim, height, width)
            
        Returns:
            weighted_context: Context weighted by attention of shape (batch_size, embedding_dim, query_length)
            attention_map: Attention map of shape (batch_size, query_length, height, width)
        """
        batch_size, embedding_dim, query_length = query.size()
        height, width = context.size(2), context.size(3)
        context_length = height * width
        
        # Reshape context: (batch_size, embedding_dim, context_length)
        context_flat = context.view(batch_size, embedding_dim, context_length)
        
        # Transpose for batched matrix multiplication: (batch_size, context_length, embedding_dim)
        context_transposed = context_flat.transpose(1, 2)
        
        # Compute raw attention scores: (batch_size, context_length, query_length)
        attention_scores = torch.bmm(context_transposed, query)
        
        # Normalize attention over query dimension
        attention_query = F.softmax(attention_scores.view(batch_size * context_length, query_length), dim=-1)
        attention_query = attention_query.view(batch_size, context_length, query_length)
        
        # Transpose and reshape attention: (batch_size, query_length, context_length)
        attention_context = attention_query.transpose(1, 2).contiguous()
        attention_context = attention_context.view(batch_size * query_length, context_length)
        
        # Apply temperature scaling and normalize over context dimension
        attention_context = attention_context * self.temperature
        attention_context = F.softmax(attention_context, dim=-1)
        attention_context = attention_context.view(batch_size, query_length, context_length)
        
        # Transpose back for weighted context computation: (batch_size, context_length, query_length)
        attention_context_transposed = attention_context.transpose(1, 2)
        
        # Compute weighted context: (batch_size, embedding_dim, query_length)
        weighted_context = torch.bmm(context_flat, attention_context_transposed)
        
        # Reshape attention map to original spatial dimensions
        attention_map = attention_context.view(batch_size, query_length, height, width)
        
        return weighted_context, attention_map


class GlobalLoss(nn.Module):
    """
    Module for computing global contrastive loss between image and text embeddings.
    """
    def __init__(self, temperature: float = 10.0, eps: float = 1e-8):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global contrastive loss between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings of shape (batch_size, embedding_dim) or (1, batch_size, embedding_dim)
            text_embeddings: Text embeddings of shape (batch_size, embedding_dim) or (1, batch_size, embedding_dim)
            
        Returns:
            image_to_text_loss: Loss for image-to-text matching
            text_to_image_loss: Loss for text-to-image matching
        """
        batch_size = image_embeddings.shape[0]
        
        # Ensure embeddings are 3D tensors (needed for batch matrix multiplication)
        if image_embeddings.dim() == 2:
            image_embeddings = image_embeddings.unsqueeze(0)
            text_embeddings = text_embeddings.unsqueeze(0)
        
        # Normalize embeddings
        image_norm = torch.norm(image_embeddings, p=2, dim=2, keepdim=True)
        text_norm = torch.norm(text_embeddings, p=2, dim=2, keepdim=True)
        
        # Compute normalized cosine similarities
        similarities = torch.bmm(image_embeddings, text_embeddings.transpose(1, 2))
        norm_factor = torch.bmm(image_norm, text_norm.transpose(1, 2)).clamp(min=self.eps)
        similarities = similarities / norm_factor * self.temperature
        
        # Convert to 2D matrix
        similarities = similarities.squeeze()
        
        # Compute bidirectional losses
        contrastive_loss = ContrastiveLoss(temperature=1.0)  # Temperature already applied
        image_to_text_loss, text_to_image_loss = contrastive_loss(similarities)
        
        return image_to_text_loss, text_to_image_loss


class ContrastiveLoss(nn.Module):
    """
    Module for computing bidirectional contrastive loss (InfoNCE) with temperature scaling.
    """
    def __init__(self, temperature: float = 10.0, eps: float = 1e-8):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        

    def forward(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional contrastive loss.
        
        Args:
            predictions: Similarity matrix of shape (batch_size, batch_size)
            
        Returns:
            forward_loss: Loss for image-to-text direction
            backward_loss: Loss for text-to-image direction
        """
        batch_size = predictions.size(0)
        labels = torch.arange(batch_size, device=predictions.device)
        
        # Temperature-scaled cross-entropy loss in both directions
        forward_loss = F.cross_entropy(predictions * self.temperature, labels)
        backward_loss = F.cross_entropy(predictions.t() * self.temperature, labels)
        
        return forward_loss, backward_loss


class LocalLoss(nn.Module):
    """
    Module for computing local contrastive loss between image regions and words.
    """
    def __init__(
            self, 
            temp_attention: float = 4.0, 
            temp_similarity: float = 5.0, 
            temp_loss: float = 10.0, 
            aggregation: Literal["sum", "mean"] = "sum"
        ):
        """
        Args:
            temp_attention: Temperature for attention computation
            temp_similarity: Temperature for word-context similarity
            temp_loss: Temperature for final contrastive loss
            aggregation: Method to aggregate word-region similarities ("sum" or "mean")
        """
        super().__init__()
        self.temp_attention = temp_attention
        self.temp_similarity = temp_similarity
        self.temp_loss = temp_loss
        self.aggregation = aggregation
        self.attention_module = AttentionModule(temperature=temp_attention)
        

    def forward(
            self, 
            image_features: torch.Tensor, 
            word_embeddings: torch.Tensor, 
            caption_lengths: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Compute local contrastive loss between image regions and words.
        
        Args:
            image_features: Visual features of shape (batch_size, embedding_dim, height, width)
            word_embeddings: Word embeddings of shape (batch_size, embedding_dim, max_seq_length)
            caption_lengths: Length of each caption (without padding) of shape (batch_size,)
            
        Returns:
            image_to_text_loss: Loss for image-to-text matching
            text_to_image_loss: Loss for text-to-image matching
            attention_maps: List of attention maps for visualization
        """
        batch_size = image_features.shape[0]
        device = image_features.device
        
        attention_maps = []
        similarities = []
        
        # Process each text in the batch
        for i in range(word_embeddings.shape[0]):
            # Get word embeddings for current caption (without padding)
            words_length = caption_lengths[i]
            words = word_embeddings[i, :, :words_length].unsqueeze(0)
            
            # Repeat the text features for each image in batch
            words_batch = words.repeat(batch_size, 1, 1)
            
            # Compute cross-attention between words and image regions
            weighted_context, attention = self.attention_module(words_batch, image_features)
            
            # Store attention map for visualization
            attention_maps.append(attention[i].unsqueeze(0))
            
            # Transpose for similarity computation
            words_batch = words_batch.transpose(1, 2).contiguous()  # (batch_size, words_length, embedding_dim)
            weighted_context = weighted_context.transpose(1, 2).contiguous()  # (batch_size, words_length, embedding_dim)
            
            # Flatten batch and word dimensions
            words_flat = words_batch.reshape(batch_size * words_length, -1)
            context_flat = weighted_context.reshape(batch_size * words_length, -1)
            
            # Compute cosine similarity for each word-region pair
            word_region_similarities = F.cosine_similarity(words_flat, context_flat)
            word_region_similarities = word_region_similarities.view(batch_size, words_length)
            
            # Apply temperature scaling
            word_region_similarities = torch.exp(word_region_similarities * self.temp_similarity)
            
            # Aggregate similarities across words
            if self.aggregation == "sum":
                caption_similarity = word_region_similarities.sum(dim=1, keepdim=True)
            else:  # mean
                caption_similarity = word_region_similarities.mean(dim=1, keepdim=True)
            
            # Apply log for numerical stability
            caption_similarity = torch.log(caption_similarity)
            similarities.append(caption_similarity)
        
        # Concatenate similarities from all captions
        similarity_matrix = torch.cat(similarities, dim=1)
        
        # Scale by temperature and compute losses
        similarity_matrix = similarity_matrix * self.temp_loss
        
        # Compute bidirectional losses
        labels = torch.arange(batch_size, device=device)
        image_to_text_loss = F.cross_entropy(similarity_matrix, labels)
        text_to_image_loss = F.cross_entropy(similarity_matrix.t(), labels)
        
        return image_to_text_loss, text_to_image_loss, attention_maps


class GloriaLoss(nn.Module):
    """
    GLoRIA: A Global-Local Representation Learning Framework for medical images.
    
    This model learns multimodal representations by contrasting image regions
    with words in paired radiology reports.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Initialize loss modules
        self.local_loss_fn = LocalLoss(
            temp_attention=config.model.gloria.temp_attention,
            temp_similarity=config.model.gloria.temp_similarity,
            temp_loss=config.model.gloria.temp_loss
        )
        
        self.global_loss_fn = GlobalLoss(
            temperature=config.model.gloria.temp_loss
        )
        
        # Get loss weights from config
        self.local_loss_weight = config.model.gloria.local_loss_weight
        self.global_loss_weight = config.model.gloria.global_loss_weight
        

    def forward(self, img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents):
        """
        Compute the combined global and local losses.
        
        Args:
            img_emb_local: Local image embeddings
            img_emb_global: Global image embedding
            text_emb_local: Local text embeddings
            text_emb_global: Global text embedding
            sents: Tokenized sentences
            
        Returns:
            LossResult: A namedtuple containing all loss components and attention maps
        """
        # Create a namedtuple to store all results
        LossResult = namedtuple('LossResult', [
            'total_loss', 'attn_maps', 'global_loss', 'local_loss',
            'local_loss_image_to_text', 'local_loss_text_to_image',
            'global_loss_image_to_text', 'global_loss_text_to_image'
        ])

        # Compute local loss (between image regions and words)
        local_loss_image_to_text, local_loss_text_to_image, attn_maps = self._compute_local_loss(
            img_emb_local, text_emb_local, sents
        )
        
        # Compute global loss (between global image and text embeddings)
        global_loss_image_to_text, global_loss_text_to_image = self._compute_global_loss(
            img_emb_global, text_emb_global
        )
        
        # Combine losses with weights
        local_loss = (local_loss_image_to_text + local_loss_text_to_image) * self.local_loss_weight
        global_loss = (global_loss_image_to_text + global_loss_text_to_image) * self.global_loss_weight
        total_loss = local_loss + global_loss
        
        return LossResult(
            total_loss=total_loss,
            attn_maps=attn_maps,
            global_loss=global_loss,
            local_loss=local_loss,
            local_loss_image_to_text=local_loss_image_to_text,
            local_loss_text_to_image=local_loss_text_to_image,
            global_loss_image_to_text=global_loss_image_to_text,
            global_loss_text_to_image=global_loss_text_to_image
        )


    def _compute_local_loss(self, img_emb_local, text_emb_local, sents):
        """
        Compute local contrastive loss between image regions and words.
        
        Args:
            img_emb_local: Local image embeddings
            text_emb_local: Local text embeddings
            sents: Tokenized sentences
            
        Returns:
            Tuple containing:
                - local_loss_i2t: Loss for image-to-text matching
                - local_loss_t2i: Loss for text-to-image matching
                - attn_maps: Attention maps for visualization
        """
        # Calculate caption lengths excluding special tokens
        caption_lengths = [
            len([word for word in sent if not word.startswith("[")]) + 1 
            for sent in sents
        ]
        
        # Convert to tensor if not already
        if not isinstance(caption_lengths, torch.Tensor):
            caption_lengths = torch.tensor(caption_lengths, device=img_emb_local.device)
        
        # Compute local loss
        local_loss_i2t, local_loss_t2i, attn_maps = self.local_loss_fn(
            img_emb_local,
            text_emb_local,
            caption_lengths
        )
        
        return local_loss_i2t, local_loss_t2i, attn_maps


    def _compute_global_loss(self, img_emb_global, text_emb_global):
        """
        Compute global contrastive loss between image and text embeddings.
        
        Args:
            img_emb_global: Global image embedding
            text_emb_global: Global text embedding
            
        Returns:
            Tuple containing:
                - global_loss_i2t: Loss for image-to-text matching
                - global_loss_t2i: Loss for text-to-image matching
        """
        return self.global_loss_fn(img_emb_global, text_emb_global)
    

class CombinedBinaryLoss(nn.Module):
    def __init__(self, config, dice_weight=0.5, focal_weight=0.5, bce_weight=0.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        """
        For binary medical image segmentation tasks, the combination of Dice and Focal Loss often provides the best results because:
        Dice Loss directly optimizes the evaluation metric
        Focal Loss addresses class imbalance while providing stable gradients
        Including BCE is less critical when using Focal Loss, as Focal Loss is an enhanced version of BCE designed specifically for class imbalance.
        """
        super().__init__()
        self.config = config
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
    
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # For binary segmentation
        self.dice = smp.losses.DiceLoss(mode=config.criterion.label_mode)
        
        # Only used if bce_weight > 0
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, y_pred, y_true):
        # For binary segmentation:
        # - y_pred should be shape (B, 1, H, W) or (B, H, W) with logits
        # - y_true should be shape (B, 1, H, W) or (B, H, W) with values 0 or 1

        # Calculate individual losses
        dice_loss = self.dice(y_pred, y_true)
        
        # Focal loss
        if self.config.criterion.label_mode == 'binary':
            y_pred = y_pred.squeeze(1)

        focal_loss = sigmoid_focal_loss(
            y_pred, y_true,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean"
        )
        
        # Calculate BCE only if weight > 0
        bce_loss = self.bce(y_pred, y_true) if self.bce_weight > 0 else 0
        
        # Combine losses with weights
        return (self.dice_weight * dice_loss + 
                self.focal_weight * focal_loss + 
                self.bce_weight * bce_loss)




import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Union


def dice_coef(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Calculate Dice coefficient.
    
    Args:
        y_pred: Predicted binary masks after sigmoid (B, H*W) or (B, 1, H*W)
        y_true: Ground truth binary masks (B, H*W) or (B, 1, H*W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient value
    """
    y_pred = y_pred.view(y_pred.shape[0], -1)
    y_true = y_true.view(y_true.shape[0], -1)
    
    intersection = (y_pred * y_true).sum(dim=1)
    union = y_pred.sum(dim=1) + y_true.sum(dim=1)
    
    return (2.0 * intersection + smooth) / (union + smooth)


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation tasks.
    
    Computes 1 - Dice coefficient as the loss value.
    """
    
    def __init__(
        self, 
        smooth: float = 1.0, 
        p: int = 1, 
        reduction: Literal["mean", "sum", "none"] = "mean"
    ):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            p: Power for input tensors (p=1 for standard dice, p=2 for squared inputs)
            reduction: Reduction method for batch loss values
        """
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass for Dice Loss calculation.
        
        Args:
            y_pred: Predicted logits (B, 1, H, W) or (B, H, W)
            y_true: Ground truth masks with values 0 or 1 (B, 1, H, W) or (B, H, W)
            
        Returns:
            Dice loss tensor based on specified reduction method
        """
        y_pred = torch.sigmoid(y_pred)
        
        # Ensure same dimensions
        if y_pred.dim() == 4 and y_pred.shape[1] == 1:
            y_pred = y_pred.squeeze(1)
        if y_true.dim() == 4 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)
            
        # Flatten predictions and targets
        y_pred = y_pred.contiguous().view(y_pred.shape[0], -1)
        y_true = y_true.contiguous().view(y_true.shape[0], -1)
        
        # Apply power if needed (p=1 is standard dice, no change)
        if self.p > 1:
            y_pred = y_pred.pow(self.p)
            y_true = y_true.pow(self.p)
        
        intersection = (y_pred * y_true).sum(dim=1)
        union = y_pred.sum(dim=1) + y_true.sum(dim=1)
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice_score
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unexpected reduction method: {self.reduction}")


class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation tasks.
    
    Adds a modulating factor to cross-entropy loss to focus more
    on hard examples and less on well-classified examples.
    """
    
    def __init__(self, gamma: float = 2.0):
        """Initialize Focal Loss.
        
        Args:
            gamma: Focusing parameter. Higher values give more weight to hard examples.
        """
        super().__init__()
        self.gamma = gamma
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass for Focal Loss calculation.
        
        Args:
            y_pred: Predicted logits (B, 1, H, W) or (B, H, W)
            y_true: Ground truth masks with values 0 or 1 (B, 1, H, W) or (B, H, W)
            
        Returns:
            Focal loss value (mean across batch)
        """
        # Ensure shapes match
        if y_pred.shape != y_true.shape:
            if y_pred.dim() == 4 and y_pred.shape[1] == 1 and y_true.dim() == 3:
                y_pred = y_pred.squeeze(1)
            elif y_true.dim() == 4 and y_true.shape[1] == 1 and y_pred.dim() == 3:
                y_true = y_true.squeeze(1)
            else:
                raise ValueError(
                    f"Target size {y_true.shape} must be compatible with input size {y_pred.shape}"
                )
        
        # Numerically stable implementation of focal loss with BCE
        max_val = (-y_pred).clamp(min=0)
        
        # Binary cross-entropy calculation
        bce_loss = y_pred - y_pred * y_true + max_val + ((-max_val).exp() + (-y_pred - max_val).exp()).log()
        
        # Apply focal weighting
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt).pow(self.gamma)
        
        return (focal_weight * bce_loss).mean()


class MixedLoss(nn.Module):
    """Combined loss using both Focal Loss and Dice Loss for binary segmentation.
    
    Helps balance pixel-wise accuracy and structural similarity.
    """
    
    def __init__(self, alpha: float = 10.0, gamma: float = 2.0, dice_smooth: float = 1.0):
        """Initialize Mixed Loss.
        
        Args:
            alpha: Weight for the Focal Loss component
            gamma: Focusing parameter for Focal Loss
            dice_smooth: Smoothing factor for Dice coefficient
        """
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.dice_smooth = dice_smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward pass for Mixed Loss calculation.
        
        Args:
            y_pred: Predicted logits (B, 1, H, W) or (B, H, W)
            y_true: Ground truth masks with values 0 or 1 (B, 1, H, W) or (B, H, W)
            
        Returns:
            Combined loss value
        """
        # Calculate individual loss components
        focal_loss = self.focal(y_pred, y_true)
        dice_score = dice_coef(torch.sigmoid(y_pred), y_true, self.dice_smooth)
        
        # Combine losses (alpha * focal_loss - log(dice_score))
        return self.alpha * focal_loss - torch.log(dice_score.mean())
    

def test_combined_binary_loss():
    """
    Test function for CombinedBinaryLoss with fixed input values
    """
    # Create test tensors with fixed values
    # Batch size = 2, 1 channel, 4x4 images
    
    # Create predictions (logits, before sigmoid)
    y_pred = torch.tensor([
        # First image in batch - checkerboard pattern of low and high values
        [[[0.2, 0.9, 0.2, 0.9],
          [0.9, 0.2, 0.9, 0.2],
          [0.2, 0.9, 0.2, 0.9],
          [0.9, 0.2, 0.9, 0.2]]],
        
        # Second image in batch - diagonal pattern with medium values
        [[[0.5, 0.6, 0.5, 0.6],
          [0.6, 0.5, 0.6, 0.5],
          [0.5, 0.6, 0.5, 0.6],
          [0.6, 0.5, 0.6, 0.5]]]
    ], dtype=torch.float32)
    
    # Create ground truth binary masks
    y_true = torch.tensor([
        # First image ground truth - checkerboard pattern
        [[[0, 1, 0, 1],
          [1, 0, 1, 0],
          [0, 1, 0, 1],
          [1, 0, 1, 0]]],
        
        # Second image ground truth - diagonal pattern
        [[[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]]
    ], dtype=torch.float64)
    
    # Test Case 1: Default weights (0.5 Dice, 0.5 Focal, 0.0 BCE)
    loss_fn1 = MixedLoss()
    loss1 = loss_fn1(y_pred, y_true)
    print(f"Test Case 1 - Default weights (0.5 Dice, 0.5 Focal):")
    print(f"Combined loss: {loss1.item()}")

    loss_fn1 = FocalLoss()
    loss1 = loss_fn1(y_pred, y_true)
    print(f"Test Case 1 - Default weights (0.5 Dice, 0.5 Focal):")
    print(f"Combined loss: {loss1.item()}")

    loss_fn1 = DiceLoss()
    loss1 = loss_fn1(y_pred, y_true)
    print(f"Test Case 1 - Default weights (0.5 Dice, 0.5 Focal):")
    print(f"Combined loss: {loss1.item()}")

if __name__ == "__main__":
    test_combined_binary_loss()