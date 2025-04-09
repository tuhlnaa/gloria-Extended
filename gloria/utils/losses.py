"""
This module implements attention and loss functions for the GLoRIA framework.
Adapted from: https://github.com/mrlibw/ControlGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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