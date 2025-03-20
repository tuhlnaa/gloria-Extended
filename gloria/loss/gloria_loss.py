"""
This module implements attention and loss functions for the GLoRIA framework.
Adapted from: https://github.com/mrlibw/ControlGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List, Optional, Literal


from torch.autograd import Variable


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def compute_attention(query: torch.Tensor, context: torch.Tensor, temperature: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention between query and context tensors.
    
    Args:
        query: Tensor of shape (batch_size, embedding_dim, query_length)
        context: Tensor of shape (batch_size, embedding_dim, height, width)
        temperature: Scaling factor for attention weights
        
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
    attention_context = attention_context * temperature
    attention_context = F.softmax(attention_context, dim=-1)
    attention_context = attention_context.view(batch_size, query_length, context_length)
    
    # Transpose back for weighted context computation: (batch_size, context_length, query_length)
    attention_context_transposed = attention_context.transpose(1, 2)
    
    # Compute weighted context: (batch_size, embedding_dim, query_length)
    weighted_context = torch.bmm(context_flat, attention_context_transposed)
    
    # Reshape attention map to original spatial dimensions
    attention_map = attention_context.view(batch_size, query_length, height, width)
    
    return weighted_context, attention_map


def global_loss(cnn_code, rnn_code, eps=1e-8, temp3=10.0):

    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    scores1 = scores0.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1


def local_loss(
        img_features, 
        words_emb, 
        cap_lens, 
        temp1=4.0, 
        temp2=5.0, 
        temp3=10.0, 
        agg="sum"
    ):

    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i]  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]

        weiContext, attn = compute_attention(
            word, context, temp1
        )  # [48, 768, 25], [48, 25, 19, 19]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [25, 19, 19]
        word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [1200, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

        row_sim = F.cosine_similarity(word, weiContext).squeeze()
        row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [48, 48]

    labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

    loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return loss0, loss1, att_maps


def global_loss(
        image_embeddings: torch.Tensor, 
        text_embeddings: torch.Tensor, 
        temperature: float = 10.0, 
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute global contrastive loss between image and text embeddings.
    
    Args:
        image_embeddings: Image embeddings of shape (batch_size, embedding_dim) or (1, batch_size, embedding_dim)
        text_embeddings: Text embeddings of shape (batch_size, embedding_dim) or (1, batch_size, embedding_dim)
        temperature: Temperature parameter for scaling similarities
        eps: Small constant for numerical stability
        
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
    norm_factor = torch.bmm(image_norm, text_norm.transpose(1, 2)).clamp(min=eps)
    similarities = similarities / norm_factor * temperature
    
    # Convert to 2D matrix
    similarities = similarities.squeeze()
    
    # Compute bidirectional losses
    image_to_text_loss, text_to_image_loss = compute_contrastive_loss(
        similarities, temperature=1.0  # Temperature already applied
    )
    
    return image_to_text_loss, text_to_image_loss


def local_loss(
        image_features: torch.Tensor, 
        word_embeddings: torch.Tensor, 
        caption_lengths: torch.Tensor,
        temp_attention: float = 4.0, 
        temp_similarity: float = 5.0, 
        temp_loss: float = 10.0, 
        aggregation: Literal["sum", "mean"] = "sum"
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Compute local contrastive loss between image regions and words.
    
    Args:
        image_features: Visual features of shape (batch_size, embedding_dim, height, width)
        word_embeddings: Word embeddings of shape (batch_size, embedding_dim, max_seq_length)
        caption_lengths: Length of each caption (without padding) of shape (batch_size,)
        temp_attention: Temperature for attention computation
        temp_similarity: Temperature for word-context similarity
        temp_loss: Temperature for final contrastive loss
        aggregation: Method to aggregate word-region similarities ("sum" or "mean")
        
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
        weighted_context, attention = compute_attention(
            words_batch, image_features, temp_attention
        )
        
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
        word_region_similarities = torch.exp(word_region_similarities * temp_similarity)
        
        # Aggregate similarities across words
        if aggregation == "sum":
            caption_similarity = word_region_similarities.sum(dim=1, keepdim=True)
        else:  # mean
            caption_similarity = word_region_similarities.mean(dim=1, keepdim=True)
        
        # Apply log for numerical stability
        caption_similarity = torch.log(caption_similarity)
        similarities.append(caption_similarity)
    
    # Concatenate similarities from all captions
    similarity_matrix = torch.cat(similarities, dim=1)
    
    # Scale by temperature and compute losses
    similarity_matrix = similarity_matrix * temp_loss
    
    # Compute bidirectional losses
    labels = torch.arange(batch_size, device=device)
    image_to_text_loss = F.cross_entropy(similarity_matrix, labels)
    text_to_image_loss = F.cross_entropy(similarity_matrix.t(), labels)
    
    return image_to_text_loss, text_to_image_loss, attention_maps


def compute_contrastive_loss(predictions: torch.Tensor, temperature: float = 10.0, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute bidirectional contrastive loss (InfoNCE) with temperature scaling.
    
    Args:
        predictions: Similarity matrix of shape (batch_size, batch_size)
        temperature: Temperature parameter for scaling similarities
        eps: Small constant for numerical stability
        
    Returns:
        forward_loss: Loss for image-to-text direction
        backward_loss: Loss for text-to-image direction
    """
    batch_size = predictions.size(0)
    labels = torch.arange(batch_size, device=predictions.device)
    
    # Temperature-scaled cross-entropy loss in both directions
    forward_loss = F.cross_entropy(predictions * temperature, labels)
    backward_loss = F.cross_entropy(predictions.t() * temperature, labels)
    
    return forward_loss, backward_loss
