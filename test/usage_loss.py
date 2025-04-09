import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

from refactored_losses import GloriaLoss


class GloriaModel(nn.Module):
    """
    Example model class that uses the GloriaLoss module.
    """
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        embedding_dim: int = 512,
        local_loss_weight: float = 1.0,
        global_loss_weight: float = 1.0,
        temp_attention: float = 4.0,
        temp_similarity: float = 5.0,
        temp_loss: float = 10.0
    ):
        """
        Initialize the GLORIA model.
        
        Args:
            image_encoder: Module for encoding images
            text_encoder: Module for encoding text
            embedding_dim: Dimension of the joint embedding space
            local_loss_weight: Weight for local loss component
            global_loss_weight: Weight for global loss component
            temp_attention: Temperature for attention computation
            temp_similarity: Temperature for word-context similarity
            temp_loss: Temperature for final contrastive loss
        """
        super().__init__()
        
        # Encoders
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # Loss module
        self.loss_module = GloriaLoss(
            local_loss_weight=local_loss_weight,
            global_loss_weight=global_loss_weight,
            temp_attention=temp_attention,
            temp_similarity=temp_similarity,
            temp_loss=temp_loss
        )
    
    def forward(
        self, 
        images: torch.Tensor, 
        texts: Dict[str, torch.Tensor],
        return_embeddings: bool = False
    ):
        """
        Forward pass of the GLORIA model.
        
        Args:
            images: Batch of images
            texts: Dictionary containing tokenized texts and attention masks
            return_embeddings: If True, return embeddings along with loss
            
        Returns:
            If training:
                loss_result: Result from the loss module
                embeddings (optional): Dictionary of embeddings if return_embeddings is True
            If not training:
                embeddings: Dictionary of embeddings
        """
        # Extract local and global image embeddings
        img_emb_local, img_emb_global = self.image_encoder(images)
        
        # Extract local and global text embeddings
        text_emb_local, text_emb_global = self.text_encoder(
            texts['input_ids'], 
            attention_mask=texts['attention_mask']
        )
        
        # Get caption lengths (assuming texts contains this information)
        caption_lengths = texts['lengths']
        
        # Store embeddings in a dictionary for possible return
        embeddings = {
            'img_emb_local': img_emb_local,
            'img_emb_global': img_emb_global,
            'text_emb_local': text_emb_local,
            'text_emb_global': text_emb_global
        }
        
        # If not in training mode, just return the embeddings
        if not self.training:
            return embeddings
        
        # Compute loss
        loss_result = self.loss_module(
            img_emb_local=img_emb_local,
            img_emb_global=img_emb_global,
            text_emb_local=text_emb_local,
            text_emb_global=text_emb_global,
            caption_lengths=caption_lengths
        )
        
        # Return embeddings along with loss if requested
        if return_embeddings:
            return loss_result, embeddings
        
        return loss_result


# Example usage:
def example_training_step(model, batch, optimizer):
    """Example of a training step using the GloriaModel."""
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    loss_result = model(batch['images'], batch['texts'])
    
    # Compute gradients and update weights
    loss_result.total_loss.backward()
    optimizer.step()
    
    # Return losses for logging
    return {
        'total_loss': loss_result.total_loss.item(),
        'global_loss': loss_result.global_loss.item(),
        'local_loss': loss_result.local_loss.item(),
    }


def example_inference(model, image, text):
    """Example of inference using the GloriaModel."""
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        embeddings = model(image, text)
        
        # Compute similarity between image and text
        similarity = torch.nn.functional.cosine_similarity(
            embeddings['img_emb_global'],
            embeddings['text_emb_global'],
            dim=1
        )
    
    return similarity