import torch
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from typing import List, Dict, Optional, Tuple, Any

from .. import utils
from ..loss.gloria_loss import compute_attention


class Retriever:
    """A retriever for finding similar text reports for medical images using GLoRIA embeddings.
    
    This class implements image-to-text retrieval functionality based on the GLoRIA
    (Global-Local Representation Learning) framework for medical images and reports.
    """
    
    def __init__(
            self, 
            ckpt_path: str, 
            targets: Optional[List[str]] = None, 
            target_classes: Optional[List[Any]] = None, 
            device: Optional[torch.device] = None, 
            top_k: int = 5
        ):
        """Initialize the GLoRIA-based retriever.
        
        Args:
            ckpt_path: Path to GLoRIA model checkpoint
            targets: List of target text reports
            target_classes: Class labels for the target reports
            device: Computation device (defaults to CUDA if available)
            top_k: Number of top results to retrieve
        """
        # Set device
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load GLoRIA model
        self.gloria = utils.load_gloria(ckpt_path).to(device=self.device)
        self.top_k = top_k
        
        # Process target reports if provided
        self.targets = self._process_targets(targets) if targets is not None else None
        self.target_classes = np.array(target_classes) if target_classes is not None else None
    

    def _process_targets(self, targets: List[str]) -> Dict[str, torch.Tensor]:
        """Process text targets to compute their embeddings."""
        # Process text through BERT tokenizer
        text_tensors = self.gloria.process_text(targets)
        
        # Stack and transfer tensors to device
        caption_ids = torch.stack([x["input_ids"] for x in text_tensors]).squeeze().to(self.device)
        attention_mask = torch.stack([x["attention_mask"] for x in text_tensors]).squeeze().to(self.device)
        token_type_ids = torch.stack([x["token_type_ids"] for x in text_tensors]).squeeze().to(self.device)
        
        # Compute text embeddings
        with torch.no_grad():
            text_emb_local, text_emb_global, sentences = self.gloria.encode_text(
                caption_ids, attention_mask, token_type_ids
            )
        
        # Calculate caption lengths (excluding special tokens)
        self.cap_lens = [len([w for w in sent if not w.startswith("[")]) for sent in sentences]
        
        # Remove [CLS] token from local embeddings
        text_emb_local = text_emb_local[:, :, 1:]
        
        return {
            "global_embeddings": text_emb_global.detach().cpu(),
            "local_embeddings": text_emb_local.detach().cpu(),
            "processed_input": targets,
        }
    

    def _process_source(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process source image to compute its embeddings."""
        with torch.no_grad():
            img_processed = self.gloria.process_img(img)
            img_tensors = torch.stack(img_processed).to(self.device)
            img_emb_local, img_emb_global = self.gloria.encode_images(img_tensors)
        
        # Free memory
        del img_tensors
        
        return {
            "global_embeddings": img_emb_global.detach().cpu(),
            "local_embeddings": img_emb_local.detach().cpu(),
            "processed_input": img_processed,
        }
    

    def _compute_local_similarity(
            self,
            img_features: torch.Tensor,
            words_emb: torch.Tensor,
            cap_lens: List[int],
            temp1: float = 4.0,
            temp2: float = 5.0,
            temp3: float = 10.0,
            agg: str = "sum"
        ) -> np.ndarray:
        """Compute local similarity between image features and word embeddings.
        
        Args:
            img_features: Image feature embeddings
            words_emb: Word embeddings from text
            cap_lens: List of caption lengths
            temp1: Temperature for attention computation
            temp2: Temperature for word-context similarity
            temp3: Temperature for final scaling
            agg: Aggregation method ('sum' or 'mean')
            
        Returns:
            Array of similarity scores
        """
        batch_size = words_emb.shape[0]
        similarities = []
        
        for i in range(batch_size):
            # Get words for this caption (excluding special tokens)
            words_num = cap_lens[i]
            word = words_emb[i, :, 1:words_num + 1].unsqueeze(0).contiguous()
            context = img_features
            
            # Compute attention-weighted context
            weighted_context, _ = compute_attention(word, context, temp1)
            
            # Transpose for similarity computation
            word = word.transpose(1, 2).contiguous().squeeze()
            weighted_context = weighted_context.transpose(1, 2).contiguous().squeeze()
            
            # Compute cosine similarity
            row_sim = F.cosine_similarity(word, weighted_context).squeeze()
            
            # Apply temperature scaling
            row_sim = row_sim.mul(temp2).exp_()
            
            # Aggregate similarities
            if agg == "sum":
                row_sim = row_sim.sum()
            else:  # mean
                row_sim = row_sim.mean()
                
            row_sim = torch.log(row_sim)
            similarities.append(row_sim.item())
        
        return np.array(similarities) * temp3
    

    def retrieve(
            self, 
            source: torch.Tensor, 
            similarity_type: str = "both"
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Retrieve similar reports for the given source image.
        
        Args:
            source: Source image tensor
            similarity_type: Type of similarity to use ('local', 'global', or 'both')
            
        Returns:
            Tuple of (retrieved_reports, retrieved_classes)
        """
        valid_similarity_types = ["both", "local", "global"]
        if similarity_type not in valid_similarity_types:
            raise ValueError(
                f"similarity_type must be one of {valid_similarity_types}"
            )
        
        # Process source image
        self.source = self._process_source(source)
        
        # Compute local similarities
        local_similarities = self._compute_local_similarity(
            self.source["local_embeddings"],
            self.targets["local_embeddings"],
            self.cap_lens,
            self.gloria.temp1,
            self.gloria.temp2,
            self.gloria.temp3,
        )
        
        # Compute global similarities using cosine similarity
        global_similarities = metrics.pairwise.cosine_similarity(
            self.source["global_embeddings"], 
            self.targets["global_embeddings"]
        )[0]
        
        # Combine similarities based on specified type
        if similarity_type == "local":
            similarities = local_similarities
        elif similarity_type == "global":
            similarities = global_similarities
        else:  # "both"
            # Normalize before combining
            def normalize(x):
                return (x - x.mean(axis=0)) / (x.std(axis=0))
            
            similarities = np.stack([
                normalize(local_similarities), 
                normalize(global_similarities)
            ]).mean(axis=0)
        
        # Get indices of top-k similar items
        sorted_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        # Return retrieved reports and their classes if available
        retrieved_classes = self.target_classes[sorted_indices] if self.target_classes is not None else None
        
        return np.array(self.targets["processed_input"])[sorted_indices], retrieved_classes