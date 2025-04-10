import torch
import torch.nn as nn

from typing import List, Tuple
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer

class BertEncoder(nn.Module):
    """
    A BERT-based encoder for text processing in multimodal medical image recognition.
    
    This encoder extracts both global and local representations from text using a pre-trained BERT model.
    It supports various aggregation methods and can be configured to handle different embedding strategies.
    """
    
    def __init__(self, config: OmegaConf):
        super().__init__()
        
        # Extract configuration parameters
        text_config = config.model.text
        self.bert_type = text_config.bert_type
        self.last_n_layers = text_config.last_n_layers
        self.aggregate_method = text_config.aggregate_method
        self.norm = text_config.norm
        self.embedding_dim = text_config.embedding_dim
        self.freeze_bert = text_config.freeze_bert
        self.agg_tokens = text_config.agg_tokens
        
        # Initialize BERT model and tokenizer
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.id_to_word = {v: k for k, v in self.tokenizer.get_vocab().items()}
        
        # These will be set later if needed
        self.emb_global = None
        self.emb_local = None
        
        # Freeze BERT parameters if specified
        if self.freeze_bert:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False
    

    def forward(
            self, 
            ids: torch.Tensor, 
            attn_mask: torch.Tensor, 
            token_type: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        """
        Forward pass through the BERT encoder.
        
        Args:
            ids: Token IDs [batch_size, seq_len]
            attn_mask: Attention mask [batch_size, seq_len]
            token_type: Token type IDs [batch_size, seq_len]
            
        Returns:
            Tuple containing:
                - Word embeddings [batch_size, embedding_dim, seq_len]
                - Sentence embeddings [batch_size, embedding_dim]
                - List of sentences, where each sentence is a list of words
        """
        outputs = self.model(ids, attn_mask, token_type)
        
        # Process multiple layers if specified
        if self.last_n_layers > 1:
            # Get the specified number of layers from the end
            all_embeddings = outputs[2]
            embeddings = torch.stack(all_embeddings[-self.last_n_layers:])  # [layers, batch, seq_len, embedding_size]
            embeddings = embeddings.permute(1, 0, 2, 3)  # [batch, layers, seq_len, embedding_size]
            
            # Aggregate tokens if requested
            if self.agg_tokens:
                embeddings, sentences = self._aggregate_tokens(embeddings, ids)
            else:
                sentences = [[self.id_to_word[w.item()] for w in sent] for sent in ids]
            
            # Calculate sentence embeddings by averaging over sequence length
            sent_embeddings = embeddings.mean(dim=2)
            
            # Aggregate over layers
            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(dim=1)
                sent_embeddings = sent_embeddings.sum(dim=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(dim=1)
                sent_embeddings = sent_embeddings.mean(dim=1)
            else:
                raise ValueError(f"Unsupported aggregation method: {self.aggregate_method}")
        else:
            # Use only the last layer
            word_embeddings, sent_embeddings = outputs[0], outputs[1]
            sentences = [[self.id_to_word[w.item()] for w in sent] for sent in ids]
        
        # Apply transformations to embeddings if needed
        batch_size, seq_len, feat_dim = word_embeddings.shape
        
        # Apply local embedding transformation if available
        if self.emb_local is not None:
            word_embeddings = word_embeddings.view(batch_size * seq_len, feat_dim)
            word_embeddings = self.emb_local(word_embeddings)
            word_embeddings = word_embeddings.view(batch_size, seq_len, self.embedding_dim)
        
        # Transpose for compatibility with convolutional operations
        word_embeddings = word_embeddings.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        # Apply global embedding transformation if available
        if self.emb_global is not None:
            sent_embeddings = self.emb_global(sent_embeddings)
        
        # Apply normalization if specified
        if self.norm:
            word_embeddings = self._normalize(word_embeddings, dim=1)
            sent_embeddings = self._normalize(sent_embeddings, dim=1)
        
        return word_embeddings, sent_embeddings, sentences
    

    def _aggregate_tokens(self, embeddings: torch.Tensor, caption_ids: torch.Tensor) -> Tuple[torch.Tensor, List[List[str]]]:
        """
        Aggregate token embeddings into word embeddings by combining subword tokens.
        
        Args:
            embeddings: Token embeddings with shape [batch_size, num_layers, num_words, dim]
            caption_ids: Token IDs with shape [batch_size, num_words]
            
        Returns:
            Tuple containing:
                - Aggregated embeddings with shape [batch_size, num_layers, num_words, dim]
                - List of sentences, where each sentence is a list of words
        """
        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)  # [batch_size, num_words, num_layers, dim]
        aggregated_embeddings_batch = []
        sentences = []
        
        # Process each sample in the batch
        for sample_embeddings, caption_id in zip(embeddings, caption_ids):
            aggregated_embeddings = []
            token_buffer = []
            words = []
            word_buffer = []
            
            # Process each token in the sentence
            for word_embedding, word_id in zip(sample_embeddings, caption_id):
                word = self.id_to_word[word_id.item()]
                
                # Handle end of sentence token
                if word == "[SEP]":
                    if token_buffer:
                        # Combine the accumulated tokens
                        new_embedding = torch.stack(token_buffer).sum(dim=0)
                        aggregated_embeddings.append(new_embedding)
                        words.append("".join(word_buffer))
                    
                    # Add the [SEP] token itself
                    aggregated_embeddings.append(word_embedding)
                    words.append(word)
                    break
                
                # Handle normal words and subwords
                if not word.startswith("##"):
                    if not word_buffer:
                        # Start of a new word
                        token_buffer.append(word_embedding)
                        word_buffer.append(word)
                    else:
                        # End of previous word, start of a new one
                        new_embedding = torch.stack(token_buffer).sum(dim=0)
                        aggregated_embeddings.append(new_embedding)
                        words.append("".join(word_buffer))
                        
                        token_buffer = [word_embedding]
                        word_buffer = [word]
                elif word.startswith("##"):
                    # Continue current word with a subword
                    token_buffer.append(word_embedding)
                    word_buffer.append(word[2:])  # Remove the "##" prefix
            
            # Stack the embeddings for this sample
            sample_aggregated_embeddings = torch.stack(aggregated_embeddings)
            
            # Add padding to match the original sequence length
            padding_size = num_words - len(sample_aggregated_embeddings)
            paddings = torch.zeros(padding_size, num_layers, dim, device=sample_aggregated_embeddings.device)
            words += ["[PAD]"] * padding_size
            
            # Combine the aggregated embeddings with padding
            aggregated_embeddings_batch.append(torch.cat([sample_aggregated_embeddings, paddings]))
            sentences.append(words)
        
        # Stack the batch and rearrange dimensions
        aggregated_embeddings_batch = torch.stack(aggregated_embeddings_batch)
        aggregated_embeddings_batch = aggregated_embeddings_batch.permute(0, 2, 1, 3)
        
        return aggregated_embeddings_batch, sentences
    

    @staticmethod
    def _normalize(embeddings: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Normalize embeddings along the specified dimension.
        
        Args:
            embeddings: Input tensor to normalize
            dim: Dimension along which to normalize
            
        Returns:
            Normalized embeddings
        """
        norm = torch.norm(embeddings, p=2, dim=dim, keepdim=True)
        return embeddings / norm.expand_as(embeddings)