import re
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn import metrics
from typing import Dict, List, Tuple, Union, Optional

from PIL import Image
from .. import builder
from .. import loss
from .. import utils
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer


class GLoRIA(nn.Module):
    """
    GLoRIA: A Global-Local Representation Learning Framework for medical images.
    
    This model learns multimodal representations by contrasting image regions
    with words in paired radiology reports.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Initialize encoders
        self.text_encoder = builder.build_text_model(config)
        self.img_encoder = builder.build_image_model(config)
        
        # Get loss functions
        self.local_loss_fn = loss.gloria_loss.local_loss
        self.global_loss_fn = loss.gloria_loss.global_loss
        
        # Get loss weights from config
        self.local_loss_weight = config.model.gloria.local_loss_weight
        self.global_loss_weight = config.model.gloria.global_loss_weight
        
        # Temperature parameters for scaling similarity scores
        self.temp_attention = config.model.gloria.temp_attention
        self.temp_similarity = config.model.gloria.temp_similarity
        self.temp_loss = config.model.gloria.temp_loss
        
        # Initialize tokenizer
        self._setup_tokenizer()
    

    def _setup_tokenizer(self):
        """Set up the tokenizer for text processing."""
        bert_type = self.config.model.text.bert_type
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.word_tokenizer = RegexpTokenizer(r"\w+")
    

    def forward(self, batch):
        """Forward pass through the GLoRIA model."""
        # Process images
        img_emb_local, img_emb_global = self.encode_images(batch["imgs"])
        
        # Process text
        text_emb_local, text_emb_global, sents = self.encode_text(
            batch["caption_ids"], 
            batch["attention_mask"], 
            batch["token_type_ids"]
        )
        
        return img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents
    

    def encode_text(self, caption_ids, attention_mask, token_type_ids):
        """
        Encode text inputs using the text encoder.
        
        Args:
            caption_ids: Token IDs from tokenizer
            attention_mask: Attention mask for BERT
            token_type_ids: Token type IDs for BERT
            
        Returns:
            Tuple containing:
                - text_emb_local: Local text embeddings
                - text_emb_global: Global text embedding
                - sents: Tokenized sentences
        """
        return self.text_encoder(caption_ids, attention_mask, token_type_ids)
    

    def encode_images(self, images):
        """
        Encode images using the image encoder.
        
        Args:
            images: Batch of images
            
        Returns:
            Tuple containing:
                - img_emb_local: Local image embeddings
                - img_emb_global: Global image embedding
        """
        # Extract features from the image encoder
        img_feat_global, img_feat_local = self.img_encoder(images, get_local=True)
        
        # Generate embeddings from extracted features
        img_emb_global, img_emb_local = self.img_encoder.generate_embeddings(
            img_feat_global, img_feat_local
        )
        
        return img_emb_local, img_emb_global
    

    def compute_loss(self, img_emb_local, img_emb_global, text_emb_local, text_emb_global, sents):
        """
        Compute the combined global and local losses.
        
        Args:
            img_emb_local: Local image embeddings
            img_emb_global: Global image embedding
            text_emb_local: Local text embeddings
            text_emb_global: Global text embedding
            sents: Tokenized sentences
            
        Returns:
            Tuple containing:
                - total_loss: Combined weighted loss
                - attn_maps: Attention maps for visualization
        """
        # Compute local loss (between image regions and words)
        local_loss_i2t, local_loss_t2i, attn_maps = self._compute_local_loss(
            img_emb_local, text_emb_local, sents
        )
        
        # Compute global loss (between global image and text embeddings)
        global_loss_i2t, global_loss_t2i = self._compute_global_loss(
            img_emb_global, text_emb_global
        )
        
        # Combine losses with weights
        local_loss = (local_loss_i2t + local_loss_t2i) * self.local_loss_weight
        global_loss = (global_loss_i2t + global_loss_t2i) * self.global_loss_weight
        total_loss = local_loss + global_loss
        
        return total_loss, attn_maps
    

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
        
        # Compute local loss with temperature parameters
        local_loss_i2t, local_loss_t2i, attn_maps = self.local_loss_fn(
            img_emb_local,
            text_emb_local,
            caption_lengths,
            temp_attention=self.temp_attention,
            temp_similarity=self.temp_similarity,
            temp_loss=self.temp_loss,
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
        return self.global_loss_fn(
            img_emb_global, 
            text_emb_global, 
            temperature=self.temp_loss
        )


    @staticmethod
    def compute_global_similarities(img_emb_global: torch.Tensor, text_emb_global: torch.Tensor) -> torch.Tensor:
        """
        Compute global similarities between image and text embeddings.
        
        Args:
            img_emb_global: Global image embeddings
            text_emb_global: Global text embeddings
            
        Returns:
            Tensor of cosine similarities
        """
        # Convert to numpy for sklearn's cosine similarity calculation
        img_emb_np = img_emb_global.detach().cpu().numpy()
        text_emb_np = text_emb_global.detach().cpu().numpy()
        
        # Compute pairwise cosine similarities
        similarities = metrics.pairwise.cosine_similarity(img_emb_np, text_emb_np)
        
        return torch.tensor(similarities)
    
    
    @staticmethod
    def compute_local_similarities(
            img_emb_local: torch.Tensor, 
            text_emb_local: torch.Tensor, 
            caption_lengths: torch.Tensor,
            attention_scale: float = 4.0,
            similarity_scale: float = 5.0
        ) -> torch.Tensor:
        """
        Compute local similarities between image and text embeddings.
        
        Args:
            img_emb_local: Local image embeddings
            text_emb_local: Local text embeddings
            caption_lengths: Length of each caption
            attention_scale: Scaling factor for attention computation
            similarity_scale: Scaling factor for similarity computation
            
        Returns:
            Tensor of local similarities
        """
        batch_size = img_emb_local.shape[0]
        similarities = []
        
        for i, words_num in enumerate(caption_lengths):
            # Extract word features for this caption, excluding special tokens
            # Shape: [1, embed_dim, caption_length], e.g. [1, 768, 32]
            word_features = text_emb_local[i, :, 1:words_num+1].unsqueeze(0)

            # Repeat word features for each image in batch
            # Shape: [batch_size, embed_dim, caption_length], e.g. [?, 768, 32]
            word_features = word_features.repeat(batch_size, 1, 1)

            # Use image embeddings as context
            # Shape: [batch_size, embed_dim, height, width], e.g. [?, 768, 19, 19]
            context = img_emb_local

            # Compute attention-weighted context
            weighted_context, _ = loss.gloria_loss.compute_attention(word_features, context, attention_scale)

            # Transpose to align dimensions for similarity computation
            # Shape: [batch_size, caption_length, embed_dim], e.g. [?, 32, 768]
            word_features = word_features.transpose(1, 2)
            weighted_context = weighted_context.transpose(1, 2)

            # Reshape for efficient computation
            # Shape: [batch_size * caption_length, embed_dim], e.g. [?, 768]
            word_features_flat = word_features.reshape(batch_size * words_num, -1)
            weighted_context_flat = weighted_context.reshape(batch_size * words_num, -1)
            
            # Compute cosine similarity
            # Shape: [batch_size, 32]
            row_sim = F.cosine_similarity(word_features_flat, weighted_context_flat).squeeze()
            row_sim = row_sim.view(batch_size, words_num)

            # Apply exponential scaling and max pooling
            # Shape: [batch_size, 1]
            row_sim = torch.exp(row_sim * similarity_scale)
            row_sim, _ = torch.max(row_sim, dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)
        
        # Concatenate all similarities
        local_similarities = torch.cat(similarities, dim=1).detach().cpu()
        
        return local_similarities


    def get_attn_maps(self, img_emb_l, text_emb_l, sents):
        _, _, attn_maps = self._calc_local_loss(img_emb_l, text_emb_l, sents)
        return attn_maps


    def plot_attn_maps(self, attn_maps, imgs, sents, epoch_idx=0, batch_idx=0):

        img_set, _ = utils.build_attention_images(
            imgs,
            attn_maps,
            max_word_num=None,
            #max_word_num=self.config.data.text.word_num,  # TODO: remove
            nvis=self.config.misc.nvis,
            rand_vis=self.config.misc.rand_vis,
            sentences=sents,
        )

        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = (
                f"{self.config.output_dir}/"
                f"attention_maps_epoch{epoch_idx}_"
                f"{batch_idx}.png"
            )
            im.save(fullpath)


    def process_text(self, text: Union[str, List[str]], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Process text input into tensors for model consumption.
        
        Args:
            text: Either a single text string or a list of text strings
            device: The device to place tensors on
            
        Returns:
            Dictionary containing processed text tensors
        """
        # Convert single string to list for consistent processing
        if isinstance(text, str):
            text = [text]
            
        processed_text_tensors = []
        
        for t in text:
            # Clean and tokenize text
            cleaned_text = self._clean_and_tokenize_text(t)
            
            # Tokenize with model tokenizer
            text_tensors = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.config.data.text.word_num,
            )
            
            # Add word representations
            text_tensors["sent"] = [
                self.ixtoword[idx] for idx in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        # Combine tensors from all texts
        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack([x["attention_mask"] for x in processed_text_tensors])
        token_type_ids = torch.stack([x["token_type_ids"] for x in processed_text_tensors])

        # Handle dimensions based on batch size
        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        # Calculate caption lengths
        cap_lens = [len([w for w in txt if not w.startswith("[")]) for txt in text]

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }
    
    
    def _clean_and_tokenize_text(self, text: str) -> str:
        """
        Clean and tokenize a single text string.
        
        Args:
            text: Text string to clean
            
        Returns:
            Cleaned and tokenized text
        """
        # Replace newlines with spaces
        text = text.replace("\n", " ")

        # Split text into sentences
        splitter = re.compile(r"[0-9]+\.")
        captions = splitter.split(text)
        captions = [point.split(".") for point in captions]
        captions = [sent for point in captions for sent in point]

        cleaned_sentences = []

        for sentence in captions:
            # Remove unicode replacement characters
            sentence = sentence.replace("\ufffd\ufffd", " ")
            
            # Tokenize sentence
            tokens = self.word_tokenizer.tokenize(sentence.lower())

            # Skip very short sentences
            if len(tokens) <= 1:
                continue

            # Filter non-ASCII characters
            filtered_tokens = []
            for token in tokens:
                ascii_token = token.encode("ascii", "ignore").decode("ascii")
                if ascii_token:
                    filtered_tokens.append(ascii_token)
                    
            if filtered_tokens:
                cleaned_sentences.append(" ".join(filtered_tokens))

        return " ".join(cleaned_sentences)


    def process_class_prompts(self, class_prompts: Dict[str, str], device: torch.device) -> Dict[str, Dict]:
        """
        Process class prompt texts into model-ready format.
        
        Args:
            class_prompts: Dictionary mapping class names to prompt texts
            device: Device to place tensors on
            
        Returns:
            Dictionary mapping class names to processed tensors
        """
        return {
            class_name: self.process_text(prompt_text, device)
            for class_name, prompt_text in class_prompts.items()
        }
    
    
    def process_images(self, paths, device):
        """
        Process images for model input.
        
        Args:
            paths: String path to single image or list of image paths
            device: PyTorch device to load tensors to
            
        Returns:
            torch.Tensor: Batch of processed images on specified device
        """
        transform = builder.build_transformation(self.config, split="test")
        
        # Ensure paths is a list
        if isinstance(paths, str):
            paths = [paths]
        
        # Process each image
        processed_images = []
        for path in paths:
            # Read grayscale image
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            
            # Resize and transform image
            img = self._resize_image(img, self.config.data.image.imsize)
            img_rgb = Image.fromarray(img).convert("RGB")
            img_tensor = transform(img_rgb)
            processed_images.append(img_tensor)
        
        # Stack and move batch to device
        batch = torch.stack(processed_images).to(device)
        
        return batch


    def _resize_image(self, img, target_size):
        """
        Resize image to target size while maintaining aspect ratio and padding if necessary.
        
        Args:
            img: Image as numpy array (grayscale)
            target_size: Desired output image size (square)
            
        Returns:
            numpy.ndarray: Resized and padded image of size target_size x target_size
        """
        height, width = img.shape
        
        # Determine which dimension to scale based on aspect ratio
        if height > width:
            # Image is taller than wide
            scale_factor = target_size / height
            new_height = target_size
            new_width = int(width * scale_factor)
        else:
            # Image is wider than tall or square
            scale_factor = target_size / width
            new_width = target_size
            new_height = int(height * scale_factor)
        
        # Resize image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        
        # Calculate padding
        pad_height = target_size - new_height
        pad_width = target_size - new_width
        
        # Distribute padding evenly on both sides
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        
        # Apply padding
        padded_img = np.pad(
            resized_img, 
            [(top, bottom), (left, right)], 
            mode="constant", 
            constant_values=0
        )
        
        return padded_img