from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from sklearn import metrics
import torch.nn.functional as F

from PIL import Image
from .. import builder
from .. import loss
from .. import utils
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer


class GLoRIA(nn.Module):
    def __init__(self, config):
        super(GLoRIA, self).__init__()

        self.config = config
        self.text_encoder = builder.build_text_model(config)
        self.img_encoder = builder.build_img_model(config)

        self.local_loss = loss.gloria_loss.local_loss
        self.global_loss = loss.gloria_loss.global_loss
        self.local_loss_weight = self.config.model.gloria.local_loss_weight
        self.global_loss_weight = self.config.model.gloria.global_loss_weight

        self.temp1 = self.config.model.gloria.temp1
        self.temp2 = self.config.model.gloria.temp2
        self.temp3 = self.config.model.gloria.temp3
        self.batch_size = self.config.train.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.word_tokenizer = RegexpTokenizer(r"\w+")


    def forward(self, x):
        # img encoder branch
        img_emb_local, img_emb_g = self.image_encoder_forward(x["imgs"])

        # text encorder branch
        text_emb_local, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        return img_emb_local, img_emb_g, text_emb_local, text_emb_g, sents
    

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        # Forward pass through the BERT encoder
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents


    def image_encoder_forward(self, imgs):
        # Forward pass through the encoder.
        img_feat_g, img_emb_local = self.img_encoder(imgs, get_local=True)
        # Generate embeddings from extracted features.
        img_emb_g, img_emb_local = self.img_encoder.generate_embeddings(img_feat_g, img_emb_local)

        return img_emb_local, img_emb_g


    def _calc_local_loss(self, img_emb_l, text_emb_l, sents):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, attn_maps = self.local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
        )
        return l_loss0, l_loss1, attn_maps


    def _calc_global_loss(self, img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1


    def calc_loss(self, img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents):

        l_loss0, l_loss1, attn_maps = self._calc_local_loss(
            img_emb_l, text_emb_l, sents
        )
        g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)

        # weighted loss
        loss = 0
        loss += (l_loss0 + l_loss1) * self.local_loss_weight
        loss += (g_loss0 + g_loss1) * self.global_loss_weight

        return loss, attn_maps


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
            max_word_num=self.config.data.text.word_num,
            nvis=self.config.train.nvis,
            rand_vis=self.config.train.rand_vis,
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