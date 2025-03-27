import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from tqdm import tqdm

from gloria.constants import (
    CHEXPERT_DATA_DIR,
    CHEXPERT_MASTER_CSV,
    CHEXPERT_PATH_COL,
    CHEXPERT_VIEW_COL,
    CHEXPERT_REPORT_COL,
    CHEXPERT_SPLIT_COL,
)


class MultimodalPretrainingDataset(Dataset):
    """Dataset for multimodal pretraining with medical images and reports."""

    def __init__(
            self, 
            config: dict, 
            split: str = "train", 
            transform: Optional[object] = None
        ):
        """Initialize the dataset with configuration parameters.
        
        Args:
            config: Configuration dictionary containing model and data parameters
            split: Dataset split (train, val, test)
            transform: Optional transforms to be applied on images
        """
        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path is not defined.\n"
                "Download data from: https://stanfordmlgroup.github.io/competitions/chexpert/ "
                "and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.config = config
        self.transform = transform
        self.max_word_num = self.config.dataset.text.captions_per_image

        # Load and preprocess the CheXpert dataframe
        self.df = self._load_chexpert_dataframe()
        
        # Filter only frontal view images
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # Load study paths and corresponding text data
        self.file_paths, self.path_to_text = self._load_text_data(split)

        # Initialize BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.text.bert_type)


    def _load_chexpert_dataframe(self) -> pd.DataFrame:
        """Load and preprocess the CheXpert CSV file."""
        csv_path = Path(CHEXPERT_DATA_DIR) / CHEXPERT_MASTER_CSV
        df = pd.read_csv(csv_path)
        
        # Convert relative paths to absolute paths
        df[CHEXPERT_PATH_COL] = df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        )
        
        return df


    def _load_text_data(self, split: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Load text data for the specified split.
        
        Args:
            split: Dataset split (train, val, test)
            
        Returns:
            Tuple containing file paths and path-to-text mapping
        """
        # Path to cached captions
        caption_path = Path(CHEXPERT_DATA_DIR) / "captions.pickle"
        
        # Create captions if they don't exist
        if not caption_path.exists():
            print(f"Caption file {caption_path} does not exist. Creating captions...")
            path_to_text, to_remove = self._create_path_to_text_mapping()
            
            with open(caption_path, "wb") as f:
                pickle.dump([path_to_text, to_remove], f, protocol=2)
                print(f"Saved captions to: {caption_path}")
        else:
            with open(caption_path, "rb") as f:
                print(f"Loading captions from {caption_path}")
                path_to_text, to_remove = pickle.load(f)

        # Filter file paths for current split
        file_paths = self.df[self.df[CHEXPERT_SPLIT_COL] == split][CHEXPERT_PATH_COL].tolist()
        file_paths = [f for f in file_paths if f not in to_remove]

        return file_paths, path_to_text


    def _create_path_to_text_mapping(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """Create mapping from image paths to report text.
        
        Returns:
            Tuple containing path-to-text mapping and list of paths to remove
        """
        sentence_lengths = []
        num_sentences = []
        paths_to_remove = []
        path_to_text = {}
        
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Processing reports"):
            # Get report text if available
            captions = ""
            if isinstance(row[CHEXPERT_REPORT_COL], str):
                captions = row[CHEXPERT_REPORT_COL]

            # Handle empty reports
            if not captions:
                paths_to_remove.append(row[CHEXPERT_PATH_COL])
                continue

            # Normalize whitespace
            captions = captions.replace("\n", " ")

            # Split into sentences
            splitter = re.compile(r"[0-9]+\.")
            caption_segments = splitter.split(captions)
            caption_sentences = [point.split(".") for point in caption_segments]
            caption_sentences = [sent for point in caption_sentences for sent in point]

            word_count = 0
            study_sentences = []
            
            # Process each sentence
            for sentence in caption_sentences:
                if not sentence:
                    continue

                # Clean and tokenize the sentence
                clean_sentence = sentence.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(clean_sentence.lower())

                # Skip very short sentences
                if len(tokens) <= 1:
                    continue

                # Filter non-ASCII characters
                filtered_tokens = []
                for token in tokens:
                    ascii_token = token.encode("ascii", "ignore").decode("ascii")
                    if ascii_token:
                        filtered_tokens.append(ascii_token)
                
                processed_sentence = " ".join(filtered_tokens)
                study_sentences.append(processed_sentence)

                # Track sentence statistics
                sentence_lengths.append(len(filtered_tokens))
                
                # Check if reached maximum word count
                word_count += len(filtered_tokens)
                if word_count >= self.max_word_num:
                    break

            num_sentences.append(len(study_sentences))

            # Store processed sentences or mark for removal
            if study_sentences:
                path_to_text[row[CHEXPERT_PATH_COL]] = study_sentences
            else:
                paths_to_remove.append(row[CHEXPERT_PATH_COL])

        # Report statistics
        sentence_lengths = np.array(sentence_lengths)
        num_sentences = np.array(num_sentences)
        
        print(
            f"Sentence lengths: min={sentence_lengths.min()}, mean={sentence_lengths.mean():.2f}, "
            f"max={sentence_lengths.max()} [p5={np.percentile(sentence_lengths, 5):.2f}, "
            f"p95={np.percentile(sentence_lengths, 95):.2f}]"
        )
        print(
            f"Sentences per report: min={num_sentences.min()}, mean={num_sentences.mean():.2f}, "
            f"max={num_sentences.max()} [p5={np.percentile(num_sentences, 5):.2f}, "
            f"p95={np.percentile(num_sentences, 95):.2f}]"
        )

        return path_to_text, paths_to_remove


    def get_caption(self, path: str) -> Tuple[dict, int]:
        """Get tokenized caption for an image."""
        sentences = self.path_to_text[path]

        if not sentences:
            raise ValueError(f"No sentences found for path: {path}")

        if self.config.data.text.full_report:
            # Use all sentences combined
            text = " ".join(sentences)
        else:
            # Randomly select one sentence
            sentence_idx = np.random.randint(0, len(sentences))
            text = sentences[sentence_idx]

        # Tokenize the text
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.config.data.text.word_num,
        )
        
        # Calculate actual token length (excluding padding)
        token_length = sum(1 for t in tokens["input_ids"][0] if t != 0)

        return tokens, token_length


    def get_image(self, img_path: str) -> Image.Image:
        """Load and preprocess an image."""
        img_array = cv2.imread(str(img_path), 0)
        resized_img = self._resize_image(img_array, self.config.data.image.imsize)
        
        # Convert to RGB PIL Image
        pil_img = Image.fromarray(resized_img).convert("RGB")
        
        # Apply transforms if provided
        if self.transform is not None:
            pil_img = self.transform(pil_img)

        return pil_img


    def _resize_image(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image preserving aspect ratio with padding."""
        height, width = img.shape
        
        if height > width:
            # Image is taller
            scale = target_size / height
            new_height = target_size
            new_width = int(width * scale)
        else:
            # Image is wider
            scale = target_size / width
            new_width = target_size
            new_height = int(height * scale)
            
        # Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        
        # Calculate padding
        pad_height = target_size - new_height
        pad_width = target_size - new_width
        
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


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict, int, str]:
        """Get an item by index."""
        path = self.file_paths[index]
        
        # Get image and caption
        img = self.get_image(path)
        caption_tokens, caption_length = self.get_caption(path)
        
        return img, caption_tokens, caption_length, path


    def __len__(self) -> int:
        """Get the dataset size."""
        return len(self.file_paths)


def multimodal_collate_fn(batch: List[Tuple]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Collate function for batching data with sorting by caption length.
    
    Args:
        batch: List of data items
        
    Returns:
        Dictionary with batched data
    """
    # Initialize lists for batch elements
    images = []
    caption_lengths = []
    input_ids = []
    token_type_ids = []
    attention_masks = []
    paths = []

    # Extract data from batch
    for image, caption_tokens, caption_length, path in batch:
        images.append(image)
        caption_lengths.append(caption_length)
        input_ids.append(caption_tokens["input_ids"])
        token_type_ids.append(caption_tokens["token_type_ids"])
        attention_masks.append(caption_tokens["attention_mask"])
        paths.append(path)

    # Stack tensors
    images = torch.stack(images)
    input_ids = torch.stack(input_ids).squeeze()
    token_type_ids = torch.stack(token_type_ids).squeeze()
    attention_masks = torch.stack(attention_masks).squeeze()

    # Sort by caption length (descending)
    caption_lengths_tensor = torch.tensor(caption_lengths)
    sorted_lengths, sorted_indices = torch.sort(caption_lengths_tensor, descending=True)

    # Return data dict with sorted elements
    return {
        "imgs": images[sorted_indices],
        "caption_ids": input_ids[sorted_indices],
        "token_type_ids": token_type_ids[sorted_indices],
        "attention_mask": attention_masks[sorted_indices],
        "cap_lens": sorted_lengths,
        "path": paths,  # Paths don't need to be sorted as they're for reference only
    }