import os
import re
import cv2
import torch
import pickle

import numpy as np
import pandas as pd
import torchvision.transforms as T

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import RegexpTokenizer

from data.dataset import build_transformation


class CheXpertMultimodalDataset(Dataset):
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
        # Get data directory and file paths from config
        self.data_dir = Path(config.data_dir)
        self.csv_path = self.data_dir / config.master_csv
        
        # Set column name mappings
        self.path_col = config.dataset.columns.path
        self.view_col = config.dataset.columns.view
        self.report_col = config.dataset.columns.report
        self.split_col = config.dataset.columns.split
        
        self.config = config
        self.transform = transform
        self.max_word_num = config.dataset.text.captions_per_image

        # Check if data directory exists
        if not self.data_dir.exists():
            raise RuntimeError(
                f"Data directory {self.data_dir} does not exist.\n"
                "Download data and update config.dataset.data_dir in your YAML file."
            )

        # Load and preprocess the CheXpert dataframe
        self.df = self._load_chexpert_dataframe()

        # Filter only frontal view images
        self.df = self.df[self.df[self.view_col] == "Frontal"]

        # Apply data fraction sampling if specified in config and we're in training mode
        if hasattr(config.dataset, 'fraction') and config.dataset.fraction != 1 and split == "train":
            orig_size = len(self.df)
            self.df = self.df.sample(frac=config.dataset.fraction, random_state=42)
            print(f"Applied dataset fraction {config.dataset.fraction}: reduced from {orig_size} to {len(self.df)} samples")
            
        # Load study paths and corresponding text data
        self.file_paths, self.path_to_text = self._load_text_data(split)

        # Initialize BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.text.bert_type)


    def _load_chexpert_dataframe(self) -> pd.DataFrame:
        """Load and preprocess the CheXpert CSV file."""
        print(f"Loading CSV from: {self.csv_path}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")
            
        df = pd.read_csv(self.csv_path)
        
        # Convert relative paths to absolute paths if needed
        if not os.path.isabs(df[self.path_col].iloc[0]):
            df[self.path_col] = df[self.path_col].apply(
                lambda x: os.path.join(self.data_dir, x)
            )
        
        print(f"Loaded dataframe with {len(df)} rows and columns: {df.columns.tolist()}")
        return df


    def _load_text_data(self, split: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Load text data for the specified split."""
        # Path to cached captions
        caption_path = self.data_dir / f"captions_{self.config.master_csv.split('.')[0]}.pickle"
        
        # Create captions if they don't exist
        if not caption_path.exists() or self.config.dataset.force_rebuild_captions:
            print(f"Caption file {caption_path} does not exist or rebuild forced. Creating captions...")
            path_to_text, to_remove = self._create_path_to_text_mapping()
            
            with open(caption_path, "wb") as f:
                pickle.dump([path_to_text, to_remove], f, protocol=4)
                print(f"Saved captions to: {caption_path}")
        else:
            with open(caption_path, "rb") as f:
                print(f"Loading captions from {caption_path}")
                path_to_text, to_remove = pickle.load(f)

        # Filter file paths for current split or use full dataset if split_col is not specified
        if self.split_col is not None:
            file_paths = self.df[self.df[self.split_col] == split][self.path_col].tolist()
            print(f"Filtering paths for split '{split}' using column '{self.split_col}'")
        else:
            file_paths = self.df[self.path_col].tolist()
            print(f"Split column not specified. Using all {len(file_paths)} paths from CSV")

        # For debugging purposes, limit to a small subset if configured
        if hasattr(self.config.dataset, "debug_sample_size") and self.config.dataset.debug_sample_size > 0:
            sample_size = min(self.config.dataset.debug_sample_size, len(file_paths))
            file_paths = file_paths[:sample_size]
            print(f"Debug mode: Using only {sample_size} samples")

        # Remove paths with missing reports
        file_paths = [f for f in file_paths if f not in to_remove]

        # Remove 'CheXpert-v1.0/' prefix from all file paths
        file_paths = [path.replace('CheXpert-v1.0/', '') for path in file_paths]

        print(f"Loaded {len(file_paths)} file paths for split '{split}' (or entire dataset if no split column)")
        return file_paths, path_to_text


    def _create_path_to_text_mapping(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """Create mapping from image paths to report text."""
        sentence_lengths = []
        num_sentences = []
        paths_to_remove = []
        path_to_text = {}
        
        # Use findings section if available, otherwise use full report
        use_findings = self.config.dataset.text.get('use_findings_section', False)
        findings_col = 'section_findings' if use_findings and 'section_findings' in self.df.columns else None
        
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Processing reports"):
            # Get report text if available
            captions = ""
            if findings_col and isinstance(row[findings_col], str) and row[findings_col].strip():
                captions = row[findings_col]
            elif isinstance(row[self.report_col], str):
                captions = row[self.report_col]

            # Handle empty reports
            if not captions or not isinstance(captions, str):
                paths_to_remove.append(row[self.path_col])
                continue

            # Process the path - remove 'CheXpert-v1.0/' prefix
            path = row[self.path_col].replace('CheXpert-v1.0/', '')

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
                path_to_text[path] = study_sentences
                # path_to_text[row[self.path_col]] = study_sentences
            else:
                # print(f"DEBUG - No valid path: {path}")
                # print(f"DEBUG - Original report: {captions[:200]}...")
                paths_to_remove.append(row[self.path_col])

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
        
        print(f"Processed {len(path_to_text)} valid reports, removed {len(paths_to_remove)} invalid paths")

        return path_to_text, paths_to_remove


    def get_caption(self, path: str) -> Tuple[dict, int]:
        """Get tokenized caption for an image."""
        sentences = self.path_to_text[path]

        if not sentences:
            raise ValueError(f"No sentences found for path: {path}")

        if self.config.dataset.text.full_report:
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
            max_length=self.config.dataset.text.word_num,
        )
        
        # Calculate actual token length (excluding padding)
        token_length = sum(1 for t in tokens["input_ids"][0] if t != 0)

        return tokens, token_length


    def get_image(self, img_path: str) -> Image.Image:
        """Load and preprocess an image."""
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at: {img_path}")
                
            img_array = cv2.imread(str(img_path), 0)
            if img_array is None:
                raise ValueError(f"Failed to read image: {img_path}")
                
            resized_img = self._resize_image(img_array, self.config.dataset.image.imsize)
            
            # Convert to RGB PIL Image
            pil_img = Image.fromarray(resized_img).convert("RGB")
            
            # Apply transforms if provided
            if self.transform is not None:
                pil_img = self.transform(pil_img)

            return pil_img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            blank = np.zeros((self.config.dataset.image.imsize, self.config.dataset.image.imsize), dtype=np.uint8)
            pil_img = Image.fromarray(blank).convert("RGB")
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
    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

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
        "paths": [paths[i] for i in sorted_indices],
    }


def get_chexpert_multimodal_dataloader(
        config,
        split: str = "train",
        transform: Optional[T.Compose] = None, 
    ) -> DataLoader:
    """
    Create a DataLoader for the CheXpert medical multimodal dataset.
    
    Args:
        config: Configuration object containing dataset parameters
            Expected structure:
            - config.data_dir: Base directory containing CheXpert data
            - config.{split}_csv: Path to the specific split CSV file
            - config.model.batch_size: Batch size
            - config.dataset.num_workers: Number of workers
            - config.dataset.fraction: Optional fraction of data to use (for training)
        split: Dataset split ('train', 'valid', or 'test')
        transform: Optional custom transformation pipeline; if None, uses transforms built from config
        
    Returns:
        DataLoader for the CheXpert dataset
    """
    # Create transformation if not provided
    if transform is None:
        transform = build_transformation(config, split)
    
    # Create dataset
    dataset = CheXpertMultimodalDataset(
        config=config,
        split=split,
        transform=transform,
    )
    
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! No images found. Please check the paths and file formats.")
    
    # Create dataloader with parameters from config
    data_loader = DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        shuffle=(split == "train"),
        num_workers=config.dataset.num_workers,
        pin_memory=getattr(config.model, "pin_memory", False),
        drop_last=getattr(config.model, "drop_last", False) if split == "train" else True,
        collate_fn=multimodal_collate_fn
    )
    print(f"DataLoader created successfully with {len(data_loader)} batches")

    return data_loader, dataset