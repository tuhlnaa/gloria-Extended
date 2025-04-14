"""
Optimized GLORIA model inference script with batch processing support.
"""
import os
import gloria
from typing import List, Dict, Tuple, Optional
import logging

import torch
import pandas as pd
from tqdm import tqdm

from gloria import builder, constants, zero_shot_classification
from configs.config import parse_args


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_gloria_model(config, checkpoint_path: str) -> torch.nn.Module:
    """
    Load and initialize the GLORIA model with pre-trained weights.
    
    Args:
        config: Configuration object with model parameters
        checkpoint_path: Path to the model checkpoint file
        
    Returns:
        Initialized GLORIA model
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Initialize model
    model = builder.build_gloria_model(config).to(config.device.device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device.device)
    model_state_dict = builder.normalize_model_state_dict(checkpoint)
    model.load_state_dict(model_state_dict)
    
    # Set to evaluation mode
    model.eval()
    return model


def prepare_dataset(csv_path: str, data_dir: str, batch_size: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and prepare dataset for inference.
    
    Args:
        csv_path: Path to the CSV file with image paths
        data_dir: Base directory for images
        batch_size: Optional number of samples to process
        
    Returns:
        Tuple of DataFrame and list of full image paths
    """
    logger.info(f"Loading dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    if batch_size is not None:
        logger.info(f"Processing {batch_size} samples out of {len(df)}")
        df = df[:batch_size]
    
    # Prepare full image paths
    full_paths = [
        os.path.join(data_dir, path.replace('CheXpert-v1.0/', '')) 
        for path in df['Path']
    ]
    
    return df, full_paths


def run_inference_in_batches(
    model: torch.nn.Module,
    images_paths: List[str],
    class_prompts: Dict[str, List[str]],
    device: str,
    batch_size: int = 16
) -> pd.DataFrame:
    """
    Run inference in batches to manage memory usage.
    
    Args:
        model: GLORIA model
        images_paths: List of image paths
        class_prompts: Dictionary of class prompts
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        DataFrame with similarity scores
    """
    # Process class prompts (only needs to be done once)
    logger.info("Processing class prompts")
    processed_txt = model.process_class_prompts(class_prompts, device)
    
    all_similarities = []
    num_batches = (len(images_paths) + batch_size - 1) // batch_size

    # Process images in batches
    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(images_paths))
        batch_paths = images_paths[start_idx:end_idx]
        
        # Process batch of images
        with torch.no_grad():
            processed_imgs = model.process_images(batch_paths, device)
            batch_similarities = zero_shot_classification(
                model, processed_imgs, processed_txt
            )
            all_similarities.append(batch_similarities)
    
    # Combine results from all batches
    return pd.concat(all_similarities, ignore_index=True)


def evaluate_accuracy(similarities: pd.DataFrame, ground_truth: pd.DataFrame) -> float:
    """
    Calculate accuracy of the model predictions.
    
    Args:
        similarities: DataFrame with model similarity scores
        ground_truth: DataFrame with ground truth labels
        
    Returns:
        Accuracy as a float
    """
    competition_tasks = constants.CHEXPERT_COMPETITION_TASKS
    
    labels = ground_truth[competition_tasks].to_numpy().argmax(axis=1)
    predictions = similarities[competition_tasks].to_numpy().argmax(axis=1)
    
    correct_predictions = (labels == predictions).sum()
    accuracy = correct_predictions / len(labels)
    
    return accuracy


def main():
    """Main function to run the inference pipeline."""
    # Initialize configuration
    config = parse_args()
    
    # Paths configuration
    CHEXPERT_5x200 = r"D:\Kai\pretrained\Gloria\chexpert_5x200.csv"
    CHECKPOINT_PATH = r"D:\Kai\pretrained\Gloria\chexpert_resnet50.ckpt"
    
    # Load model
    gloria_model = load_gloria_model(config, CHECKPOINT_PATH)
    
    # Load dataset
    batch_size = 1000  # Adjust based on available GPU memory
    df, full_paths = prepare_dataset(
        CHEXPERT_5x200, 
        config.data_dir, 
        batch_size=batch_size
    )
    
    # Generate class prompts
    class_prompts = gloria.generate_chexpert_class_prompts()
    
    # Run inference in batches
    similarities = run_inference_in_batches(
        gloria_model,
        full_paths,
        class_prompts,
        config.device.device,
        batch_size=1  # Process in smaller batches to save memory
    )
    
    # Calculate accuracy
    accuracy = evaluate_accuracy(similarities, df)
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    return {
        "accuracy": accuracy,
        "similarities": similarities,
        "num_samples": len(df)
    }


if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    
    # Run the main function
    results = main()