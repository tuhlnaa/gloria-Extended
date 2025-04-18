"""
Oxford Pet Dataset Segmentation Analysis Tool

This script analyzes the Oxford Pet Dataset from the segmentation_models_pytorch library
to count and visualize the distribution of segmentation masks between training and validation sets.
"""

import argparse
import mplcyberpunk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import sys
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# For loading the dataset
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize

# Allow import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OxfordPetAnalyzer:
    """Class for analyzing and visualizing the Oxford Pet Dataset."""

    def __init__(self, data_dir: str):
        """Initialize the Oxford Pet analyzer with the data directory."""
        self.data_dir = data_dir
        
        # Load datasets
        self.train_dataset = SimpleOxfordPetDataset(data_dir, "train")
        self.valid_dataset = SimpleOxfordPetDataset(data_dir, "valid")
        
        # Basic transforms to inspect images
        self.transforms = Compose([
            Resize(256, 256),
            ToTensorV2(),
        ])
        
        # Analyze data
        self.train_stats = self.analyze_dataset(self.train_dataset)
        self.valid_stats = self.analyze_dataset(self.valid_dataset)
        
        # Class mapping (from SimpleOxfordPetDataset documentation)
        self.class_names = {
            1: "Cat", 
            2: "Dog", 
            3: "Background"
        }


    def analyze_dataset(self, dataset) -> Dict:
        """Analyze the dataset to extract statistics."""
        stats = {
            "size": len(dataset),
            "has_segmentation": 0,  # Count of images with segmentation (any non-zero label)
            "no_segmentation": 0,   # Count of images without segmentation (only zero labels)
            "mask_stats": {
                "min_ratio": float('inf'),
                "max_ratio": 0,
                "avg_ratio": 0,
            }
        }
        
        # Process the entire dataset
        total_mask_ratio = 0
        sample_for_ratio = min(100, len(dataset))  # Use sampling only for mask ratio stats to save time
        
        # First pass: count all images with/without segmentation
        for idx in range(len(dataset)):
            item = dataset[idx]
            mask = item["mask"]
            
            # Convert to numpy if needed
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # Check if the mask has any segmentation (any non-zero label)
            unique_values = np.unique(mask)
            has_segmentation = any(val != 0 for val in unique_values)
            
            if has_segmentation:
                stats["has_segmentation"] += 1
            else:
                stats["no_segmentation"] += 1
            
            # Only calculate mask ratios for a sample to save processing time
            if idx < sample_for_ratio:
                # Calculate foreground ratio (assuming 0 is background)
                if 0 in unique_values:
                    background_pixels = np.sum(mask == 0)
                    total_pixels = mask.size
                    foreground_ratio = 1.0 - (background_pixels / total_pixels)
                    
                    # Update mask stats
                    stats["mask_stats"]["min_ratio"] = min(stats["mask_stats"]["min_ratio"], foreground_ratio)
                    stats["mask_stats"]["max_ratio"] = max(stats["mask_stats"]["max_ratio"], foreground_ratio)
                    total_mask_ratio += foreground_ratio
        
        # Calculate average mask ratio
        if sample_for_ratio > 0:
            stats["mask_stats"]["avg_ratio"] = total_mask_ratio / sample_for_ratio
            
        return stats


    def create_dataset_comparison(self) -> pd.DataFrame:
        """Create a comparison of training and validation datasets focusing on segmentation counts."""
        result_data = []
        
        # Add segmentation statistics (has pet vs no pet)
        result_data.extend([
            {'Dataset': 'Training', 'Category': 'Has Pet', 'Count': self.train_stats["has_segmentation"]},
            {'Dataset': 'Training', 'Category': 'No Pet', 'Count': self.train_stats["no_segmentation"]},
            {'Dataset': 'Validation', 'Category': 'Has Pet', 'Count': self.valid_stats["has_segmentation"]},
            {'Dataset': 'Validation', 'Category': 'No Pet', 'Count': self.valid_stats["no_segmentation"]}
        ])
       
        return pd.DataFrame(result_data)


    def visualize_dataset_comparison(self, comparison_df: pd.DataFrame, output_path: str) -> None:
        """Create a bar plot visualization comparing segmentation counts between training and validation sets."""
        print(comparison_df)
        plt.figure(figsize=(8, 6))
        
        # Define color palette (Data volume ranking)
        palette_name = 'coolwarm'
        pal = sns.color_palette(palette_name, len(comparison_df['Dataset']))
        rank = comparison_df['Count'].argsort().argsort()
        palette_list = list(np.array(pal)[rank])

        plt.style.use('cyberpunk')
        ax = sns.barplot(
            x='Dataset', 
            y='Count', 
            hue='Category',
            data=comparison_df,
            gap=0.05,
            width=0.7
        )

        # Set title and labels
        ax.set_title('Pet Segmentation Counts by Dataset', fontsize=16)
        ax.set_xlabel('Dataset', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        
        # Add count labels on the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=13)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)


    def visualize_sample_images(self, num_samples: int, output_path: str) -> None:
        """Visualize random sample images and their segmentation masks."""
        # Sample random images from the training set
        indices = random.sample(range(len(self.train_dataset)), num_samples)
        
        # Create subplot grid
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
        plt.style.use('cyberpunk')
        
        for i, idx in enumerate(indices):
            item = self.train_dataset[idx]
            image = item["image"]
            mask = item["mask"]
            
            # Convert to numpy if needed
            if isinstance(image, torch.Tensor):
                image = image.numpy()
                
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # Convert image to format for display if needed
            if image.shape[0] == 3:  # If the image is in channel-first format
                image = np.transpose(image, (1, 2, 0))
            
            # Plot image
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f"Image {idx}")
            axes[i, 0].axis('off')
            
            # Plot mask
            if len(mask.shape) == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
                
            axes[i, 1].imshow(mask, cmap='viridis')
            axes[i, 1].set_title(f"Mask {idx}")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)


    def print_summary(self) -> None:
        """Print a summary of the dataset."""
        print(f"Oxford Pet Dataset Summary:")
        print(f"  - Data directory: {self.data_dir}")
        print(f"  - Training set: {self.train_stats['size']} images")
        print(f"  - Validation set: {self.valid_stats['size']} images")
        
        print("\nSegmentation Statistics:")
        print(f"  - Training set:")
        print(f"    * Images with pet segmentation: {self.train_stats['has_segmentation']}")
        print(f"    * Images without pet segmentation: {self.train_stats['no_segmentation']}")
        print(f"  - Validation set:")
        print(f"    * Images with pet segmentation: {self.valid_stats['has_segmentation']}")
        print(f"    * Images without pet segmentation: {self.valid_stats['no_segmentation']}")
        
        print("\nMask Statistics:")
        for dataset_name, stats in [("Training", self.train_stats), ("Validation", self.valid_stats)]:
            print(f"  - {dataset_name} set:")
            print(f"    * Average foreground ratio: {stats['mask_stats']['avg_ratio']:.4f}")
            print(f"    * Min foreground ratio: {stats['mask_stats']['min_ratio']:.4f}")
            print(f"    * Max foreground ratio: {stats['mask_stats']['max_ratio']:.4f}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze and visualize Oxford Pet Dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the Oxford Pet Dataset directory')
    parser.add_argument('--stats_output', type=str, required=True, help='Path to save the dataset statistics visualization')
    parser.add_argument('--samples_output', type=str, required=False, help='Path to save the sample images visualization')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of sample images to visualize')
    
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create analyzer
    analyzer = OxfordPetAnalyzer(args.data_dir)
    
    # Ensure the output directories exist
    for output_path in [args.stats_output, args.samples_output]:
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Print summary information
        analyzer.print_summary()
        
        # Create dataset comparison and visualize
        comparison_df = analyzer.create_dataset_comparison()
        analyzer.visualize_dataset_comparison(comparison_df, args.stats_output)
        print(f"Dataset statistics visualization saved to {args.stats_output}")
        
        # Visualize sample images if requested
        if args.samples_output:
            analyzer.visualize_sample_images(args.num_samples, args.samples_output)
            print(f"Sample images visualization saved to {args.samples_output}")
            
        print("Analysis complete.")
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()

"""
Usage example:

python cat_analyzer.py --data_dir "./data/oxford_pet_dataset" --stats_output "./output/dataset_stats.png" --samples_output "./output/sample_images.png"

python data\pet_dataset_analyzer.py --data_dir "D:\Kai\DATA_Set_2" --stats_output "./output/dataset_stats.png" --samples_output "./output/sample_images.png"

"""