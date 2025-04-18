"""
Oxford Pet Dataset Cat Analysis Tool

This script analyzes the Oxford Pet Dataset from the segmentation_models_pytorch library
to count and visualize the distribution of cat/dog classes and segmentation masks.
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
        """
        Analyze the dataset to extract statistics.
        
        Args:
            dataset: SimpleOxfordPetDataset to analyze
            
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "size": len(dataset),
            "segmentation_classes": Counter(),
            "mask_stats": {
                "min_ratio": float('inf'),
                "max_ratio": 0,
                "avg_ratio": 0,
            }
        }
        
        # Sample some images to analyze segmentation masks
        sample_size = min(100, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)
        
        total_mask_ratio = 0
        for idx in indices:
            item = dataset[idx]
            image = item["image"]
            mask = item["mask"]
            
            # Convert to numpy if needed
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # Count unique segmentation classes
            unique_classes, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique_classes, counts):
                stats["segmentation_classes"][int(cls)] += 1
                
            # Calculate foreground ratio (assuming 0 is background)
            if 0 in unique_classes:
                background_idx = np.where(unique_classes == 0)[0][0]
                background_pixels = counts[background_idx]
                total_pixels = mask.size
                foreground_ratio = 1.0 - (background_pixels / total_pixels)
                
                # Update mask stats
                stats["mask_stats"]["min_ratio"] = min(stats["mask_stats"]["min_ratio"], foreground_ratio)
                stats["mask_stats"]["max_ratio"] = max(stats["mask_stats"]["max_ratio"], foreground_ratio)
                total_mask_ratio += foreground_ratio
        
        # Calculate average mask ratio
        if sample_size > 0:
            stats["mask_stats"]["avg_ratio"] = total_mask_ratio / sample_size
            
        return stats

    def create_dataset_comparison(self) -> pd.DataFrame:
        """
        Create a comparison of training and validation datasets.
        
        Returns:
            DataFrame comparing dataset statistics
        """
        comparison_data = []
        
        # Add dataset size information
        comparison_data.extend([
            {"Dataset": "Training", "Metric": "Number of Images", "Value": self.train_stats["size"]},
            {"Dataset": "Validation", "Metric": "Number of Images", "Value": self.valid_stats["size"]},
        ])
        
        # Add segmentation class information
        for dataset_name, stats in [("Training", self.train_stats), ("Validation", self.valid_stats)]:
            for cls, count in stats["segmentation_classes"].items():
                class_name = self.class_names.get(cls, f"Class {cls}")
                comparison_data.append({
                    "Dataset": dataset_name,
                    "Metric": f"{class_name} Instances",
                    "Value": count
                })
                
        # Add mask statistics
        for dataset_name, stats in [("Training", self.train_stats), ("Validation", self.valid_stats)]:
            for stat_name, value in stats["mask_stats"].items():
                if stat_name != "min_ratio" and stat_name != "max_ratio":
                    formatted_name = " ".join(s.capitalize() for s in stat_name.split("_"))
                    comparison_data.append({
                        "Dataset": dataset_name,
                        "Metric": f"{formatted_name} Mask",
                        "Value": value
                    })
                
        return pd.DataFrame(comparison_data)

    def visualize_dataset_comparison(self, comparison_df: pd.DataFrame, output_path: str) -> None:
        """Create a grouped bar plot visualization of the dataset comparison."""
        plt.figure(figsize=(10, 6))
        
        # Define color palette
        palette_name = 'coolwarm'
        
        plt.style.use('cyberpunk')
        
        # Filter dataframe to include only numeric metrics
        numeric_df = comparison_df[comparison_df["Metric"].isin([
            "Number of Images", 
            "Cat Instances", 
            "Dog Instances", 
            "Background Instances",
            "Avg Ratio Mask"
        ])]
        
        # Pivot the dataframe for easier plotting
        pivot_df = numeric_df.pivot(index="Metric", columns="Dataset", values="Value")
        
        ax = pivot_df.plot(kind='bar', figsize=(10, 6))
        
        # Set title and labels
        ax.set_title('Oxford Pet Dataset Statistics', fontsize=16)
        ax.set_xlabel('Metric', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        
        # Add count labels on the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)

    def visualize_sample_images(self, num_samples: int, output_path: str) -> None:
        """
        Visualize random sample images and their segmentation masks.
        
        Args:
            num_samples: Number of samples to visualize
            output_path: Path to save the visualization
        """
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
        
        print("\nSegmentation Classes:")
        for cls, class_name in self.class_names.items():
            train_count = self.train_stats["segmentation_classes"].get(cls, 0)
            valid_count = self.valid_stats["segmentation_classes"].get(cls, 0)
            print(f"  - {class_name}: {train_count} in training, {valid_count} in validation")
        
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

python data\pet_analyzer.py --data_dir "D:\Kai\DATA_Set_2" --stats_output "./output/dataset_stats.png" --samples_output "./output/sample_images.png"

"""