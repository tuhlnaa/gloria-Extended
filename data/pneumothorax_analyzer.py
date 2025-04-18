"""
SIIM-ACR Pneumothorax Segmentation Analysis Tool

This script analyzes the SIIM-ACR Pneumothorax Segmentation dataset
to count and visualize the distribution of pneumothorax cases in both
training and testing datasets.
"""

import argparse
import mplcyberpunk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


class PneumothoraxAnalyzer:
    """Class for analyzing and visualizing SIIM-ACR Pneumothorax Segmentation dataset."""

    def __init__(self, train_csv_path: str, test_csv_path: str):
        """Initialize the Pneumothorax analyzer."""
        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)
        
        # Validate the expected column structure
        required_columns = ['new_filename', 'ImageId', 'has_pneumo']
        for col in required_columns:
            if col not in self.train_df.columns or col not in self.test_df.columns:
                raise ValueError(f"Missing required column: {col}")
    

    def get_dataset_stats(self) -> pd.DataFrame:
        """
        Count positive (pneumothorax) and negative cases in both datasets.

        Returns:
            DataFrame with counts of pneumothorax and no-pneumothorax cases for each dataset
        """
        result_data = []
        
        # Count training set cases
        train_positive = (self.train_df['has_pneumo'] == 1).sum()
        train_negative = (self.train_df['has_pneumo'] == 0).sum()
        
        # Count testing set cases
        test_positive = (self.test_df['has_pneumo'] == 1).sum()
        test_negative = (self.test_df['has_pneumo'] == 0).sum()
        
        # Compile the results
        result_data.extend([
            {'Dataset': 'Training', 'Category': 'Pneumothorax', 'Count': train_positive},
            {'Dataset': 'Training', 'Category': 'No Pneumothorax', 'Count': train_negative},
            {'Dataset': 'Validation', 'Category': 'Pneumothorax', 'Count': test_positive},
            {'Dataset': 'Validation', 'Category': 'No Pneumothorax', 'Count': test_negative}
        ])
        
        return pd.DataFrame(result_data)


    def visualize_distribution(self, stats_df: pd.DataFrame, output_path: str) -> None:
        """Create a grouped bar plot visualization of the pneumothorax distribution."""
        plt.figure(figsize=(8, 6))

        # Define color palette (Data volume ranking)
        palette_name = 'coolwarm'
        pal = sns.color_palette(palette_name, len(stats_df['Dataset']))
        rank = stats_df['Count'].argsort().argsort()
        palette_list = list(np.array(pal)[rank])

        plt.style.use('cyberpunk')
        ax = sns.barplot(
            x='Dataset', 
            y='Count', 
            hue='Category',
            data=stats_df,
            gap=0.05,
            width=0.7
        )

        # Set title and labels
        ax.set_title('Distribution of Pneumothorax Cases', fontsize=16)
        ax.set_xlabel('Dataset', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        
        # Add count labels on the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=13)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)


    def print_summary(self) -> None:
        """Print a summary of the dataset."""
        train_count = len(self.train_df)
        test_count = len(self.test_df)
        train_positive = (self.train_df['has_pneumo'] == 1).sum()
        test_positive = (self.test_df['has_pneumo'] == 1).sum()
        
        print(f"Dataset Summary:")
        print(f"  - Training set: {train_count} images ({train_positive} with pneumothorax)")
        print(f"  - Testing set: {test_count} images ({test_positive} with pneumothorax)")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze and visualize SIIM-ACR Pneumothorax Segmentation dataset.')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the testing CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the visualization')
    
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    analyzer = PneumothoraxAnalyzer(args.train_csv, args.test_csv)
    
    # Ensure the output directory exists
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get dataset statistics
        stats_df = analyzer.get_dataset_stats()
        
        # Print summary information
        analyzer.print_summary()
        
        # Create and save visualization
        analyzer.visualize_distribution(stats_df, args.output_path)
        
        print(f"Analysis complete. Visualization saved to {args.output_path}")
    except ValueError as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()

"""
Usage example:

python data/pneumothorax_analyzer.py --train_csv path/to/stage_1_train_images.csv --test_csv path/to/stage_1_test_images.csv --output_path .output/output.png

python data/pneumothorax_analyzer.py --train_csv "D:\Kai\DATA_Set_2\X-ray\SIIM-ACR-Pneumothorax\stage_1_train_images.csv" --test_csv "D:\Kai\DATA_Set_2\X-ray\SIIM-ACR-Pneumothorax\stage_1_test_images.csv" --output_path ./pneumothorax.png

"""