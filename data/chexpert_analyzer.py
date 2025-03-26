"""
CheXpert Data Analysis and Visualization Tool

This script analyzes CheXpert CSV data to count and visualize sick (1.0) and uncertain (-1.0) 
findings, generating bar plots grouped by these categories.
"""

import argparse
import mplcyberpunk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Tuple, Optional

class CheXpertAnalyzer:
    """Class for analyzing and visualizing CheXpert dataset."""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        # Drop non-finding columns and first row column
        self.feature_cols = [col for col in self.df.columns if col not in 
                            ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
        

    def count_conditions(self, tags: List[str]) -> pd.DataFrame:
        """
        Count sick (1.0) and uncertain (-1.0) cases for specified tags.

        Args:
            tags: List of condition tags to analyze

        Returns:
            DataFrame with counts of sick and uncertain cases for each tag
        """
        # Validate that all requested tags exist in the dataset
        invalid_tags = [tag for tag in tags if tag not in self.feature_cols]
        if invalid_tags:
            raise ValueError(f"Invalid tags: {invalid_tags}. Available tags: {self.feature_cols}")
        
        # Filter to requested tags only
        tag_data = self.df[tags]
        
        # Count occurrences of each condition status
        result_data = []
        
        for tag in tags:
            sick_count = (tag_data[tag] == 1.0).sum()
            uncertain_count = (tag_data[tag] == -1.0).sum()
            
            result_data.extend([
                {'Tag': tag, 'Category': 'Sick', 'Count': sick_count},
                {'Tag': tag, 'Category': 'Uncertain', 'Count': uncertain_count}
            ])
            
        return pd.DataFrame(result_data)


    def visualize_counts(self, count_df: pd.DataFrame, output_path: str) -> None:
        """Create a grouped bar plot visualization of the counts."""
        #plt.figure(figsize=(12, 8))
        
        # Define color palette (Data volume ranking)
        palette_name = 'coolwarm'
        pal = sns.color_palette(palette_name, len(count_df['Tag']))
        print(count_df['Tag'])
        rank = count_df['Count'].argsort().argsort()
        palette_list = list(np.array(pal)[rank])

        # Create the grouped bar plot
        plt.style.use('cyberpunk')
        ax = sns.barplot(
            x='Tag', 
            y='Count', 
            hue='Category',
            data=count_df,
            palette=palette_list,
            dodge=False
            #palette={'Sick': 'coral', 'Uncertain': 'lightblue'}, 
        )
        
        # Set title and labels
        ax.set_title('Counts of Sick and Uncertain Cases by Condition', fontsize=16)
        ax.set_xlabel('Conditions', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        
        # Rotate x-axis labels if there are many conditions
        plt.xticks(rotation=35, ha='right')
        
        # Add count labels on the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=16)
            
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze and visualize CheXpert dataset.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CheXpert CSV file')
    parser.add_argument('--tags', type=str, nargs='+', required=True, help='List of condition tags to analyze')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the visualization')
    
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    
    # Ensure the output directory exists
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer and process data
    analyzer = CheXpertAnalyzer(args.csv_path)
    
    try:
        count_df = analyzer.count_conditions(args.tags)
        analyzer.visualize_counts(count_df, args.output_path)
        print(f"Analysis complete. Visualization saved to {args.output_path}")
    except ValueError as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()

"""
python chexpert_analyzer.py --csv_path path/to/chexpert.csv --tags "No Finding" "Pneumonia" "Edema" --output_path path/to/output.png

python data\chexpert_analyzer.py --csv_path D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0-small\valid.csv --tags "Atelectasis" "Cardiomegaly" "Consolidation" "Edema" "Pleural Effusion" --output_path output/output.png


"""