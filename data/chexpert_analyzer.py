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

# Based on original CheXpert paper
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

class CheXpertAnalyzer:
    """Class for analyzing and visualizing CheXpert dataset."""

    def __init__(self, csv_path: str, convert_uncertain: bool = False):
        """
        Initialize the CheXpert analyzer.
        
        Args:
            csv_path: Path to the CheXpert CSV file
            convert_uncertain: Whether to convert uncertain labels (-1) using the mapping
        """
        self.df = pd.read_csv(csv_path)
        self.convert_uncertain = convert_uncertain
        
        # Drop non-finding columns and first row column
        self.feature_cols = [col for col in self.df.columns if col not in 
                            ['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
        
        # Apply uncertain mappings if requested
        if self.convert_uncertain:
            self._apply_uncertain_mappings()
    
    def _apply_uncertain_mappings(self) -> None:
        """Apply the uncertain value mappings from the CheXpert paper."""
        for condition, mapping in CHEXPERT_UNCERTAIN_MAPPINGS.items():
            if condition in self.df.columns:
                # Replace -1.0 values with the mapping (0 or 1)
                self.df.loc[self.df[condition] == -1.0, condition] = mapping

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
            
            # Only count uncertain if we're not converting them
            if self.convert_uncertain:
                # When converted, we don't have uncertain values anymore
                result_data.append({'Tag': tag, 'Category': 'Positive', 'Count': sick_count})
            else:
                uncertain_count = (tag_data[tag] == -1.0).sum()
                result_data.extend([
                    {'Tag': tag, 'Category': 'Positive', 'Count': sick_count},
                    {'Tag': tag, 'Category': 'Uncertain', 'Count': uncertain_count}
                ])
            
        return pd.DataFrame(result_data)


    def visualize_counts(self, count_df: pd.DataFrame, output_path: str) -> None:
        """Create a grouped bar plot visualization of the counts."""
        plt.figure(figsize=(8, 6))

        # Define color palette (Data volume ranking)
        palette_name = 'coolwarm'
        pal = sns.color_palette(palette_name, len(count_df['Tag']))
        rank = count_df['Count'].argsort().argsort()
        palette_list = list(np.array(pal)[rank])


        plt.style.use('cyberpunk')
        if self.convert_uncertain:
            # Create the grouped bar plot
            ax = sns.barplot(
                x='Tag', 
                y='Count', 
                hue='Tag',
                data=count_df,
                palette=palette_list,
                dodge=False, 
                width=0.5
            )
        else:
            ax = sns.barplot(
                x='Tag', 
                y='Count', 
                hue='Category',
                data=count_df,
                gap=0.05,
                width=0.7
            )

        # Set title and labels
        title = 'Counts of Positive Cases by Condition'
        if not self.convert_uncertain:
            title = 'Counts of Positive and Uncertain Cases by Condition'
            
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Conditions', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        
        # Rotate x-axis labels if there are many conditions
        plt.xticks(rotation=30, ha='right')
        
        # Add count labels on the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=13)
            
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def get_available_tags(self) -> List[str]:
        """Return a list of available condition tags in the dataset."""
        return self.feature_cols

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze and visualize CheXpert dataset.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CheXpert CSV file')
    parser.add_argument('--tags', type=str, nargs='+', required=True, help='List of condition tags to analyze')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the visualization')
    parser.add_argument('--convert_uncertain', action='store_true', 
                        help='Convert uncertain labels (-1) using CheXpert paper mappings')
    parser.add_argument('--list_tags', action='store_true',
                        help='List all available condition tags in the dataset')
    
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    
    # Initialize analyzer
    analyzer = CheXpertAnalyzer(args.csv_path, convert_uncertain=args.convert_uncertain)
    
    # List available tags if requested
    if args.list_tags:
        print("Available condition tags:")
        for tag in analyzer.get_available_tags():
            print(f"  - {tag}")
        return
    
    # Ensure the output directory exists
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        count_df = analyzer.count_conditions(args.tags)
        analyzer.visualize_counts(count_df, args.output_path)
        
        # Print summary of what was done
        conversion_status = "with uncertainty conversion" if args.convert_uncertain else "without uncertainty conversion"
        print(f"Analysis complete {conversion_status}. Visualization saved to {args.output_path}")
    except ValueError as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()

"""
python chexpert_analyzer.py --csv_path path/to/chexpert.csv --tags "No Finding" "Pneumonia" "Edema" --output_path path/to/output.png

python data\chexpert_analyzer.py --csv_path D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0-small\valid.csv --tags "Atelectasis" "Cardiomegaly" "Consolidation" "Edema" "Pleural Effusion" --output_path output/chexpert_val.png  --convert_uncertain

python data\chexpert_analyzer.py --csv_path D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0-small\train.csv --tags "Atelectasis" "Cardiomegaly" "Consolidation" "Edema" "Pleural Effusion" --output_path output/chexpert_train.png --convert_uncertain


"""