import os
import shutil
import pandas as pd
import argparse
from tqdm import tqdm

def copy_images_from_csv(input_dir, output_dir, csv_path):
    """
    Copy images listed in a CSV file from input directory to output directory.
    
    Args:
        input_dir (str): Base directory containing the image files
        output_dir (str): Directory where images will be copied to
        csv_path (str): Path to the CSV file containing image paths
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV file with {len(df)} entries")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if 'Path' column exists
    if 'Path' not in df.columns:
        print("Error: CSV file does not contain a 'Path' column")
        return
    
    # Copy each image file
    success_count = 0
    error_count = 0
    
    print("Starting to copy image files...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        # Get relative image path from CSV
        relative_path = row['Path']
        
        # Calculate the full source path
        src_path = os.path.join(input_dir, relative_path)
        
        # Create a filename for the destination
        # Use the original filename to preserve the file extension
        filename = os.path.basename(relative_path)
        dst_path = os.path.join(output_dir, filename)
        
        # Check for duplicate filenames and adjust if necessary
        if os.path.exists(dst_path):
            base_name, extension = os.path.splitext(filename)
            dst_path = os.path.join(output_dir, f"{base_name}_{index}{extension}")
        
        try:
            # Make sure source directory exists
            if not os.path.exists(src_path):
                print(f"Warning: Source file does not exist: {src_path}")
                error_count += 1
                continue
            
            # Copy the file
            shutil.copy2(src_path, dst_path)
            success_count += 1
            
        except Exception as e:
            print(f"Error copying file {src_path}: {e}")
            error_count += 1
    
    print(f"\nCopying complete! Successfully copied {success_count} images.")
    if error_count > 0:
        print(f"Encountered {error_count} errors during copying.")

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Copy images listed in a CSV file to a new location')
    parser.add_argument('--input', required=True, help='Input data directory containing the images')
    parser.add_argument('--output', required=True, help='Output directory where images will be copied')
    parser.add_argument('--csv', required=True, help='Path to the CSV file with image paths')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print configuration
    print("Configuration:")
    print(f"  Input directory: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  CSV file path: {args.csv}")
    print()
    
    # Call the copy function
    copy_images_from_csv(args.input, args.output, args.csv)

if __name__ == "__main__":
    main()