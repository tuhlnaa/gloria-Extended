import shutil
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path

def copy_images_from_csv(input_dir, output_dir, csv_path):
    """Copy images listed in a CSV file from input directory to output directory."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        # Remove the 'CheXpert-v1.0/' prefix from the path
        relative_path = row['Path'].replace('CheXpert-v1.0/', '')
        src_path = Path(input_dir) / relative_path
       
        # Preserve directory structure by using the relative path for destination
        # This will maintain patient/study structure
        dst_path = output_dir / relative_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
       
        # Check for duplicate filenames and adjust if necessary
        if dst_path.exists():
            base_name = dst_path.stem
            extension = dst_path.suffix
            dst_path = dst_path.parent / f"{base_name}_{index}{extension}"
       
        try:
            # Make sure source directory exists
            if not src_path.exists():
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

"""
python utils\image_copy_script.py --input "D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0" --output "D:\Kai\DATA_Set_2\X-ray\chexpert_5x200" --csv "E:\Kai_2\CODE_Repository\gloria-Extended\pretrained\chexpert_5x200.csv"

"""