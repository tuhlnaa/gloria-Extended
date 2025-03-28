import os
import csv
import argparse
from pathlib import Path

def check_missing_images(csv_file, root_dir, output_dir):
    """
    Checks if images listed in a CSV file exist in the root directory.
    Writes paths of missing images to a text file in the output directory.
    
    Args:
        csv_file (str): Path to the CSV file containing path_to_image field
        root_dir (str): Root directory where images should be located
        output_dir (str): Directory where the missing images list will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path for missing images
    missing_images_file = os.path.join(output_dir, "missing_images.txt")
    
    missing_count = 0
    total_count = 0
    
    # Open the missing images file for writing
    with open(missing_images_file, 'w', encoding='utf-8') as out_file:
        # Read the CSV file
        with open(csv_file, 'r', encoding='utf-8') as csv_in:
            reader = csv.DictReader(csv_in)
            
            # Check if path_to_image field exists
            if 'path_to_image' not in reader.fieldnames:
                raise ValueError("CSV file does not contain 'path_to_image' field")
            
            # Process each row
            for row in reader: 
                total_count += 1
                image_path = row['path_to_image']
                full_path = os.path.join(root_dir, image_path)
                
                # Check if the file exists
                if not os.path.isfile(full_path):
                    missing_count += 1
                    out_file.write(f"{image_path}\n")
    
    print(f"Total images checked: {total_count}")
    print(f"Missing images found: {missing_count}")
    print(f"List of missing images saved to: {missing_images_file}")

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Check for missing image files listed in a CSV.')
    parser.add_argument('csv_file', help='Path to the CSV file containing path_to_image field')
    parser.add_argument('--root_dir', help='Root directory where images should be located')
    parser.add_argument('--output_dir', help='Directory where the missing images list will be saved')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the function to check for missing images
    check_missing_images(args.csv_file, args.root_dir, args.output_dir)

if __name__ == "__main__":
    main()