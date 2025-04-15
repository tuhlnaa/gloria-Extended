"""
Utility for combining multiple segmentation masks into a single colorized output.
Supports blending with original images, handles variable number of input masks,
and properly displays overlapping areas.
"""
import cv2
import argparse
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple


class MaskCombiner:
    """Combines multiple binary segmentation masks into a single colorized output."""
    
    def __init__(
        self,
        input_folders: List[str],
        output_folder: str,
        original_images: Optional[str] = None,
        alpha: float = 0.5,
        color_map: Optional[List[Tuple[int, int, int]]] = None
    ):
        """
        Initialize the MaskCombiner.

        Args:
            input_folders: List of paths to folders containing segmentation masks
            output_folder: Path to save combined outputs
            original_images: Optional path to original images for blending
            alpha: Transparency value for blending (0.0 to 1.0)
            color_map: List of RGB tuples for mask colors. If None, generates automatically
        """
        self.input_paths = [Path(folder) for folder in input_folders]
        self.output_path = Path(output_folder)
        self.original_images_path = Path(original_images) if original_images else None
        self.alpha = alpha
        
        # Validate inputs
        self._validate_paths()
        
        # Generate or use provided color map
        self.color_map = (color_map if color_map is not None 
                         else self._generate_color_map(len(input_folders)))


    def _validate_paths(self) -> None:
        """Validate input and output paths exist or can be created."""
        for path in self.input_paths:
            if not path.exists():
                raise FileNotFoundError(f"Input folder not found: {path}")
                
        if self.original_images_path and not self.original_images_path.exists():
            raise FileNotFoundError(
                f"Original images folder not found: {self.original_images_path}"
            )
            
        self.output_path.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def _generate_color_map(num_classes: int) -> List[Tuple[int, int, int]]:
        """
        Generate visually distinct colors for each mask.
        
        Args:
            num_classes: Number of colors needed
            
        Returns:
            List of RGB color tuples
        """
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            # Convert HSV to RGB for better visual distinction
            rgb = cv2.cvtColor(np.uint8([[[hue * 180, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
            colors.append(tuple(map(int, rgb)))
        return colors


    def _load_mask(self, path: Path) -> np.ndarray:
        """Load and normalize a mask image."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return (mask > 0).astype(np.uint8) * 255


    def _combine_masks(self, masks: List[np.ndarray], original_img: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combine multiple masks into a single colorized output with proper overlap handling.
        
        Args:
            masks: List of binary mask arrays
            original_img: Optional original image for blending
            
        Returns:
            Combined and colorized mask array
        """
        # Create a float32 array for accumulating colors with alpha blending
        height, width = masks[0].shape
        overlay = np.zeros((height, width, 3), dtype=np.float32)
        
        # Create an alpha channel to track overlapping regions
        alpha_acc = np.zeros((height, width), dtype=np.float32)
        
        # Apply each mask with its corresponding color using alpha blending
        for mask, color in zip(masks, self.color_map):
            mask_float = mask.astype(np.float32) / 255.0
            for c in range(3):
                # Add color contribution where mask is active
                overlay[:, :, c] += mask_float * color[c]
            # Accumulate alpha for overlapping regions
            alpha_acc += mask_float
            
        # Normalize the overlay based on accumulated alpha
        alpha_acc = np.maximum(alpha_acc, 1.0)[:, :, np.newaxis]
        overlay = (overlay / alpha_acc).astype(np.uint8)
        
        # Create binary mask for any segmentation
        any_mask = (alpha_acc > 0).astype(np.uint8)

        # Blend with original image if provided
        if original_img is not None:

            # Resize original image if needed
            if original_img.shape[:2] != (height, width):
                original_img = cv2.resize(original_img, (width, height))

            # Apply alpha blending only where we have segmentation
            result = np.where(any_mask, cv2.addWeighted(
                    original_img,
                    1.0, # 1 - self.alpha,
                    overlay,
                    self.alpha,
                    0
                ),
                original_img
            )
            return result
            
        return overlay


    def process_all(self) -> None:
        """Process all masks in the input folders."""
        # Get list of mask files (assuming same names across folders)
        mask_files = sorted(self.input_paths[0].glob('*.png'))
        
        for mask_file in tqdm(mask_files, desc="Processing masks"):
            # Load masks from each input folder
            masks = [
                self._load_mask(path / mask_file.name)
                for path in self.input_paths
            ]
            
            # Load original image if path provided
            original_img = None
            if self.original_images_path:
                img_path = self.original_images_path / mask_file.name
                if img_path.exists():
                    original_img = cv2.imread(str(img_path))
            
            # Combine masks
            result = self._combine_masks(masks, original_img)
            
            # Save result
            output_file = self.output_path / mask_file.name
            cv2.imwrite(str(output_file), result)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Combine multiple segmentation masks into colorized outputs."
    )
    
    parser.add_argument('input_folders', nargs='+', help='Paths to folders containing segmentation masks')
    parser.add_argument('--output', required=True, help='Path to output folder')
    parser.add_argument('--original-images', help='Path to original images for blending')
    parser.add_argument('--alpha', type=float, default=0.5, help='Transparency for blending (0.0 to 1.0)')
    args = parser.parse_args()
    
    # Initialize and run mask combiner
    combiner = MaskCombiner(
        args.input_folders,
        args.output,
        args.original_images,
        args.alpha
    )
    
    combiner.process_all()

if __name__ == '__main__':
    main()