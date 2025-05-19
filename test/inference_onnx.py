"""
ONNX Medical Image Inference Script

This script performs inference on medical images using an ONNX model.
It can process a single image or recursively process all images in a directory.
"""
import sys
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
import cv2
import time
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

class ImageFormat(Enum):
    """Supported image formats for medical image processing."""
    JPG = "jpg"
    PNG = "png"
    DICOM = "dicom"

@dataclass
class ImageProcessorConfig:
    """Configuration for the medical image processor."""
    # Image size for resizing
    image_size: int = 224
    
    # Normalization parameters
    normalization: str = "imagenet"  # Options: "none", "half", "imagenet"
    
    # Normalization means and stds for different strategies
    norm_params: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "half": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        }
    })

class MedicalImageProcessor:
    """
    A lightweight processor for medical images without PyTorch dependencies.
    Uses OpenCV and NumPy for image operations.
    """
    
    def __init__(self, config: Optional[ImageProcessorConfig] = None):
        """Initialize the medical image processor."""
        self.config = config or ImageProcessorConfig()
    

    def read_image(self, img_path: Union[str, Path], format: ImageFormat = None) -> np.ndarray:
        """Read image from file path based on format."""
        img_path = Path(img_path)
        
        # Determine format from file extension if not specified
        if format is None:
            ext = img_path.suffix.lower()[1:]  # Remove the dot
            try:
                format = ImageFormat(ext)
            except ValueError:
                format = ImageFormat.JPG  # Default to JPG
        
        if format == ImageFormat.JPG or format == ImageFormat.PNG:
            return self._read_from_file(img_path)
        elif format == ImageFormat.DICOM:
            return self._read_from_dicom(img_path)
        else:
            raise ValueError(f"Unsupported image format: {format}")
    

    def _read_from_file(self, img_path: Union[str, Path]) -> np.ndarray:
        """Read and preprocess a JPG/PNG image."""
        img_path = Path(img_path)
        # Read as grayscale for medical images
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise FileNotFoundError(f"Image not found or couldn't be read: {img_path}")
        
        # Process the image
        processed_img = self.preprocess_image(img)
        return processed_img
    

    def _read_from_dicom(self, img_path: Union[str, Path]) -> np.ndarray:
        """Read and preprocess a DICOM image."""
        try:
            import pydicom
            dicom = pydicom.dcmread(str(img_path))
            img = dicom.pixel_array.astype(np.float32)
            # Normalize to 0-255 range
            img = ((img - img.min()) / (img.max() - img.min())) * 255.0
            img = img.astype(np.uint8)
            # Process the image
            processed_img = self.preprocess_image(img)
            return processed_img
        except ImportError:
            raise ImportError("DICOM reading requires pydicom. Install with: pip install pydicom")
    

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps to the image."""
        # Resize the image while preserving aspect ratio
        resized_img = self._resize_img(img, self.config.image_size)
        
        # Convert to RGB (3 channels) by repeating grayscale
        if len(resized_img.shape) == 2:
            rgb_img = np.stack([resized_img] * 3, axis=-1)
        else:
            rgb_img = resized_img
            
        # Apply normalization
        normalized_img = self._normalize_image(rgb_img)
        
        # Transpose from HWC to CHW format for ONNX
        normalized_img = normalized_img.transpose(2, 0, 1)
        
        # Add batch dimension
        normalized_img = np.expand_dims(normalized_img, axis=0)
        
        return normalized_img
    

    def _resize_img(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize image while preserving aspect ratio and padding shorter dimension.
        
        Args:
            img: Input image
            target_size: Target size for the image
            
        Returns:
            Resized and padded image
        """
        height, width = img.shape[:2]
        
        # Determine which dimension to scale
        if height > width:
            # Image is taller
            scale_factor = target_size / height
            new_height = target_size
            new_width = int(width * scale_factor)
        else:
            # Image is wider or square
            scale_factor = target_size / width
            new_width = target_size
            new_height = int(height * scale_factor)
            
        # Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        
        # Calculate padding
        pad_height = target_size - new_height
        pad_width = target_size - new_width
        
        # Calculate padding for each side
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        
        # Apply padding
        if len(resized_img.shape) == 2:
            # Grayscale image
            padded_img = np.pad(
                resized_img, [(top, bottom), (left, right)], 
                mode="constant", constant_values=0
            )
        else:
            # Color image
            padded_img = np.pad(
                resized_img, [(top, bottom), (left, right), (0, 0)], 
                mode="constant", constant_values=0
            )
        
        return padded_img
    

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply normalization to the image.
        
        Args:
            img: Input image as NumPy array (RGB)
            
        Returns:
            Normalized image
        """
        # Convert to float32 and scale to [0, 1]
        img = img.astype(np.float32) / 255.0

        if self.config.normalization == "none":
            return img
        
        # Get normalization parameters
        norm_type = self.config.normalization
        if norm_type not in self.config.norm_params:
            raise ValueError(f"Unsupported normalization method: {norm_type}")
            
        mean = np.array(self.config.norm_params[norm_type]["mean"])
        std = np.array(self.config.norm_params[norm_type]["std"])
        
        # Apply normalization
        normalized_img = (img - mean) / std
        normalized_img = normalized_img.astype(np.float32)

        return normalized_img

def save_visualization(input_image, output_path, probabilities, class_names=None):
    """
    Save a visualization of the original image with predictions.
    
    Args:
        input_image: Path to the original input image
        output_path: Path to save the visualization
        probabilities: Model prediction probabilities
        class_names: List of class names (if available)
    """
    # Read the original image
    img = cv2.imread(str(input_image))
    if img is None:
        # Try to read as grayscale and convert to BGR
        img = cv2.imread(str(input_image), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise FileNotFoundError(f"Could not read image: {input_image}")
    
    # Create a white background for the text
    h, w = img.shape[:2]
    text_height = 30 * len(probabilities)
    vis = np.ones((h + text_height, w, 3), dtype=np.uint8) * 255
    vis[:h, :w] = img
    
    # Add prediction text
    for i, prob in enumerate(probabilities[0]):
        y_pos = h + 25 * (i + 1)
        class_label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
        text = f"{class_label}: {prob:.4f}"
        cv2.putText(vis, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save the visualization
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(str(output_path), vis)
    print(f"Visualization saved to {output_path}")

def run_onnx_inference(image_path, onnx_model_path, output_path=None, 
                       class_names=None, image_size=224, normalization="imagenet",
                       use_random_inference=False, random_seed=None):
    """
    Run inference on a single image using an ONNX model.
    
    Args:
        image_path: Path to the input image
        onnx_model_path: Path to the ONNX model
        output_path: Path to save the output visualization (optional)
        class_names: List of class names (optional)
        image_size: Size for image preprocessing
        normalization: Normalization method ("none", "half", "imagenet")
        use_random_inference: Whether to use random numbers for inference (testing mode)
        random_seed: Seed for random number generation (optional)
        
    Returns:
        Probability predictions for the image
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Configure image processor
    config = ImageProcessorConfig(image_size=image_size, normalization=normalization)
    processor = MedicalImageProcessor(config)
    
    # Preprocess the image
    start_time = time.time()
    print(f"Processing image: {image_path}")
    img_input = processor.read_image(image_path)
    
    # Create ONNX Runtime session
    session = None
    if not hasattr(run_onnx_inference, "session") or run_onnx_inference.session is None:
        print(f"Loading ONNX model from {onnx_model_path}...")
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(onnx_model_path, providers=providers)
        run_onnx_inference.session = session
    else:
        session = run_onnx_inference.session
    
    input_name = session.get_inputs()[0].name
    
    if use_random_inference:
        print("TESTING MODE: Using random numbers for inference")
        # Get output shape from the model
        output_shape = session.get_outputs()[0].shape
        print(f"Model output shape: {output_shape}")
        
        # Handle dynamic or symbolic dimensions in output_shape
        numeric_shape = []
        for dim in output_shape:
            if dim is None or isinstance(dim, str):
                # For batch dimension, use 1
                if dim == 'batch_size' or len(numeric_shape) == 0:
                    numeric_shape.append(1)
                # For class dimension, use length of class_names or default
                else:
                    numeric_shape.append(len(class_names) if class_names else 5)
            else:
                # Keep numeric dimensions as they are
                numeric_shape.append(int(dim))
        
        print(f"Using shape for random inference: {numeric_shape}")
        logits = np.random.randn(*numeric_shape).astype(np.float32)
    else:
        # Run inference
        print("Running inference...")
        outputs = session.run(None, {input_name: img_input})
        logits = outputs[0]
    
    # Convert logits to probabilities
    probabilities = 1 / (1 + np.exp(-logits))  # sigmoid
    
    # Print results
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.4f} seconds")
    
    print("\nPrediction Results:")
    for i, prob in enumerate(probabilities[0]):
        class_label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
        print(f"{class_label}: {(prob)*100:.1f} %")
    
    # Save visualization if requested
    if output_path:
        save_visualization(image_path, output_path, probabilities, class_names)
    
    return probabilities

def find_images_in_directory(directory_path):
    """
    Recursively find all supported image files in a directory and its subdirectories.
    
    Args:
        directory_path: Path to the directory to search
        
    Returns:
        List of image file paths
    """
    directory_path = Path(directory_path)
    supported_extensions = [f".{fmt.value}" for fmt in ImageFormat]
    
    image_files = []
    for ext in supported_extensions:
        # Use recursive glob to find files with the extension
        image_files.extend(list(directory_path.glob(f"**/*{ext}")))
        # Also check for uppercase extensions
        image_files.extend(list(directory_path.glob(f"**/*{ext.upper()}")))
    
    # Additional check for DICOM files without extension
    try:
        import pydicom
        # Find files without extension that might be DICOM
        for file_path in directory_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix == "":
                try:
                    # Try to read as DICOM
                    pydicom.dcmread(str(file_path))
                    image_files.append(file_path)
                except:
                    # Not a DICOM file, skip
                    pass
    except ImportError:
        print("pydicom not installed, skipping DICOM file detection for files without extension")
    
    return sorted(image_files)

def main():
    parser = argparse.ArgumentParser(description="ONNX Medical Image Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, 
                       help='Path to a single input image')
    group.add_argument('--folder', type=str, 
                       help='Path to a folder of images (will process all images recursively)')
    
    parser.add_argument('--onnx_model', type=str, required=True, 
                        help='Path to the ONNX model')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output visualizations (required if --folder is used)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output visualization (for single image)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size for image preprocessing (default: 224)')
    parser.add_argument('--normalization', type=str, default="imagenet",
                        choices=["none", "half", "imagenet"],
                        help='Normalization method (default: imagenet)')
    parser.add_argument('--classes', type=str, default=None,
                        help='Path to a text file with class names (one per line)')
    parser.add_argument('--random_inference', action='store_true',
                        help='Use random numbers for inference (test mode)')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Seed for random number generation (optional)')
    
    args = parser.parse_args()
    
    # Load class names if provided
    class_names = None
    if args.classes:
        try:
            with open(args.classes, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")
    
    # Process folder of images
    if args.folder:
        # if not args.output_dir:
        #     parser.error("--output_dir is required when using --folder")
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        
        # Find all images in the directory
        image_files = find_images_in_directory(args.folder)
        if not image_files:
            print(f"No supported image files found in {args.folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Initialize the run_onnx_inference.session to None to load model only once
        run_onnx_inference.session = None
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"\nProcessing image {i+1}/{len(image_files)}: {image_path}")
            
            # Create output filename with same relative path structure
            rel_path = image_path.relative_to(Path(args.folder))
            output_path = output_dir / f"{rel_path.stem}_result{rel_path.suffix}"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Run inference
                run_onnx_inference(
                    image_path, 
                    args.onnx_model,
                    output_path,
                    class_names,
                    args.image_size,
                    args.normalization,
                    args.random_inference,
                    args.random_seed
                )
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"\nProcessed {len(image_files)} images. Results saved to {args.output_dir}")
    
    # Process single image
    else:
        output_path = args.output
        run_onnx_inference(
            args.image, 
            args.onnx_model,
            output_path,
            class_names,
            args.image_size,
            args.normalization,
            args.random_inference,
            args.random_seed
        )

if __name__ == "__main__":
    main()

"""
Example usage:
# Single image inference
python infer_medical_images.py --image path/to/image.jpg --onnx_model path/to/model.onnx --output visualization.jpg

# Folder inference (recursive)
python infer_medical_images.py --folder path/to/image_folder --onnx_model path/to/model.onnx --output_dir path/to/results

# With custom class names
python infer_medical_images.py --folder path/to/image_folder --onnx_model path/to/model.onnx --output_dir path/to/results --classes class_names.txt

# With custom processing parameters
python infer_medical_images.py --folder path/to/image_folder --onnx_model path/to/model.onnx --output_dir path/to/results --image_size 256 --normalization half

# With random inference for testing
python infer_medical_images.py --folder path/to/image_folder --onnx_model path/to/model.onnx --output_dir path/to/results --random_inference


python test\inference_onnx.py --normalization "half" --image E:/Kai_2/CODE_Repository/CppTest/input.jpg --onnx_model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx" --random_inference --random_seed 42

python test\inference_onnx.py --normalization "half" --image "D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0\valid\patient64545\study1\view1_frontal.jpg" --onnx_model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx"



python test\inference_onnx.py --folder D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0\valid --onnx_model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx" --output_dir output

"""


"""
Example usage:
python infer_single_image.py --image path/to/image.jpg --onnx_model path/to/model.onnx --output visualization.jpg

# With custom class names
python infer_single_image.py --image path/to/image.jpg --onnx_model path/to/model.onnx --classes class_names.txt

# With custom processing parameters
python infer_single_image.py --image path/to/image.jpg --onnx_model path/to/model.onnx --image_size 256 --normalization half

# With random inference for testing
python infer_single_image.py --image path/to/image.jpg --onnx_model path/to/model.onnx --random_inference

# With random inference and fixed seed for reproducible testing
python infer_single_image.py --image path/to/image.jpg --onnx_model path/to/model.onnx --random_inference --random_seed 42
"""