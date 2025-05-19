"""
ONNX Single Image Inference Script

This script performs inference on a single medical image using an ONNX model.
It uses the MedicalImageProcessor for preprocessing without PyTorch dependencies.
"""
import sys
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
import cv2
import time
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
    cv2.imwrite(str(output_path), vis)
    print(f"Visualization saved to {output_path}")


def run_onnx_inference(image_path, onnx_model_path, output_path=None, 
                       class_names=None, image_size=224, normalization="imagenet"):
    """
    Run inference on a single image using an ONNX model.
    
    Args:
        image_path: Path to the input image
        onnx_model_path: Path to the ONNX model
        output_path: Path to save the output visualization (optional)
        class_names: List of class names (optional)
        image_size: Size for image preprocessing
        normalization: Normalization method ("none", "half", "imagenet")
        
    Returns:
        Probability predictions for the image
    """
    # Configure image processor
    config = ImageProcessorConfig(image_size=image_size, normalization=normalization)
    processor = MedicalImageProcessor(config)
    
    # Preprocess the image
    start_time = time.time()
    print(f"Processing image: {image_path}")
    img_input = processor.read_image(image_path)
    
    # Create ONNX Runtime session
    print(f"Loading ONNX model from {onnx_model_path}...")
    providers = ['CPUExecutionProvider']
    if ort.get_device() == 'GPU':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
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
        print(f"{class_label}: {prob:.4f}")
    
    # Save visualization if requested
    if output_path:
        save_visualization(image_path, output_path, probabilities, class_names)
    
    return probabilities


def main():
    parser = argparse.ArgumentParser(description="ONNX Single Image Inference")
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the input image')
    parser.add_argument('--onnx_model', type=str, required=True, 
                        help='Path to the ONNX model')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output visualization')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size for image preprocessing (default: 224)')
    parser.add_argument('--normalization', type=str, default="imagenet",
                        choices=["none", "half", "imagenet"],
                        help='Normalization method (default: imagenet)')
    parser.add_argument('--classes', type=str, default=None,
                        help='Path to a text file with class names (one per line)')
    
    args = parser.parse_args()
    
    # Load class names if provided
    class_names = None
    if args.classes:
        try:
            with open(args.classes, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")
    
    # Run inference
    probabilities = run_onnx_inference(
        args.image, 
        args.onnx_model,
        args.output,
        class_names,
        args.image_size,
        args.normalization
    )


if __name__ == "__main__":
    main()

"""
Example usage:
python infer_single_image.py --image path/to/image.jpg --onnx_model path/to/model.onnx --output visualization.jpg

# With custom class names
python infer_single_image.py --image path/to/image.jpg --onnx_model path/to/model.onnx --classes class_names.txt

# With custom processing parameters
python infer_single_image.py --image path/to/image.jpg --onnx_model path/to/model.onnx --image_size 256 --normalization half


python test\inference_onnx.py --image E:/Kai_2/CODE_Repository/CppTest/input.jpg --onnx_model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx"


"""