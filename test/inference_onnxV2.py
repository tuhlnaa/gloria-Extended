import cv2
import torch
import argparse
import random
import numpy as np

from PIL import Image
from pathlib import Path
import onnxruntime as ort
    
def preprocess_image(img_path, target_size=256, crop_size=224, norm_type="imagenet"):
    """
    Preprocess an image for inference with the ONNX model.
    
    Args:
        img_path (str or Path): Path to the image
        target_size (int): Target size for the image
        crop_size (int): Size for center crop
        norm_type (str): Type of normalization to apply ("imagenet" or "half")
        
    Returns:
        numpy.ndarray: Preprocessed image as a numpy array
    """
    # Read image
    img_path = Path(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Image not found or couldn't be read: {img_path}")
    
    # Resize image while preserving aspect ratio
    height, width = img.shape
    
    # Determine which dimension to scale
    if height > width:
        scale_factor = target_size / height
        new_height = target_size
        new_width = int(width * scale_factor)
    else:
        scale_factor = target_size / width
        new_width = target_size
        new_height = int(height * scale_factor)
        
    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Calculate padding
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    
    # Calculate padding for each side
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    
    # Apply padding
    padded_img = np.pad(
        resized_img, [(top, bottom), (left, right)], 
        mode="constant", constant_values=0
    )
    
    # Apply center crop
    if crop_size < target_size:
        # Calculate crop coordinates
        crop_top = (target_size - crop_size) // 2
        crop_bottom = crop_top + crop_size
        crop_left = (target_size - crop_size) // 2
        crop_right = crop_left + crop_size
        
        # Perform center crop
        cropped_img = padded_img[crop_top:crop_bottom, crop_left:crop_right]
    else:
        cropped_img = padded_img
    
    # Convert to PIL Image and to RGB
    pil_img = Image.fromarray(cropped_img).convert("RGB")
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(np.array(pil_img)).float().permute(2, 0, 1) / 255.0
    
    # Apply normalization
    if norm_type == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    else:  # half
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.numpy()


def run_inference(session, image, input_name):
    """
    Run inference with ONNX model on a preprocessed image.
    
    Args:
        session (onnxruntime.InferenceSession): ONNX Runtime session
        image (numpy.ndarray): Preprocessed image
        input_name (str): Input name for the ONNX model
        
    Returns:
        dict: Dictionary with probabilities and predictions
    """
    # Add batch dimension if not present
    if len(image.shape) == 3:
        image = image.reshape(1, *image.shape)
    
    # Run inference
    outputs = session.run(None, {input_name: image})
    logits = torch.from_numpy(outputs[0])
    
    # Apply sigmoid to get probabilities
    probabilities = torch.sigmoid(logits)
    
    # Get predictions using threshold of 0.5
    predictions = (probabilities > 0.5).int()
    
    return {
        'probabilities': probabilities[0],
        'predictions': predictions[0]
    }


def process_images(img_paths, session, input_name, class_names, target_size, crop_size, norm_type):
    """
    Process images and print results.
    
    Args:
        img_paths (list): List of image paths
        session (onnxruntime.InferenceSession): ONNX Runtime session
        input_name (str): Input name for the ONNX model
        class_names (list): List of class names
        target_size (int): Target size for image preprocessing
        crop_size (int): Size for center crop
        norm_type (str): Type of normalization to apply
        
    Returns:
        None
    """
    for i, img_path in enumerate(img_paths, 1):
        try:
            # Preprocess image
            preprocessed_img = preprocess_image(img_path, target_size, crop_size, norm_type)
            
            # Run inference
            results = run_inference(session, preprocessed_img, input_name)
            
            # Print results
            print(f"  Image {i}:")
            
            for j, prob in enumerate(results['probabilities']):
                pred = "✓" if results['predictions'][j] else "✗"
                print(f"    {class_names[j]}: {prob.item():.4f} {pred}")
                
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")


def get_image_files(dir_path, recursive=True):
    """
    Get all image files in a directory.
    
    Args:
        dir_path (str or Path): Path to the directory
        recursive (bool): Whether to process subdirectories recursively
        
    Returns:
        list: List of image file paths
    """
    dir_path = Path(dir_path)
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    # Get all image files
    image_files = []
    
    if recursive:
        for ext in image_extensions:
            image_files.extend(list(dir_path.glob(f"**/*{ext}")))
    else:
        for ext in image_extensions:
            image_files.extend(list(dir_path.glob(f"*{ext}")))
    
    # Sort image files for consistent output
    image_files.sort()
    
    return image_files


def main():
    parser = argparse.ArgumentParser(description="Run inference with ONNX model on images")
    parser.add_argument('--model', type=str, required=True, help='Path to the ONNX model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--recursive', action='store_true', help='Process subdirectories recursively')
    parser.add_argument('--classes', type=str, default="Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion", 
                        help='Comma-separated list of class names')
    parser.add_argument('--target_size', type=int, default=256, 
                        help='Target size for image preprocessing (default: 256)')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Size for center crop (default: 224)')
    parser.add_argument('--norm', type=str, default="half", choices=["imagenet", "half"],
                        help='Type of normalization to apply (default: imagenet)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution')
    
    args = parser.parse_args()
    
    # Parse class names
    class_names = args.classes.split(',')
    
    # Create ONNX Runtime session
    print(f"Loading ONNX model from {args.model}...")
    
    # Use CPU provider by default, or CUDA if available and not forced to CPU
    providers = ['CPUExecutionProvider']
    if not args.cpu and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("Using CUDA for inference")
    else:
        print("Using CPU for inference")
    
    session = ort.InferenceSession(args.model, providers=providers)
    input_name = session.get_inputs()[0].name
    
    # Process input path
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single image
        img_paths = [input_path]
    elif input_path.is_dir():
        # Process directory
        img_paths = get_image_files(input_path, args.recursive)
        print(f"Found {len(img_paths)} images in {input_path}")
    else:
        print(f"Error: Input path '{input_path}' does not exist")
        return
    
    if not img_paths:
        print("No images found to process")
        return
    
    # Process images and print results
    process_images(img_paths, session, input_name, class_names, args.target_size, args.crop_size, args.norm)


if __name__ == "__main__":
    main()

"""
# Process a single image
python onnx_inference.py --model model.onnx --input path/to/image.jpg

# Process all images in a directory
python onnx_inference.py --model model.onnx --input path/to/directory

# Process all images recursively with custom class names
python onnx_inference.py --model model.onnx --input path/to/directory --recursive --classes "Atelectasis,Cardiomegaly,Consolidation,Edema,Pleural Effusion"

# Additional options
# Force CPU execution
python onnx_inference.py --model model.onnx --input path/to/image.jpg --cpu

# Change target image size
python onnx_inference.py --model model.onnx --input path/to/image.jpg --target_size 512

# Change normalization type
python onnx_inference.py --model model.onnx --input path/to/image.jpg --norm half


python test\inference_onnxV2.py --model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx" --norm "half" --input "E:\Kai_2\view1_frontal.jpg"

python test\inference_onnxV2.py --model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx" --norm "half" --input "D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0\valid" --recursive

python test\inference_onnxV2.py --model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx" --norm "half" --input "D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0-small\valid\patient64603\study1\view1_frontal.jpg"

python test\inference_onnxV2.py --model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx" --norm "half" --input "D:\Kai\DATA_Set_2\X-ray\CheXpert-v1.0-small\valid\patient64541\study1\view1_frontal.jpg"

"""