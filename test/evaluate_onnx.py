import sys
import torch
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm.auto import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.factory import dataset_factory
from gloria.utils.metrics import ClassificationMetrics
from configs.config import parse_args


def evaluate_onnx_model(onnx_path, config_path, batch_size=None):
    """
    Evaluate the ONNX model using the validation dataset.
    
    Args:
        onnx_path (str): Path to the ONNX model
        config_path (str): Path to the configuration file
        batch_size (int, optional): Batch size for evaluation. 
                                    If None, use the batch size from config.
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Load configuration
    config = parse_args(['--config', config_path])
    
    # Override batch size if provided
    if batch_size is not None:
        config.data.batch_size = batch_size
    
    # Set device
    device = config.device.device
    
    # Create ONNX Runtime session
    print(f"Loading ONNX model from {onnx_path}...")
    # Use CPU provider by default, or CUDA if available and specified in config
    providers = ['CPUExecutionProvider']
    if device.startswith('cuda') and ort.get_device() == 'GPU':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_loader, _ = dataset_factory.get_dataloader(config, split="valid")
    
    # Create metrics tracker
    metrics = ClassificationMetrics(split='val')
    metrics.reset()
    
    # Setup loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # if config.model.n_classes > 1 and config.model.multi_label:
    #     criterion = torch.nn.BCEWithLogitsLoss()
    # elif config.model.n_classes > 1 and not config.model.multi_label:
    #     criterion = torch.nn.CrossEntropyLoss()
    # else:
    #     criterion = torch.nn.BCEWithLogitsLoss()
    
    # Replace with actual class names
    class_names = ["class1", "class2", "class3", "class4", "class5"]  

    # Evaluate model
    print("Evaluating ONNX model...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validate")):
            images, labels, img_path = batch
            
            # Convert to numpy for ONNX Runtime
            images_np = images.numpy()
            
            # Run inference
            outputs = session.run(None, {input_name: images_np})
            logits = torch.from_numpy(outputs[0])
            
            # Move tensors to the appropriate device for loss computation
            labels = labels.to(device)
            logits = logits.to(device)
            
            # ==========================================================
            probabilities = torch.sigmoid(logits)
            
            # Print results for this batch
            print(f"\nBatch {batch_idx+1} results:")
            
            # For multi-label classification, we typically use a threshold (e.g., 0.5)
            predictions = (probabilities > 0.5).int()
            
            # Print results for each image in the batch (or just a few to avoid overwhelming output)
            images_to_show = min(100, len(images))  # Show at most 5 images per batch
            
            for i in range(images_to_show):
                print(f"  Image {i+1}:")
                print(img_path[i])
                if class_names:
                    # Print with class names
                    for j, prob in enumerate(probabilities[i]):
                        pred = "✓" if predictions[i][j] else "✗"
                        print(f"    {class_names[j]}: {prob.item():.4f} {pred}")
                else:
                    # Print without class names
                    for j, prob in enumerate(probabilities[i]):
                        pred = "✓" if predictions[i][j] else "✗"
                        print(f"    Class {j}: {prob.item():.4f} {pred}")
                
                # Print ground truth
                true_classes = [j for j, val in enumerate(labels[i]) if val == 1]
                if class_names:
                    true_class_names = [class_names[j] for j in true_classes]
                    print(f"    Ground truth: {true_class_names}")
                else:
                    print(f"    Ground truth: {true_classes}")
                quit()
            # Optional: calculate and print batch accuracy metrics
            # With 5 classes, each sample can have between 0 and 5 correct predictions. 
            # The value of 3.7344 means that on average, each image in this batch had approximately 3.73 out of 5 class predictions correct.
            correct = (predictions == labels).float().sum(dim=1)
            accuracy = correct.sum() / len(predictions)
            print(f"  Batch accuracy: {accuracy.item():.4f}")
            # ==========================================================

            # Compute loss
            loss = criterion(logits, labels)
            
            # Update metrics
            metrics.update(logits, labels, loss)
    
    # Compute final metrics
    results = metrics.compute()
    
    # Print results
    print("\nONNX Model Evaluation Results:")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    return results


def compare_with_pytorch_model(onnx_results, pytorch_checkpoint, config_path):
    """
    Compare ONNX model results with PyTorch model results.
    
    Args:
        onnx_results (dict): Metrics from ONNX model evaluation
        pytorch_checkpoint (str): Path to PyTorch checkpoint
        config_path (str): Path to configuration file
    """
    # Load configuration
    config = parse_args(['--config', config_path])
    
    # Load PyTorch model
    from gloria.builder import build_image_model
    model = build_image_model(config).eval()
    
    # Load checkpoint
    checkpoint = torch.load(pytorch_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to appropriate device
    device = config.device.device
    model = model.to(device)
    
    # Setup validation
    from gloria.engine.factory import validator_factory
    
    # Setup loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # if config.model.n_classes > 1 and config.model.multi_label:
    #     criterion = torch.nn.BCEWithLogitsLoss()
    # elif config.model.n_classes > 1 and not config.model.multi_label:
    #     criterion = torch.nn.CrossEntropyLoss()
    # else:
    #     criterion = torch.nn.BCEWithLogitsLoss()
    
    # Initialize validator
    validator = validator_factory.get_validator(config, model, criterion)
    
    # Load validation dataset
    val_loader, _ = dataset_factory.get_dataloader(config, split="valid")
    
    # Run validation
    print("\nEvaluating PyTorch model...")
    pytorch_results = validator.validate(val_loader)
    
    # Print comparison
    print("\nModel Comparison (ONNX vs PyTorch):")
    print("{:<20} {:<15} {:<15}".format("Metric", "ONNX", "PyTorch"))
    print("-" * 65)
    
    for metric in onnx_results:
        if metric in pytorch_results:
            diff = onnx_results[metric] - pytorch_results[metric]
            print("{:<20} {:<15.4f} {:<15.4f}".format(
                metric, onnx_results[metric], pytorch_results[metric]
            ))
    
    return pytorch_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ONNX model")
    parser.add_argument('--onnx_model', type=str, required=True, 
                        help='Path to the ONNX model')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the configuration file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for evaluation (default: use config value)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with PyTorch model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to PyTorch checkpoint for comparison')
    
    args = parser.parse_args()
    
    # Evaluate ONNX model
    onnx_results = evaluate_onnx_model(args.onnx_model, args.config, args.batch_size)
    
    # Compare with PyTorch model if requested
    if args.compare and args.checkpoint:
        pytorch_results = compare_with_pytorch_model(onnx_results, args.checkpoint, args.config)


if __name__ == "__main__":
    main()

"""
Example usage:
python test/evaluate_onnx.py --onnx_model model.onnx --config configs/default_config.yaml

# Compare with PyTorch model
python test/evaluate_onnx.py --onnx_model model.onnx --config configs/default_config.yaml --compare --checkpoint path/to/checkpoint.pth


python test/evaluate_onnx.py --onnx_model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx" --config configs/default_config.yaml

python test/evaluate_onnx.py --onnx_model "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx" --config configs/default_config.yaml --checkpoint "D:\Kai\training-results\output\experiment01\checkpoint_best.pth" --compare

"""