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
    
    # Evaluate model
    print("Evaluating ONNX model...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validate"):
            images, labels = batch
            
            # Convert to numpy for ONNX Runtime
            images_np = images.numpy()
            
            # Run inference
            outputs = session.run(None, {input_name: images_np})
            logits = torch.from_numpy(outputs[0])
            
            # Move tensors to the appropriate device for loss computation
            labels = labels.to(device)
            logits = logits.to(device)
            
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
    if config.model.n_classes > 1 and config.model.multi_label:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config.model.n_classes > 1 and not config.model.multi_label:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # Initialize validator
    validator = validator_factory.get_validator(config, model, criterion)
    
    # Load validation dataset
    val_loader, _ = dataset_factory.get_dataloader(config, split="valid")
    
    # Run validation
    print("\nEvaluating PyTorch model...")
    pytorch_results = validator.validate(val_loader)
    
    # Print comparison
    print("\nModel Comparison (ONNX vs PyTorch):")
    print("{:<20} {:<15} {:<15} {:<15}".format("Metric", "ONNX", "PyTorch", "Difference"))
    print("-" * 65)
    
    for metric in onnx_results:
        if metric in pytorch_results:
            diff = onnx_results[metric] - pytorch_results[metric]
            print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                metric, onnx_results[metric], pytorch_results[metric], diff
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
"""