import sys
import time
import torch
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from gloria.utils.metrics import ClassificationMetrics


def benchmark_compute_time(device='cpu', num_samples=64000, num_classes=5):
    """
    Simple benchmark to measure the computation time of the ClassificationMetrics.compute method.
    
    Args:
        device: Device to run on ('cpu' or 'cuda')
        num_samples: Number of samples to process
        num_classes: Number of classes for multilabel classification
    
    Returns:
        Time taken by the compute method in seconds
    """
    print(f"\nRunning benchmark on {device} with {num_samples} samples and {num_classes} classes")
    
    # Initialize metrics tracker
    metrics_tracker = ClassificationMetrics(num_classes=num_classes, split='benchmark').to(device)
    
    # Generate random logits and labels
    logits = torch.randn(num_samples, num_classes, device=device) * 2
    labels = (torch.rand(num_samples, num_classes, device=device) > 0.8).to(torch.int32)
    
    # Add some correlation between logits and labels for realistic metrics
    logits = logits + labels.float() * 2
    
    # Update the metrics tracker with all data at once
    metrics_tracker.update(logits, labels)
    
    # Measure compute time
    torch.cuda.synchronize() if device == 'cuda' else None  # Ensure CUDA operations are completed
    start_time = time.time()
    
    metrics = metrics_tracker.compute()
    
    torch.cuda.synchronize() if device == 'cuda' else None  # Ensure CUDA operations are completed
    compute_time = time.time() - start_time
    
    print(f"  Compute time: {compute_time:.4f} seconds")
    return compute_time


def main():
    parser = argparse.ArgumentParser(description='Benchmark the ClassificationMetrics.compute method')
    parser.add_argument('--num_samples', type=int, default=640000, help='Number of samples to process')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--cpu_only', action='store_true', help='Run only on CPU')
    
    args = parser.parse_args()
    
    # Run CPU benchmark
    cpu_time = benchmark_compute_time('cpu', args.num_samples, args.num_classes)
    
    # Run GPU benchmark if available and not disabled
    if torch.cuda.is_available() and not args.cpu_only:
        gpu_time = benchmark_compute_time('cuda', args.num_samples, args.num_classes)
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\nGPU vs CPU Speedup: {speedup:.2f}x faster on GPU")

if __name__ == "__main__":
    main()