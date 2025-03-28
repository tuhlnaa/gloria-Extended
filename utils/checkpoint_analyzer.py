import torch
import argparse
import os
import numpy as np
from collections import OrderedDict
import json

def explore_checkpoint(checkpoint_path, save_to_file=None):
    """
    Explore the contents of a PyTorch checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the PyTorch checkpoint file (.pt, .pth)
        save_to_file (str, optional): Path to save the report as a text file
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Get basic information
    print("\n" + "="*50)
    print("CHECKPOINT STRUCTURE OVERVIEW")
    print("="*50)
    
    # If it's a state_dict directly (common format)
    if isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict):
        print("Checkpoint is a dictionary/state_dict")
        top_level_keys = list(checkpoint.keys())
    else:
        print(f"Checkpoint is type: {type(checkpoint)}")
        if hasattr(checkpoint, "__dict__"):
            print("Checkpoint has attributes")
            top_level_keys = list(checkpoint.__dict__.keys())
        else:
            top_level_keys = []
            print("Checkpoint has no dictionary-like structure")
    
    # Display top-level keys
    print("\nTop-level keys:")
    for i, key in enumerate(top_level_keys):
        print(f"  {i+1}. {key}")
    
    # Function to recursively explore and report on the checkpoint structure
    def explore_structure(obj, prefix="", max_depth=3, current_depth=0, max_items=10):
        """Recursively explore the structure of an object"""
        result = []
        
        if current_depth >= max_depth:
            return [f"{prefix} ... (max depth reached)"]
        
        if isinstance(obj, dict) or isinstance(obj, OrderedDict):
            if not obj:
                return [f"{prefix} (empty dict)"]
            
            count = 0
            for k, v in obj.items():
                if count >= max_items:
                    result.append(f"{prefix} ... ({len(obj) - max_items} more items)")
                    break
                
                if isinstance(v, (dict, OrderedDict, list, tuple, torch.Tensor, np.ndarray)):
                    result.append(f"{prefix}{k}:")
                    result.extend(explore_structure(v, prefix + "  ", max_depth, current_depth + 1, max_items))
                else:
                    result.append(f"{prefix}{k}: {type(v).__name__} = {v}")
                
                count += 1
        
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return [f"{prefix} (empty {type(obj).__name__})"]
            
            result.append(f"{prefix} {type(obj).__name__} with {len(obj)} items:")
            if len(obj) > max_items:
                for i, item in enumerate(obj[:max_items]):
                    if isinstance(item, (dict, OrderedDict, list, tuple, torch.Tensor, np.ndarray)):
                        result.append(f"{prefix}  {i}:")
                        result.extend(explore_structure(item, prefix + "    ", max_depth, current_depth + 1, max_items))
                    else:
                        result.append(f"{prefix}  {i}: {type(item).__name__} = {item}")
                result.append(f"{prefix}  ... ({len(obj) - max_items} more items)")
            else:
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, OrderedDict, list, tuple, torch.Tensor, np.ndarray)):
                        result.append(f"{prefix}  {i}:")
                        result.extend(explore_structure(item, prefix + "    ", max_depth, current_depth + 1, max_items))
                    else:
                        result.append(f"{prefix}  {i}: {type(item).__name__} = {item}")
        
        elif isinstance(obj, torch.Tensor):
            result.append(f"{prefix} Tensor: shape={obj.shape}, dtype={obj.dtype}, device={obj.device}")
            if obj.numel() <= 10:
                result.append(f"{prefix}  Values: {obj.tolist()}")
            elif obj.ndim == 0:
                result.append(f"{prefix}  Value: {obj.item()}")
            else:
                # Add some statistical info for larger tensors
                result.append(f"{prefix}  Stats: min={obj.min().item()}, max={obj.max().item()}, mean={obj.float().mean().item():.4f}, std={obj.float().std().item():.4f}")
        
        elif isinstance(obj, np.ndarray):
            result.append(f"{prefix} ndarray: shape={obj.shape}, dtype={obj.dtype}")
            if obj.size <= 10:
                result.append(f"{prefix}  Values: {obj.tolist()}")
            else:
                # Add some statistical info for larger arrays
                result.append(f"{prefix}  Stats: min={obj.min()}, max={obj.max()}, mean={obj.mean():.4f}, std={obj.std():.4f}")
        
        else:
            result.append(f"{prefix} {type(obj).__name__} = {obj}")
        
        return result
    
    # Explore and report detailed structure
    print("\n" + "="*50)
    print("DETAILED CHECKPOINT STRUCTURE")
    print("="*50)
    
    if isinstance(checkpoint, dict) or isinstance(checkpoint, OrderedDict):
        structure_report = explore_structure(checkpoint)
    else:
        if hasattr(checkpoint, "__dict__"):
            structure_report = explore_structure(checkpoint.__dict__)
        else:
            structure_report = ["Checkpoint is not explorable with dictionary-like methods"]
    
    for line in structure_report:
        print(line)
    
    # Save to file if requested
    if save_to_file:
        with open(save_to_file, 'w') as f:
            f.write(f"Checkpoint Analysis: {checkpoint_path}\n\n")
            f.write("="*50 + "\n")
            f.write("TOP-LEVEL KEYS\n")
            f.write("="*50 + "\n")
            for i, key in enumerate(top_level_keys):
                f.write(f"{i+1}. {key}\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("DETAILED STRUCTURE\n")
            f.write("="*50 + "\n")
            for line in structure_report:
                f.write(line + "\n")
            
            print(f"\nReport saved to {save_to_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore a PyTorch checkpoint file")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    parser.add_argument("--output", "-o", type=str, help="Save the report to this file")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth to explore (default: 3)")
    
    args = parser.parse_args()
    
    explore_checkpoint(args.checkpoint_path, args.output)