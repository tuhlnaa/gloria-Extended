def analyze_checkpoint_mismatch(model, checkpoint_path):
    """
    Analyze mismatches between a model and a checkpoint.
    
    Args:
        model: The PyTorch model to compare
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        Dict containing analysis information
    """
    import torch
    from collections import defaultdict
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if "model_state_dict" in checkpoint:
        checkpoint_state_dict = checkpoint["model_state_dict"]
    else:
        checkpoint_state_dict = checkpoint  # Assume it's just the state dict directly
    
    # Get model state dict
    model_state_dict = model.state_dict()
    
    # Collect keys
    checkpoint_keys = set(checkpoint_state_dict.keys())
    model_keys = set(model_state_dict.keys())
    
    # Find differences
    unexpected_keys = checkpoint_keys - model_keys
    missing_keys = model_keys - checkpoint_keys
    
    # Group keys by prefix to identify structural differences
    def get_key_prefix(key, level=1):
        parts = key.split('.')
        return '.'.join(parts[:level])
    
    # Analyze checkpoint structure
    checkpoint_prefixes = defaultdict(list)
    for key in checkpoint_keys:
        prefix = get_key_prefix(key, 1)
        checkpoint_prefixes[prefix].append(key)
    
    # Analyze model structure
    model_prefixes = defaultdict(list)
    for key in model_keys:
        prefix = get_key_prefix(key, 1)
        model_prefixes[prefix].append(key)
    
    # Print summary
    print(f"{'='*20} CHECKPOINT ANALYSIS {'='*20}")
    print(f"Total keys in checkpoint: {len(checkpoint_keys)}")
    print(f"Total keys in model: {len(model_keys)}")
    print(f"Unexpected keys (in checkpoint but not in model): {len(unexpected_keys)}")
    print(f"Missing keys (in model but not in checkpoint): {len(missing_keys)}")
    
    # Print checkpoint structure
    print(f"\n{'='*20} CHECKPOINT STRUCTURE {'='*20}")
    for prefix, keys in sorted(checkpoint_prefixes.items()):
        print(f"{prefix}: {len(keys)} parameters")
        # Print a few examples if there are many
        if len(keys) > 5:
            for key in sorted(keys)[:3]:
                shape_str = 'unknown'
                if key in checkpoint_state_dict:
                    shape_str = str(tuple(checkpoint_state_dict[key].shape))
                print(f"  - {key} {shape_str}")
            print(f"  ... and {len(keys)-3} more")
        else:
            for key in sorted(keys):
                shape_str = 'unknown'
                if key in checkpoint_state_dict:
                    shape_str = str(tuple(checkpoint_state_dict[key].shape))
                print(f"  - {key} {shape_str}")
    
    # Print model structure
    print(f"\n{'='*20} MODEL STRUCTURE {'='*20}")
    for prefix, keys in sorted(model_prefixes.items()):
        print(f"{prefix}: {len(keys)} parameters")
        # Print a few examples if there are many
        if len(keys) > 5:
            for key in sorted(keys)[:3]:
                shape_str = str(tuple(model_state_dict[key].shape))
                print(f"  - {key} {shape_str}")
            print(f"  ... and {len(keys)-3} more")
        else:
            for key in sorted(keys):
                shape_str = str(tuple(model_state_dict[key].shape))
                print(f"  - {key} {shape_str}")
    
    # Print some examples of unexpected keys
    if unexpected_keys:
        print(f"\n{'='*20} EXAMPLE UNEXPECTED KEYS {'='*20}")
        for key in sorted(list(unexpected_keys))[:10]:
            shape_str = 'unknown'
            if key in checkpoint_state_dict:
                shape_str = str(tuple(checkpoint_state_dict[key].shape))
            print(f"  - {key} {shape_str}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys)-10} more")
    
    # Print some examples of missing keys
    if missing_keys:
        print(f"\n{'='*20} EXAMPLE MISSING KEYS {'='*20}")
        for key in sorted(list(missing_keys))[:10]:
            shape_str = str(tuple(model_state_dict[key].shape))
            print(f"  - {key} {shape_str}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys)-10} more")
    
    # Create a modified state dict that might work with the current model
    # This is for more advanced cases where structure is similar but key names differ
    
    # Check if it appears to be the same model but with keys organized differently
    # For example, if model uses 'img_encoder' but checkpoint uses 'img_encoder.model'
    
    # Create a mapping of similar keys
    compatible_state_dict = {}
    if len(unexpected_keys) > 0 and len(missing_keys) > 0:
        print(f"\n{'='*20} ATTEMPTED KEY MAPPING {'='*20}")
        
        # Try to map unexpected keys to missing keys by removing prefixes
        for unexpected_key in sorted(list(unexpected_keys)):
            # Try removing potential extra prefixes/nested structure
            potential_simplified_key = unexpected_key.replace('.model.', '.')
            
            # Check if this simplified key exists in the model's keys
            if potential_simplified_key in model_keys:
                compatible_state_dict[potential_simplified_key] = checkpoint_state_dict[unexpected_key]
                print(f"Mapped: {unexpected_key} -> {potential_simplified_key}")
        
        # Alternatively, try adding model prefix to model keys to match checkpoint
        matched_count = 0
        for missing_key in sorted(list(missing_keys)):
            # Try adding potential missing prefix structure
            potential_expanded_key = missing_key.replace('img_encoder.', 'img_encoder.model.')
            
            # Check if this expanded key exists in the checkpoint's keys
            if potential_expanded_key in checkpoint_keys:
                compatible_state_dict[missing_key] = checkpoint_state_dict[potential_expanded_key]
                matched_count += 1
                if matched_count <= 5:  # Only print a few examples
                    print(f"Mapped: {potential_expanded_key} -> {missing_key}")
        
        if matched_count > 5:
            print(f"... and {matched_count-5} more mappings")
    
    return {
        "unexpected_keys": unexpected_keys,
        "missing_keys": missing_keys,
        "checkpoint_prefixes": checkpoint_prefixes,
        "model_prefixes": model_prefixes,
        "compatible_state_dict": compatible_state_dict
    }


def load_checkpoint_with_remapping(model, checkpoint_path, verbose=True):
    """
    Load a checkpoint with potential key remapping to handle structural differences.
    
    Args:
        model: The PyTorch model to load into
        checkpoint_path: Path to the checkpoint file
        verbose: Whether to print detailed information
    
    Returns:
        Tuple of (successfully_loaded_keys, remaining_missing_keys)
    """
    import torch
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if "model_state_dict" in checkpoint:
        checkpoint_state_dict = checkpoint["model_state_dict"]
    else:
        checkpoint_state_dict = checkpoint  # Assume it's just the state dict directly
    
    # Get model state dict
    model_state_dict = model.state_dict()
    
    # Collect keys
    checkpoint_keys = set(checkpoint_state_dict.keys())
    model_keys = set(model_state_dict.keys())
    
    # Direct matches first
    direct_matches = checkpoint_keys.intersection(model_keys)
    
    # Prepare a state dict with direct matches
    new_state_dict = {k: checkpoint_state_dict[k] for k in direct_matches}
    
    if verbose:
        print(f"Direct matches: {len(direct_matches)} parameters")
    
    # Identify keys that need remapping
    remaining_checkpoint_keys = checkpoint_keys - direct_matches
    remaining_model_keys = model_keys - direct_matches
    
    # Try to map remaining keys by applying transformation patterns
    mapping_patterns = [
        # Add patterns to map from checkpoint to model keys
        # Example: remove '.model.' from keys
        (lambda k: k.replace('img_encoder.model.', 'img_encoder.')),
        # Example: add 'module.' prefix for DataParallel models
        (lambda k: f"module.{k}" if not k.startswith('module.') else k),
        # Example: remove 'module.' prefix
        (lambda k: k[7:] if k.startswith('module.') else k),
    ]
    
    # Try each mapping pattern
    for pattern_idx, pattern_func in enumerate(mapping_patterns):
        if verbose:
            print(f"\nTrying mapping pattern {pattern_idx+1}...")
        
        mapped_keys = 0
        
        # Map checkpoint keys to model keys
        for ck in list(remaining_checkpoint_keys):  # Create a copy to modify during iteration
            transformed_key = pattern_func(ck)
            if transformed_key in remaining_model_keys:
                # This is a match with the current pattern
                new_state_dict[transformed_key] = checkpoint_state_dict[ck]
                remaining_checkpoint_keys.remove(ck)
                remaining_model_keys.remove(transformed_key)
                mapped_keys += 1
        
        if verbose:
            print(f"Pattern {pattern_idx+1} mapped {mapped_keys} keys")
            if mapped_keys > 0:
                print(f"Remaining unmapped checkpoint keys: {len(remaining_checkpoint_keys)}")
                print(f"Remaining unmapped model keys: {len(remaining_model_keys)}")
    
    # Load the new state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    if verbose:
        print("\nFinal loading results:")
        print(f"Successfully loaded keys: {len(new_state_dict)}")
        print(f"Remaining missing keys: {len(remaining_model_keys)}")
    
    return new_state_dict.keys(), remaining_model_keys