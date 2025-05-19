import sys
import torch
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

#from gloria import builder
from gloria.builder import build_image_model
from gloria.engine.factory import validator_factory
from configs.config import parse_args

def convert_checkpoint_to_onnx(checkpoint_path, config_path, output_path, input_shape=(1, 3, 224, 224)):
    """
    Convert a PyTorch checkpoint to ONNX format.
    
    Args:
        checkpoint_path: Path to the PyTorch checkpoint
        config_path: Path to the configuration file
        output_path: Path to save the ONNX model
        input_shape: Input shape for the model (batch_size, channels, height, width)
    """
    # Load configuration
    config = parse_args(['--config', config_path])
    
    # Initialize the model
    model = build_image_model(config).eval()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,                      # Model being run
        dummy_input,                # Model input (or a tuple for multiple inputs)
        output_path,                # Where to save the model
        export_params=True,         # Store the trained parameter weights inside the model file
        # opset_version=13,           # The ONNX version to export the model to
        do_constant_folding=True,   # Whether to execute constant folding for optimization
        input_names=['input'],      # Input tensor names
        output_names=['output'],    # Output tensor names
        dynamic_axes={              # Variable length axes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )
    
    print(f"Model exported to ONNX format at: {output_path}")
    
    # Verify the model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to ONNX")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the checkpoint file')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the configuration file')
    parser.add_argument('--output', type=str, required=True, 
                        help='Path to save the ONNX model')
    parser.add_argument('--height', type=int, default=224, 
                        help='Input height')
    parser.add_argument('--width', type=int, default=224, 
                        help='Input width')
    parser.add_argument('--channels', type=int, default=3, 
                        help='Input channels')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create input shape
    input_shape = (args.batch_size, args.channels, args.height, args.width)
    
    # Convert checkpoint to ONNX
    convert_checkpoint_to_onnx(args.checkpoint, args.config, args.output, input_shape)

if __name__ == "__main__":
    main()

"""
python test\convert_to_onnx.py --checkpoint path/to/checkpoint.pt --config configs/default_gloria_classification_config.yaml --output model.onnx

python test\convert_to_onnx.py --checkpoint "D:\Kai\training-results\output\experiment01\checkpoint_best.pth" --config configs\default_config.yaml  --output model.onnx


pnnx "E:\Kai_2\CODE_Repository\ChestDx-Intelligence-Models\model.onnx" "inputshape=[1,3,224,224]f32"
"""
