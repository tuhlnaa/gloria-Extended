import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict
from rich import print

from data.factory import dataset_factory
from configs.config import parse_args
from gloria import builder
from gloria.engine.gloria_validator import GloriaValidator
from gloria.models import pytorch
from utils.checkpoint import CheckpointHandler


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def setup_evaluation(config: OmegaConf) -> Dict:
    """Setup evaluation environment."""
    # Set random seed for reproducibility
    set_seed(config.misc.seed)
    
    # Setup data loader for validation
    val_loader, _ = dataset_factory.get_dataloader(config, split="valid")
    
    # Initialize model
    model = builder.build_gloria_model(config)
    criterion = builder.build_loss(config)

    return {
        'model': model,
        'val_loader': val_loader,
        "criterion": criterion
    }


def load_model_checkpoint(model, checkpoint_path):
    """Load model weights from checkpoint."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model_state_dict = builder.normalize_model_state_dict(checkpoint)
    model.load_state_dict(model_state_dict)

    # start_epochs = checkpoint.get("epochs")
    # best_val_metrics = checkpoint.get("best_metrics")
    # val_loss = best_val_metrics["val_loss"]
    print(f"[bold blue]Loaded checkpoint from '{checkpoint_path}'.[/bold blue]")

    return model


def evaluate_model(config: OmegaConf) -> Dict[str, float]:
    """Evaluate model on validation set."""
    # Setup evaluation components
    eval_setup = setup_evaluation(config)

    model = eval_setup['model']
    val_loader = eval_setup['val_loader']
    criterion = eval_setup["criterion"]

    # Load model checkpoint if specified
    if config.model.resume:
        model = load_model_checkpoint(model, config.model.resume)
    
    # Initialize validator
    validator = GloriaValidator(config, model, criterion)
    
    # Run validation
    print("Starting evaluation...")
    val_metrics = validator.validate(val_loader)
    
    # Print results
    print("\nEvaluation Results:")
    for metric_name, metric_value in val_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    return val_metrics


def main():
    """Main evaluation function."""
    config = parse_args()
    
    # Create output directory for results if needed
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    metrics = evaluate_model(config)
    
    # Save metrics to file
    results_file = output_path / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.6f}\n")
    
    print(f"Evaluation results saved to {results_file}")


if __name__ == '__main__':
    main()

"""
Usage:
python evaluate.py --config configs/default_gloria_config.yaml
"""