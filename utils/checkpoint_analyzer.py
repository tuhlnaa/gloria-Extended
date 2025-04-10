"""
Checkpoint Inspector: A tool to display PyTorch checkpoint information.
"""
import argparse
import json
import torch

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress


class CheckpointInspector:
    """Tool for inspecting PyTorch checkpoint files with rich output formatting."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a PyTorch checkpoint file."""
        path = Path(checkpoint_path)
        if not path.exists():
            self.console.print(f"[bold red]Error:[/] Checkpoint file {path} not found")
            return {}
            
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Loading checkpoint...", total=1)
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
                progress.update(task, completed=1)
            return checkpoint
        except Exception as e:
            self.console.print(f"[bold red]Error loading checkpoint:[/] {str(e)}")
            return {}
    

    def _format_timestamp(self, timestamp: str) -> str:
        """Format ISO timestamp to a more readable form."""
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            return "Invalid timestamp"
    

    def _get_state_dict_info(self, state_dict: Dict[str, Any]) -> List[Tuple[str, str, int]]:
        """Extract parameter information from a state dictionary.
        
        Args:
            state_dict: PyTorch state dictionary
            
        Returns:
            List of (name, shape, param_count) tuples
        """
        info = []
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                shape_str = "×".join(str(dim) for dim in param.shape)
                param_count = param.numel()
                info.append((name, shape_str, param_count))
        return info
    

    def display_checkpoint_info(self, checkpoint_path: str) -> None:
        """Display formatted information about a checkpoint."""
        checkpoint = self.load_checkpoint(checkpoint_path)
        if not checkpoint:
            return
        
        # Create main tree
        tree = Tree(f"[bold cyan]Checkpoint: {Path(checkpoint_path).name}")
        
        # Basic information
        basic_info = tree.add("[bold green]Basic Information")
        if 'epochs' in checkpoint:
            basic_info.add(f"Epoch: [yellow]{checkpoint['epochs']}")
        if 'timestamp' in checkpoint:
            formatted_time = self._format_timestamp(checkpoint['timestamp'])
            basic_info.add(f"Saved at: [yellow]{formatted_time}")
        if 'best_val_loss' in checkpoint and checkpoint['best_val_loss'] is not None:
            basic_info.add(f"Validation Loss: [yellow]{checkpoint['best_val_loss']:.6f}")
        
        # Metrics
        if 'best_metrics' in checkpoint and checkpoint['best_metrics']:
            metrics_branch = tree.add("[bold green]Metrics")
            metrics_table = Table(show_header=True, header_style="bold magenta")
            metrics_table.add_column("Metric")
            metrics_table.add_column("Value")
            
            for metric, value in checkpoint['best_metrics'].items():
                if isinstance(value, (int, float)):
                    metrics_table.add_row(metric, f"{value:.6f}")
                else:
                    metrics_table.add_row(metric, str(value))
            metrics_branch.add(metrics_table)
        
        # Model parameters summary
        if 'model_state_dict' in checkpoint:
            model_branch = tree.add("[bold green]Model Parameters")
            param_info = self._get_state_dict_info(checkpoint['model_state_dict'])
            
            # Calculate total parameters
            total_params = sum(count for _, _, count in param_info)
            model_branch.add(f"Total parameters: [yellow]{total_params:,}")
            
            # Create parameter table
            param_table = Table(show_header=True, header_style="bold magenta")
            param_table.add_column("Layer")
            param_table.add_column("Shape")
            param_table.add_column("Parameters")
            
            # Add top 10 largest layers by parameter count
            sorted_params = sorted(param_info, key=lambda x: x[2], reverse=True)
            for name, shape, count in sorted_params[:10]:
                param_table.add_row(name, shape, f"{count:,}")

            if len(sorted_params) > 10:
                param_table.add_row("...", "...", "...")
                
            for name, shape, count in sorted_params[-6:]:
                param_table.add_row(name, shape, f"{count:,}")

            model_branch.add(param_table)
        
        # Optimizer information
        if 'optimizer_state_dict' in checkpoint:
            optimizer_branch = tree.add("[bold green]Optimizer")
            if 'param_groups' in checkpoint['optimizer_state_dict']:
                for i, group in enumerate(checkpoint['optimizer_state_dict']['param_groups']):
                    group_info = []
                    for k, v in group.items():
                        if k != 'params':
                            group_info.append(f"{k}: {v}")
                    optimizer_branch.add(f"Group {i}: {', '.join(group_info)}")
        
        # Scheduler information
        if 'scheduler_state_dict' in checkpoint:
            scheduler_branch = tree.add("[bold green]Scheduler")
            scheduler_state = checkpoint['scheduler_state_dict']
            
            # Display common scheduler parameters
            common_params = ['base_lrs', 'last_epoch']
            for param in common_params:
                if param in scheduler_state:
                    scheduler_branch.add(f"{param}: {scheduler_state[param]}")

        # Render the tree
        self.console.print()
        self.console.print(Panel(tree, title="Checkpoint Inspector", border_style="blue"))
        self.console.print()


def main():
    parser = argparse.ArgumentParser(description='Inspect PyTorch checkpoint files')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('--json', action='store_true', help='Export checkpoint info as JSON')
    args = parser.parse_args()
    
    console = Console()
    inspector = CheckpointInspector(console)
    
    if args.json:
        # JSON export mode
        checkpoint = inspector.load_checkpoint(args.checkpoint_path)
        if checkpoint:
            # Clean up non-serializable parts for JSON export
            clean_checkpoint = {}
            for key, value in checkpoint.items():
                if key in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']:
                    # Include summary instead of full state dict
                    if key == 'model_state_dict':
                        param_info = inspector._get_state_dict_info(value)
                        clean_checkpoint[f"{key}_summary"] = {
                            'total_params': sum(count for _, _, count in param_info),
                            'layers': len(param_info)
                        }
                else:
                    clean_checkpoint[key] = value
                    
            # Print JSON
            print(json.dumps(clean_checkpoint, indent=2, default=str))
    else:
        # Rich display mode
        inspector.display_checkpoint_info(args.checkpoint_path)


if __name__ == "__main__":
    main()

"""
python utils\checkpoint_analyzer.py "E:\Kai_2\CODE_Repository\gloria-Extended\output\checkpoint_best.pth"
╭────────────────────────────────────────────────────────── Checkpoint Inspector ───────────────────────────────────────────────────────────╮
│ Checkpoint: checkpoint_latest.pth                                                                                                         │
│ ├── Basic Information                                                                                                                     │
│ │   ├── Epoch: 17                                                                                                                         │
│ │   └── Saved at: 2025-04-08 13:36:59                                                                                                     │
│ ├── Metrics                                                                                                                               │
│ │   └── ┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓                                                                                                  │
│ │       ┃ Metric            ┃ Value    ┃                                                                                                  │
│ │       ┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩                                                                                                  │
│ │       │ val_loss          │ 1.057028 │                                                                                                  │
│ │       │ val_mean_auroc    │ 0.697875 │                                                                                                  │
│ │       │ val_mean_auprc    │ 0.504642 │                                                                                                  │
│ │       │ val_auroc_class_0 │ 0.653472 │                                                                                                  │
│ │       │ val_auprc_class_0 │ 0.551278 │                                                                                                  │
│ │       │ val_auroc_class_1 │ 0.633545 │                                                                                                  │
│ │       │ val_auprc_class_1 │ 0.513680 │                                                                                                  │
│ │       │ val_auroc_class_2 │ 0.643945 │                                                                                                  │
│ │       │ val_auprc_class_2 │ 0.299151 │                                                                                                  │
│ │       │ val_auroc_class_3 │ 0.792693 │                                                                                                  │
│ │       │ val_auprc_class_3 │ 0.488719 │                                                                                                  │
│ │       │ val_auroc_class_4 │ 0.765719 │                                                                                                  │
│ │       │ val_auprc_class_4 │ 0.670382 │                                                                                                  │
│ │       └───────────────────┴──────────┘                                                                                                  │
│ ├── Model Parameters                                                                                                                      │
│ │   ├── Total parameters: 23,571,450                                                                                                      │
│ │   └── ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓                                                         │
│ │       ┃ Layer                                    ┃ Shape         ┃ Parameters ┃                                                         │
│ │       ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩                                                         │
│ │       │ img_encoder.layer4.0.conv2.weight        │ 512×512×3×3   │ 2,359,296  │                                                         │
│ │       │ img_encoder.layer4.1.conv2.weight        │ 512×512×3×3   │ 2,359,296  │                                                         │
│ │       │ img_encoder.layer4.2.conv2.weight        │ 512×512×3×3   │ 2,359,296  │                                                         │
│ │       │ img_encoder.layer4.0.downsample.0.weight │ 2048×1024×1×1 │ 2,097,152  │                                                         │
│ │       │ img_encoder.layer4.0.conv3.weight        │ 2048×512×1×1  │ 1,048,576  │                                                         │
│ │       │ img_encoder.layer4.1.conv1.weight        │ 512×2048×1×1  │ 1,048,576  │                                                         │
│ │       │ img_encoder.layer4.1.conv3.weight        │ 2048×512×1×1  │ 1,048,576  │                                                         │
│ │       │ img_encoder.layer4.2.conv1.weight        │ 512×2048×1×1  │ 1,048,576  │                                                         │
│ │       │ img_encoder.layer4.2.conv3.weight        │ 2048×512×1×1  │ 1,048,576  │                                                         │
│ │       │ img_encoder.layer3.0.conv2.weight        │ 256×256×3×3   │ 589,824    │                                                         │
│ │       │ ...                                      │ ...           │ ...        │                                                         │
│ │       └──────────────────────────────────────────┴───────────────┴────────────┘                                                         │
│ ├── Optimizer                                                                                                                             │
│ │   └── Group 0: lr: 0.0001, betas: (0.5, 0.999), eps: 1e-08, weight_decay: 1e-06, amsgrad: False, maximize: False, foreach: None,        │
│ │       capturable: False, differentiable: False, fused: None                                                                             │
│ └── Scheduler                                                                                                                             │
│     └── last_epoch: 6                                                                                                                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
"""