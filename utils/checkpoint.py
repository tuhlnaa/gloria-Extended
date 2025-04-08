import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class CheckpointHandler:
    def __init__(
            self, 
            save_dir: str, 
            filename_prefix: str = "checkpoint",
            max_save_num: int = 2,
            save_interval: int = 1  # Save checkpoints every N epochs
        ):
        """Initialize CheckpointHandler for managing model checkpoints.
        
        Args:
            save_dir: Directory to save checkpoints
            filename_prefix: Prefix for checkpoint filenames
            max_save_num: Maximum number of checkpoint files to keep
            save_interval: Save checkpoints every N epochs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.max_save_num = max_save_num
        self.save_interval = save_interval
        self.last_saved_epoch = 0


    def _should_save_checkpoint(self, epoch: int) -> bool:
        """Determine if we should save a checkpoint at this epoch."""
        return epoch % self.save_interval == 0


    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints while keeping the most recent ones.
        
        Maintains:
        - Latest checkpoint
        - Best checkpoint
        - Most recent max_save_num epoch checkpoints
        """
        # Don't delete latest or best checkpoints
        protected_files = {
            self.save_dir / f"{self.filename_prefix}_latest.pth",
            self.save_dir / f"{self.filename_prefix}_best.pth"
        }
        
        # Get all epoch checkpoints
        checkpoints = sorted(
            [f for f in self.save_dir.glob(f"{self.filename_prefix}_epoch_*.pth")],
            key=lambda x: int(x.stem.split('_')[-1])
        )

        if len(checkpoints) <= self.max_save_num:
            return

        # Keep the most recent max_save_num checkpoints
        keep_files = set(checkpoints[-self.max_save_num:])

        # Delete checkpoints that aren't protected or meant to be kept
        for checkpoint in checkpoints:
            if checkpoint not in keep_files and checkpoint.resolve() not in protected_files:
                try:
                    checkpoint.unlink()
                except Exception as e:
                    print(f"Error deleting checkpoint {checkpoint}: {e}")


    def save_checkpoint(
            self,
            epochs: int,
            model_state: Dict[str, Any],
            optimizer_state: Dict[str, Any],
            scheduler_state: Optional[Dict[str, Any]] = None,
            metrics: Optional[Dict[str, float]] = None,
            is_best: bool = False,
        ) -> None:
        """Save model checkpoint with metadata.
        
        Args:
            epochs: Current epoch number
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            scheduler_state: Optional scheduler state dictionary
            metrics: Optional dictionary of training metrics
            is_best: Whether this is the best model so far
        """
        # Check if we should save at this epoch
        if not self._should_save_checkpoint(epochs) and not is_best:
            return

        checkpoint = {
            'epochs': epochs,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'best_metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
            
        # Save latest checkpoint
        latest_path = self.save_dir / f"{self.filename_prefix}_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint if it meets our saving criteria
        if self._should_save_checkpoint(epochs):
            epoch_path = self.save_dir / f"{self.filename_prefix}_epoch_{epochs}.pth"
            torch.save(checkpoint, epoch_path)
            self.last_saved_epoch = epochs
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = self.save_dir / f"{self.filename_prefix}_best.pth"
            torch.save(checkpoint, best_path)
            
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()


    def load_checkpoint(
            self,
            path: str,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            strict: bool = True
        ) -> Dict[str, Any]:
        """Load checkpoint with error handling.
        
        Args:
            path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            strict: Whether to strictly enforce that the keys match
            
        Returns:
            Dict containing checkpoint data
            
        Raises:
            Exception: If checkpoint loading fails
        """
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(path)
            else:
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
                
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
            # Load scheduler state if provided
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            return checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise