"""
Checkpoint Management Utility for XTTS Fine-tuning

This module provides functionality to:
- List all available checkpoints from training runs
- Extract metadata (epoch, step, eval loss) from checkpoints
- Select and copy checkpoints to the ready folder
- Recommend best checkpoints based on evaluation metrics
"""

import os
import re
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class CheckpointInfo:
    """Container for checkpoint metadata"""
    
    def __init__(self, path: str, step: int, epoch: Optional[int] = None, 
                 eval_loss: Optional[float] = None, is_best: bool = False):
        self.path = path
        self.step = step
        self.epoch = epoch
        self.eval_loss = eval_loss
        self.is_best = is_best
        self.name = os.path.basename(path)
        self.size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        self.modified_time = datetime.fromtimestamp(os.path.getmtime(path)) if os.path.exists(path) else None
    
    def __repr__(self):
        return f"CheckpointInfo(step={self.step}, epoch={self.epoch}, eval_loss={self.eval_loss}, is_best={self.is_best})"
    
    def display_name(self) -> str:
        """Generate a user-friendly display name"""
        parts = []
        
        if self.is_best:
            parts.append("🏆 BEST")
        
        if self.epoch is not None:
            parts.append(f"Epoch {self.epoch}")
        
        parts.append(f"Step {self.step}")
        
        if self.eval_loss is not None:
            parts.append(f"(Loss: {self.eval_loss:.4f})")
        
        if self.size_mb > 0:
            parts.append(f"[{self.size_mb:.1f} MB]")
        
        return " | ".join(parts)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "path": self.path,
            "name": self.name,
            "step": self.step,
            "epoch": self.epoch,
            "eval_loss": self.eval_loss,
            "is_best": self.is_best,
            "size_mb": round(self.size_mb, 2),
            "modified_time": self.modified_time.isoformat() if self.modified_time else None,
            "display_name": self.display_name()
        }


def find_training_runs(output_path: str) -> List[Path]:
    """
    Find all training run directories in the output path.
    
    Args:
        output_path: Base output directory (e.g., finetune_models)
    
    Returns:
        List of training run directories
    """
    training_base = Path(output_path) / "run" / "training"
    
    if not training_base.exists():
        return []
    
    # Find directories that contain checkpoints
    run_dirs = []
    for item in training_base.iterdir():
        if item.is_dir():
            # Check if this directory contains checkpoint files
            checkpoints = list(item.glob("checkpoint_*.pth")) + list(item.glob("best_model*.pth"))
            if checkpoints:
                run_dirs.append(item)
    
    # Sort by modification time (newest first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return run_dirs


def extract_step_from_checkpoint_name(checkpoint_name: str) -> Optional[int]:
    """
    Extract step number from checkpoint filename.
    
    Examples:
        checkpoint_1000.pth -> 1000
        best_model_569.pth -> 569
        model_5000.pth -> 5000
    """
    # Try patterns: checkpoint_STEP.pth, best_model_STEP.pth, model_STEP.pth
    patterns = [
        r'checkpoint_(\d+)\.pth',
        r'best_model_(\d+)\.pth',
        r'model_(\d+)\.pth',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, checkpoint_name)
        if match:
            return int(match.group(1))
    
    return None


def parse_training_log_for_eval_losses(log_path: str) -> Dict[int, float]:
    """
    Parse training log to extract evaluation losses per step.
    
    Args:
        log_path: Path to trainer_0_log.txt
    
    Returns:
        Dictionary mapping step -> eval_loss
    """
    eval_losses = {}
    
    if not os.path.exists(log_path):
        return eval_losses
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Look for evaluation average loss lines
                # Example: " > EVALUATION AVERAGE LOSS: 3.415"
                if "EVALUATION AVERAGE LOSS" in line or "avg_loss" in line:
                    # Try to extract loss value
                    match = re.search(r'(?:AVERAGE LOSS:|avg_loss:?)\s*([\d.]+)', line)
                    if match:
                        loss = float(match.group(1))
                        
                        # Try to find corresponding step in nearby lines
                        # This is approximate - we'll associate with the closest step
                        # Look back in recent lines for step info
                        # For now, we'll just store it and match later
                        # (This is a simplified approach)
                        pass
        
        # Alternative: parse from the structured eval logs if available
        # Example: extract from lines like "STEP: 569 | EVAL | loss: 3.415"
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            current_step = None
            for line in f:
                # Extract step number
                step_match = re.search(r'STEP:\s*(\d+)', line)
                if step_match:
                    current_step = int(step_match.group(1))
                
                # Extract eval loss if this is an eval line
                if current_step and ("EVAL" in line or "Evaluation" in line):
                    loss_match = re.search(r'(?:loss:|avg_loss:)\s*([\d.]+)', line)
                    if loss_match:
                        eval_losses[current_step] = float(loss_match.group(1))
    
    except Exception as e:
        print(f"Warning: Could not parse training log: {e}")
    
    return eval_losses


def estimate_epoch_from_step(step: int, steps_per_epoch: int = 569) -> int:
    """
    Estimate epoch number from step count.
    
    Args:
        step: Current step number
        steps_per_epoch: Number of steps per epoch (default based on your logs)
    
    Returns:
        Estimated epoch number (0-indexed)
    """
    return step // steps_per_epoch


def list_checkpoints_from_run(run_dir: Path, log_path: Optional[Path] = None) -> List[CheckpointInfo]:
    """
    List all checkpoints from a training run with metadata.
    
    Args:
        run_dir: Training run directory
        log_path: Optional path to training log for eval loss extraction
    
    Returns:
        List of CheckpointInfo objects
    """
    checkpoints = []
    
    # Find all checkpoint files
    checkpoint_files = list(run_dir.glob("checkpoint_*.pth")) + list(run_dir.glob("best_model*.pth"))
    
    # Parse eval losses from log if available
    eval_losses = {}
    if log_path and log_path.exists():
        eval_losses = parse_training_log_for_eval_losses(str(log_path))
    
    # Determine steps per epoch from the first checkpoint spacing
    # (Assuming checkpoints are saved at regular intervals)
    steps_per_epoch = 569  # Default from your logs
    
    for ckpt_file in checkpoint_files:
        step = extract_step_from_checkpoint_name(ckpt_file.name)
        
        if step is None:
            continue
        
        # Determine if this is the "best" model
        is_best = "best_model" in ckpt_file.name
        
        # Estimate epoch
        epoch = estimate_epoch_from_step(step, steps_per_epoch)
        
        # Get eval loss if available
        eval_loss = eval_losses.get(step)
        
        checkpoint_info = CheckpointInfo(
            path=str(ckpt_file),
            step=step,
            epoch=epoch,
            eval_loss=eval_loss,
            is_best=is_best
        )
        
        checkpoints.append(checkpoint_info)
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x.step)
    
    return checkpoints


def get_latest_training_run_checkpoints(output_path: str) -> Tuple[Optional[Path], List[CheckpointInfo]]:
    """
    Get checkpoints from the most recent training run.
    
    Args:
        output_path: Base output directory
    
    Returns:
        Tuple of (run_directory, list_of_checkpoints)
    """
    runs = find_training_runs(output_path)
    
    if not runs:
        return None, []
    
    latest_run = runs[0]  # Already sorted by modification time
    
    # Look for training log in the run directory
    log_path = latest_run / "trainer_0_log.txt"
    
    checkpoints = list_checkpoints_from_run(latest_run, log_path if log_path.exists() else None)
    
    return latest_run, checkpoints


def recommend_best_checkpoint(checkpoints: List[CheckpointInfo]) -> Optional[CheckpointInfo]:
    """
    Recommend the best checkpoint based on evaluation loss.
    
    Strategy:
    1. If eval losses are available, choose the checkpoint with lowest eval loss
    2. Otherwise, prefer early checkpoints (to avoid overfitting)
    3. If a "best_model" exists, consider it
    
    Args:
        checkpoints: List of available checkpoints
    
    Returns:
        Recommended checkpoint or None
    """
    if not checkpoints:
        return None
    
    # Filter checkpoints with eval loss data
    checkpoints_with_loss = [c for c in checkpoints if c.eval_loss is not None]
    
    if checkpoints_with_loss:
        # Choose checkpoint with lowest eval loss
        best = min(checkpoints_with_loss, key=lambda x: x.eval_loss)
        return best
    
    # Fallback: prefer early checkpoints (epochs 0-5) to avoid overfitting
    early_checkpoints = [c for c in checkpoints if c.epoch is not None and c.epoch <= 5]
    
    if early_checkpoints:
        # Among early checkpoints, prefer the one marked as "best" or the latest
        best_in_early = next((c for c in early_checkpoints if c.is_best), None)
        if best_in_early:
            return best_in_early
        return early_checkpoints[-1]  # Latest early checkpoint
    
    # Final fallback: return the "best_model" if it exists
    best_model = next((c for c in checkpoints if c.is_best), None)
    if best_model:
        return best_model
    
    # Last resort: return first checkpoint (earliest)
    return checkpoints[0] if checkpoints else None


def copy_checkpoint_to_ready(checkpoint_path: str, output_path: str, 
                             as_name: str = "model.pth") -> Tuple[bool, str]:
    """
    Copy a selected checkpoint to the ready folder.
    
    Args:
        checkpoint_path: Source checkpoint path
        output_path: Base output directory
        as_name: Target filename in ready folder
    
    Returns:
        Tuple of (success, message)
    """
    try:
        import shutil
        
        ready_dir = Path(output_path) / "ready"
        ready_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = ready_dir / as_name
        
        # Backup existing model if it exists
        if target_path.exists():
            backup_name = f"{target_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            backup_path = ready_dir / backup_name
            shutil.copy(target_path, backup_path)
            print(f" > Backed up existing model to: {backup_name}")
        
        # Copy checkpoint
        shutil.copy(checkpoint_path, target_path)
        
        return True, f"✅ Checkpoint copied to ready/{as_name}\n(Backup created if previous model existed)"
    
    except Exception as e:
        return False, f"❌ Error copying checkpoint: {str(e)}"


def format_checkpoint_list_for_display(checkpoints: List[CheckpointInfo], 
                                       recommended: Optional[CheckpointInfo] = None) -> str:
    """
    Format checkpoint list for display in Gradio UI.
    
    Args:
        checkpoints: List of checkpoints
        recommended: Recommended checkpoint to highlight
    
    Returns:
        Formatted string for display
    """
    if not checkpoints:
        return "No checkpoints found in the latest training run.\n\nPlease train a model first."
    
    lines = []
    lines.append("📦 **Available Checkpoints from Latest Training Run**")
    lines.append("=" * 70)
    lines.append("")
    
    if recommended:
        lines.append(f"💡 **RECOMMENDED**: {recommended.display_name()}")
        lines.append(f"   Reason: {'Lowest eval loss' if recommended.eval_loss else 'Early checkpoint (less overfitting)'}")
        lines.append("")
    
    lines.append("**All Checkpoints:**")
    lines.append("")
    
    for i, ckpt in enumerate(checkpoints, 1):
        prefix = "  ➜" if ckpt == recommended else "   "
        lines.append(f"{prefix} {i}. {ckpt.display_name()}")
        if ckpt.modified_time:
            lines.append(f"      Saved: {ckpt.modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("ℹ️  **Note**: Select a checkpoint from the dropdown and click 'Use Selected Checkpoint'")
    
    return "\n".join(lines)


def get_checkpoint_dropdown_choices(checkpoints: List[CheckpointInfo]) -> List[Tuple[str, str]]:
    """
    Generate choices for Gradio Dropdown.
    
    Args:
        checkpoints: List of available checkpoints
    
    Returns:
        List of (display_name, checkpoint_path) tuples
    """
    if not checkpoints:
        return [("No checkpoints available", "")]
    
    choices = []
    for ckpt in checkpoints:
        display = ckpt.display_name()
        choices.append((display, ckpt.path))
    
    return choices


def analyze_checkpoints_for_overfitting(checkpoints: List[CheckpointInfo]) -> Dict:
    """
    Analyze checkpoints to detect overfitting patterns.
    
    Args:
        checkpoints: List of checkpoints with eval loss data
    
    Returns:
        Analysis report dictionary
    """
    analysis = {
        "overfitting_detected": False,
        "safe_checkpoint": None,
        "warning_message": "",
        "eval_loss_trend": []
    }
    
    # Filter checkpoints with eval loss
    checkpoints_with_loss = [c for c in checkpoints if c.eval_loss is not None]
    
    if len(checkpoints_with_loss) < 3:
        analysis["warning_message"] = "Not enough eval data to determine overfitting"
        return analysis
    
    # Sort by step
    checkpoints_with_loss.sort(key=lambda x: x.step)
    
    # Build trend
    for ckpt in checkpoints_with_loss:
        analysis["eval_loss_trend"].append({
            "step": ckpt.step,
            "epoch": ckpt.epoch,
            "eval_loss": ckpt.eval_loss
        })
    
    # Check if eval loss is increasing
    first_loss = checkpoints_with_loss[0].eval_loss
    last_loss = checkpoints_with_loss[-1].eval_loss
    
    if last_loss > first_loss * 1.5:  # 50% increase indicates overfitting
        analysis["overfitting_detected"] = True
        
        # Find the checkpoint with lowest eval loss
        best_ckpt = min(checkpoints_with_loss, key=lambda x: x.eval_loss)
        analysis["safe_checkpoint"] = best_ckpt
        
        analysis["warning_message"] = (
            f"⚠️  OVERFITTING DETECTED!\n\n"
            f"Eval loss increased from {first_loss:.4f} (epoch 0) to {last_loss:.4f} (final).\n"
            f"Recommended safe checkpoint: {best_ckpt.display_name()}\n\n"
            f"This checkpoint has the lowest evaluation loss."
        )
    else:
        analysis["warning_message"] = "✅ No significant overfitting detected"
    
    return analysis
