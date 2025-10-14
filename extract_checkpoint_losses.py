#!/usr/bin/env python3
"""
Extract Checkpoint Evaluation Losses from Training Log

This script parses trainer_0_log.txt to extract evaluation losses
for each checkpoint. Useful for manual inspection or debugging.

Usage:
    python extract_checkpoint_losses.py
    python extract_checkpoint_losses.py path/to/trainer_0_log.txt
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_training_log(log_path: str) -> Dict[int, float]:
    """
    Parse training log to extract evaluation losses per epoch.
    
    Args:
        log_path: Path to trainer_0_log.txt
    
    Returns:
        Dictionary mapping epoch -> eval_loss
    """
    eval_losses = {}
    
    print(f"📖 Reading log file: {log_path}")
    print("")
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print(f"✅ Found {len(lines)} lines in log file")
        print("")
        
        # Pattern 1: " > EPOCH: X --> EVALUATION AVERAGE LOSS: Y.YYY"
        pattern1 = re.compile(r'>\s*EPOCH:\s*(\d+)\s*-->\s*EVALUATION\s+AVERAGE\s+LOSS:\s*([\d.]+)')
        
        # Pattern 2: "Epoch X ended with avg_loss: Y.YYY"
        pattern2 = re.compile(r'Epoch\s+(\d+)\s+.*?avg_loss:\s*([\d.]+)')
        
        # Pattern 3: Extract from lines like: " > avg_loss: X.XXX (epoch Y)"
        current_epoch = None
        
        for i, line in enumerate(lines):
            # Try pattern 1
            match1 = pattern1.search(line)
            if match1:
                epoch = int(match1.group(1))
                loss = float(match1.group(2))
                eval_losses[epoch] = loss
                continue
            
            # Try pattern 2
            match2 = pattern2.search(line)
            if match2:
                epoch = int(match2.group(1))
                loss = float(match2.group(2))
                eval_losses[epoch] = loss
                continue
            
            # Track current epoch
            epoch_match = re.search(r'EPOCH:\s*(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Look for "EVALUATION" section with avg_loss
            if current_epoch is not None and "EVALUATION" in line:
                # Look in next few lines for avg_loss
                for j in range(i, min(i + 10, len(lines))):
                    loss_match = re.search(r'avg_loss:\s*([\d.]+)', lines[j])
                    if loss_match:
                        loss = float(loss_match.group(1))
                        if current_epoch not in eval_losses:  # Don't overwrite
                            eval_losses[current_epoch] = loss
                        break
        
        return eval_losses
    
    except Exception as e:
        print(f"❌ Error parsing log: {e}")
        return {}


def find_training_log() -> str:
    """Find the most recent trainer_0_log.txt in the finetune_models directory."""
    
    base_dir = Path("finetune_models") / "run" / "training"
    
    if not base_dir.exists():
        print(f"❌ Training directory not found: {base_dir}")
        return None
    
    # Find all trainer_0_log.txt files
    log_files = list(base_dir.glob("*/trainer_0_log.txt"))
    
    if not log_files:
        print(f"❌ No training logs found in {base_dir}")
        return None
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    latest_log = log_files[0]
    print(f"📁 Found latest training log:")
    print(f"   {latest_log}")
    print(f"   Modified: {latest_log.stat().st_mtime}")
    print("")
    
    return str(latest_log)


def format_results(eval_losses: Dict[int, float]) -> str:
    """Format evaluation losses for display."""
    
    if not eval_losses:
        return "❌ No evaluation losses found in log file."
    
    lines = []
    lines.append("=" * 70)
    lines.append("📊 EVALUATION LOSSES PER EPOCH")
    lines.append("=" * 70)
    lines.append("")
    
    # Sort by epoch
    epochs = sorted(eval_losses.keys())
    
    lines.append(f"{'Epoch':<8} {'Eval Loss':<12} {'Change':<12} {'Status'}")
    lines.append("-" * 70)
    
    prev_loss = None
    for epoch in epochs:
        loss = eval_losses[epoch]
        
        # Calculate change
        if prev_loss is not None:
            change = loss - prev_loss
            change_str = f"{change:+.4f}"
            
            if change > 0:
                status = "⚠️ INCREASE"
            elif change < 0:
                status = "✅ DECREASE"
            else:
                status = "→ SAME"
        else:
            change_str = "baseline"
            status = "🔵 START"
        
        lines.append(f"{epoch:<8} {loss:<12.4f} {change_str:<12} {status}")
        prev_loss = loss
    
    lines.append("")
    lines.append("=" * 70)
    
    # Summary statistics
    first_loss = eval_losses[epochs[0]]
    last_loss = eval_losses[epochs[-1]]
    min_loss = min(eval_losses.values())
    max_loss = max(eval_losses.values())
    
    best_epoch = [e for e, l in eval_losses.items() if l == min_loss][0]
    worst_epoch = [e for e, l in eval_losses.items() if l == max_loss][0]
    
    lines.append("📈 SUMMARY STATISTICS")
    lines.append("=" * 70)
    lines.append(f"First Loss (Epoch {epochs[0]}):        {first_loss:.4f}")
    lines.append(f"Last Loss (Epoch {epochs[-1]}):        {last_loss:.4f}")
    lines.append(f"Min Loss (Epoch {best_epoch}):         {min_loss:.4f} ← BEST")
    lines.append(f"Max Loss (Epoch {worst_epoch}):        {max_loss:.4f} ← WORST")
    lines.append("")
    
    # Calculate overall trend
    change = last_loss - first_loss
    change_pct = (change / first_loss) * 100
    
    lines.append(f"Overall Change:                {change:+.4f} ({change_pct:+.1f}%)")
    lines.append("")
    
    if change_pct > 50:
        lines.append("⚠️  SEVERE OVERFITTING DETECTED!")
        lines.append(f"   Eval loss increased by {change_pct:.1f}%")
        lines.append(f"   Recommended: Use checkpoint from epoch {best_epoch}")
    elif change_pct > 20:
        lines.append("⚠️  OVERFITTING DETECTED!")
        lines.append(f"   Eval loss increased by {change_pct:.1f}%")
        lines.append(f"   Recommended: Use checkpoint from epoch {best_epoch}")
    elif change_pct < -20:
        lines.append("✅ EXCELLENT TRAINING!")
        lines.append(f"   Eval loss decreased by {abs(change_pct):.1f}%")
        lines.append(f"   Model improved significantly")
    else:
        lines.append("✅ STABLE TRAINING")
        lines.append(f"   Eval loss change: {change_pct:+.1f}%")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    
    print("=" * 70)
    print("🔍 CHECKPOINT EVALUATION LOSS EXTRACTOR")
    print("=" * 70)
    print("")
    
    # Determine log file path
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
        print(f"📁 Using provided log file: {log_path}")
        print("")
    else:
        log_path = find_training_log()
        if not log_path:
            print("❌ No training log found.")
            print("")
            print("Usage:")
            print("  python extract_checkpoint_losses.py [path/to/trainer_0_log.txt]")
            return
    
    # Parse the log
    eval_losses = parse_training_log(log_path)
    
    # Format and display results
    print("")
    print(format_results(eval_losses))
    print("")
    
    # Save to file
    output_path = "checkpoint_eval_losses.txt"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(format_results(eval_losses))
        print(f"💾 Results saved to: {output_path}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    print("")
    print("✅ Extraction complete!")
    print("")
    print("💡 TIP: Use the Checkpoint Selection feature in the WebUI (Inference tab)")
    print("   to automatically select the best checkpoint based on these losses.")
    print("")


if __name__ == "__main__":
    main()
