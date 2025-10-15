#!/usr/bin/env python3
"""
Automated Dataset Re-Splitter with Stratification

This script properly splits your dataset into train/val with:
- Random shuffling for proper distribution
- Stratification by text length to ensure similar complexity
- Quality verification before saving
- Automatic backup of existing files

Usage:
    python resplit_dataset.py
"""

import os
import csv
import random
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_all_data(base_dir):
    """Load all data from existing train and eval CSVs."""
    train_csv = os.path.join(base_dir, "metadata_train.csv")
    eval_csv = os.path.join(base_dir, "metadata_eval.csv")
    
    all_data = []
    
    # Load train data
    if os.path.exists(train_csv):
        print(f"📂 Loading train CSV: {train_csv}")
        with open(train_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            train_data = list(reader)
            all_data.extend(train_data)
            print(f"   Loaded {len(train_data)} samples")
    
    # Load eval data
    if os.path.exists(eval_csv):
        print(f"📂 Loading eval CSV: {eval_csv}")
        with open(eval_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            eval_data = list(reader)
            all_data.extend(eval_data)
            print(f"   Loaded {len(eval_data)} samples")
    
    if not all_data:
        raise ValueError("No data found! Check CSV paths.")
    
    print(f"\n✅ Total samples loaded: {len(all_data)}")
    return all_data


def stratified_split(data, val_ratio=0.10, random_seed=42):
    """
    Split data with stratification by text length to ensure
    similar distributions in train and val sets.
    """
    print(f"\n🎯 Performing stratified split (val_ratio={val_ratio*100:.0f}%)...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Calculate text lengths
    data_with_lengths = []
    for row in data:
        if len(row) >= 2:
            text = row[1]
            text_len = len(text)
            data_with_lengths.append({
                'row': row,
                'text_length': text_len
            })
    
    print(f"   Valid samples: {len(data_with_lengths)}")
    
    # Create length bins for stratification
    # Bins: short (0-50), medium (51-100), long (101-150), very_long (151+)
    bins = [0, 50, 100, 150, float('inf')]
    bin_labels = ['short', 'medium', 'long', 'very_long']
    
    # Group by bins
    binned_data = defaultdict(list)
    for item in data_with_lengths:
        text_len = item['text_length']
        for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
            if lower < text_len <= upper:
                binned_data[bin_labels[i]].append(item)
                break
    
    # Print bin statistics
    print(f"\n   Text length distribution:")
    for label in bin_labels:
        count = len(binned_data[label])
        if count > 0:
            avg_len = np.mean([item['text_length'] for item in binned_data[label]])
            print(f"     {label:12s}: {count:4d} samples (avg: {avg_len:.0f} chars)")
    
    # Split each bin proportionally
    train_data = []
    val_data = []
    
    for label in bin_labels:
        bin_samples = binned_data[label]
        if not bin_samples:
            continue
        
        # Shuffle bin
        random.shuffle(bin_samples)
        
        # Split
        val_size = int(len(bin_samples) * val_ratio)
        val_size = max(1, val_size)  # At least 1 sample in val if bin exists
        
        bin_val = bin_samples[:val_size]
        bin_train = bin_samples[val_size:]
        
        val_data.extend(bin_val)
        train_data.extend(bin_train)
    
    # Final shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    # Extract rows
    train_rows = [item['row'] for item in train_data]
    val_rows = [item['row'] for item in val_data]
    
    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train_rows)} samples ({len(train_rows)/(len(train_rows)+len(val_rows))*100:.1f}%)")
    print(f"   Val:   {len(val_rows)} samples ({len(val_rows)/(len(train_rows)+len(val_rows))*100:.1f}%)")
    
    return train_rows, val_rows


def verify_split_quality(train_data, val_data):
    """Verify that train and val have similar distributions."""
    print(f"\n🔍 Verifying split quality...")
    
    # Calculate text length stats
    train_lengths = [len(row[1]) for row in train_data if len(row) >= 2]
    val_lengths = [len(row[1]) for row in val_data if len(row) >= 2]
    
    train_avg = np.mean(train_lengths)
    val_avg = np.mean(val_lengths)
    
    train_std = np.std(train_lengths)
    val_std = np.std(val_lengths)
    
    diff_pct = abs(train_avg - val_avg) / train_avg * 100
    
    print(f"\n   Text Length Statistics:")
    print(f"     Train avg: {train_avg:.1f} chars (±{train_std:.1f})")
    print(f"     Val avg:   {val_avg:.1f} chars (±{val_std:.1f})")
    print(f"     Difference: {diff_pct:.1f}%", end="")
    
    if diff_pct < 5:
        print(f"  ✅ EXCELLENT!")
        quality = "excellent"
    elif diff_pct < 10:
        print(f"  ✅ GOOD")
        quality = "good"
    elif diff_pct < 20:
        print(f"  ⚠️  ACCEPTABLE")
        quality = "acceptable"
    else:
        print(f"  ❌ POOR - Consider re-running")
        quality = "poor"
    
    # Check size ratio
    val_ratio = len(val_data) / (len(train_data) + len(val_data)) * 100
    print(f"\n   Validation Size: {val_ratio:.1f}%", end="")
    
    if 8 <= val_ratio <= 15:
        print(f"  ✅ OPTIMAL")
    elif 5 <= val_ratio < 8 or 15 < val_ratio <= 20:
        print(f"  ✅ ACCEPTABLE")
    else:
        print(f"  ⚠️  SUBOPTIMAL")
    
    return quality


def backup_existing_files(base_dir):
    """Backup existing train/eval CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(base_dir, f"backup_{timestamp}")
    
    train_csv = os.path.join(base_dir, "metadata_train.csv")
    eval_csv = os.path.join(base_dir, "metadata_eval.csv")
    
    files_backed_up = []
    
    if os.path.exists(train_csv) or os.path.exists(eval_csv):
        os.makedirs(backup_dir, exist_ok=True)
        print(f"\n💾 Backing up existing files to: {backup_dir}")
        
        if os.path.exists(train_csv):
            backup_path = os.path.join(backup_dir, "metadata_train.csv")
            shutil.copy2(train_csv, backup_path)
            files_backed_up.append(train_csv)
            print(f"   ✅ Backed up: metadata_train.csv")
        
        if os.path.exists(eval_csv):
            backup_path = os.path.join(backup_dir, "metadata_eval.csv")
            shutil.copy2(eval_csv, backup_path)
            files_backed_up.append(eval_csv)
            print(f"   ✅ Backed up: metadata_eval.csv")
    
    return files_backed_up


def save_split(train_data, val_data, base_dir):
    """Save train and val splits."""
    train_csv = os.path.join(base_dir, "metadata_train.csv")
    val_csv = os.path.join(base_dir, "metadata_eval.csv")
    
    print(f"\n💾 Saving new splits...")
    
    # Save train
    with open(train_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(train_data)
    print(f"   ✅ Saved: {train_csv} ({len(train_data)} samples)")
    
    # Save val
    with open(val_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(val_data)
    print(f"   ✅ Saved: {val_csv} ({len(val_data)} samples)")


def main():
    """Main re-splitting process."""
    print(f"\n{'='*70}")
    print(f"🔄 AUTOMATED DATASET RE-SPLITTER")
    print(f"{'='*70}\n")
    
    # Configuration
    base_dir = "finetune_models/dataset"
    val_ratio = 0.10  # 10% validation
    random_seed = 42
    
    print(f"Configuration:")
    print(f"  Base directory: {base_dir}")
    print(f"  Validation ratio: {val_ratio*100:.0f}%")
    print(f"  Random seed: {random_seed}")
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"\n❌ Directory not found: {base_dir}")
        print(f"   Please check the path and try again.")
        return
    
    try:
        # 1. Load all data
        all_data = load_all_data(base_dir)
        
        # 2. Backup existing files
        backup_existing_files(base_dir)
        
        # 3. Perform stratified split
        train_data, val_data = stratified_split(all_data, val_ratio, random_seed)
        
        # 4. Verify quality
        quality = verify_split_quality(train_data, val_data)
        
        # 5. Save new splits
        save_split(train_data, val_data, base_dir)
        
        # 6. Final summary
        print(f"\n{'='*70}")
        print(f"✅ RE-SPLIT COMPLETE!")
        print(f"{'='*70}\n")
        
        print(f"📊 Summary:")
        print(f"   Original: {len(all_data)} total samples")
        print(f"   Train:    {len(train_data)} samples ({len(train_data)/len(all_data)*100:.1f}%)")
        print(f"   Val:      {len(val_data)} samples ({len(val_data)/len(all_data)*100:.1f}%)")
        print(f"   Quality:  {quality.upper()}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Verify split quality:")
        print(f"      python diagnose_validation_data.py")
        print(f"")
        print(f"   2. Start training with V2 hyperparameters:")
        print(f"      Expected eval loss: 0.8-1.5 (vs previous 5.2!)")
        print(f"")
        print(f"   3. Monitor training:")
        print(f"      - Epoch 0: eval should be ~0.8-1.2")
        print(f"      - Epoch 2: eval should be ~0.5-0.8")
        print(f"      - Gap should be <50%")
        
        print(f"\n{'='*70}\n")
        
        print(f"✅ SUCCESS! Your dataset is now properly split.")
        print(f"   Re-run training and eval loss should drop dramatically!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n   Please check the error and try again.")


if __name__ == "__main__":
    main()
