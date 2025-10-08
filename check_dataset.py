#!/usr/bin/env python3
"""
Dataset Validation Script for XTTS Training
Diagnoses common issues that cause infinite recursion in dataset loading
"""

import os
import sys
import csv
from pathlib import Path
import torchaudio

def check_dataset(csv_path, max_audio_length=11):
    """
    Validates dataset CSV and audio files
    
    Args:
        csv_path: Path to train.csv or eval.csv
        max_audio_length: Maximum audio length in seconds (default: 11)
    """
    print(f"\n{'='*60}")
    print(f"Checking dataset: {csv_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV file not found: {csv_path}")
        return False
    
    csv_dir = Path(csv_path).parent
    issues = []
    valid_samples = 0
    total_samples = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        
        for idx, row in enumerate(reader):
            total_samples += 1
            audio_file = row.get('audio_file', '')
            text = row.get('text', '')
            speaker_name = row.get('speaker_name', '')
            
            # Check if audio file path is absolute or relative
            if not os.path.isabs(audio_file):
                audio_path = csv_dir / audio_file
            else:
                audio_path = Path(audio_file)
            
            # Issue 1: Missing audio file
            if not audio_path.exists():
                issues.append(f"Row {idx}: Audio file missing: {audio_path}")
                continue
            
            # Issue 2: Empty text
            if not text or len(text.strip()) < 2:
                issues.append(f"Row {idx}: Text too short or empty: '{text}'")
                continue
            
            # Issue 3: Check audio duration
            try:
                waveform, sample_rate = torchaudio.load(str(audio_path))
                duration = waveform.shape[1] / sample_rate
                
                if duration < 1.0:
                    issues.append(f"Row {idx}: Audio too short ({duration:.2f}s): {audio_path}")
                    continue
                
                if duration > max_audio_length:
                    issues.append(f"Row {idx}: Audio too long ({duration:.2f}s > {max_audio_length}s): {audio_path}")
                    continue
                
                valid_samples += 1
                
            except Exception as e:
                issues.append(f"Row {idx}: Cannot load audio {audio_path}: {e}")
                continue
    
    # Print results
    print(f"📊 Results:")
    print(f"  Total samples: {total_samples}")
    print(f"  ✅ Valid samples: {valid_samples}")
    print(f"  ❌ Invalid samples: {total_samples - valid_samples}")
    print()
    
    if issues:
        print(f"⚠️  Found {len(issues)} issues:\n")
        for issue in issues[:20]:  # Show first 20 issues
            print(f"  • {issue}")
        
        if len(issues) > 20:
            print(f"\n  ... and {len(issues) - 20} more issues")
        print()
    
    if valid_samples == 0:
        print("❌ CRITICAL: No valid samples found! Training cannot proceed.")
        print("\nPossible fixes:")
        print("  1. Check that audio files exist in the correct location")
        print("  2. Verify CSV file paths are correct (relative to CSV location)")
        print("  3. Ensure audio files are not corrupted")
        print("  4. Check audio durations (must be 1-11 seconds by default)")
        return False
    elif valid_samples < total_samples:
        print(f"⚠️  WARNING: Only {valid_samples}/{total_samples} samples are valid")
        print("Training will proceed but may fail if too few valid samples exist.")
        return True
    else:
        print("✅ All samples are valid!")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 check_dataset.py <path_to_train.csv> [max_audio_length]")
        print("\nExample:")
        print("  python3 check_dataset.py finetune_models/dataset/metadata_train.csv 11")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    max_length = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    
    success = check_dataset(csv_path, max_length)
    sys.exit(0 if success else 1)
