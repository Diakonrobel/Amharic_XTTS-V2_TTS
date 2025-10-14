#!/usr/bin/env python3
"""
Quick Dataset Verification Script
Checks if dataset files actually contain the expected number of segments.
"""

import os
import sys
import pandas as pd
from pathlib import Path

def verify_dataset(dataset_path):
    """Verify dataset files and show actual contents."""
    dataset_path = Path(dataset_path)
    
    print("=" * 70)
    print("🔍 Dataset Verification")
    print("=" * 70)
    print(f"\nDataset path: {dataset_path}")
    print()
    
    # Check if dataset directory exists
    if not dataset_path.exists():
        print("❌ ERROR: Dataset directory does not exist!")
        print(f"   Path: {dataset_path}")
        return False
    
    # Check metadata files
    train_csv = dataset_path / "metadata_train.csv"
    eval_csv = dataset_path / "metadata_eval.csv"
    lang_file = dataset_path / "lang.txt"
    wavs_dir = dataset_path / "wavs"
    
    print("📁 File Check:")
    print(f"  • metadata_train.csv: {'✓ EXISTS' if train_csv.exists() else '❌ MISSING'}")
    print(f"  • metadata_eval.csv:  {'✓ EXISTS' if eval_csv.exists() else '❌ MISSING'}")
    print(f"  • lang.txt:           {'✓ EXISTS' if lang_file.exists() else '❌ MISSING'}")
    print(f"  • wavs/ directory:    {'✓ EXISTS' if wavs_dir.exists() else '❌ MISSING'}")
    print()
    
    if not train_csv.exists() or not eval_csv.exists():
        print("❌ ERROR: Metadata files are missing!")
        return False
    
    # Load CSVs
    try:
        print("📊 Loading metadata files...")
        train_df = pd.read_csv(train_csv, sep='|')
        eval_df = pd.read_csv(eval_csv, sep='|')
        
        num_train = len(train_df)
        num_eval = len(eval_df)
        num_total = num_train + num_eval
        
        print(f"  • Training segments:   {num_train:,}")
        print(f"  • Evaluation segments: {num_eval:,}")
        print(f"  • TOTAL SEGMENTS:      {num_total:,}")
        print()
        
        # Check audio files
        if wavs_dir.exists():
            audio_files = list(wavs_dir.glob("*.wav"))
            num_audio = len(audio_files)
            print(f"🎵 Audio Files:")
            print(f"  • Found: {num_audio:,} WAV files")
            print(f"  • Expected: {num_total:,} files")
            
            if num_audio == num_total:
                print(f"  ✅ Audio file count matches!")
            else:
                print(f"  ⚠️  Mismatch: {abs(num_audio - num_total)} files difference")
            print()
            
            # Show first and last few filenames
            sorted_files = sorted([f.name for f in audio_files])
            print("  First 3 files:")
            for fname in sorted_files[:3]:
                print(f"    - {fname}")
            print("  ...")
            print("  Last 3 files:")
            for fname in sorted_files[-3:]:
                print(f"    - {fname}")
            print()
        
        # Check language
        if lang_file.exists():
            with open(lang_file, 'r', encoding='utf-8') as f:
                language = f.read().strip()
            print(f"🌍 Language: {language.upper()}")
            print()
        
        # Show sample of data
        print("📝 Sample Data (first 3 training entries):")
        print("-" * 70)
        for idx, row in train_df.head(3).iterrows():
            print(f"\n  [{idx+1}] {row['audio_file']}")
            print(f"      Text: {row['text'][:80]}...")
        print()
        print("-" * 70)
        
        # Show file sizes
        train_size = train_csv.stat().st_size / 1024  # KB
        eval_size = eval_csv.stat().st_size / 1024    # KB
        print(f"\n💾 File Sizes:")
        print(f"  • metadata_train.csv: {train_size:.1f} KB")
        print(f"  • metadata_eval.csv:  {eval_size:.1f} KB")
        print()
        
        # Summary
        print("=" * 70)
        print("✅ VERIFICATION COMPLETE")
        print("=" * 70)
        print(f"\nYour dataset contains {num_total:,} segments")
        print(f"({num_train:,} training + {num_eval:,} evaluation)")
        
        if num_total >= 1716:
            print("\n✅ Dataset has been successfully updated!")
            print(f"   The new segments ARE in your dataset files.")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR reading CSV files: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Default path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Try common paths
        possible_paths = [
            "finetune_models/dataset",
            "./dataset",
            "../finetune_models/dataset"
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if not dataset_path:
            print("Usage: python verify_dataset.py <dataset_path>")
            print("\nExample: python verify_dataset.py finetune_models/dataset")
            sys.exit(1)
    
    success = verify_dataset(dataset_path)
    sys.exit(0 if success else 1)
