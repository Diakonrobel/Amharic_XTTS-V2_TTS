"""
Clean Merged Dataset - Remove VAD Mismatch Entries
==================================================

This script cleans a merged dataset by removing segments with empty/invalid text
that resulted from VAD-SRT mismatch bug.

Perfect for datasets that combine multiple YouTube videos where you don't have
the original SRT files anymore.

Usage:
    python clean_merged_dataset.py --dataset_dir <path>
"""

import os
import argparse
import pandas as pd
import shutil
from pathlib import Path
from typing import Set


def clean_dataset(
    dataset_dir: str,
    min_text_length: int = 3,
    remove_duplicate_text: bool = True,
    dry_run: bool = False
):
    """
    Clean merged dataset by removing invalid entries.
    
    Args:
        dataset_dir: Path to dataset directory
        min_text_length: Minimum text length to keep (chars)
        remove_duplicate_text: Remove segments with identical text (likely VAD duplicates)
        dry_run: Don't actually modify files, just show what would be done
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    print("=" * 70)
    print("Clean Merged Dataset - Remove VAD Mismatch Entries")
    print("=" * 70)
    print(f"\nDataset: {dataset_dir}")
    print(f"Min text length: {min_text_length} chars")
    print(f"Remove duplicates: {remove_duplicate_text}")
    print(f"Dry run: {dry_run}")
    print()
    
    # Process both metadata files
    for csv_name in ["metadata_train.csv", "metadata_eval.csv"]:
        csv_path = dataset_path / csv_name
        
        if not csv_path.exists():
            print(f"⚠ Skipping {csv_name} (not found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {csv_name}")
        print(f"{'='*70}")
        
        # Read CSV
        try:
            df = pd.read_csv(csv_path, sep="|")
        except Exception as e:
            print(f"❌ Error reading {csv_name}: {e}")
            continue
        
        original_count = len(df)
        print(f"Original entries: {original_count}")
        
        # Backup if not dry run
        if not dry_run:
            backup_path = dataset_path / f"{csv_name}.backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            df.to_csv(backup_path, sep="|", index=False)
            print(f"✓ Backup created: {backup_path.name}")
        
        # Track what we're removing
        removed_empty = 0
        removed_short = 0
        removed_duplicates = 0
        audio_files_to_remove = set()
        
        # Filter 1: Remove empty text
        mask_empty = df['text'].isna() | (df['text'].str.strip() == "")
        removed_empty = mask_empty.sum()
        if removed_empty > 0:
            audio_files_to_remove.update(df[mask_empty]['audio_file'].tolist())
            df = df[~mask_empty]
            print(f"\n🗑 Removing {removed_empty} entries with empty text")
        
        # Filter 2: Remove very short text
        mask_short = df['text'].str.len() < min_text_length
        removed_short = mask_short.sum()
        if removed_short > 0:
            audio_files_to_remove.update(df[mask_short]['audio_file'].tolist())
            df = df[~mask_short]
            print(f"🗑 Removing {removed_short} entries with text < {min_text_length} chars")
        
        # Filter 3: Remove duplicate text (likely VAD bug artifacts)
        if remove_duplicate_text:
            # Find segments with same text appearing multiple times
            # Group by text and check if it appears > 3 times (suspicious)
            text_counts = df['text'].value_counts()
            suspicious_texts = text_counts[text_counts > 3].index.tolist()
            
            if suspicious_texts:
                print(f"\n🔍 Found {len(suspicious_texts)} texts appearing >3 times (likely VAD duplicates)")
                print(f"   Example duplicates:")
                for text in suspicious_texts[:3]:
                    print(f"   - \"{text[:50]}...\" appears {text_counts[text]} times")
                
                # For each suspicious text, keep only 1 occurrence, remove the rest
                for text in suspicious_texts:
                    duplicate_mask = df['text'] == text
                    duplicate_indices = df[duplicate_mask].index.tolist()
                    
                    # Keep the first occurrence, mark rest for removal
                    indices_to_remove = duplicate_indices[1:]  # Keep first, remove rest
                    
                    if indices_to_remove:
                        audio_files_to_remove.update(df.loc[indices_to_remove, 'audio_file'].tolist())
                        df = df.drop(indices_to_remove)
                        removed_duplicates += len(indices_to_remove)
                
                print(f"🗑 Removing {removed_duplicates} duplicate text entries")
        
        final_count = len(df)
        removed_total = original_count - final_count
        
        print(f"\n📊 Summary for {csv_name}:")
        print(f"   Original entries: {original_count}")
        print(f"   Removed (empty): {removed_empty}")
        print(f"   Removed (short): {removed_short}")
        print(f"   Removed (duplicates): {removed_duplicates}")
        print(f"   Total removed: {removed_total}")
        print(f"   Final entries: {final_count}")
        print(f"   Reduction: {removed_total/original_count*100:.1f}%")
        
        # Save cleaned CSV
        if not dry_run:
            df.to_csv(csv_path, sep="|", index=False)
            print(f"✓ Saved cleaned {csv_name}")
        
        # Remove audio files
        if not dry_run and audio_files_to_remove:
            print(f"\n🗑 Removing {len(audio_files_to_remove)} orphaned audio files...")
            removed_audio_count = 0
            
            for audio_file in audio_files_to_remove:
                audio_path = dataset_path / audio_file
                if audio_path.exists():
                    try:
                        audio_path.unlink()
                        removed_audio_count += 1
                    except Exception as e:
                        print(f"   ⚠ Could not remove {audio_file}: {e}")
            
            print(f"✓ Removed {removed_audio_count} audio files")
    
    print(f"\n{'='*70}")
    print("✅ Dataset cleaning complete!")
    print(f"{'='*70}")
    
    if dry_run:
        print("\n⚠ DRY RUN - No files were actually modified")
        print("   Remove --dry_run flag to apply changes")


def analyze_dataset(dataset_dir: str):
    """
    Analyze dataset and show statistics about potential issues.
    """
    dataset_path = Path(dataset_dir)
    
    print("=" * 70)
    print("Dataset Analysis")
    print("=" * 70)
    
    for csv_name in ["metadata_train.csv", "metadata_eval.csv"]:
        csv_path = dataset_path / csv_name
        
        if not csv_path.exists():
            continue
        
        print(f"\n{csv_name}:")
        df = pd.read_csv(csv_path, sep="|")
        
        # Basic stats
        print(f"  Total entries: {len(df)}")
        
        # Empty text
        empty = df['text'].isna() | (df['text'].str.strip() == "")
        print(f"  Empty text: {empty.sum()}")
        
        # Short text
        short = df['text'].str.len() < 3
        print(f"  Short text (<3 chars): {short.sum()}")
        
        # Text length stats
        text_lengths = df['text'].str.len()
        print(f"  Text length - min: {text_lengths.min()}, max: {text_lengths.max()}, avg: {text_lengths.mean():.1f}")
        
        # Duplicate text
        text_counts = df['text'].value_counts()
        duplicates = text_counts[text_counts > 1]
        print(f"  Duplicate texts: {len(duplicates)} texts appear multiple times")
        
        if len(duplicates) > 0:
            suspicious = text_counts[text_counts > 3]
            print(f"  Suspicious (>3 occurrences): {len(suspicious)} texts")
            if len(suspicious) > 0:
                print(f"  Top duplicates:")
                for text, count in suspicious.head(5).items():
                    print(f"    - \"{text[:40]}...\" appears {count} times")


def main():
    parser = argparse.ArgumentParser(
        description="Clean merged dataset by removing VAD-SRT mismatch entries"
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=3,
        help="Minimum text length to keep (default: 3)"
    )
    
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Don't remove duplicate text entries"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without modifying files"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze dataset, don't clean"
    )
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.dataset_dir)
    else:
        clean_dataset(
            dataset_dir=args.dataset_dir,
            min_text_length=args.min_text_length,
            remove_duplicate_text=not args.keep_duplicates,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()
