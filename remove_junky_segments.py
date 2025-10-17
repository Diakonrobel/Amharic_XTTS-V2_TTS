"""
Remove Junky Audio Segments from Dataset
=========================================

Removes specific audio files and their metadata entries from the dataset.

Usage:
    python remove_junky_segments.py --dataset_dir <path> --start_index <num>
    
Example:
    python remove_junky_segments.py \
        --dataset_dir "/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models/dataset" \
        --start_index 9968
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import shutil


def remove_junky_segments(
    dataset_dir: str,
    start_index: int,
    pattern: str = "merged",
    dry_run: bool = False
):
    """
    Remove audio segments starting from a specific index.
    
    Args:
        dataset_dir: Path to dataset directory
        start_index: Starting index to remove (e.g., 9968 for merged_00009968.wav)
        pattern: Filename pattern (default: "merged")
        dry_run: Preview changes without actually modifying files
    """
    dataset_path = Path(dataset_dir)
    wavs_dir = dataset_path / "wavs"
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    if not wavs_dir.exists():
        print(f"❌ Wavs directory not found: {wavs_dir}")
        return
    
    print("=" * 70)
    print("Remove Junky Audio Segments")
    print("=" * 70)
    print(f"\nDataset: {dataset_dir}")
    print(f"Pattern: {pattern}_*")
    print(f"Starting from index: {start_index}")
    print(f"Dry run: {dry_run}\n")
    
    # Find all audio files matching the pattern and index
    audio_files_to_remove = []
    
    for audio_file in sorted(wavs_dir.glob(f"{pattern}_*.wav")):
        # Extract index from filename (e.g., merged_00009968.wav -> 9968)
        try:
            # Get the numeric part after pattern_
            filename = audio_file.stem  # merged_00009968
            index_str = filename.split('_')[-1]  # 00009968
            index = int(index_str)  # 9968
            
            if index >= start_index:
                audio_files_to_remove.append(audio_file)
        except (ValueError, IndexError):
            continue
    
    if not audio_files_to_remove:
        print(f"✓ No audio files found matching pattern '{pattern}' with index >= {start_index}")
        return
    
    print(f"📁 Found {len(audio_files_to_remove)} audio files to remove:")
    print(f"   First: {audio_files_to_remove[0].name}")
    print(f"   Last: {audio_files_to_remove[-1].name}")
    print()
    
    # Process metadata files
    for csv_name in ["metadata_train.csv", "metadata_eval.csv"]:
        csv_path = dataset_path / csv_name
        
        if not csv_path.exists():
            print(f"⚠ Skipping {csv_name} (not found)")
            continue
        
        print(f"{'='*70}")
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
        
        # Create backup
        if not dry_run:
            backup_path = dataset_path / f"{csv_name}.backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            df.to_csv(backup_path, sep="|", index=False)
            print(f"✓ Backup created: {backup_path.name}")
        
        # Find entries to remove
        audio_files_to_remove_set = set(f"wavs/{f.name}" for f in audio_files_to_remove)
        mask = df['audio_file'].isin(audio_files_to_remove_set)
        removed_count = mask.sum()
        
        if removed_count > 0:
            print(f"\n🗑 Removing {removed_count} metadata entries")
            
            # Show some examples
            removed_entries = df[mask].head(5)
            print(f"\nExample entries being removed:")
            for idx, row in removed_entries.iterrows():
                text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
                print(f"  - {row['audio_file']}: \"{text_preview}\"")
            
            # Remove entries
            df_cleaned = df[~mask]
            
            # Save cleaned CSV
            if not dry_run:
                df_cleaned.to_csv(csv_path, sep="|", index=False)
                print(f"\n✓ Saved cleaned {csv_name}")
            
            print(f"\n📊 Summary for {csv_name}:")
            print(f"   Original: {original_count}")
            print(f"   Removed: {removed_count}")
            print(f"   Final: {len(df_cleaned)}")
        else:
            print(f"\n✓ No matching entries found in {csv_name}")
        
        print()
    
    # Remove audio files
    if audio_files_to_remove:
        print(f"{'='*70}")
        print(f"Removing Audio Files")
        print(f"{'='*70}\n")
        
        removed_audio_count = 0
        total_size = 0
        
        for audio_file in audio_files_to_remove:
            try:
                file_size = audio_file.stat().st_size
                total_size += file_size
                
                if not dry_run:
                    audio_file.unlink()
                
                removed_audio_count += 1
                
                # Show progress for every 100 files
                if removed_audio_count % 100 == 0:
                    print(f"  Removed {removed_audio_count}/{len(audio_files_to_remove)} files...")
                
            except Exception as e:
                print(f"  ⚠ Could not remove {audio_file.name}: {e}")
        
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"\n✓ Removed {removed_audio_count} audio files")
        print(f"  Total space freed: {total_size_mb:.2f} MB")
    
    print(f"\n{'='*70}")
    print("✅ Cleanup complete!")
    print(f"{'='*70}")
    
    if dry_run:
        print("\n⚠ DRY RUN - No files were actually modified")
        print("   Remove --dry_run flag to apply changes")


def main():
    parser = argparse.ArgumentParser(
        description="Remove junky audio segments from dataset"
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--start_index",
        type=int,
        required=True,
        help="Starting index to remove (e.g., 9968 for merged_00009968.wav)"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="merged",
        help="Filename pattern (default: merged)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    
    args = parser.parse_args()
    
    remove_junky_segments(
        dataset_dir=args.dataset_dir,
        start_index=args.start_index,
        pattern=args.pattern,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
