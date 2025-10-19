#!/usr/bin/env python3
"""
Dataset Merger for XTTS Fine-tuning
Merges multiple temp_dataset_* folders into the main dataset folder
"""

import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Configuration
BASE_DIR = Path("/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models")
MAIN_DATASET = BASE_DIR / "dataset"
TEMP_PATTERN = "temp_dataset_*"

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_info(msg):
    print(f"{Colors.CYAN}ℹ {msg}{Colors.RESET}")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}{Colors.RESET}\n")

def create_backup(file_path):
    """Create a timestamped backup of a file"""
    if not file_path.exists():
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.parent / f"{file_path.name}.backup_{timestamp}"
    shutil.copy2(file_path, backup_path)
    print_success(f"Backup created: {backup_path.name}")
    return backup_path

def find_temp_datasets():
    """Find all temp_dataset_* folders"""
    temp_dirs = sorted([d for d in BASE_DIR.glob(TEMP_PATTERN) if d.is_dir()])
    return temp_dirs

def validate_dataset_structure(dataset_path):
    """Validate that a dataset has the expected structure"""
    required = ['lang.txt', 'metadata_train.csv', 'metadata_eval.csv', 'wavs']
    missing = []
    
    for item in required:
        if not (dataset_path / item).exists():
            missing.append(item)
    
    return len(missing) == 0, missing

def count_wavs(dataset_path):
    """Count wav files in a dataset"""
    wavs_dir = dataset_path / "wavs"
    if not wavs_dir.exists():
        return 0
    return len(list(wavs_dir.glob("*.wav")))

def merge_csv_files(main_csv, temp_csvs, output_csv):
    """Merge multiple CSV files, removing duplicates"""
    print_info(f"Merging {len(temp_csvs) + 1} CSV files...")
    
    # Read main dataset
    if main_csv.exists():
        df_main = pd.read_csv(main_csv, sep='|')
        print_info(f"Main CSV has {len(df_main)} rows")
    else:
        df_main = pd.DataFrame()
        print_warning("Main CSV doesn't exist, starting fresh")
    
    # Read and concatenate all temp CSVs
    dfs = [df_main]
    total_temp_rows = 0
    
    for temp_csv in temp_csvs:
        if temp_csv.exists():
            df_temp = pd.read_csv(temp_csv, sep='|')
            dfs.append(df_temp)
            print_info(f"  {temp_csv.parent.name}: {len(df_temp)} rows")
            total_temp_rows += len(df_temp)
        else:
            print_warning(f"  {temp_csv} not found, skipping")
    
    # Merge
    if len(dfs) > 0:
        df_merged = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates based on audio_file column (assuming it exists)
        if 'audio_file' in df_merged.columns:
            before = len(df_merged)
            df_merged = df_merged.drop_duplicates(subset=['audio_file'], keep='first')
            duplicates = before - len(df_merged)
            if duplicates > 0:
                print_warning(f"Removed {duplicates} duplicate entries")
        
        # Save
        df_merged.to_csv(output_csv, sep='|', index=False)
        print_success(f"Merged CSV saved: {len(df_merged)} total rows")
        return len(df_merged), total_temp_rows
    else:
        print_error("No valid CSVs to merge")
        return 0, 0

def merge_wav_files(main_wavs_dir, temp_wavs_dirs):
    """Copy wav files from temp datasets to main dataset"""
    main_wavs_dir.mkdir(exist_ok=True)
    
    total_copied = 0
    total_skipped = 0
    
    for temp_wavs_dir in temp_wavs_dirs:
        if not temp_wavs_dir.exists():
            print_warning(f"  {temp_wavs_dir} doesn't exist, skipping")
            continue
        
        wav_files = list(temp_wavs_dir.glob("*.wav"))
        print_info(f"Processing {len(wav_files)} files from {temp_wavs_dir.parent.name}")
        
        for wav_file in wav_files:
            dest = main_wavs_dir / wav_file.name
            
            if dest.exists():
                # Check if files are identical
                if dest.stat().st_size == wav_file.stat().st_size:
                    total_skipped += 1
                else:
                    # Rename to avoid overwrite
                    new_name = f"{dest.stem}_{temp_wavs_dir.parent.name}{dest.suffix}"
                    dest = main_wavs_dir / new_name
                    shutil.copy2(wav_file, dest)
                    total_copied += 1
            else:
                shutil.copy2(wav_file, dest)
                total_copied += 1
    
    print_success(f"Copied {total_copied} wav files")
    if total_skipped > 0:
        print_info(f"Skipped {total_skipped} existing files")
    
    return total_copied, total_skipped

def generate_report(temp_dirs, stats):
    """Generate a merge report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "temp_datasets_merged": [d.name for d in temp_dirs],
        "statistics": stats
    }
    
    report_path = BASE_DIR / f"merge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print_success(f"Report saved: {report_path.name}")
    return report_path

def main():
    print_header("Dataset Merger for XTTS Fine-tuning")
    
    # Check if main dataset exists
    if not MAIN_DATASET.exists():
        print_error(f"Main dataset not found: {MAIN_DATASET}")
        sys.exit(1)
    
    # Find temp datasets
    temp_dirs = find_temp_datasets()
    
    if not temp_dirs:
        print_warning("No temp datasets found!")
        print_info(f"Looking for: {BASE_DIR}/{TEMP_PATTERN}")
        sys.exit(0)
    
    print_info(f"Found {len(temp_dirs)} temp datasets:")
    for temp_dir in temp_dirs:
        wav_count = count_wavs(temp_dir)
        print(f"  • {temp_dir.name}: {wav_count} wav files")
    
    # Validate structures
    print_info("\nValidating dataset structures...")
    invalid_datasets = []
    
    for temp_dir in temp_dirs:
        is_valid, missing = validate_dataset_structure(temp_dir)
        if not is_valid:
            print_warning(f"  {temp_dir.name} is missing: {', '.join(missing)}")
            invalid_datasets.append(temp_dir)
    
    if invalid_datasets:
        response = input(f"\n⚠ {len(invalid_datasets)} datasets have missing files. Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print_info("Merge cancelled")
            sys.exit(0)
    
    # Confirm merge
    print_warning(f"\nAbout to merge {len(temp_dirs)} temp datasets into main dataset")
    response = input("Continue? (yes/no): ")
    if response.lower() != 'yes':
        print_info("Merge cancelled")
        sys.exit(0)
    
    # Create backups
    print_header("Creating Backups")
    create_backup(MAIN_DATASET / "metadata_train.csv")
    create_backup(MAIN_DATASET / "metadata_eval.csv")
    
    # Statistics
    stats = {
        "main_dataset_wavs_before": count_wavs(MAIN_DATASET),
        "temp_datasets_count": len(temp_dirs)
    }
    
    # Merge CSV files
    print_header("Merging Metadata Files")
    
    # Train metadata
    temp_train_csvs = [d / "metadata_train.csv" for d in temp_dirs]
    train_total, train_temp = merge_csv_files(
        MAIN_DATASET / "metadata_train.csv",
        temp_train_csvs,
        MAIN_DATASET / "metadata_train.csv"
    )
    
    # Eval metadata
    temp_eval_csvs = [d / "metadata_eval.csv" for d in temp_dirs]
    eval_total, eval_temp = merge_csv_files(
        MAIN_DATASET / "metadata_eval.csv",
        temp_eval_csvs,
        MAIN_DATASET / "metadata_eval.csv"
    )
    
    stats["metadata_train_rows"] = train_total
    stats["metadata_eval_rows"] = eval_total
    
    # Merge wav files
    print_header("Merging Audio Files")
    temp_wavs_dirs = [d / "wavs" for d in temp_dirs]
    copied, skipped = merge_wav_files(MAIN_DATASET / "wavs", temp_wavs_dirs)
    
    stats["wavs_copied"] = copied
    stats["wavs_skipped"] = skipped
    stats["main_dataset_wavs_after"] = count_wavs(MAIN_DATASET)
    
    # Generate report
    print_header("Merge Summary")
    print(f"{'='*60}")
    print(f"Train metadata: {train_total} rows (added {train_temp} from temp)")
    print(f"Eval metadata:  {eval_total} rows (added {eval_temp} from temp)")
    print(f"Wav files:      {stats['main_dataset_wavs_after']} total")
    print(f"  - Copied:     {copied}")
    print(f"  - Skipped:    {skipped}")
    print(f"  - Before:     {stats['main_dataset_wavs_before']}")
    print(f"{'='*60}")
    
    report_path = generate_report(temp_dirs, stats)
    
    # Ask about cleanup
    print_warning("\nCleanup Options:")
    print("1. Keep temp datasets")
    print("2. Delete temp datasets (PERMANENT)")
    print("3. Move temp datasets to 'merged_temps' folder")
    
    cleanup = input("\nSelect option (1-3): ").strip()
    
    if cleanup == "2":
        confirm = input(f"⚠ DELETE {len(temp_dirs)} temp datasets? Type 'DELETE' to confirm: ")
        if confirm == "DELETE":
            print_info("Deleting temp datasets...")
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir)
                print_success(f"  Deleted: {temp_dir.name}")
        else:
            print_info("Deletion cancelled")
    
    elif cleanup == "3":
        merged_dir = BASE_DIR / "merged_temps"
        merged_dir.mkdir(exist_ok=True)
        print_info(f"Moving temp datasets to {merged_dir.name}...")
        for temp_dir in temp_dirs:
            dest = merged_dir / temp_dir.name
            shutil.move(str(temp_dir), str(dest))
            print_success(f"  Moved: {temp_dir.name}")
    
    print_success("\n✓ Merge completed successfully!")
    print_info(f"Report: {report_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nMerge interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
