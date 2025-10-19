#!/usr/bin/env python3
"""
Standardize Dataset Filenames
Renames all non-merged files to follow the merged_XXXXXXXX.wav pattern
and updates metadata CSVs accordingly
"""

import os
import sys
import shutil
import csv
from pathlib import Path
from datetime import datetime
import json
import re

# Configuration
BASE_DIR = Path("/teamspace/studios/this_studio/Amharic_XTTS-V2_TTS/finetune_models")
DATASET_DIR = BASE_DIR / "dataset"
WAVS_DIR = DATASET_DIR / "wavs"
METADATA_TRAIN = DATASET_DIR / "metadata_train.csv"
METADATA_EVAL = DATASET_DIR / "metadata_eval.csv"

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

def find_highest_merged_number():
    """Find the highest number in merged_XXXXXXXX.wav files"""
    merged_pattern = re.compile(r'^merged_(\d+)\.wav$')
    max_num = -1
    
    for file in WAVS_DIR.glob("merged_*.wav"):
        match = merged_pattern.match(file.name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    
    return max_num

def get_files_to_rename():
    """Get list of all non-merged wav files"""
    merged_pattern = re.compile(r'^merged_\d+\.wav$')
    files_to_rename = []
    
    for file in sorted(WAVS_DIR.glob("*.wav")):
        if not merged_pattern.match(file.name):
            files_to_rename.append(file)
    
    return files_to_rename

def create_rename_mapping(files_to_rename, start_num):
    """Create mapping of old filenames to new filenames"""
    rename_map = {}
    
    for i, file in enumerate(files_to_rename):
        new_num = start_num + i + 1
        new_name = f"merged_{new_num:08d}.wav"
        rename_map[file.name] = new_name
    
    return rename_map

def rename_wav_files(rename_map):
    """Rename wav files according to the mapping"""
    print_info(f"Renaming {len(rename_map)} wav files...")
    
    # First pass: rename to temporary names to avoid conflicts
    temp_map = {}
    for old_name, new_name in rename_map.items():
        old_path = WAVS_DIR / old_name
        temp_name = f"_temp_{new_name}"
        temp_path = WAVS_DIR / temp_name
        
        if old_path.exists():
            old_path.rename(temp_path)
            temp_map[temp_name] = new_name
    
    print_info("Phase 1: Temporary rename complete")
    
    # Second pass: rename from temporary to final names
    for temp_name, new_name in temp_map.items():
        temp_path = WAVS_DIR / temp_name
        new_path = WAVS_DIR / new_name
        
        if temp_path.exists():
            temp_path.rename(new_path)
    
    print_success(f"Renamed {len(rename_map)} wav files")

def update_metadata_csv(csv_path, rename_map):
    """Update metadata CSV with new filenames"""
    if not csv_path.exists():
        print_warning(f"{csv_path.name} not found, skipping")
        return 0
    
    print_info(f"Updating {csv_path.name}...")
    
    # Read CSV
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            rows.append(row)
    
    if not rows:
        print_warning(f"{csv_path.name} is empty")
        return 0
    
    # Update rows
    header = rows[0]
    updated_count = 0
    
    for i in range(1, len(rows)):
        if rows[i] and len(rows[i]) > 0:
            old_path = rows[i][0]  # First column is audio_file path
            
            # Extract just the filename from the path
            if '/' in old_path:
                path_parts = old_path.split('/')
                old_filename = path_parts[-1]
                prefix = '/'.join(path_parts[:-1]) + '/'
            else:
                old_filename = old_path
                prefix = ''
            
            # Check if this file needs to be renamed
            if old_filename in rename_map:
                new_filename = rename_map[old_filename]
                rows[i][0] = prefix + new_filename
                updated_count += 1
    
    # Write updated CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(rows)
    
    print_success(f"Updated {updated_count} entries in {csv_path.name}")
    return updated_count

def generate_report(rename_map, train_updates, eval_updates):
    """Generate a rename report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "files_renamed": len(rename_map),
        "metadata_train_updates": train_updates,
        "metadata_eval_updates": eval_updates,
        "sample_renames": dict(list(rename_map.items())[:10])  # First 10 as sample
    }
    
    report_path = BASE_DIR / f"rename_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print_success(f"Report saved: {report_path.name}")
    return report_path

def main():
    print_header("Dataset Filename Standardizer")
    
    # Validate paths
    if not WAVS_DIR.exists():
        print_error(f"Wavs directory not found: {WAVS_DIR}")
        sys.exit(1)
    
    # Find highest merged number
    print_info("Finding highest merged file number...")
    max_merged_num = find_highest_merged_number()
    print_success(f"Highest merged number: {max_merged_num} (merged_{max_merged_num:08d}.wav)")
    
    # Get files to rename
    print_info("\nScanning for non-merged files...")
    files_to_rename = get_files_to_rename()
    print_info(f"Found {len(files_to_rename)} files to rename")
    
    if not files_to_rename:
        print_success("All files already follow the merged_XXXXXXXX.wav pattern!")
        sys.exit(0)
    
    # Show sample files
    print_info("\nSample files to rename:")
    for i, file in enumerate(files_to_rename[:5]):
        print(f"  {i+1}. {file.name}")
    if len(files_to_rename) > 5:
        print(f"  ... and {len(files_to_rename) - 5} more")
    
    # Create rename mapping
    print_info(f"\nNew files will be numbered from merged_{max_merged_num + 1:08d}.wav onwards")
    rename_map = create_rename_mapping(files_to_rename, max_merged_num)
    
    # Show sample mapping
    print_info("\nSample rename mapping:")
    for i, (old, new) in enumerate(list(rename_map.items())[:3]):
        print(f"  {old[:50]}... → {new}")
    
    # Confirm
    print_warning(f"\n🤖 AUTO-MODE: Will rename {len(rename_map)} files")
    
    # Create backups
    print_header("Creating Backups")
    create_backup(METADATA_TRAIN)
    create_backup(METADATA_EVAL)
    
    # Rename wav files
    print_header("Renaming WAV Files")
    rename_wav_files(rename_map)
    
    # Update metadata CSVs
    print_header("Updating Metadata CSVs")
    train_updates = update_metadata_csv(METADATA_TRAIN, rename_map)
    eval_updates = update_metadata_csv(METADATA_EVAL, rename_map)
    
    # Generate report
    print_header("Summary")
    print(f"{'='*60}")
    print(f"WAV files renamed:           {len(rename_map)}")
    print(f"Metadata train entries updated: {train_updates}")
    print(f"Metadata eval entries updated:  {eval_updates}")
    print(f"New number range:            merged_{max_merged_num + 1:08d}.wav")
    print(f"                         to  merged_{max_merged_num + len(rename_map):08d}.wav")
    print(f"{'='*60}")
    
    report_path = generate_report(rename_map, train_updates, eval_updates)
    
    # Verify
    print_header("Verification")
    remaining = get_files_to_rename()
    if not remaining:
        print_success("✓ All files now follow the merged_XXXXXXXX.wav pattern!")
    else:
        print_warning(f"⚠ {len(remaining)} files still don't match pattern")
    
    print_success("\n✓ Filename standardization completed!")
    print_info(f"Report: {report_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\n\nStandardization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
