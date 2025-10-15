#!/usr/bin/env python3
"""
Fix CSV Headers for Coqui TTS Format

This script adds the required header row to metadata CSV files if missing.
The Coqui TTS formatter expects the first line to contain: audio_file|text|speaker_name

Usage:
    python utils/fix_csv_headers.py [dataset_dir]
    
    If no dataset_dir is provided, defaults to: finetune_models/dataset/
    
Example:
    python utils/fix_csv_headers.py
    python utils/fix_csv_headers.py /path/to/dataset/
"""

import os
import sys
from pathlib import Path

# Header for Coqui format
COQUI_HEADER = 'audio_file|text|speaker_name\n'

def add_header_if_missing(filepath, backup=True):
    """
    Add header to CSV file if it's missing.
    
    Args:
        filepath: Path to the CSV file
        backup: Whether to create a backup before modifying
    """
    if not os.path.exists(filepath):
        print(f'⚠️  {filepath} does not exist, skipping...')
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        
    # Check if header is missing (first line should contain 'audio_file' if header exists)
    if 'audio_file' in first_line:
        print(f'✅ {filepath} already has correct header')
        return True
    
    print(f'🔧 Adding header to {filepath}...')
    
    # Create backup if requested
    if backup:
        backup_path = f'{filepath}.backup'
        if not os.path.exists(backup_path):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'   📦 Backup created: {backup_path}')
    
    # Add header
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(COQUI_HEADER + content)
    
    print(f'✅ Header added to {filepath}')
    return True

def fix_dataset_csvs(dataset_dir):
    """
    Fix CSV headers in the dataset directory.
    
    Args:
        dataset_dir: Path to the dataset directory
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f'❌ Dataset directory not found: {dataset_dir}')
        print(f'   Please create the directory or provide the correct path.')
        return False
    
    print(f'📁 Checking CSV files in: {dataset_dir}\n')
    
    # Files to fix
    train_csv = dataset_path / 'metadata_train.csv'
    eval_csv = dataset_path / 'metadata_eval.csv'
    
    results = []
    results.append(add_header_if_missing(str(train_csv)))
    results.append(add_header_if_missing(str(eval_csv)))
    
    if all(results):
        print(f'\n✅ CSV headers verified/fixed successfully!')
        return True
    else:
        print(f'\n⚠️  Some files could not be processed')
        return False

def main():
    """Main function."""
    # Determine dataset directory
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]
    else:
        # Default to finetune_models/dataset relative to script location
        script_dir = Path(__file__).parent.parent
        dataset_dir = script_dir / 'finetune_models' / 'dataset'
    
    print('=' * 70)
    print('CSV Header Fix Utility for Coqui TTS Format')
    print('=' * 70)
    print()
    
    success = fix_dataset_csvs(dataset_dir)
    
    print()
    print('=' * 70)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
