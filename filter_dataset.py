#!/usr/bin/env python3
"""
Filter dataset CSV to remove samples with audio longer than max_length
"""

import os
import sys
import csv
from pathlib import Path
import subprocess

def get_audio_duration(audio_path):
    """Get audio duration using ffprobe"""
    try:
        result = subprocess.run(
            ['ffprobe', '-i', str(audio_path), '-show_entries', 
             'format=duration', '-v', 'quiet', '-of', 'csv=p=0'],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {audio_path}: {e}")
        return None

def filter_dataset(csv_path, output_path, max_length=18, min_length=1):
    """
    Filter dataset CSV to only include audio within duration range
    
    Args:
        csv_path: Input CSV file path
        output_path: Output CSV file path
        max_length: Maximum audio duration in seconds
        min_length: Minimum audio duration in seconds
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return False
    
    csv_dir = Path(csv_path).parent
    valid_rows = []
    filtered_count = 0
    total_count = 0
    
    print(f"\n{'='*60}")
    print(f"Filtering dataset: {csv_path}")
    print(f"Duration range: {min_length}s - {max_length}s")
    print(f"{'='*60}\n")
    
    # Read the CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_count += 1
            audio_file = row.get('audio_file', '')
            
            # Handle relative/absolute paths
            if not os.path.isabs(audio_file):
                audio_path = csv_dir / audio_file
            else:
                audio_path = Path(audio_file)
            
            if not audio_path.exists():
                print(f"⚠️  Skipping missing file: {audio_file}")
                filtered_count += 1
                continue
            
            # Get duration
            duration = get_audio_duration(audio_path)
            if duration is None:
                filtered_count += 1
                continue
            
            # Check if within range
            if min_length <= duration <= max_length:
                valid_rows.append(row)
            else:
                filtered_count += 1
                if total_count <= 10:  # Show first 10 filtered samples
                    print(f"❌ Filtered: {audio_file} ({duration:.2f}s)")
    
    # Write filtered CSV
    if valid_rows:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
            writer.writeheader()
            writer.writerows(valid_rows)
        
        print(f"\n✅ Results:")
        print(f"   Total samples: {total_count}")
        print(f"   Valid samples: {len(valid_rows)}")
        print(f"   Filtered out: {filtered_count}")
        print(f"   Output saved to: {output_path}\n")
        return True
    else:
        print(f"\n❌ ERROR: No valid samples found!")
        print(f"   All {total_count} samples were outside the {min_length}-{max_length}s range.")
        print(f"\n💡 Suggestions:")
        print(f"   1. Increase max_length (current: {max_length}s)")
        print(f"   2. Split longer audio files into shorter segments")
        print(f"   3. Check if audio files are corrupted\n")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 filter_dataset.py <input_csv> <output_csv> [max_length] [min_length]")
        print("\nExample:")
        print("  python3 filter_dataset.py metadata_train.csv metadata_train_filtered.csv 18 1")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 18
    min_len = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    success = filter_dataset(input_csv, output_csv, max_len, min_len)
    sys.exit(0 if success else 1)
