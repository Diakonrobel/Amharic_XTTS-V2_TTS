#!/usr/bin/env python3
"""
Find where audio files are actually located on Lightning AI
"""

import os
import csv
from pathlib import Path

print("=" * 70)
print("🔍 AUDIO FILE LOCATION FINDER")
print("=" * 70)
print()

# Read CSV to see what paths are expected
train_csv = "finetune_models/dataset/metadata_train.csv"
eval_csv = "finetune_models/dataset/metadata_eval.csv"

print(f"Reading CSV: {train_csv}")
print()

# Get first few audio paths from CSV
sample_paths = []
with open(train_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    for i, row in enumerate(reader):
        if i >= 5:  # Just check first 5
            break
        if len(row) >= 1:
            audio_path = row[0]
            sample_paths.append(audio_path)
            print(f"CSV references: {audio_path}")
            print(f"  Exists? {os.path.exists(audio_path)}")
            if os.path.exists(audio_path):
                print(f"  ✅ Found!")
            else:
                print(f"  ❌ Missing!")
            print()

print()
print("=" * 70)
print("🔍 Searching for wav files...")
print("=" * 70)
print()

# Search common locations
search_dirs = [
    "finetune_models/dataset",
    "finetune_models",
    ".",
    "wavs",
    "audio",
]

for search_dir in search_dirs:
    if not os.path.exists(search_dir):
        continue
    
    print(f"Checking: {search_dir}")
    
    # Count wav files
    wav_files = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    print(f"  Found {len(wav_files)} .wav files")
    
    if wav_files:
        print(f"  Sample files:")
        for f in wav_files[:5]:
            print(f"    - {f}")
    print()

print()
print("=" * 70)
print("💡 NEXT STEPS")
print("=" * 70)
print()
print("1. Check if audio files exist in a different directory")
print("2. Verify CSV paths match actual audio file locations")
print("3. Either:")
print("   a) Move audio files to match CSV paths, OR")
print("   b) Update CSV paths to match audio file locations")
print()
