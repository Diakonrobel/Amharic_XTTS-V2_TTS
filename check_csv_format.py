#!/usr/bin/env python3
"""
Check and display CSV format to diagnose training failure
"""

import csv
import os

print("=" * 70)
print("🔍 CSV FORMAT CHECKER")
print("=" * 70)
print()

csv_files = [
    "finetune_models/dataset/metadata_train.csv",
    "finetune_models/dataset/metadata_eval.csv"
]

for csv_path in csv_files:
    print(f"Checking: {csv_path}")
    print("-" * 70)
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found!")
        print()
        continue
    
    # Read first few lines
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = [f.readline().strip() for _ in range(5)]
    
    print(f"First 5 lines (raw):")
    for i, line in enumerate(lines, 1):
        print(f"  {i}: {line[:100]}{'...' if len(line) > 100 else ''}")
    
    print()
    
    # Try parsing with pipe delimiter
    print("Parsing with pipe delimiter (|):")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for i, row in enumerate(reader):
            if i >= 3:
                break
            print(f"  Row {i+1}: {len(row)} columns")
            if len(row) >= 3:
                print(f"    Column 0 (audio): {row[0][:50]}")
                print(f"    Column 1 (text):  {row[1][:50]}")
                print(f"    Column 2 (lang):  {row[2]}")
            else:
                print(f"    ⚠️  Expected 3 columns, got {len(row)}")
                print(f"    Data: {row}")
    
    print()
    
    # Check if audio files exist
    print("Checking audio file paths:")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        row = next(reader)
        if len(row) >= 1:
            audio_path = row[0]
            print(f"  Sample path: {audio_path}")
            print(f"  Exists? {os.path.exists(audio_path)}")
            
            # Try to find the file in different locations
            if not os.path.exists(audio_path):
                print(f"  ❌ Audio file not found!")
                print(f"  Trying alternative paths...")
                
                # Check if it's an absolute path
                if os.path.isabs(audio_path):
                    print(f"    Path is absolute: {audio_path}")
                else:
                    print(f"    Path is relative: {audio_path}")
                
                # Try common prefixes
                for prefix in ["finetune_models/dataset/", "finetune_models/", ""]:
                    alt_path = os.path.join(prefix, os.path.basename(audio_path))
                    if os.path.exists(alt_path):
                        print(f"    ✅ Found at: {alt_path}")
                        break
    
    print()
    print("=" * 70)
    print()

print()
print("💡 EXPECTED FORMAT:")
print("-" * 70)
print("Each line should be: audio_file.wav|text content|language")
print()
print("Example:")
print("  wavs/audio_001.wav|ሰላም ዓለም|am")
print("  wavs/audio_002.wav|ኢትዮጵያ አማርኛ|am")
print()
print("Common issues:")
print("  1. Audio file paths don't exist")
print("  2. Wrong number of columns (should be 3)")
print("  3. Empty lines or malformed data")
print("=" * 70)
