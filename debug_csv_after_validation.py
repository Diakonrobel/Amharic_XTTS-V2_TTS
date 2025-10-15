#!/usr/bin/env python3
"""
Debug CSV format after validation to see what TTS formatter sees
"""

import csv
import os

csv_path = "finetune_models/dataset/metadata_train.csv"

print("=" * 70)
print("🔍 DEBUG: CSV FORMAT AFTER VALIDATION")
print("=" * 70)
print()

if not os.path.exists(csv_path):
    print(f"❌ File not found: {csv_path}")
    exit(1)

print(f"File: {csv_path}")
print(f"Size: {os.path.getsize(csv_path)} bytes")
print()

# Read raw lines
print("RAW LINES (first 5):")
print("-" * 70)
with open(csv_path, 'r', encoding='utf-8') as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        print(f"{i+1}: {repr(line[:120])}")
print()

# Read with CSV reader (pipe delimiter)
print("PARSED WITH CSV READER (pipe delimiter):")
print("-" * 70)
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    for i, row in enumerate(reader):
        if i >= 5:
            break
        print(f"Row {i+1}:")
        print(f"  Columns: {len(row)}")
        if len(row) >= 1:
            print(f"  [0]: {row[0][:60]}")
        if len(row) >= 2:
            print(f"  [1]: {row[1][:60]}")
        if len(row) >= 3:
            print(f"  [2]: {row[2]}")
        print()

# Check what TTS formatter sees
print("TTS FORMATTER SIMULATION:")
print("-" * 70)
try:
    from TTS.tts.datasets.formatters import coqui
    
    root_path = os.path.dirname(csv_path)
    meta_file = os.path.basename(csv_path)
    
    print(f"root_path: {root_path}")
    print(f"meta_file: {meta_file}")
    print()
    
    try:
        metadata = coqui(root_path, meta_file)
        print(f"✅ TTS formatter succeeded!")
        print(f"   Returned {len(metadata)} samples")
        print(f"   First sample keys: {metadata[0].keys() if metadata else 'N/A'}")
        if metadata:
            print(f"   First sample:")
            for key, value in list(metadata[0].items())[:5]:
                print(f"      {key}: {str(value)[:60]}")
    except Exception as e:
        print(f"❌ TTS formatter failed: {e}")
        print(f"   This is why training fails!")
        import traceback
        traceback.print_exc()
        
except ImportError:
    print("⚠️  Could not import TTS formatters")

print()
print("=" * 70)
