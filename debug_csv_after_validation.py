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
    import inspect
    
    root_path = os.path.dirname(csv_path)
    meta_file = os.path.basename(csv_path)
    
    print(f"root_path: {root_path}")
    print(f"meta_file: {meta_file}")
    print(f"Full CSV path: {os.path.join(root_path, meta_file)}")
    print()
    
    # Show what coqui formatter expects
    print("Coqui formatter signature:")
    print(f"  {inspect.signature(coqui)}")
    print()
    
    try:
        # Read CSV manually to see what formatter will see
        csv_full_path = os.path.join(root_path, meta_file)
        print(f"Reading CSV from: {csv_full_path}")
        print(f"CSV exists: {os.path.exists(csv_full_path)}")
        print()
        
        metadata = coqui(root_path, meta_file)
        print(f"✅ TTS formatter succeeded!")
        print(f"   Returned {len(metadata)} samples")
        print(f"   First sample type: {type(metadata[0])}")
        print(f"   First sample keys: {metadata[0].keys() if hasattr(metadata[0], 'keys') else 'Not a dict'}")
        if metadata:
            print(f"   First sample:")
            if hasattr(metadata[0], 'items'):
                for key, value in list(metadata[0].items())[:5]:
                    print(f"      {key}: {str(value)[:60]}")
            else:
                print(f"      {metadata[0]}")
    except Exception as e:
        print(f"❌ TTS formatter failed: {e}")
        print(f"   This is why training fails!")
        print()
        
        # Try to debug what went wrong
        try:
            import csv as csv_lib
            csv_full_path = os.path.join(root_path, meta_file)
            print("Manual CSV read test:")
            with open(csv_full_path, 'r', encoding='utf-8') as f:
                reader = csv_lib.reader(f, delimiter='|')
                first_row = next(reader)
                print(f"  First row: {first_row}")
                print(f"  Columns: {len(first_row)}")
        except Exception as debug_e:
            print(f"  Manual read also failed: {debug_e}")
        
        print()
        import traceback
        traceback.print_exc()
        
except ImportError:
    print("⚠️  Could not import TTS formatters")

print()
print("=" * 70)
