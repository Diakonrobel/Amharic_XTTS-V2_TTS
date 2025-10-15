#!/usr/bin/env python3
"""
Fix CSV format issues:
1. Replace 'speaker' with 'am' in language column
2. Fix audio paths: wavs/file.wav -> finetune_models/dataset/wavs/file.wav
"""

import csv
import os
import shutil
from datetime import datetime

print("=" * 70)
print("🔧 CSV PATH FIXER")
print("=" * 70)
print()

csv_files = [
    "finetune_models/dataset/metadata_train.csv",
    "finetune_models/dataset/metadata_eval.csv"
]

for csv_path in csv_files:
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        continue
    
    print(f"Processing: {csv_path}")
    print("-" * 70)
    
    # Backup original file
    backup_path = f"{csv_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(csv_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Read and fix
    fixed_rows = []
    path_fixes = 0
    lang_fixes = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        
        for row_num, row in enumerate(reader, 1):
            if len(row) != 3:
                print(f"  ⚠️  Row {row_num}: Wrong number of columns ({len(row)}), skipping")
                continue
            
            audio_path, text, lang = row
            
            # Fix 1: Replace 'speaker' with 'am'
            if lang.strip().lower() == 'speaker':
                lang = 'am'
                lang_fixes += 1
            
            # Fix 2: Fix audio path
            # From: wavs/file.wav
            # To:   finetune_models/dataset/wavs/file.wav
            if audio_path.startswith('wavs/'):
                audio_path = audio_path.replace('wavs/', 'finetune_models/dataset/wavs/', 1)
                path_fixes += 1
            
            fixed_rows.append([audio_path, text, lang])
    
    # Write fixed data
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(fixed_rows)
    
    print(f"✅ Fixed {path_fixes} audio paths")
    print(f"✅ Fixed {lang_fixes} language codes")
    print(f"✅ Written {len(fixed_rows)} rows")
    print()
    
    # Verify fixes
    print("Verification (first row):")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        first_row = next(reader)
        
        print(f"  Audio: {first_row[0]}")
        print(f"  Text:  {first_row[1][:50]}...")
        print(f"  Lang:  {first_row[2]}")
        
        # Check if audio file exists now
        audio_exists = os.path.exists(first_row[0])
        if audio_exists:
            print(f"  ✅ Audio file exists!")
        else:
            print(f"  ℹ️  Audio file path: {first_row[0]}")
            print(f"     (Will be checked on Lightning AI)")
    
    print()
    print("=" * 70)
    print()

print()
print("✅ CSV fixing complete!")
print()
print("Summary:")
print("  - Language codes: 'speaker' → 'am'")
print("  - Audio paths: 'wavs/file.wav' → 'finetune_models/dataset/wavs/file.wav'")
print()
print("Next steps:")
print("  1. Run: python check_csv_format.py (verify fixes)")
print("  2. Push to GitHub: git add, git commit, git push")
print("  3. Pull on Lightning AI and try training again")
print()
