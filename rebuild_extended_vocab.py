#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild extended vocabulary with Amharic/Ethiopic characters."""

import json
import sys
from pathlib import Path
from collections import OrderedDict

print("="*70)
print("REBUILDING EXTENDED AMHARIC VOCABULARY")
print("="*70)

ready_dir = Path("finetune_models/ready")
vocab_base_path = ready_dir / "vocab.json"
vocab_extended_path = ready_dir / "vocab_extended_amharic.json"

# 1. Load base vocabulary
if not vocab_base_path.exists():
    print(f"❌ ERROR: Base vocab not found at {vocab_base_path}")
    sys.exit(1)

with open(vocab_base_path, 'r', encoding='utf-8') as f:
    base_vocab = json.load(f, object_pairs_hook=OrderedDict)

print(f"✅ Loaded base vocabulary: {len(base_vocab)} tokens")

# 2. Generate comprehensive Ethiopic character set
print("\nGenerating Ethiopic character set...")

# Ethiopic Unicode blocks
ethiopic_chars = set()

# Main Ethiopic block (U+1200 - U+137F)
for code in range(0x1200, 0x1380):
    ethiopic_chars.add(chr(code))

# Ethiopic Supplement (U+1380 - U+139F)
for code in range(0x1380, 0x13A0):
    ethiopic_chars.add(chr(code))

# Ethiopic Extended (U+2D80 - U+2DDF)
for code in range(0x2D80, 0x2DE0):
    ethiopic_chars.add(chr(code))

# Ethiopic Extended-A (U+AB00 - U+AB2F)
for code in range(0xAB00, 0xAB30):
    ethiopic_chars.add(chr(code))

print(f"✅ Generated {len(ethiopic_chars)} Ethiopic characters")

# 3. Create extended vocabulary
extended_vocab = OrderedDict(base_vocab)
start_idx = len(base_vocab)

# Add Ethiopic characters
for idx, char in enumerate(sorted(ethiopic_chars), start=start_idx):
    if char not in extended_vocab:
        extended_vocab[char] = idx

print(f"✅ Extended vocabulary: {len(extended_vocab)} tokens")
print(f"   Added: {len(extended_vocab) - len(base_vocab)} new tokens")

# 4. Verify key Amharic characters are present
test_chars = "ሰላም ዓለም አንድ ሁለት ሶስት"
print(f"\nVerifying key Amharic characters: {test_chars}")

missing = []
for char in test_chars:
    if char not in extended_vocab and char not in ' ':
        missing.append(char)

if missing:
    print(f"⚠️  Missing characters: {missing}")
else:
    print(f"✅ All test characters present!")

# 5. Save extended vocabulary
vocab_extended_path.parent.mkdir(parents=True, exist_ok=True)
with open(vocab_extended_path, 'w', encoding='utf-8') as f:
    json.dump(extended_vocab, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved extended vocabulary to: {vocab_extended_path}")

# 6. Create backup of old vocab if it exists
if vocab_extended_path.exists():
    backup_path = ready_dir / "vocab_extended_amharic.json.backup"
    import shutil
    shutil.copy2(vocab_extended_path, backup_path)
    print(f"✅ Backed up old vocab to: {backup_path}")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. This new vocab file will be used automatically in inference")
print("2. But your CHECKPOINT still has the OLD vocab size!")
print("3. You need to either:")
print("   a) Re-train from scratch with this new vocab, OR")
print("   b) Use a checkpoint that was trained with extended vocab")
print("="*70)
