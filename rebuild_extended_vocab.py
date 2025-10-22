#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild extended vocabulary with Amharic/Ethiopic characters (tokenizers JSON).

This script reads ready/vocab.json (HF Tokenizers JSON), adds Ethiopic characters
into model.vocab (ensuring dict schema), and writes ready/vocab_extended_amharic.json.
"""

import json
import sys
from pathlib import Path

print("="*70)
print("REBUILDING EXTENDED AMHARIC VOCABULARY")
print("="*70)

ready_dir = Path("finetune_models/ready")
vocab_base_path = ready_dir / "vocab.json"
vocab_extended_path = ready_dir / "vocab_extended_amharic.json"

# 1. Load base tokenizer JSON
if not vocab_base_path.exists():
    print(f"❌ ERROR: Base vocab not found at {vocab_base_path}")
    sys.exit(1)

with open(vocab_base_path, 'r', encoding='utf-8') as f:
    base_tokenizer = json.load(f)

# Normalize model.vocab schema to dict
model = base_tokenizer.get('model', {})
vocab = model.get('vocab')
if vocab is None:
    print("❌ ERROR: Invalid tokenizer JSON (missing model.vocab)")
    sys.exit(1)

if isinstance(vocab, list):
    print("ℹ️  Normalizing model.vocab list → dict mapping")
    vocab = {tok: idx for idx, tok in enumerate(vocab)}
    model['vocab'] = vocab
    base_tokenizer['model'] = model
elif not isinstance(vocab, dict):
    print(f"❌ ERROR: Unsupported model.vocab type: {type(vocab)}")
    sys.exit(1)

print(f"✅ Loaded base tokenizer JSON; base vocab size: {len(model['vocab'])}")

# 2. Generate comprehensive Ethiopic character set
print("\nGenerating Ethiopic character set...")
ethiopic_chars = set()
# Main Ethiopic block (U+1200 - U+137F)
ethiopic_chars.update(chr(cp) for cp in range(0x1200, 0x1380))
# Ethiopic Supplement (U+1380 - U+139F)
ethiopic_chars.update(chr(cp) for cp in range(0x1380, 0x13A0))
# Ethiopic Extended (U+2D80 - U+2DDF)
ethiopic_chars.update(chr(cp) for cp in range(0x2D80, 0x2DE0))
# Ethiopic Extended-A (U+AB00 - U+AB2F)
ethiopic_chars.update(chr(cp) for cp in range(0xAB00, 0xAB30))
print(f"✅ Generated {len(ethiopic_chars)} Ethiopic characters")

# 3. Create extended vocabulary inside model.vocab
vocab_dict = base_tokenizer['model']['vocab']
try:
    next_id = max(int(i) for i in vocab_dict.values()) + 1
except Exception:
    next_id = len(vocab_dict)

added = 0
for ch in sorted(ethiopic_chars):
    if ch not in vocab_dict:
        vocab_dict[ch] = next_id
        next_id += 1
        added += 1

print(f"✅ Extended vocabulary entries added: {added}")
print(f"📊 New vocab size: {len(vocab_dict)}")

# 4. Verify key Amharic characters are present
test_chars = "ሰላም ዓለም አንድ ሁለት ሶስት"
print(f"\nVerifying key Amharic characters: {test_chars}")
missing = [c for c in test_chars if c.strip() and c not in vocab_dict]
if missing:
    print(f"⚠️  Missing characters: {missing}")
else:
    print(f"✅ All test characters present!")

# 5. Save extended tokenizer JSON
vocab_extended_path.parent.mkdir(parents=True, exist_ok=True)
with open(vocab_extended_path, 'w', encoding='utf-8') as f:
    json.dump(base_tokenizer, f, ensure_ascii=False, indent=2)
print(f"\n✅ Saved extended vocabulary to: {vocab_extended_path}")

# 6. Backup old extended file if exists
try:
    if vocab_extended_path.exists():
        backup_path = ready_dir / "vocab_extended_amharic.json.backup"
        import shutil
        shutil.copy2(vocab_extended_path, backup_path)
        print(f"✅ Backed up previous extended vocab to: {backup_path}")
except Exception as e:
    print(f"⚠️  Could not create backup: {e}")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. This new vocab file will be auto-detected by the WebUI/inference")
print("2. Ensure your checkpoint was trained with this vocab size (or retrain)")
print("3. You can copy it over ready/vocab.json to force usage")
print("="*70)
