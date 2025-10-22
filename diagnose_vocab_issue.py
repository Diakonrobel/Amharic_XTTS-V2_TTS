#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose why Amharic text doesn't work in inference."""

import sys
import torch
import json
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

print("="*70)
print("AMHARIC VOCABULARY DIAGNOSTIC")
print("="*70)

# 1. Check if extended vocab file exists
ready_dir = Path("finetune_models/ready")
vocab_extended = ready_dir / "vocab_extended_amharic.json"
vocab_base = ready_dir / "vocab.json"

print("\n1. VOCABULARY FILES:")
def _count_tokens(tokenizer_json):
    try:
        mv = tokenizer_json.get('model', {}).get('vocab', {})
        if isinstance(mv, dict):
            base = len(mv)
        elif isinstance(mv, list):
            base = len(mv)
        else:
            base = 0
        added = len(tokenizer_json.get('added_tokens', []))
        return base + added, base, added
    except Exception:
        return 0, 0, 0

if vocab_extended.exists():
    with open(vocab_extended, 'r', encoding='utf-8') as f:
        vocab_ext = json.load(f)
    total, base, added = _count_tokens(vocab_ext)
    print(f"   ✅ Extended vocab found: {total} tokens (base={base}, added={added})")
else:
    print(f"   ❌ Extended vocab NOT found at {vocab_extended}")

if vocab_base.exists():
    with open(vocab_base, 'r', encoding='utf-8') as f:
        vocab_b = json.load(f)
    total_b, base_b, added_b = _count_tokens(vocab_b)
    print(f"   ✅ Base vocab found: {total_b} tokens (base={base_b}, added={added_b})")
else:
    print(f"   ⚠️  Base vocab not found")

# 2. Check checkpoint embedding size
checkpoint_path = ready_dir / "best_model.pth"
if checkpoint_path.exists():
    print("\n2. CHECKPOINT ANALYSIS:")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check embedding layer size
    if 'model' in checkpoint:
        model_dict = checkpoint['model']
    else:
        model_dict = checkpoint
    
    # Look for text embedding weight
    emb_key = None
    for key in model_dict.keys():
        if 'text_embedding' in key and 'weight' in key:
            emb_key = key
            break
    
    if emb_key:
        emb_weight = model_dict[emb_key]
        vocab_size = emb_weight.shape[0]
        print(f"   Text embedding shape: {emb_weight.shape}")
        print(f"   Checkpoint vocab size: {vocab_size}")
        
        if vocab_extended.exists():
            expected_size = len(vocab_ext)
            if vocab_size == expected_size:
                print(f"   ✅ MATCH: Checkpoint has extended vocab ({vocab_size} tokens)")
            else:
                print(f"   ❌ MISMATCH: Expected {expected_size}, got {vocab_size}")
                print(f"   ⚠️  PROBLEM: Checkpoint doesn't have extended vocabulary!")
    else:
        print(f"   ❌ Could not find text embedding in checkpoint")
else:
    print(f"\n2. ❌ Checkpoint not found at {checkpoint_path}")

# 3. Test tokenization
print("\n3. TOKENIZATION TEST:")
try:
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
    
    # Test with extended vocab
    if vocab_extended.exists():
        tokenizer = VoiceBpeTokenizer(vocab_file=str(vocab_extended))
        
        test_texts = [
            ("Hello world", "English"),
            ("ሰላም ዓለም", "Amharic"),
            ("አንድ ሁለት ሶስት", "Amharic numbers")
        ]
        
        for text, label in test_texts:
            tokens = tokenizer.encode(text)
            print(f"\n   {label}: '{text}'")
            print(f"   Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"   Tokens: {tokens}")
            print(f"   Length: {len(tokens)} tokens")
            
            # Check for UNK tokens
            unk_count = sum(1 for t in tokens if t == tokenizer.tokenizer.unk_token_id)
            if unk_count > 0:
                print(f"   ⚠️  Contains {unk_count} UNK tokens!")
            else:
                print(f"   ✅ No UNK tokens")
    else:
        print("   ❌ Cannot test - extended vocab not found")
except Exception as e:
    print(f"   ❌ Tokenization test failed: {e}")

# 4. Summary
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY:")
print("="*70)

if not vocab_extended.exists():
    print("❌ CRITICAL: Extended vocabulary file missing!")
    print("   Solution: Re-run training to generate extended vocab")
elif checkpoint_path.exists() and emb_key and vocab_size != len(vocab_ext):
    print("❌ CRITICAL: Checkpoint has wrong vocabulary size!")
    print(f"   Checkpoint: {vocab_size} tokens")
    print(f"   Extended vocab: {len(vocab_ext)} tokens")
    print("   Solution: Use correct checkpoint that was trained with extended vocab")
else:
    print("✅ Vocabulary files look correct")
    print("   Next: Check inference code is loading extended vocab")

print("="*70)
