#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose why Amharic text doesn't work in inference."""

import sys
import torch
import json
import argparse
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

print("="*70)
print("AMHARIC VOCABULARY DIAGNOSTIC")
print("="*70)

# Args
parser = argparse.ArgumentParser(description="Diagnose vocab/checkpoint/tokenizer alignment")
parser.add_argument("--ready_dir", type=str, default="finetune_models/ready")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth (defaults to best_model.pth under ready_dir)")
parser.add_argument("--vocab", type=str, default=None, help="Path to extended vocab json (defaults to vocab_extended_amharic.json under ready_dir)")
parser.add_argument("--base_vocab", type=str, default=None, help="Path to base vocab json (defaults to vocab.json under ready_dir)")
parser.add_argument("--am_lang", type=str, default="am", help="Language code to use for Amharic in tests (am or amh)")
args = parser.parse_args()

# 1. Check if extended vocab file exists
ready_dir = Path(args.ready_dir)
vocab_extended = Path(args.vocab) if args.vocab else (ready_dir / "vocab_extended_amharic.json")
vocab_base = Path(args.base_vocab) if args.base_vocab else (ready_dir / "vocab.json")

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
checkpoint_path = Path(args.checkpoint) if args.checkpoint else (ready_dir / "best_model.pth")
if checkpoint_path.exists():
    print("\n2. CHECKPOINT ANALYSIS:")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Determine model dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_dict = checkpoint['state_dict']
    else:
        model_dict = checkpoint

    # Look for text embedding weight (robust search)
    def _find_embedding_key(keys):
        candidates = []
        for k in keys:
            lk = k.lower()
            if lk.endswith("text_embedding.weight"):
                candidates.append((0, k))
            elif lk.endswith("wte.weight") and ("gpt" in lk or "transformer" in lk):
                candidates.append((1, k))
            elif lk.endswith("embeddings.weight") and ("gpt" in lk or "transformer" in lk):
                candidates.append((2, k))
        return sorted(candidates, key=lambda x: x[0])[0][1] if candidates else None

    emb_key = _find_embedding_key(list(model_dict.keys()))

    if emb_key:
        emb_weight = model_dict[emb_key]
        vocab_size = emb_weight.shape[0]
        print(f"   Embedding key: {emb_key}")
        print(f"   Text embedding shape: {emb_weight.shape}")
        print(f"   Checkpoint vocab size: {vocab_size}")

        if 'vocab_ext' in locals():
            # Use robust token count (model.vocab + added_tokens)
            def _count_total(tok_json):
                mv = tok_json.get('model', {}).get('vocab', {})
                base = len(mv) if isinstance(mv, (dict, list)) else 0
                added = len(tok_json.get('added_tokens', []))
                return base + added
            expected_size = _count_total(vocab_ext)
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
            ("Hello world", "en", "English"),
            ("ሰላም ዓለም", args.am_lang, "Amharic"),
            ("አንድ ሁለት ሶስት", args.am_lang, "Amharic numbers"),
        ]

        for text, lang, label in test_texts:
            tokens = tokenizer.encode(text, lang)
            print(f"\n   {label}: '{text}'")
            print(f"   Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"   Tokens: {tokens}")
            print(f"   Length: {len(tokens)} tokens")

            # Check for UNK tokens (robustly)
            unk_id = None
            try:
                unk_id = tokenizer.tokenizer.unk_token_id
            except Exception:
                try:
                    unk_id = tokenizer.tokenizer.token_to_id("[UNK]")
                except Exception:
                    unk_id = None
            if unk_id is not None:
                unk_count = sum(1 for t in tokens if t == unk_id)
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
else:
    expected_total = None
    if 'vocab_ext' in locals():
        expected_total, _, _ = _count_tokens(vocab_ext)
    if checkpoint_path.exists() and 'emb_key' in locals() and emb_key and 'vocab_size' in locals() and expected_total is not None and vocab_size != expected_total:
        print("❌ CRITICAL: Checkpoint has wrong vocabulary size!")
        print(f"   Checkpoint: {vocab_size} tokens")
        print(f"   Extended vocab: {expected_total} tokens")
        print("   Solution: Use the checkpoint that matches the extended vocab OR regenerate extended vocab from this checkpoint")
    else:
        print("✅ Vocabulary files look correct")
        print("   Next: Check inference code is loading extended vocab")

print("="*70)
