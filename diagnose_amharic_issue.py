#!/usr/bin/env python3
"""
Amharic TTS Pronunciation Issue Diagnostics
=============================================

This script diagnoses why Amharic pronunciation is incorrect/nonsense.
It checks for common issues that cause pronunciation problems.
"""

import torch
import json
import os
from pathlib import Path

def diagnose_checkpoint_and_vocab(checkpoint_path, vocab_path):
    """Diagnose vocabulary and checkpoint compatibility issues."""
    
    print("=" * 70)
    print("AMHARIC TTS PRONUNCIATION DIAGNOSTICS")
    print("=" * 70)
    print()
    
    # Load checkpoint
    print("📦 Loading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        print(f"   ✅ Checkpoint loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load checkpoint: {e}")
        return
    
    # Load vocab
    print("\n📚 Loading vocabulary...")
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        vocab_list = vocab_data['model']['vocab']
        print(f"   ✅ Vocabulary loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load vocab: {e}")
        return
    
    # Check vocab size match
    print("\n🔍 ISSUE #1: Vocabulary Size Mismatch Check")
    print("-" * 70)
    
    # Get checkpoint vocab size from embedding weights
    embed_key = None
    for k in state_dict.keys():
        if k.endswith("text_embedding.weight") or ".wte.weight" in k:
            embed_key = k
            break
    
    if embed_key:
        ckpt_vocab_size = state_dict[embed_key].shape[0]
        vocab_file_size = len(vocab_list)
        
        print(f"   Checkpoint vocabulary size: {ckpt_vocab_size}")
        print(f"   Vocab file vocabulary size: {vocab_file_size}")
        
        if ckpt_vocab_size != vocab_file_size:
            print(f"   ❌ MISMATCH DETECTED! ({ckpt_vocab_size} vs {vocab_file_size})")
            print(f"   🔧 This is causing token ID misalignment!")
            print(f"   📝 Solution: You need to retrain with matching vocab")
        else:
            print(f"   ✅ Vocabulary sizes match")
    else:
        print(f"   ⚠️  Could not find embedding weights in checkpoint")
    
    # Check for G2P markers in vocab
    print("\n🔍 ISSUE #2: G2P Phoneme Check")
    print("-" * 70)
    
    ipa_markers = ['ə', 'ɨ', 'ʔ', 'ʕ', 'ʷ', 'ː', 'ʼ', 'ʃ', 'ʧ', 'ʤ', 'ɲ', 'ɡ', 'ʲ']
    amharic_chars = ['ሀ', 'ለ', 'ሐ', 'መ', 'ሰ', 'ረ', 'ሸ', 'ቀ', 'በ', 'ተ', 'ቸ', 'ነ']
    
    has_ipa = any(marker in ''.join(vocab_list) for marker in ipa_markers)
    has_amharic = any(char in ''.join(vocab_list) for char in amharic_chars)
    
    print(f"   IPA phoneme markers in vocab: {'✅ YES' if has_ipa else '❌ NO'}")
    print(f"   Amharic characters in vocab: {'✅ YES' if has_amharic else '❌ NO'}")
    
    if has_ipa and has_amharic:
        print(f"   ℹ️  Vocab contains BOTH phonemes and characters (hybrid)")
        print(f"   📝 Training mode: Likely mixed/hybrid")
    elif has_ipa and not has_amharic:
        print(f"   ℹ️  Vocab contains ONLY phonemes (G2P-only)")
        print(f"   📝 Training mode: Pure G2P phoneme-based")
    elif has_amharic and not has_ipa:
        print(f"   ℹ️  Vocab contains ONLY Amharic characters (no G2P)")
        print(f"   📝 Training mode: Character-based (no G2P)")
    else:
        print(f"   ⚠️  Vocab appears to be neither phoneme nor Amharic")
    
    # Check checkpoint metadata for training config
    print("\n🔍 ISSUE #3: Training Configuration Check")
    print("-" * 70)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"   ✅ Found config in checkpoint")
        print(f"   Config keys: {list(config.keys())[:10]}")
    else:
        print(f"   ⚠️  No config found in checkpoint")
    
    # Check for training metadata
    training_meta = checkpoint.get('training_metadata', {})
    if training_meta:
        print(f"   ✅ Found training metadata:")
        for k, v in list(training_meta.items())[:10]:
            print(f"      {k}: {v}")
    else:
        print(f"   ⚠️  No training metadata found")
    
    # Final diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    if ckpt_vocab_size != vocab_file_size:
        print("\n❌ CRITICAL ISSUE FOUND:")
        print("   The checkpoint and vocab file have different sizes.")
        print("   This causes every token to map to the wrong embedding!")
        print()
        print("🔧 SOLUTION:")
        print("   You MUST retrain the model with the correct vocabulary file.")
        print("   Option 1: Find the original vocab file used during training")
        print(f"            (look for vocab with exactly {ckpt_vocab_size} tokens)")
        print("   Option 2: Retrain from scratch with your current vocab file")
        print(f"            (which has {vocab_file_size} tokens)")
        return False
    
    if has_ipa and has_amharic:
        print("\n⚠️  POTENTIAL ISSUE:")
        print("   Your vocab appears to be hybrid (phonemes + characters).")
        print("   Make sure inference uses the SAME mode as training:")
        print("   - If trained WITH G2P: enable G2P at inference")
        print("   - If trained WITHOUT G2P: disable G2P at inference")
    
    print("\n✅ No critical vocabulary mismatch detected.")
    print("   If pronunciation is still wrong, the issue is likely:")
    print("   1. G2P mode mismatch (training vs inference)")
    print("   2. Insufficient training data or epochs")
    print("   3. Language code mismatch during training")
    
    return True


def check_ready_directory(ready_dir):
    """Check files in ready directory."""
    print("\n🔍 Checking ready directory...")
    print("-" * 70)
    
    ready_path = Path(ready_dir)
    if not ready_path.exists():
        print(f"   ❌ Ready directory not found: {ready_dir}")
        return
    
    files = list(ready_path.glob("*"))
    print(f"   Found {len(files)} files:")
    
    for f in files:
        print(f"      - {f.name}")
        if 'vocab' in f.name.lower():
            # Check vocab size
            try:
                with open(f, 'r', encoding='utf-8') as vf:
                    vocab_data = json.load(vf)
                    vocab_size = len(vocab_data['model']['vocab'])
                    print(f"        Size: {vocab_size} tokens")
            except:
                pass


if __name__ == "__main__":
    import sys
    
    print()
    print("This script helps diagnose Amharic pronunciation issues.")
    print()
    
    # Try to find checkpoint and vocab automatically
    checkpoint_path = input("Enter path to checkpoint file (.pth): ").strip()
    if checkpoint_path.startswith('"') and checkpoint_path.endswith('"'):
        checkpoint_path = checkpoint_path[1:-1]
    
    vocab_path = input("Enter path to vocab file (.json): ").strip()
    if vocab_path.startswith('"') and vocab_path.endswith('"'):
        vocab_path = vocab_path[1:-1]
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not os.path.exists(vocab_path):
        print(f"\n❌ Vocab not found: {vocab_path}")
        sys.exit(1)
    
    # Run diagnostics
    success = diagnose_checkpoint_and_vocab(checkpoint_path, vocab_path)
    
    # Check ready directory if available
    ready_dir = os.path.dirname(vocab_path)
    if os.path.exists(ready_dir):
        check_ready_directory(ready_dir)
    
    print("\n" + "=" * 70)
    if not success:
        print("⚠️  RETRAINING REQUIRED")
        print("=" * 70)
        print("\nRecommended retraining steps:")
        print("1. Ensure you use the same vocab file throughout")
        print("2. Enable/disable G2P consistently (training and inference)")
        print("3. Use language code 'amh' (not 'am')")
        print("4. Train for at least 10-20 epochs with good quality data")
    else:
        print("✅ VOCABULARY COMPATIBLE")
        print("=" * 70)
        print("\nIf pronunciation is still wrong:")
        print("1. Check G2P mode matches between training and inference")
        print("2. Verify you're using language code 'amh' consistently")
        print("3. Try disabling G2P at inference if you didn't use it in training")
    
    print()
