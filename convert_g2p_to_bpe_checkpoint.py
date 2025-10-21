"""
Convert G2P+BPE Checkpoint to BPE-Only Checkpoint
=================================================

This script attempts to convert a checkpoint trained with G2P+BPE
to be compatible with BPE-only mode by:

1. Loading the G2P+BPE checkpoint
2. Creating new BPE-only vocabulary
3. Transferring non-text-embedding weights
4. Initializing new text embeddings (random/knowledge distillation)
5. Saving as a new checkpoint

WARNING: This is experimental and may result in degraded quality!
Consider starting fresh training instead.

Usage:
    python convert_g2p_to_bpe_checkpoint.py \
        --input_checkpoint output/run/training/GPT_XTTS_FT-October-21-2024_12+00AM-abc123/best_model.pth \
        --output_checkpoint output/converted_bpe_only_model.pth \
        --vocab_path base_models/v2.0.2/vocab.json \
        --new_vocab_size 2048
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Tuple


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict, Dict]:
    """
    Load checkpoint and extract state dict
    
    Returns:
        Tuple of (full_checkpoint, model_state_dict)
    """
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    try:
        # PyTorch 2.6+ requires weights_only=False for older checkpoints
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract model state dict (may be nested)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        print(f"✅ Checkpoint loaded")
        print(f"   Keys in checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")
        print(f"   Model state dict keys: {len(state_dict)}")
        
        return checkpoint, state_dict
    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise


def analyze_vocabulary(state_dict: Dict) -> Dict:
    """
    Analyze current vocabulary from embeddings
    
    Returns:
        Dictionary with vocab info
    """
    print(f"\n🔍 Analyzing vocabulary from embeddings...")
    
    # Find text embedding layer
    embed_key = None
    head_w_key = None
    head_b_key = None
    
    for key in state_dict.keys():
        if 'text_embedding.weight' in key:
            embed_key = key
        if 'text_head.weight' in key:
            head_w_key = key
        if 'text_head.bias' in key:
            head_b_key = key
    
    if not embed_key:
        print("❌ Could not find text_embedding.weight in checkpoint")
        print(f"   Available keys: {list(state_dict.keys())[:20]}")
        raise KeyError("text_embedding.weight not found")
    
    old_vocab_size = state_dict[embed_key].shape[0]
    embed_dim = state_dict[embed_key].shape[1]
    
    vocab_info = {
        "embed_key": embed_key,
        "head_w_key": head_w_key,
        "head_b_key": head_b_key,
        "old_vocab_size": old_vocab_size,
        "embed_dim": embed_dim
    }
    
    print(f"✅ Vocabulary analysis complete:")
    print(f"   Old vocab size: {old_vocab_size}")
    print(f"   Embedding dimension: {embed_dim}")
    print(f"   Embedding key: {embed_key}")
    print(f"   Head weight key: {head_w_key}")
    print(f"   Head bias key: {head_b_key}")
    
    return vocab_info


def create_bpe_only_vocabulary(base_vocab_path: str, target_size: int) -> Dict:
    """
    Create a BPE-only vocabulary (without phonemes)
    
    For this conversion, we'll use the base XTTS vocab
    as a starting point since we're converting TO standard BPE.
    """
    print(f"\n📝 Loading base vocabulary: {base_vocab_path}")
    
    try:
        with open(base_vocab_path, 'r', encoding='utf-8') as f:
            base_vocab = json.load(f)
        
        # XTTS vocab.json structure: {"model": {"vocab": {...}}}
        if "model" in base_vocab and "vocab" in base_vocab["model"]:
            vocab_dict = base_vocab["model"]["vocab"]
        else:
            vocab_dict = base_vocab
        
        vocab_size = len(vocab_dict)
        print(f"✅ Base vocabulary loaded: {vocab_size} tokens")
        
        return {
            "vocab_dict": vocab_dict,
            "vocab_size": vocab_size,
            "source": base_vocab_path
        }
    
    except Exception as e:
        print(f"❌ Error loading vocabulary: {e}")
        raise


def transfer_weights(
    old_state_dict: Dict,
    vocab_info: Dict,
    new_vocab_size: int,
    transfer_strategy: str = "random"
) -> Dict:
    """
    Transfer weights from G2P+BPE to BPE-only
    
    Strategies:
    - "random": Initialize new embeddings randomly (safest)
    - "copy_partial": Copy embeddings for overlapping tokens (experimental)
    - "mean": Initialize new embeddings as mean of old embeddings (experimental)
    """
    print(f"\n🔄 Transferring weights (strategy: {transfer_strategy})...")
    
    new_state_dict = {}
    
    embed_key = vocab_info["embed_key"]
    head_w_key = vocab_info["head_w_key"]
    head_b_key = vocab_info["head_b_key"]
    old_vocab_size = vocab_info["old_vocab_size"]
    embed_dim = vocab_info["embed_dim"]
    
    # Copy all non-text-embedding weights
    for key, value in old_state_dict.items():
        if 'text_embedding' not in key and 'text_head' not in key:
            new_state_dict[key] = value.clone()
    
    print(f"   ✅ Copied {len(new_state_dict)} non-embedding layers")
    
    # Create new text embeddings
    print(f"   Creating new text embeddings ({old_vocab_size} → {new_vocab_size})...")
    
    if transfer_strategy == "random":
        # Initialize randomly (safest, model will learn from scratch)
        new_text_embedding = torch.randn(new_vocab_size, embed_dim) * 0.02
        new_text_head_weight = torch.randn(new_vocab_size, embed_dim) * 0.02
        new_text_head_bias = torch.zeros(new_vocab_size)
        print(f"   ✅ Initialized embeddings randomly (will learn from scratch)")
    
    elif transfer_strategy == "mean":
        # Initialize as mean of old embeddings (may help with initialization)
        mean_embedding = old_state_dict[embed_key].mean(dim=0)
        new_text_embedding = mean_embedding.unsqueeze(0).expand(new_vocab_size, -1).clone()
        new_text_embedding += torch.randn(new_vocab_size, embed_dim) * 0.01  # Add noise
        
        mean_head = old_state_dict[head_w_key].mean(dim=0)
        new_text_head_weight = mean_head.unsqueeze(0).expand(new_vocab_size, -1).clone()
        new_text_head_weight += torch.randn(new_vocab_size, embed_dim) * 0.01
        
        new_text_head_bias = torch.zeros(new_vocab_size)
        print(f"   ✅ Initialized embeddings from mean (may converge faster)")
    
    elif transfer_strategy == "copy_partial":
        # Copy what we can, random init rest (experimental)
        min_vocab = min(old_vocab_size, new_vocab_size)
        
        new_text_embedding = torch.randn(new_vocab_size, embed_dim) * 0.02
        new_text_embedding[:min_vocab] = old_state_dict[embed_key][:min_vocab].clone()
        
        new_text_head_weight = torch.randn(new_vocab_size, embed_dim) * 0.02
        new_text_head_weight[:min_vocab] = old_state_dict[head_w_key][:min_vocab].clone()
        
        new_text_head_bias = torch.zeros(new_vocab_size)
        new_text_head_bias[:min_vocab] = old_state_dict[head_b_key][:min_vocab].clone()
        
        print(f"   ✅ Copied {min_vocab} embeddings, random init rest")
    
    else:
        raise ValueError(f"Unknown strategy: {transfer_strategy}")
    
    # Add new embeddings to state dict
    new_state_dict[embed_key] = new_text_embedding
    new_state_dict[head_w_key] = new_text_head_weight
    new_state_dict[head_b_key] = new_text_head_bias
    
    print(f"   ✅ New embeddings created:")
    print(f"      text_embedding: {new_text_embedding.shape}")
    print(f"      text_head: {new_text_head_weight.shape}")
    
    return new_state_dict


def save_converted_checkpoint(
    output_path: str,
    new_state_dict: Dict,
    original_checkpoint: Dict,
    metadata: Dict
):
    """Save converted checkpoint with metadata"""
    print(f"\n💾 Saving converted checkpoint: {output_path}")
    
    # Create output checkpoint structure
    if isinstance(original_checkpoint, dict):
        output_checkpoint = original_checkpoint.copy()
        output_checkpoint["model"] = new_state_dict
        
        # Add conversion metadata
        output_checkpoint["conversion_metadata"] = {
            "converted_from": "g2p+bpe",
            "converted_to": "bpe_only",
            "original_vocab_size": metadata.get("old_vocab_size", "unknown"),
            "new_vocab_size": metadata.get("new_vocab_size", "unknown"),
            "transfer_strategy": metadata.get("strategy", "unknown"),
            "warning": "This checkpoint was converted. Quality may be degraded. Consider retraining."
        }
    else:
        output_checkpoint = new_state_dict
    
    # Save checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(output_checkpoint, output_path)
    
    print(f"✅ Checkpoint saved successfully")
    print(f"   Path: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / (1024**2):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert G2P+BPE checkpoint to BPE-only")
    parser.add_argument("--input_checkpoint", required=True, help="Path to G2P+BPE checkpoint")
    parser.add_argument("--output_checkpoint", required=True, help="Path to save converted checkpoint")
    parser.add_argument("--vocab_path", required=True, help="Path to base vocab.json (BPE-only)")
    parser.add_argument("--new_vocab_size", type=int, default=None, help="New vocabulary size (auto-detect if not specified)")
    parser.add_argument("--strategy", choices=["random", "mean", "copy_partial"], default="mean", 
                       help="Weight transfer strategy (default: mean)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔄 G2P+BPE → BPE-ONLY CHECKPOINT CONVERTER")
    print("=" * 70)
    print()
    print("⚠️  WARNING: This is experimental!")
    print("   Converted checkpoints may have degraded quality.")
    print("   Consider training from scratch for best results.")
    print()
    print("=" * 70)
    
    # Step 1: Load checkpoint
    checkpoint, state_dict = load_checkpoint(args.input_checkpoint)
    
    # Step 2: Analyze vocabulary
    vocab_info = analyze_vocabulary(state_dict)
    
    # Step 3: Load BPE-only vocabulary
    bpe_vocab = create_bpe_only_vocabulary(args.vocab_path, args.new_vocab_size)
    new_vocab_size = args.new_vocab_size or bpe_vocab["vocab_size"]
    
    # Step 4: Transfer weights
    new_state_dict = transfer_weights(
        state_dict,
        vocab_info,
        new_vocab_size,
        args.strategy
    )
    
    # Step 5: Save converted checkpoint
    metadata = {
        "old_vocab_size": vocab_info["old_vocab_size"],
        "new_vocab_size": new_vocab_size,
        "strategy": args.strategy
    }
    save_converted_checkpoint(
        args.output_checkpoint,
        new_state_dict,
        checkpoint,
        metadata
    )
    
    print()
    print("=" * 70)
    print("✅ CONVERSION COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Copy standard BPE vocab to your model directory:")
    print(f"   cp {args.vocab_path} output/ready/vocab.json")
    print()
    print("2. Fine-tune the converted checkpoint with BPE-only mode:")
    print(f"   python utils/gpt_train.py \\")
    print(f"       --custom_model {args.output_checkpoint} \\")
    print(f"       --use_amharic_g2p False \\")
    print(f"       --epochs 5  # Fine-tune for a few epochs")
    print()
    print("3. Test thoroughly - quality may be degraded!")
    print()


if __name__ == "__main__":
    main()
