"""
Extract Extended Vocabulary from Checkpoint

This script creates a vocabulary file (vocab_extended.json) from a trained
checkpoint that was created with extended vocabulary (Amharic G2P).

Usage:
    python utils/extract_vocab_from_checkpoint.py \
        --checkpoint path/to/model.pth \
        --base_vocab path/to/base_vocab.json \
        --output path/to/vocab_extended.json
"""

import torch
import json
import argparse
from pathlib import Path
import sys


def extract_vocab_from_checkpoint(checkpoint_path, base_vocab_path, output_path):
    """
    Extract extended vocabulary from checkpoint
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        base_vocab_path: Path to base vocabulary file
        output_path: Path to save extended vocabulary
    """
    print("=" * 70)
    print("🔧 Extended Vocabulary Extraction Tool")
    print("=" * 70)
    
    # Load checkpoint (robust to different layouts)
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Determine model dict
        if isinstance(ckpt, dict) and "model" in ckpt:
            model_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model_dict = ckpt["state_dict"]
        else:
            model_dict = ckpt
        
        # Find embedding key across known variants
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
        if emb_key is None:
            print("   ❌ Could not locate text embedding weight in checkpoint")
            return False
        
        emb_weight = model_dict[emb_key]
        checkpoint_vocab_size = emb_weight.shape[0]
        print(f"   ✅ Checkpoint loaded")
        print(f"   🔎 Embedding key: {emb_key}")
        print(f"   📊 Checkpoint vocabulary size: {checkpoint_vocab_size}")
    except Exception as e:
        print(f"   ❌ Failed to load checkpoint: {e}")
        return False
    
    # Load base vocabulary (normalize schema)
    print(f"\n📂 Loading base vocabulary: {base_vocab_path}")
    try:
        with open(base_vocab_path, 'r', encoding='utf-8') as f:
            base_vocab = json.load(f)
        mv = base_vocab.get('model', {}).get('vocab', {})
        if isinstance(mv, list):
            mv = {tok: idx for idx, tok in enumerate(mv)}
            base_vocab['model']['vocab'] = mv
        elif not isinstance(mv, dict):
            print("   ❌ Unexpected model.vocab type in base vocab")
            return False
        base_added = len(base_vocab.get('added_tokens', []))
        base_vocab_size = len(mv) + base_added
        print(f"   ✅ Base vocabulary loaded")
        print(f"   📊 Base vocabulary size: {base_vocab_size} (base={len(mv)}, added={base_added})")
    except Exception as e:
        print(f"   ❌ Failed to load base vocabulary: {e}")
        return False
    
    # Check if extension is needed
    if checkpoint_vocab_size == base_vocab_size:
        print(f"\n✅ Vocabulary sizes match - no extension needed")
        print(f"   Checkpoint uses standard vocabulary")
        # Just copy base vocab to output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(base_vocab, f, ensure_ascii=False, indent=2)
        print(f"   💾 Standard vocabulary saved to: {output_path}")
        return True
    
    # Calculate number of new tokens
    num_new_tokens = checkpoint_vocab_size - base_vocab_size
    print(f"\n⚠️  Vocabulary size mismatch detected!")
    print(f"   Base vocabulary: {base_vocab_size} tokens")
    print(f"   Checkpoint vocabulary: {checkpoint_vocab_size} tokens")
    print(f"   New tokens to add: {num_new_tokens}")
    
    # Create extended vocabulary
    print(f"\n🔨 Creating extended vocabulary...")
    import copy
    extended_vocab = copy.deepcopy(base_vocab)
    
    # Get existing vocab dictionary
    vocab_dict = extended_vocab['model']['vocab']
    
    print(f"   Adding {num_new_tokens} extended tokens...")
    start_idx = max(int(i) for i in vocab_dict.values()) + 1 if vocab_dict else 0
    
    # Generate Ethiopic character set (comprehensive)
    ethiopic_chars = []
    ethiopic_chars += [chr(cp) for cp in range(0x1200, 0x1380)]  # Main
    ethiopic_chars += [chr(cp) for cp in range(0x1380, 0x13A0)]  # Supplement
    ethiopic_chars += [chr(cp) for cp in range(0x2D80, 0x2DE0)]  # Extended
    ethiopic_chars += [chr(cp) for cp in range(0xAB00, 0xAB30)]  # Extended-A
    
    # IPA symbols commonly used in Amharic G2P
    ipa_symbols = ['ɨ','ə','ʔ','ʕ','ʃ','ʒ','ʷ','ʼ','ː','ɲ','ŋ','ɾ','ɡ','ʤ','ʧ']
    
    extended_tokens = []
    # Prioritize Ethiopic
    for ch in ethiopic_chars:
        if ch not in vocab_dict and len(extended_tokens) < num_new_tokens:
            extended_tokens.append(ch)
    # Then IPA
    for sym in ipa_symbols:
        if sym not in vocab_dict and len(extended_tokens) < num_new_tokens:
            extended_tokens.append(sym)
    # Fill remaining
    while len(extended_tokens) < num_new_tokens:
        token_name = f"<ext_{len(extended_tokens)}>"
        if token_name not in vocab_dict:
            extended_tokens.append(token_name)
    
    # Add tokens to vocabulary with sequential ids starting at start_idx
    for i, token in enumerate(extended_tokens):
        vocab_dict[token] = start_idx + i
    
    print(f"   ✅ Added {len(extended_tokens[:num_new_tokens])} tokens")
    print(f"   📊 New vocabulary size: {len(vocab_dict)}")
    
    # Verify size and adjust if needed
    if len(vocab_dict) != checkpoint_vocab_size:
        print(f"\n⚠️  Warning: Vocabulary size mismatch!")
        print(f"   Expected: {checkpoint_vocab_size}")
        print(f"   Got: {len(vocab_dict)}")
        print(f"   Adjusting...")
        
        while len(vocab_dict) < checkpoint_vocab_size:
            token_name = f"<pad_{len(vocab_dict)}>"
            if token_name not in vocab_dict:
                vocab_dict[token_name] = len(vocab_dict)
    
    # Save extended vocabulary
    print(f"\n💾 Saving extended vocabulary...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extended_vocab, f, ensure_ascii=False, indent=2)
        print(f"   ✅ Extended vocabulary saved to: {output_path}")
        print(f"   📊 Final size: {len(vocab_dict)} tokens")
    except Exception as e:
        print(f"   ❌ Failed to save vocabulary: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Extended Vocabulary Created Successfully!")
    print("=" * 70)
    print(f"\n📄 Output file: {output_path}")
    print(f"📊 Vocabulary size: {len(vocab_dict)} tokens")
    print(f"🆕 Extended tokens: {num_new_tokens}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Use this vocabulary file for inference")
    print(f"   2. Load model with: vocab_path={output_path}")
    print(f"   3. Enable Amharic G2P during inference")
    print("=" * 70 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract extended vocabulary from trained checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--base_vocab",
        type=str,
        required=True,
        help="Path to base vocabulary file (vocab.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save extended vocabulary (vocab_extended.json)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    checkpoint_path = Path(args.checkpoint)
    base_vocab_path = Path(args.base_vocab)
    output_path = Path(args.output)
    
    # Validate inputs
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not base_vocab_path.exists():
        print(f"❌ Error: Base vocabulary not found: {base_vocab_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract vocabulary
    success = extract_vocab_from_checkpoint(
        checkpoint_path,
        base_vocab_path,
        output_path
    )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
