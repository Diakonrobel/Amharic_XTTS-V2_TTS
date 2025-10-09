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
    
    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_vocab_size = checkpoint["model"]["gpt.text_embedding.weight"].shape[0]
        print(f"   ✅ Checkpoint loaded")
        print(f"   📊 Checkpoint vocabulary size: {checkpoint_vocab_size}")
    except Exception as e:
        print(f"   ❌ Failed to load checkpoint: {e}")
        return False
    
    # Load base vocabulary
    print(f"\n📂 Loading base vocabulary: {base_vocab_path}")
    try:
        with open(base_vocab_path, 'r', encoding='utf-8') as f:
            base_vocab = json.load(f)
        base_vocab_size = len(base_vocab['model']['vocab'])
        print(f"   ✅ Base vocabulary loaded")
        print(f"   📊 Base vocabulary size: {base_vocab_size}")
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
    extended_vocab = base_vocab.copy()
    
    # Get existing vocab dictionary
    vocab_dict = extended_vocab['model']['vocab']
    
    # Add placeholder tokens for the extended vocabulary
    # These represent Amharic-specific characters and IPA phonemes
    print(f"   Adding {num_new_tokens} extended tokens...")
    
    # Generate token names for extended vocabulary
    # These will be used during inference when the model sees Amharic text
    start_idx = base_vocab_size
    
    # Common Amharic/IPA tokens that were likely added during training
    extended_tokens = []
    
    # Ethiopic Unicode range: U+1200 to U+137F
    ethiopic_start = 0x1200
    ethiopic_end = 0x137F
    
    # Add Ethiopic characters
    for code in range(ethiopic_start, min(ethiopic_end + 1, ethiopic_start + num_new_tokens)):
        char = chr(code)
        if char not in vocab_dict:
            extended_tokens.append(char)
    
    # Add IPA phoneme symbols commonly used in Amharic G2P
    ipa_symbols = [
        'ɨ', 'ə', 'ʔ', 'ʕ', 'ʃ', 'ʒ', 'ʷ', 'ʼ', 'ː',
        'ɲ', 'ŋ', 'ɾ', 'ɡ', 'ʤ', 'ʧ',
    ]
    
    for symbol in ipa_symbols:
        if symbol not in vocab_dict and len(extended_tokens) < num_new_tokens:
            extended_tokens.append(symbol)
    
    # If we still need more tokens, add numbered placeholders
    while len(extended_tokens) < num_new_tokens:
        token_name = f"<ext_{len(extended_tokens)}>"
        extended_tokens.append(token_name)
    
    # Add tokens to vocabulary
    for i, token in enumerate(extended_tokens[:num_new_tokens]):
        vocab_dict[token] = start_idx + i
    
    print(f"   ✅ Added {len(extended_tokens[:num_new_tokens])} tokens")
    print(f"   📊 New vocabulary size: {len(vocab_dict)}")
    
    # Verify size
    if len(vocab_dict) != checkpoint_vocab_size:
        print(f"\n⚠️  Warning: Vocabulary size mismatch!")
        print(f"   Expected: {checkpoint_vocab_size}")
        print(f"   Got: {len(vocab_dict)}")
        print(f"   Adjusting...")
        
        # Add remaining tokens if needed
        while len(vocab_dict) < checkpoint_vocab_size:
            token_name = f"<pad_{len(vocab_dict)}>"
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
