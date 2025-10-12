#!/usr/bin/env python3
"""
Create 7536-token vocabulary from 7537-token vocabulary
========================================================

This removes the last added token to match the checkpoint's expected size.
This is a TEMPORARY workaround - proper solution is to retrain.
"""

import json
import sys
from pathlib import Path

def create_matching_vocab(input_vocab_path, output_vocab_path):
    """Remove the extra token to match checkpoint size."""
    
    print("=" * 70)
    print("Creating 7536-token vocabulary")
    print("=" * 70)
    
    # Load current vocab
    print(f"\n📂 Loading: {input_vocab_path}")
    with open(input_vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    vocab_list = vocab_data['model']['vocab']
    current_size = len(vocab_list)
    
    print(f"   Current size: {current_size} tokens")
    
    if current_size == 7536:
        print("   ✅ Already 7536 tokens! No change needed.")
        return
    
    if current_size != 7537:
        print(f"   ❌ Unexpected size: {current_size}")
        print(f"   Expected 7537. This script is for 7537→7536 conversion.")
        return
    
    # Identify what was added (likely the last token)
    print(f"\n🔍 Last 5 tokens in vocab:")
    for i, token in enumerate(vocab_list[-5:], start=len(vocab_list)-5):
        print(f"   [{i}] {repr(token)}")
    
    # Remove the last token
    new_vocab_list = vocab_list[:-1]
    print(f"\n✂️  Removing last token: {repr(vocab_list[-1])}")
    
    # Create new vocab data
    new_vocab_data = vocab_data.copy()
    new_vocab_data['model']['vocab'] = new_vocab_list
    
    print(f"   New size: {len(new_vocab_list)} tokens")
    
    # Save
    print(f"\n💾 Saving to: {output_vocab_path}")
    with open(output_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(new_vocab_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Saved!")
    
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT NOTES")
    print("=" * 70)
    print("""
This is a TEMPORARY workaround that may or may not work:

Pros:
- Quick to try
- Might restore correct pronunciation if the removed token wasn't used

Cons:
- If the removed token WAS used in training, output will still be wrong
- Not a proper long-term solution
- Future training will need this exact vocab

Next Steps:
1. Use this 7536-token vocab for inference
2. Test again with lang='amh' and G2P enabled
3. If still wrong → proceed to Path B (retrain)

For proper fix: Retrain model with consistent vocab from start to finish.
""")


if __name__ == "__main__":
    print()
    
    # Default paths
    default_input = "vocab_extended_amharic.json"
    default_output = "vocab_7536_compatible.json"
    
    input_path = input(f"Input vocab path [{default_input}]: ").strip() or default_input
    output_path = input(f"Output vocab path [{default_output}]: ").strip() or default_output
    
    # Remove quotes if present
    input_path = input_path.strip('"')
    output_path = output_path.strip('"')
    
    if not Path(input_path).exists():
        print(f"\n❌ File not found: {input_path}")
        sys.exit(1)
    
    create_matching_vocab(input_path, output_path)
    
    print(f"\n📝 To use this vocab:")
    print(f"   1. On your server, replace the vocab file:")
    print(f"      cp {output_path} /path/to/your/ready/vocab.json")
    print(f"   2. Restart the application")
    print(f"   3. Test with lang='amh' and G2P enabled")
    print()
