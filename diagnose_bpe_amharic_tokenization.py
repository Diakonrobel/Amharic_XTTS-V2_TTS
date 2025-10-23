#!/usr/bin/env python3
"""
BPE Tokenization Diagnostic for Amharic
========================================

This script analyzes how your BPE tokenizer handles Amharic characters,
specifically the problematic ones: ቀ, ጨ, ፀ

It will show:
1. Whether Amharic chars are in vocabulary
2. How they're being tokenized (whole char vs byte-level)
3. Token ID consistency across multiple runs
4. Suggested fixes for improving consistency

Usage:
    python diagnose_bpe_amharic_tokenization.py
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional


def find_vocab_files():
    """Find all vocab.json files in the project."""
    print("Searching for vocab.json files...")
    vocab_files = []
    
    # Search in common locations
    search_dirs = [
        Path.cwd() / "ready",
        Path.cwd() / "models",
        Path.cwd(),
    ]
    
    # Also check subdirectories
    for root_dir in search_dirs:
        if root_dir.exists():
            for vocab_path in root_dir.rglob("vocab*.json"):
                vocab_files.append(vocab_path)
    
    return vocab_files


def analyze_vocabulary(vocab_path: Path) -> Dict:
    """Analyze vocabulary for Amharic character coverage."""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {vocab_path.name}")
    print(f"{'='*70}")
    
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab_list = vocab_data.get('model', {}).get('vocab', {})
        vocab_size = len(vocab_list)
        
        print(f"\n📊 Vocabulary Size: {vocab_size} tokens")
        
        # Check for Amharic characters
        amharic_chars_found = []
        problem_chars = ['ቀ', 'ጨ', 'ፀ', 'ሰ', 'ለ', 'መ', 'ረ', 'ቸ']
        
        print(f"\n🔍 Checking for Amharic characters in vocabulary...")
        print(f"{'='*70}")
        
        # Convert vocab to string for searching
        if isinstance(vocab_list, dict):
            vocab_str = ''.join(str(v) for v in vocab_list.keys())
        elif isinstance(vocab_list, list):
            vocab_str = ''.join(str(v) for v in vocab_list)
        else:
            vocab_str = str(vocab_list)
        
        # Check specific problem characters
        for char in problem_chars:
            if char in vocab_str:
                amharic_chars_found.append(char)
                print(f"✅ '{char}' - FOUND in vocabulary")
            else:
                print(f"❌ '{char}' - NOT FOUND (will use byte-level encoding)")
        
        # Check broader Ethiopic Unicode range
        ethiopic_count = 0
        for char_code in range(0x1200, 0x1380):
            char = chr(char_code)
            if char in vocab_str:
                ethiopic_count += 1
        
        print(f"\n📈 Total Ethiopic characters in vocab: {ethiopic_count}/384")
        
        result = {
            'path': str(vocab_path),
            'size': vocab_size,
            'has_amharic': len(amharic_chars_found) > 0,
            'amharic_chars_found': amharic_chars_found,
            'ethiopic_coverage': ethiopic_count,
            'coverage_percent': (ethiopic_count / 384) * 100
        }
        
        if ethiopic_count == 0:
            print(f"\n⚠️  WARNING: No Ethiopic characters in vocabulary!")
            print(f"   This means ALL Amharic text uses byte-level encoding")
            print(f"   → Inconsistent tokenization likely!")
        elif ethiopic_count < 100:
            print(f"\n⚠️  WARNING: Only {ethiopic_count} Ethiopic chars in vocab")
            print(f"   → Limited coverage may cause inconsistencies")
        else:
            print(f"\n✅ Good Ethiopic coverage: {ethiopic_count} characters")
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing {vocab_path}: {e}")
        return None


def test_tokenization(vocab_path: Path):
    """Test actual tokenization of problematic characters."""
    
    print(f"\n{'='*70}")
    print(f"TOKENIZATION TEST")
    print(f"{'='*70}")
    
    try:
        # Try to load tokenizer
        print(f"\n🔧 Loading tokenizer from: {vocab_path}")
        
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(vocab_path))
            print(f"✅ Tokenizer loaded successfully")
        except ImportError:
            print(f"❌ tokenizers library not available")
            print(f"   Install: pip install tokenizers")
            return
        except Exception as e:
            print(f"❌ Could not load tokenizer: {e}")
            return
        
        # Test problematic characters
        test_texts = [
            "ቀ",
            "ጨ", 
            "ፀ",
            "ቀጨፀ",
            "ሰላም",
            "ቀን ጨለመ ፀሐይ",
        ]
        
        print(f"\n🧪 Testing tokenization consistency...")
        print(f"{'='*70}")
        
        for text in test_texts:
            print(f"\nText: '{text}'")
            print(f"Unicode: {' '.join(f'U+{ord(c):04X}' for c in text if ord(c) >= 0x1200)}")
            
            # Tokenize multiple times to check consistency
            encodings = []
            for i in range(3):
                encoding = tokenizer.encode(text)
                encodings.append(encoding.ids)
            
            # Check if all encodings are identical
            all_same = all(enc == encodings[0] for enc in encodings)
            
            print(f"Token IDs: {encodings[0]}")
            print(f"Tokens: {tokenizer.encode(text).tokens}")
            print(f"Num tokens: {len(encodings[0])}")
            
            if all_same:
                print(f"✅ Consistent: Same tokenization across 3 runs")
            else:
                print(f"⚠️  INCONSISTENT: Different tokenizations detected!")
                for i, enc in enumerate(encodings):
                    print(f"   Run {i+1}: {enc}")
            
            # Check if using byte-level encoding
            tokens = tokenizer.encode(text).tokens
            byte_level = any(token.startswith('Ġ') or len(token) <= 2 for token in tokens)
            
            if byte_level and any(ord(c) >= 0x1200 for c in text):
                print(f"⚠️  Using BYTE-LEVEL encoding for Amharic chars")
                print(f"   → This can cause inconsistent pronunciation")
            
    except Exception as e:
        print(f"❌ Error during tokenization test: {e}")
        import traceback
        traceback.print_exc()


def suggest_fixes(analysis_results: List[Dict]):
    """Suggest fixes based on analysis."""
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDED FIXES")
    print(f"{'='*70}")
    
    if not analysis_results:
        print(f"\n❌ No vocabulary files analyzed")
        return
    
    # Find the best vocab (highest Ethiopic coverage)
    best_vocab = max(analysis_results, key=lambda x: x.get('ethiopic_coverage', 0) if x else 0)
    
    print(f"\n📋 Current Situation:")
    for result in analysis_results:
        if result:
            print(f"   • {Path(result['path']).name}: {result['ethiopic_coverage']} Ethiopic chars")
    
    if best_vocab['ethiopic_coverage'] == 0:
        print(f"\n🔴 CRITICAL ISSUE: No Amharic characters in vocabulary!")
        print(f"\n💡 SOLUTION 1: Use Extended Vocabulary (Recommended)")
        print(f"   1. Create extended vocab with Amharic characters:")
        print(f"      python rebuild_extended_vocab.py")
        print(f"   2. Retrain model with extended vocab for better results")
        print(f"\n💡 SOLUTION 2: Add Unicode Normalization (Quick Fix)")
        print(f"   Add normalization before inference:")
        print(f"   ```python")
        print(f"   import unicodedata")
        print(f"   text = unicodedata.normalize('NFC', text)")
        print(f"   ```")
        print(f"\n💡 SOLUTION 3: Fine-tune with Character-Level Loss")
        print(f"   Continue training with learning rate 1e-6 for 100-200 steps")
        print(f"   This helps model learn byte-level patterns better")
        
    elif best_vocab['ethiopic_coverage'] < 100:
        print(f"\n🟡 PARTIAL COVERAGE: Only {best_vocab['ethiopic_coverage']} Ethiopic chars")
        print(f"\n💡 RECOMMENDED: Use fully extended vocabulary")
        print(f"   Run: python rebuild_extended_vocab.py")
        print(f"   Then retrain from checkpoint with --vocab vocab_extended.json")
        
    else:
        print(f"\n🟢 GOOD COVERAGE: {best_vocab['ethiopic_coverage']} Ethiopic chars found")
        print(f"\n💡 If still having issues:")
        print(f"   1. Ensure you're using the CORRECT vocab file at inference")
        print(f"      (should match training vocab)")
        print(f"   2. Check checkpoint was trained with this vocab")
        print(f"      Run: python diagnose_amharic_issue.py")
        print(f"   3. Add Unicode normalization as safety measure")
    
    print(f"\n{'='*70}")
    print(f"NEXT STEPS")
    print(f"{'='*70}")
    print(f"\n1. ✅ Run this diagnostic (you just did!)")
    print(f"2. 📝 Note the coverage percentage above")
    print(f"3. 🔧 Apply recommended solution")
    print(f"4. 🧪 Test with: python test_amharic_modes.py")


def main():
    print(f"\n╔{'='*68}╗")
    print(f"║{'BPE TOKENIZATION DIAGNOSTIC FOR AMHARIC'.center(68)}║")
    print(f"╚{'='*68}╝")
    
    # Find vocab files
    vocab_files = find_vocab_files()
    
    if not vocab_files:
        print(f"\n❌ No vocab.json files found!")
        print(f"\n💡 Please provide vocab file path:")
        vocab_path = input(f"   Enter path: ").strip().strip('"')
        if vocab_path and os.path.exists(vocab_path):
            vocab_files = [Path(vocab_path)]
        else:
            print(f"❌ File not found. Exiting.")
            sys.exit(1)
    
    print(f"\n✅ Found {len(vocab_files)} vocabulary file(s)")
    for vf in vocab_files:
        print(f"   • {vf}")
    
    # Analyze each vocabulary
    results = []
    for vocab_path in vocab_files:
        result = analyze_vocabulary(vocab_path)
        if result:
            results.append(result)
    
    # Test tokenization with the first vocab file
    if vocab_files:
        test_tokenization(vocab_files[0])
    
    # Provide recommendations
    suggest_fixes(results)
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
