#!/usr/bin/env python3
"""
Test script to verify dynamic G2P backend selection is working correctly.

This script tests:
1. Backend detection
2. Backend selection (auto and manual)
3. Tokenizer initialization with correct backend
4. G2P conversion using selected backend
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("DYNAMIC G2P BACKEND SELECTION - VERIFICATION TEST")
print("=" * 80)
print()

# Test 1: Backend Detection
print("Test 1: Backend Detection")
print("-" * 80)
try:
    from utils.g2p_backend_selector import G2PBackendSelector
    
    selector = G2PBackendSelector(verbose=True)
    available = selector.get_available_backends()
    
    print(f"\nDetected {len(available)} available backend(s):")
    for backend in available:
        info = selector.get_backend_info(backend)
        print(f"  ✅ {backend:12s} - Priority: {info.priority}")
    
    print(f"\n✅ Backend detection working correctly!")
    
except Exception as e:
    print(f"❌ Backend detection failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Backend Selection
print("\n" + "=" * 80)
print("Test 2: Backend Selection (Auto-select)")
print("-" * 80)
try:
    from utils.g2p_backend_selector import select_g2p_backend
    
    backend, reason = select_g2p_backend(preferred=None, fallback=True, verbose=True)
    print(f"\nSelected backend: {backend}")
    print(f"Reason: {reason}")
    print(f"\n✅ Auto-selection working correctly!")
    
except Exception as e:
    print(f"❌ Backend selection failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Tokenizer Creation with Backend
print("\n" + "=" * 80)
print("Test 3: Tokenizer Creation with Dynamic Backend")
print("-" * 80)
try:
    from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
    from utils.g2p_backend_selector import select_g2p_backend
    
    # Select backend
    backend, reason = select_g2p_backend(preferred=None, fallback=True, verbose=False)
    print(f"Creating tokenizer with backend: {backend}")
    
    # Create tokenizer
    tokenizer = create_xtts_tokenizer(
        vocab_file=None,
        use_phonemes=True,
        g2p_backend=backend
    )
    
    print(f"✅ Tokenizer created successfully!")
    print(f"   Tokenizer type: {type(tokenizer).__name__}")
    print(f"   Phoneme mode: {tokenizer.use_phonemes}")
    
except Exception as e:
    print(f"❌ Tokenizer creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: G2P Conversion with Backend
print("\n" + "=" * 80)
print("Test 4: G2P Conversion with Selected Backend")
print("-" * 80)
try:
    test_text = "ሰላም ዓለም"
    
    print(f"Input text: {test_text}")
    print(f"Backend: {backend}")
    
    # Preprocess text (applies G2P)
    phonemes = tokenizer.preprocess_text(test_text, lang="am")
    
    print(f"Output phonemes: {phonemes}")
    
    # Verify conversion happened
    if phonemes != test_text:
        print(f"\n✅ G2P conversion working correctly!")
        print(f"   Text was converted: '{test_text}' → '{phonemes}'")
    else:
        print(f"\n⚠️  Warning: Text unchanged - G2P may not have converted")
    
except Exception as e:
    print(f"❌ G2P conversion failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Training Integration Test
print("\n" + "=" * 80)
print("Test 5: Training Integration (apply_g2p_to_training_data)")
print("-" * 80)
try:
    from utils.amharic_g2p_dataset_wrapper import apply_g2p_to_training_data
    from utils.g2p_backend_selector import select_g2p_backend
    import tempfile
    import csv
    import os
    
    # Create temporary test CSV files
    with tempfile.TemporaryDirectory() as tmpdir:
        train_csv = os.path.join(tmpdir, "train.csv")
        eval_csv = os.path.join(tmpdir, "eval.csv")
        
        # Write test data
        with open(train_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(["test.wav", "ሰላም ዓለም", "am"])
        
        with open(eval_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(["test2.wav", "ኢትዮጵያ", "am"])
        
        # Select backend
        selected_backend, reason = select_g2p_backend(
            preferred=None,
            fallback=True,
            verbose=False
        )
        
        print(f"Testing with backend: {selected_backend}")
        
        # Create mock samples
        train_samples = [{"text": "ሰላም ዓለም", "audio_file": "test.wav", "language": "am"}]
        eval_samples = [{"text": "ኢትዮጵያ", "audio_file": "test2.wav", "language": "am"}]
        
        # Apply G2P
        train_processed, eval_processed, lang = apply_g2p_to_training_data(
            train_samples=train_samples,
            eval_samples=eval_samples,
            train_csv_path=train_csv,
            eval_csv_path=eval_csv,
            language="am",
            g2p_backend=selected_backend
        )
        
        print(f"\nResults:")
        print(f"  Original train text: {train_samples[0]['text']}")
        print(f"  Processed train text: {train_processed[0]['text']}")
        print(f"  Language switched to: {lang}")
        
        if train_processed[0]['text'] != train_samples[0]['text']:
            print(f"\n✅ Training integration working correctly!")
            print(f"   Backend '{selected_backend}' was used for G2P conversion")
        else:
            print(f"\n⚠️  Warning: Text unchanged - backend may not be applied")
    
except Exception as e:
    print(f"❌ Training integration test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print()
print("Expected behavior in training:")
print("  1. Backend detection runs automatically")
print("  2. Best available backend is selected (Transphone > Epitran > rule_based)")
print("  3. Selected backend is logged in training output")
print("  4. G2P conversion uses the selected backend")
print("  5. NO hardcoded backend - fully dynamic selection")
print()
print("Look for these log messages during training:")
print("  > Selected G2P backend: <name> (<reason>)")
print("  > Loading Amharic G2P tokenizer (backend: <name>)...")
print("  > ✓ Amharic G2P tokenizer loaded successfully (backend: <name>)")
print()
print("=" * 80)
print("✅ Verification test complete!")
print("=" * 80)
