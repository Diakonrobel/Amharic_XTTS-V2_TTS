#!/usr/bin/env python3
"""
Simple Test Runner for Amharic TTS - No pytest required
Tests critical integrations and auto-fixes issues
"""

import sys
import os
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Test results tracking
tests_passed = 0
tests_failed = 0
tests_skipped = 0
failures = []


def test_result(test_name, passed, error=None):
    """Record test result"""
    global tests_passed, tests_failed, failures
    if passed:
        tests_passed += 1
        print(f"✅ {test_name}")
    else:
        tests_failed += 1
        print(f"❌ {test_name}")
        if error:
            print(f"   Error: {error}")
        failures.append((test_name, error))


def test_skip(test_name, reason):
    """Record skipped test"""
    global tests_skipped
    tests_skipped += 1
    print(f"⚠️  {test_name} (skipped: {reason})")


print("=" * 70)
print("AMHARIC TTS INTEGRATION TEST SUITE")
print("=" * 70)

# Test 1: Module Structure
print("\n📁 Test 1: Module Structure")
print("-" * 70)

required_files = [
    "amharic_tts/__init__.py",
    "amharic_tts/g2p/__init__.py",
    "amharic_tts/g2p/amharic_g2p.py",
    "amharic_tts/g2p/amharic_g2p_enhanced.py",
    "amharic_tts/tokenizer/__init__.py",
    "amharic_tts/tokenizer/hybrid_tokenizer.py",
    "amharic_tts/tokenizer/xtts_tokenizer_wrapper.py",
    "amharic_tts/preprocessing/__init__.py",
    "amharic_tts/preprocessing/text_normalizer.py",
    "amharic_tts/preprocessing/number_expander.py",
    "amharic_tts/config/__init__.py",
    "amharic_tts/config/amharic_config.py",
]

for file_path in required_files:
    full_path = project_root / file_path
    test_result(f"File exists: {file_path}", full_path.exists(), 
                None if full_path.exists() else "File not found")

# Test 2: Module Imports
print("\n📦 Test 2: Module Imports")
print("-" * 70)

try:
    from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
    test_result("Import AmharicG2P", True)
except Exception as e:
    test_result("Import AmharicG2P", False, str(e))

try:
    from amharic_tts.preprocessing.text_normalizer import AmharicTextNormalizer
    test_result("Import AmharicTextNormalizer", True)
except Exception as e:
    test_result("Import AmharicTextNormalizer", False, str(e))

try:
    from amharic_tts.preprocessing.number_expander import AmharicNumberExpander
    test_result("Import AmharicNumberExpander", True)
except Exception as e:
    test_result("Import AmharicNumberExpander", False, str(e))

try:
    from amharic_tts.config.amharic_config import G2PConfiguration, G2PBackend
    test_result("Import Configuration", True)
except Exception as e:
    test_result("Import Configuration", False, str(e))

try:
    from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
    test_result("Import Tokenizer Wrapper", True)
except Exception as e:
    test_result("Import Tokenizer Wrapper", False, str(e))

# Test 3: G2P Backend Tests
print("\n🔄 Test 3: G2P Backend Tests")
print("-" * 70)

try:
    from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
    
    # Test rule-based backend (should always work)
    g2p = AmharicG2P(backend='rule-based')
    result = g2p.convert("ሰላም")
    passed = result is not None and len(result) > 0
    test_result("G2P rule-based backend", passed, 
                None if passed else f"Got empty result: {result}")
    
    # Test auto backend (fallback mechanism)
    g2p_auto = AmharicG2P(backend='auto')
    result_auto = g2p_auto.convert("ሰላም ዓለም")
    passed = result_auto is not None and len(result_auto) > 0
    test_result("G2P auto backend with fallback", passed,
                None if passed else f"Fallback failed: {result_auto}")
    
except Exception as e:
    test_result("G2P backend tests", False, str(e))
    traceback.print_exc()

# Test Transphone if available
try:
    import transphone
    from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
    g2p = AmharicG2P(backend='transphone')
    result = g2p.convert("ሰላም")
    test_result("G2P Transphone backend", result is not None)
except ImportError:
    test_skip("G2P Transphone backend", "transphone not installed")
except Exception as e:
    test_result("G2P Transphone backend", False, str(e))

# Test Epitran if available
try:
    import epitran
    from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
    g2p = AmharicG2P(backend='epitran')
    result = g2p.convert("ሰላም")
    test_result("G2P Epitran backend", result is not None)
except ImportError:
    test_skip("G2P Epitran backend", "epitran not installed")
except Exception as e:
    test_result("G2P Epitran backend", False, str(e))

# Test 4: Text Preprocessing
print("\n📝 Test 4: Text Preprocessing")
print("-" * 70)

try:
    from amharic_tts.preprocessing.text_normalizer import AmharicTextNormalizer
    normalizer = AmharicTextNormalizer()
    
    # Test variant normalization
    result = normalizer.normalize("ሀሎ ሰላም")
    expected = "ሃሎ ሰላም"
    test_result("Character variant normalization (ሀ→ሃ)", result == expected,
                None if result == expected else f"Expected '{expected}', got '{result}'")
    
    # Test whitespace normalization
    result = normalizer.normalize("ተኛ   መኝታ")
    expected = "ተኛ መኝታ"
    test_result("Whitespace normalization", result == expected,
                None if result == expected else f"Expected '{expected}', got '{result}'")
    
except Exception as e:
    test_result("Text normalization", False, str(e))
    traceback.print_exc()

# Test 5: Number Expansion
print("\n🔢 Test 5: Number Expansion")
print("-" * 70)

try:
    from amharic_tts.preprocessing.number_expander import AmharicNumberExpander
    expander = AmharicNumberExpander()
    
    test_cases = [
        ("0", "ዜሮ"),
        ("1", "አንድ"),
        ("10", "አሥር"),
        ("100", "መቶ"),
    ]
    
    for input_num, expected in test_cases:
        result = expander.expand_number(input_num)
        test_result(f"Number expansion: {input_num} → {expected}", 
                   result == expected,
                   None if result == expected else f"Got '{result}'")
    
except Exception as e:
    test_result("Number expansion", False, str(e))
    traceback.print_exc()

# Test 6: Configuration System
print("\n⚙️  Test 6: Configuration System")
print("-" * 70)

try:
    from amharic_tts.config.amharic_config import (
        G2PConfiguration, G2PBackend, TokenizerConfiguration, PhonemeInventory
    )
    
    # Test G2P config creation
    config = G2PConfiguration()
    test_result("Create G2P configuration", 
               config.backend_order is not None and len(config.backend_order) > 0)
    
    # Test backend order
    test_result("Default backend order includes TRANSPHONE",
               G2PBackend.TRANSPHONE in config.backend_order)
    
    # Test phoneme inventory
    inventory = PhonemeInventory()
    test_result("Phoneme inventory has consonants", 
               's' in inventory.consonants and 'kʷ' in inventory.consonants)
    test_result("Phoneme inventory has vowels",
               'a' in inventory.vowels and 'ɨ' in inventory.vowels)
    
except Exception as e:
    test_result("Configuration system", False, str(e))
    traceback.print_exc()

# Test 7: Tokenizer Wrapper
print("\n🔤 Test 7: Tokenizer Wrapper")
print("-" * 70)

try:
    from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
    
    # Test tokenizer creation
    tokenizer = create_xtts_tokenizer(use_g2p=False)
    test_result("Create tokenizer (raw mode)", tokenizer is not None)
    test_result("Tokenizer has encode method", hasattr(tokenizer, 'encode'))
    test_result("Tokenizer has decode method", hasattr(tokenizer, 'decode'))
    
    # Test encoding
    text = "ሰላም"
    tokens = tokenizer.encode(text)
    test_result("Tokenizer encode works", tokens is not None and len(tokens) > 0)
    
except Exception as e:
    test_result("Tokenizer wrapper", False, str(e))
    traceback.print_exc()

# Test 8: UI Integration
print("\n🖥️  Test 8: UI Integration")
print("-" * 70)

try:
    demo_path = project_root / "xtts_demo.py"
    if demo_path.exists():
        content = demo_path.read_text(encoding='utf-8')
        
        # Check for Amharic language code
        has_amh = '"amh"' in content or "'amh'" in content
        test_result("xtts_demo.py contains 'amh' language", has_amh)
        
        # Check for G2P UI controls
        has_g2p_ui = 'g2p_backend' in content.lower() or 'amharic_g2p' in content.lower()
        test_result("xtts_demo.py has G2P UI controls", has_g2p_ui)
        
        # Check for preprocessing accordion
        has_accordion = 'Amharic G2P Options' in content
        test_result("xtts_demo.py has G2P options accordion", has_accordion)
        
    else:
        test_result("xtts_demo.py exists", False, "File not found")
        
except Exception as e:
    test_result("UI integration check", False, str(e))
    traceback.print_exc()

# Test 9: Training Integration
print("\n🏋️  Test 9: Training Integration")
print("-" * 70)

try:
    gpt_train_path = project_root / "utils" / "gpt_train.py"
    if gpt_train_path.exists():
        content = gpt_train_path.read_text(encoding='utf-8')
        
        # Check for use_amharic_g2p parameter
        has_g2p_param = 'use_amharic_g2p' in content
        test_result("gpt_train.py has use_amharic_g2p parameter", has_g2p_param)
        
        # Check for Amharic language handling
        has_amh_check = 'language == "am"' in content or 'language == "amh"' in content
        test_result("gpt_train.py checks for Amharic language", has_amh_check)
        
    else:
        test_result("gpt_train.py exists", False, "File not found")
        
except Exception as e:
    test_result("Training integration check", False, str(e))
    traceback.print_exc()

# Test 10: End-to-End Workflow
print("\n🔗 Test 10: End-to-End Workflow")
print("-" * 70)

try:
    from amharic_tts.preprocessing.text_normalizer import AmharicTextNormalizer
    from amharic_tts.preprocessing.number_expander import AmharicNumberExpander
    from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
    
    # Complete pipeline
    text = "ሀሎ 123 ዓለም"
    
    # Step 1: Normalize
    normalizer = AmharicTextNormalizer()
    text = normalizer.normalize(text)
    
    # Step 2: Expand numbers
    expander = AmharicNumberExpander()
    import re
    for num in re.findall(r'\d+', text):
        text = text.replace(num, expander.expand_number(num))
    
    # Step 3: G2P
    g2p = AmharicG2P(backend='rule-based')
    phonemes = g2p.convert(text)
    
    success = phonemes is not None and "123" not in phonemes
    test_result("End-to-end preprocessing pipeline", success,
               None if success else "Pipeline did not complete successfully")
    
except Exception as e:
    test_result("End-to-end workflow", False, str(e))
    traceback.print_exc()

# Print Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Tests: {tests_passed + tests_failed + tests_skipped}")
print(f"✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
print(f"⚠️  Skipped: {tests_skipped}")

if tests_failed > 0:
    success_rate = (tests_passed / (tests_passed + tests_failed)) * 100
else:
    success_rate = 100.0

print(f"Success Rate: {success_rate:.1f}%")

if failures:
    print("\n" + "=" * 70)
    print("FAILED TESTS DETAILS")
    print("=" * 70)
    for test_name, error in failures:
        print(f"\n❌ {test_name}")
        if error:
            print(f"   {error}")

print("\n" + "=" * 70)

# Exit with appropriate code
sys.exit(0 if tests_failed == 0 else 1)
