#!/usr/bin/env python3
"""
Comprehensive Test Suite for Amharic TTS Implementation
Tests all newly implemented Amharic modules and integration points
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all Amharic modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Module Import Tests")
    print("="*60)
    
    results = []
    
    # Test G2P module
    try:
        from amharic_tts.g2p.amharic_g2p import AmharicG2P
        print("✅ AmharicG2P module imported successfully")
        results.append(("G2P Import", True, None))
    except Exception as e:
        print(f"❌ Failed to import AmharicG2P: {e}")
        results.append(("G2P Import", False, str(e)))
    
    # Test text normalizer
    try:
        from amharic_tts.preprocessing.text_normalizer import AmharicTextNormalizer
        print("✅ AmharicTextNormalizer imported successfully")
        results.append(("Text Normalizer Import", True, None))
    except Exception as e:
        print(f"❌ Failed to import AmharicTextNormalizer: {e}")
        results.append(("Text Normalizer Import", False, str(e)))
    
    # Test number expander
    try:
        from amharic_tts.preprocessing.number_expander import AmharicNumberExpander
        print("✅ AmharicNumberExpander imported successfully")
        results.append(("Number Expander Import", True, None))
    except Exception as e:
        print(f"❌ Failed to import AmharicNumberExpander: {e}")
        results.append(("Number Expander Import", False, str(e)))
    
    # Test tokenizer integration
    try:
        from utils.tokenizer import multilingual_cleaners
        print("✅ Tokenizer module imported successfully")
        results.append(("Tokenizer Import", True, None))
    except Exception as e:
        print(f"❌ Failed to import tokenizer: {e}")
        results.append(("Tokenizer Import", False, str(e)))
    
    return results


def test_text_normalization():
    """Test Amharic text normalization"""
    print("\n" + "="*60)
    print("TEST 2: Text Normalization Tests")
    print("="*60)
    
    results = []
    
    try:
        from amharic_tts.preprocessing.text_normalizer import AmharicTextNormalizer
        normalizer = AmharicTextNormalizer()
        
        # Test cases
        test_cases = [
            ("ሀሎ ሰላም", "ሃሎ ሰላም", "Character variant normalization (ሀ→ሃ)"),
            ("ተኛ   መኝታ", "ተኛ መኝታ", "Whitespace normalization"),
            ("ሰላም።", "ሰላም።", "Punctuation handling"),
            ("ዓለም", "አለም", "ዓ→አ normalization"),
        ]
        
        for input_text, expected, description in test_cases:
            result = normalizer.normalize(input_text)
            passed = result == expected
            status = "✅" if passed else "❌"
            print(f"{status} {description}")
            print(f"   Input:    '{input_text}'")
            print(f"   Expected: '{expected}'")
            print(f"   Got:      '{result}'")
            results.append((description, passed, None if passed else f"Expected '{expected}', got '{result}'"))
            
    except Exception as e:
        print(f"❌ Text normalization test failed: {e}")
        results.append(("Text Normalization", False, str(e)))
    
    return results


def test_number_expansion():
    """Test Amharic number expansion"""
    print("\n" + "="*60)
    print("TEST 3: Number Expansion Tests")
    print("="*60)
    
    results = []
    
    try:
        from amharic_tts.preprocessing.number_expander import AmharicNumberExpander
        expander = AmharicNumberExpander()
        
        # Test cases
        test_cases = [
            ("0", "ዜሮ", "Zero"),
            ("1", "አንድ", "One"),
            ("5", "አምስት", "Five"),
            ("10", "አሥር", "Ten"),
            ("42", "አርባ ሁለት", "Forty-two"),
            ("100", "መቶ", "One hundred"),
            ("1000", "ሺህ", "One thousand"),
            ("2024", "ሁለት ሺህ ሃያ አራት", "Year 2024"),
        ]
        
        for input_num, expected, description in test_cases:
            result = expander.expand_number(input_num)
            passed = result == expected
            status = "✅" if passed else "❌"
            print(f"{status} {description}: {input_num} → '{result}'")
            results.append((f"Number Expansion: {description}", passed, 
                          None if passed else f"Expected '{expected}', got '{result}'"))
            
    except Exception as e:
        print(f"❌ Number expansion test failed: {e}")
        results.append(("Number Expansion", False, str(e)))
    
    return results


def test_g2p_conversion():
    """Test G2P conversion with different backends"""
    print("\n" + "="*60)
    print("TEST 4: G2P Conversion Tests")
    print("="*60)
    
    results = []
    
    try:
        from amharic_tts.g2p.amharic_g2p import AmharicG2P
        
        # Test with rule-based backend (should always work)
        print("\nTesting with rule-based backend...")
        g2p = AmharicG2P(backend='rule-based')
        
        test_cases = [
            "ሰላም",
            "ኢትዮጵያ",
            "አማርኛ",
            "መልካም",
        ]
        
        for text in test_cases:
            try:
                phonemes = g2p.convert(text)
                passed = phonemes is not None and len(phonemes) > 0
                status = "✅" if passed else "❌"
                print(f"{status} '{text}' → {phonemes}")
                results.append((f"G2P: {text}", passed, None))
            except Exception as e:
                print(f"❌ Failed to convert '{text}': {e}")
                results.append((f"G2P: {text}", False, str(e)))
        
        # Test backend fallback
        print("\nTesting backend fallback mechanism...")
        g2p_auto = AmharicG2P(backend='auto')
        phonemes = g2p_auto.convert("ሰላም")
        print(f"✅ Backend fallback works: '{phonemes}'")
        results.append(("G2P Backend Fallback", True, None))
        
    except Exception as e:
        print(f"❌ G2P conversion test failed: {e}")
        results.append(("G2P Conversion", False, str(e)))
    
    return results


def test_tokenizer_integration():
    """Test Amharic integration in tokenizer"""
    print("\n" + "="*60)
    print("TEST 5: Tokenizer Integration Tests")
    print("="*60)
    
    results = []
    
    try:
        from utils.tokenizer import multilingual_cleaners
        
        # Test Amharic text processing
        test_cases = [
            ("ሰላም 123 ዓለም", "Amharic text with numbers"),
            ("አማርኛ መልእክት።", "Amharic with punctuation"),
            ("ኢትዮጵያ", "Simple Amharic word"),
        ]
        
        for text, description in test_cases:
            try:
                result = multilingual_cleaners(text, language='amh')
                passed = result is not None
                status = "✅" if passed else "❌"
                print(f"{status} {description}")
                print(f"   Input:  '{text}'")
                print(f"   Output: '{result}'")
                results.append((f"Tokenizer: {description}", passed, None))
            except Exception as e:
                print(f"❌ Failed: {description} - {e}")
                results.append((f"Tokenizer: {description}", False, str(e)))
        
    except Exception as e:
        print(f"❌ Tokenizer integration test failed: {e}")
        results.append(("Tokenizer Integration", False, str(e)))
    
    return results


def test_ui_integration():
    """Test UI integration for Amharic language option"""
    print("\n" + "="*60)
    print("TEST 6: UI Integration Tests")
    print("="*60)
    
    results = []
    
    try:
        # Check xtts_demo.py for Amharic
        demo_path = project_root / "xtts_demo.py"
        if demo_path.exists():
            with open(demo_path, 'r', encoding='utf-8') as f:
                content = f.read()
                has_amh = '"amh"' in content or "'amh'" in content
                status = "✅" if has_amh else "❌"
                print(f"{status} xtts_demo.py contains Amharic language option")
                results.append(("UI: xtts_demo.py", has_amh, 
                              None if has_amh else "Amharic not found in xtts_demo.py"))
        else:
            print("⚠️  xtts_demo.py not found")
            results.append(("UI: xtts_demo.py", False, "File not found"))
        
        # Check headlessXttsTrain.py for Amharic
        headless_path = project_root / "headlessXttsTrain.py"
        if headless_path.exists():
            with open(headless_path, 'r', encoding='utf-8') as f:
                content = f.read()
                has_amh = '"amh"' in content or "'amh'" in content
                status = "✅" if has_amh else "❌"
                print(f"{status} headlessXttsTrain.py contains Amharic language option")
                results.append(("UI: headlessXttsTrain.py", has_amh,
                              None if has_amh else "Amharic not found in headlessXttsTrain.py"))
        else:
            print("⚠️  headlessXttsTrain.py not found")
            results.append(("UI: headlessXttsTrain.py", False, "File not found"))
            
    except Exception as e:
        print(f"❌ UI integration test failed: {e}")
        results.append(("UI Integration", False, str(e)))
    
    return results


def test_module_structure():
    """Test that all expected files and directories exist"""
    print("\n" + "="*60)
    print("TEST 7: Module Structure Tests")
    print("="*60)
    
    results = []
    
    expected_paths = [
        "amharic_tts/__init__.py",
        "amharic_tts/g2p/__init__.py",
        "amharic_tts/g2p/amharic_g2p.py",
        "amharic_tts/tokenizer/__init__.py",
        "amharic_tts/preprocessing/__init__.py",
        "amharic_tts/preprocessing/text_normalizer.py",
        "amharic_tts/preprocessing/number_expander.py",
        "amharic_tts/config/__init__.py",
    ]
    
    for path in expected_paths:
        full_path = project_root / path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {path}")
        results.append((f"File: {path}", exists, None if exists else "File not found"))
    
    return results


def print_summary(all_results):
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(all_results)
    passed = sum(1 for _, result, _ in all_results if result)
    failed = total - passed
    
    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    
    if failed > 0:
        print("\n" + "="*60)
        print("FAILED TESTS:")
        print("="*60)
        for name, result, error in all_results:
            if not result:
                print(f"❌ {name}")
                if error:
                    print(f"   Error: {error}")
    
    return passed == total


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("AMHARIC TTS IMPLEMENTATION TEST SUITE")
    print("="*60)
    print(f"Project Root: {project_root}")
    
    all_results = []
    
    # Run all test suites
    all_results.extend(test_module_structure())
    all_results.extend(test_imports())
    all_results.extend(test_text_normalization())
    all_results.extend(test_number_expansion())
    all_results.extend(test_g2p_conversion())
    all_results.extend(test_tokenizer_integration())
    all_results.extend(test_ui_integration())
    
    # Print summary
    success = print_summary(all_results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
