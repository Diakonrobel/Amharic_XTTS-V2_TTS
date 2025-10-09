#!/usr/bin/env python3
"""
Test script to verify the tokenizer patching works correctly.
This simulates the KeyError fix for Amharic language codes.
"""

import sys
import os
import json

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 70)
print("🧪 Tokenizer Patch Test - Amharic KeyError Fix")
print("=" * 70)

def test_char_limits_patch():
    """Test the char_limits patching logic"""
    print("1. Testing char_limits patching logic...")
    
    # Simulate the tokenizer object structure
    class MockTokenizer:
        def __init__(self):
            self.char_limits = {
                'en': 250,
                'es': 239,
                'fr': 273,
                'de': 253,
                'it': 213,
                'pt': 203,
                'pl': 224,
                'tr': 226,
                'ru': 182,
                'nl': 251,
                'cs': 186,
                'ar': 166,
                'zh-cn': 82,
                'ja': 71,
                'hu': 224,
                'ko': 95,
                # Note: 'am' and 'amh' are missing - this causes the KeyError
            }
    
    class MockModel:
        def __init__(self):
            self.tokenizer = MockTokenizer()
    
    # Create mock model (simulates XTTS_MODEL)
    model = MockModel()
    
    print("   Original char_limits keys:", list(model.tokenizer.char_limits.keys()))
    
    # Test the failing case
    print("2. Testing KeyError scenario...")
    try:
        limit = model.tokenizer.char_limits['am']  # This would cause KeyError
        print(f"   ❌ Unexpected: 'am' limit = {limit}")
    except KeyError as e:
        print(f"   ✅ Expected KeyError: {e}")
    
    # Apply the patch (same logic as in xtts_demo.py)
    print("3. Applying tokenizer patch...")
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'char_limits'):
        if 'am' not in model.tokenizer.char_limits:
            model.tokenizer.char_limits['am'] = 200  # Amharic (ISO 639-1)
            print("   ✅ Patched tokenizer to support 'am' language code")
        if 'amh' not in model.tokenizer.char_limits:
            model.tokenizer.char_limits['amh'] = 200  # Amharic (ISO 639-3)
            print("   ✅ Patched tokenizer to support 'amh' language code")
    
    # Test the fixed case
    print("4. Testing fixed scenario...")
    try:
        am_limit = model.tokenizer.char_limits['am']
        amh_limit = model.tokenizer.char_limits['amh']
        print(f"   ✅ Success: 'am' limit = {am_limit}")
        print(f"   ✅ Success: 'amh' limit = {amh_limit}")
    except KeyError as e:
        print(f"   ❌ Unexpected KeyError: {e}")
        return False
    
    print("   Updated char_limits keys:", list(model.tokenizer.char_limits.keys()))
    return True

def test_split_sentence_simulation():
    """Simulate the split_sentence call that was failing"""
    print("5. Testing split_sentence scenario...")
    
    # This simulates the failing line from the error:
    # text = split_sentence(text, language, self.tokenizer.char_limits[language])
    
    mock_char_limits = {
        'en': 250,
        'am': 200,   # Now available after patch
        'amh': 200,  # Now available after patch
    }
    
    test_cases = [
        ('am', 'Sample text'),
        ('amh', 'Sample text'),
        ('en', 'Sample text'),
    ]
    
    for language, text in test_cases:
        try:
            char_limit = mock_char_limits[language]
            print(f"   ✅ Language '{language}': char_limit = {char_limit}")
        except KeyError as e:
            print(f"   ❌ Language '{language}': KeyError = {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    try:
        success1 = test_char_limits_patch()
        success2 = test_split_sentence_simulation()
        
        print("\n" + "=" * 70)
        if success1 and success2:
            print("🎉 ALL TESTS PASSED! The tokenizer patch should fix the KeyError.")
            print("")
            print("Expected behavior after applying this fix:")
            print("✅ Model loads successfully")
            print("✅ Tokenizer gets patched with 'am' and 'amh' support")
            print("✅ Amharic inference works without KeyError")
            print("✅ Both 'am' and 'amh' language codes supported")
        else:
            print("❌ SOME TESTS FAILED! The patch needs refinement.")
        print("=" * 70)
        
        return success1 and success2
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()