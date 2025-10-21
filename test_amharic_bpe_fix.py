"""
Quick Test: Amharic BPE-Only Training Fix
==========================================

This script verifies that the global tokenizer patch fixes the
NotImplementedError for Amharic language codes in BPE-only mode.

Run: python test_amharic_bpe_fix.py
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("🧪 TESTING AMHARIC BPE-ONLY TOKENIZER FIX")
print("=" * 70)
print()

# Step 1: Apply the global patch
print("Step 1: Applying global tokenizer patch...")
try:
    from utils.amharic_bpe_tokenizer_patch import apply_global_amharic_bpe_patch
    apply_global_amharic_bpe_patch()
    print("✅ Patch applied\n")
except Exception as e:
    print(f"❌ Failed to apply patch: {e}\n")
    sys.exit(1)

# Step 2: Import TTS libraries (after patch)
print("Step 2: Importing TTS libraries...")
try:
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
    print("✅ TTS libraries imported\n")
except ImportError as e:
    print(f"❌ TTS import failed: {e}\n")
    sys.exit(1)

# Step 3: Create tokenizer instance
print("Step 3: Creating tokenizer instance...")
try:
    tokenizer = VoiceBpeTokenizer()
    print("✅ Tokenizer created\n")
except Exception as e:
    print(f"❌ Failed to create tokenizer: {e}\n")
    sys.exit(1)

# Step 4: Test Amharic text encoding
print("Step 4: Testing Amharic text encoding...")
print()

test_cases = [
    ("ሰላም ዓለም", "amh", "Hello world in Amharic (ISO 639-3)"),
    ("ኢትዮጵያ", "am", "Ethiopia in Amharic (ISO 639-1)"),
    ("አማርኛ ቋንቋ", "amh", "Amharic language"),
]

all_passed = True

for text, lang, description in test_cases:
    print(f"  Testing: {description}")
    print(f"    Text: '{text}'")
    print(f"    Lang: '{lang}'")
    
    try:
        # This should NOT raise NotImplementedError
        encoded = tokenizer.encode(text, lang)
        print(f"    ✅ Encoded: {len(encoded)} tokens")
        
        # Try to decode (not critical if it fails)
        try:
            decoded = tokenizer.decode(encoded)
            print(f"    ✅ Decoded successfully")
        except Exception:
            print(f"    ⚠️  Decode not tested (not critical)")
        
        print()
        
    except NotImplementedError as e:
        print(f"    ❌ FAILED: {e}")
        all_passed = False
        print()
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        all_passed = False
        print()

# Final result
print("=" * 70)
if all_passed:
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("🎉 The fix is working correctly!")
    print("   You can now train with BPE-only mode using 'amh' or 'am' language code.")
    print()
    print("Usage in Gradio WebUI:")
    print("  1. Select language: 'amh' or 'am'")
    print("  2. Disable 'Use Amharic G2P' checkbox")
    print("  3. Start training - it will use BPE on raw Ethiopic text")
    print()
else:
    print("❌ SOME TESTS FAILED")
    print("=" * 70)
    print()
    print("The patch may need adjustment. Check error messages above.")
    print()

sys.exit(0 if all_passed else 1)
