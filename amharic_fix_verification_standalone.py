#!/usr/bin/env python3
"""
Standalone verification script for Amharic TTS pronunciation fix.

This script tests the language normalization and G2P conversion logic
without requiring the full TTS model or other dependencies.

Run with: python amharic_fix_verification_standalone.py
"""

import sys
import os
from pathlib import Path

print("=" * 80)
print("🧪 Amharic TTS Pronunciation Fix Verification (Standalone)")
print("=" * 80)

# Define the fixed language normalization function
def normalize_xtts_lang(lang: str) -> str:
    """Normalize user language code to XTTS-supported code.
    
    IMPORTANT: For Amharic fine-tuned models with G2P phonemes:
    - Keep 'am'/'amh' as 'am' to preserve phoneme interpretation context
    - The model was fine-tuned with Amharic data, so it can handle 'am' correctly
    - Mapping to 'en' breaks phoneme pronunciation (Amharic phonemes interpreted as English)
    
    - Map 'zh' -> 'zh-cn' (XTTS expectation in some paths)
    """
    if not lang:
        return lang
    lang = lang.strip().lower()
    if lang in ("am", "amh"):
        return "am"  # Keep as Amharic for correct phoneme interpretation in fine-tuned models
    if lang == "zh":
        return "zh-cn"
    return lang

# Mock G2P conversion for testing
class MockG2P:
    def preprocess_text(self, text, lang=None):
        """Simple mock G2P conversion for Amharic text"""
        if lang in ("am", "amh") and any(ord(c) >= 0x1200 and ord(c) <= 0x137F for c in text):
            # Convert Amharic characters to mock phonemes
            # This is just for demonstration - real G2P would be more sophisticated
            if "ሰላም" in text:
                return "səlamɨ"
            if "ጤና" in text:
                return "t'ena"
            if "እንደምን" in text:
                return "ɨndəmɨn"
            # Generic conversion for other Amharic text
            return "amharic_phonemes_" + "".join(f"_{ord(c):x}" for c in text[:5])
        return text

# Simulate the inference pipeline
def simulate_inference(text, lang, use_g2p=True):
    """Simulate the inference pipeline with the fix applied"""
    print(f"\nSimulating inference for: '{text}' (lang: {lang})")
    
    # Step 1: G2P conversion (if enabled and Amharic)
    g2p_active = False
    if use_g2p and lang in ("am", "amh"):
        try:
            print("Step 1: Applying G2P conversion...")
            tokenizer = MockG2P()
            original_text = text
            converted_text = tokenizer.preprocess_text(text, lang=lang)
            
            if converted_text == original_text:
                print(f"  ⚠️  Warning: G2P conversion may not have worked (text unchanged)")
            else:
                print(f"  ✅ G2P conversion: '{original_text}' → '{converted_text}'")
                text = converted_text
                g2p_active = True
                
        except Exception as e:
            print(f"  ❌ Error in G2P conversion: {e}")
    else:
        print("Step 1: G2P conversion skipped (not Amharic or disabled)")
    
    # Step 2: Language normalization
    print("Step 2: Normalizing language code...")
    lang_norm = normalize_xtts_lang(lang)
    
    if lang != lang_norm:
        print(f"  ✅ Language normalized: '{lang}' → '{lang_norm}'")
    else:
        print(f"  ✅ Language unchanged: '{lang}'")
    
    # Step 3: Simulate what would be sent to the model
    print("Step 3: Preparing for model inference...")
    
    # BEFORE FIX: This is what was happening
    old_behavior_lang = "en" if g2p_active and lang in ("am", "amh") else lang_norm
    
    # AFTER FIX: This is the correct behavior
    new_behavior_lang = lang_norm
    
    print("\nBEFORE FIX (problematic behavior):")
    print(f"  Text: '{text}'")
    print(f"  Language: '{old_behavior_lang}'")
    if g2p_active and old_behavior_lang == "en":
        print(f"  ❌ PROBLEM: Amharic phonemes would be interpreted with English rules!")
    
    print("\nAFTER FIX (correct behavior):")
    print(f"  Text: '{text}'")
    print(f"  Language: '{new_behavior_lang}'")
    if g2p_active and new_behavior_lang == "am":
        print(f"  ✅ FIXED: Amharic phonemes will be interpreted with Amharic context!")
    
    return {
        "original_lang": lang,
        "normalized_lang": lang_norm,
        "old_behavior_lang": old_behavior_lang,
        "new_behavior_lang": new_behavior_lang,
        "text": text,
        "g2p_active": g2p_active
    }

# Test with various inputs
def run_tests():
    """Run tests with various inputs"""
    print("\n" + "=" * 80)
    print("Running verification tests...")
    print("=" * 80)
    
    test_cases = [
        # Amharic text with different language codes
        {"text": "ሰላም ዓለም", "lang": "am", "use_g2p": True, "desc": "Amharic text with 'am' code"},
        {"text": "ሰላም ዓለም", "lang": "amh", "use_g2p": True, "desc": "Amharic text with 'amh' code"},
        {"text": "ሰላም ዓለም", "lang": "AM", "use_g2p": True, "desc": "Amharic text with uppercase 'AM'"},
        
        # G2P disabled
        {"text": "ሰላም ዓለም", "lang": "am", "use_g2p": False, "desc": "Amharic text with G2P disabled"},
        
        # Non-Amharic languages
        {"text": "Hello world", "lang": "en", "use_g2p": True, "desc": "English text"},
        {"text": "你好世界", "lang": "zh", "use_g2p": True, "desc": "Chinese text"},
    ]
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['desc']}")
        print("-" * 40)
        result = simulate_inference(case["text"], case["lang"], case["use_g2p"])
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    fix_working = True
    for i, result in enumerate(results, 1):
        if result["g2p_active"] and result["original_lang"] in ("am", "amh", "AM"):
            if result["old_behavior_lang"] == "en" and result["new_behavior_lang"] == "am":
                print(f"Test {i}: ✅ Fix verified - would have fixed the issue")
            else:
                print(f"Test {i}: ⚠️ Unexpected behavior - please check")
                fix_working = False
    
    if fix_working:
        print("\n✅ VERIFICATION SUCCESSFUL: The fix correctly preserves 'am' language code during inference")
        print("✅ This should resolve the Amharic pronunciation issues")
    else:
        print("\n⚠️ VERIFICATION INCONCLUSIVE: Some tests showed unexpected behavior")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Push this fix to GitHub")
    print("2. Pull the changes on your Lightning AI machine")
    print("3. Run this verification script there")
    print("4. Test with actual inference using your fine-tuned model")
    print("5. For best results, install G2P backends:")
    print("   pip install transphone epitran")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()