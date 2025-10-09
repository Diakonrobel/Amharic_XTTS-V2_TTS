#!/usr/bin/env python3
"""
Core Tests for Amharic Language Normalization Fix

Tests the language normalization function and proposes fixes.
This is a focused test that doesn't require external dependencies.
"""

import sys
import os
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def normalize_xtts_lang_current(lang: str) -> str:
    """Current problematic implementation from xtts_demo.py"""
    if not lang:
        return lang
    lang = lang.strip().lower()
    if lang in ("am", "amh"):
        return "en"  # PROBLEM: Maps Amharic to English!
    if lang == "zh":
        return "zh-cn"
    return lang


def normalize_xtts_lang_fixed(lang: str) -> str:
    """Fixed implementation for Amharic G2P inference"""
    if not lang:
        return lang
    lang = lang.strip().lower()
    
    # For Amharic with G2P phonemes, preserve the language context
    # Don't map to English as it breaks phoneme interpretation
    if lang in ("am", "amh"):
        return "am"  # Keep as Amharic for phoneme-based inference
    
    if lang == "zh":
        return "zh-cn"
    return lang


class TestLanguageNormalizationFix(unittest.TestCase):
    """Test the language normalization fix"""
    
    def test_current_problematic_behavior(self):
        """Test current problematic behavior"""
        result_am = normalize_xtts_lang_current("am")
        result_amh = normalize_xtts_lang_current("amh")
        
        # Current behavior - maps to English (WRONG for G2P phonemes!)
        self.assertEqual(result_am, "en")
        self.assertEqual(result_amh, "en")
        
        print("❌ Current behavior: am → en, amh → en (problematic for phoneme inference)")
    
    def test_fixed_behavior(self):
        """Test fixed behavior preserves Amharic language context"""
        result_am = normalize_xtts_lang_fixed("am")
        result_amh = normalize_xtts_lang_fixed("amh")
        
        # Fixed behavior - preserves Amharic context
        self.assertEqual(result_am, "am")
        self.assertEqual(result_amh, "am")
        
        print("✅ Fixed behavior: am → am, amh → am (preserves phoneme context)")
    
    def test_other_languages_unchanged(self):
        """Test that other languages are not affected by the fix"""
        test_langs = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja"]
        
        for lang in test_langs:
            current = normalize_xtts_lang_current(lang)
            fixed = normalize_xtts_lang_fixed(lang)
            
            if lang == "zh":
                self.assertEqual(current, "zh-cn")
                self.assertEqual(fixed, "zh-cn")
            else:
                self.assertEqual(current, lang)
                self.assertEqual(fixed, lang)
        
        print("✅ Other languages remain unchanged")
    
    def test_edge_cases(self):
        """Test edge cases like None, empty, whitespace"""
        edge_cases = [None, "", "  ", "\t", "\n"]
        
        for case in edge_cases:
            current = normalize_xtts_lang_current(case)
            fixed = normalize_xtts_lang_fixed(case)
            self.assertEqual(current, fixed)
        
        print("✅ Edge cases handled consistently")
    
    def test_case_insensitive(self):
        """Test case insensitive handling"""
        variants = ["AM", "am", "Am", "AMH", "amh", "Amh"]
        
        for variant in variants:
            current = normalize_xtts_lang_current(variant)
            fixed = normalize_xtts_lang_fixed(variant)
            
            self.assertEqual(current, "en")  # Current maps to English
            self.assertEqual(fixed, "am")    # Fixed preserves as Amharic
        
        print("✅ Case insensitive handling works")


class TestInferencePipelineAnalysis(unittest.TestCase):
    """Analyze the inference pipeline issues"""
    
    def test_inference_pipeline_problem_analysis(self):
        """Analyze what goes wrong in the current pipeline"""
        
        print("\n" + "="*80)
        print("AMHARIC INFERENCE PIPELINE PROBLEM ANALYSIS")
        print("="*80)
        
        # Simulate the problematic flow
        original_text = "ሰላም ዓለም"
        lang = "am"
        
        print(f"1. User input: '{original_text}' (lang: {lang})")
        print(f"2. G2P converts to: 'səlamɨ ʔələmɨ' (Amharic phonemes)")
        
        # The problem happens here:
        normalized_lang = normalize_xtts_lang_current(lang)
        print(f"3. Language normalization: '{lang}' → '{normalized_lang}' ❌")
        
        print(f"4. Model receives:")
        print(f"   - Text: 'səlamɨ ʔələmɨ' (Amharic phonemes)")
        print(f"   - Language: '{normalized_lang}' (English!)")
        print(f"   - Result: Model interprets Amharic phonemes as English → BAD")
        
        print(f"\nFixed pipeline:")
        fixed_lang = normalize_xtts_lang_fixed(lang)
        print(f"3. Fixed normalization: '{lang}' → '{fixed_lang}' ✅")
        print(f"4. Model receives:")
        print(f"   - Text: 'səlamɨ ʔələmɨ' (Amharic phonemes)")
        print(f"   - Language: '{fixed_lang}' (Amharic)")
        print(f"   - Result: Model correctly interprets as Amharic phonemes → GOOD")
        
        print("="*80)
        
        # Verify the fix
        self.assertEqual(normalized_lang, "en")  # Current problem
        self.assertEqual(fixed_lang, "am")       # Fixed behavior
    
    def test_why_current_mapping_is_wrong(self):
        """Explain why mapping am->en is wrong for G2P inference"""
        
        print("\nWhy mapping 'am' → 'en' breaks Amharic G2P inference:")
        print("1. Amharic text gets converted to IPA phonemes: ሰላም → səlamɨ")
        print("2. These phonemes are Amharic-specific (e.g., ɨ is common in Amharic)")
        print("3. Mapping language to 'en' tells model to interpret as English")
        print("4. Model applies English pronunciation rules to Amharic phonemes")
        print("5. Result: Incorrect pronunciation and poor audio quality")
        
        print("\nCorrect approach:")
        print("1. Keep language as 'am' to preserve context")
        print("2. Model (fine-tuned on Amharic) correctly handles 'am' + phonemes")
        print("3. Result: Accurate Amharic pronunciation")
        
        # The mapping should preserve language context for fine-tuned models
        self.assertTrue(True)  # This is an explanatory test


if __name__ == "__main__":
    print("Testing Amharic Language Normalization Fix")
    print("=" * 80)
    unittest.main(verbosity=2)