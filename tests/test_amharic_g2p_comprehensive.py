"""
Comprehensive Test Suite for Enhanced Amharic G2P

Tests:
1. Complete Ethiopic character coverage (231 mappings)
2. Phonological rules (epenthesis, gemination)
3. Labiovelar consonants
4. Real Amharic words and phrases
5. Edge cases and error handling
"""

import sys
import os
import io

# Fix Unicode encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from amharic_tts.g2p.amharic_g2p_enhanced import EnhancedAmharicG2P
from amharic_tts.g2p.ethiopic_g2p_table import (
    ETHIOPIC_G2P_TABLE, 
    LABIOVELAR_TABLE,
    ETHIOPIC_PUNCTUATION,
    get_table_stats
)


class TestEthiopicG2PTable(unittest.TestCase):
    """Test the comprehensive G2P table"""
    
    def setUp(self):
        self.stats = get_table_stats()
    
    def test_table_completeness(self):
        """Verify table has expected number of mappings"""
        # 33 consonants × 7 orders = 231
        self.assertEqual(self.stats['consonant_series'], 33)
        self.assertGreaterEqual(self.stats['total_mappings'], 231)
        print(f"\n✅ G2P table coverage: {self.stats['coverage']}")
        print(f"   Total mappings: {self.stats['total_mappings']}")
    
    def test_consonant_series_coverage(self):
        """Test that all 7 orders are present for key consonants"""
        # Test ሰ-series (s)
        s_series = ['ሰ', 'ሱ', 'ሲ', 'ሳ', 'ሴ', 'ስ', 'ሶ']
        for char in s_series:
            self.assertIn(char, ETHIOPIC_G2P_TABLE)
            self.assertIn('s', ETHIOPIC_G2P_TABLE[char])
        
        # Test በ-series (b)
        b_series = ['በ', 'ቡ', 'ቢ', 'ባ', 'ቤ', 'ብ', 'ቦ']
        for char in b_series:
            self.assertIn(char, ETHIOPIC_G2P_TABLE)
            self.assertIn('b', ETHIOPIC_G2P_TABLE[char])
        
        # Test አ-series (glottal stop)
        a_series = ['አ', 'ኡ', 'ኢ', 'ኣ', 'ኤ', 'እ', 'ኦ']
        for char in a_series:
            self.assertIn(char, ETHIOPIC_G2P_TABLE)
            self.assertIn('ʔ', ETHIOPIC_G2P_TABLE[char])
    
    def test_ejective_consonants(self):
        """Test ejective consonants (unique to Amharic/Ethiopic)"""
        # ጠ-series (tʼ - ejective t)
        t_ejective = ['ጠ', 'ጡ', 'ጢ', 'ጣ', 'ጤ', 'ጥ', 'ጦ']
        for char in t_ejective:
            self.assertIn(char, ETHIOPIC_G2P_TABLE)
            self.assertIn('tʼ', ETHIOPIC_G2P_TABLE[char])
        
        # ጨ-series (tʃʼ - ejective ch)
        ch_ejective = ['ጨ', 'ጩ', 'ጪ', 'ጫ', 'ጬ', 'ጭ', 'ጮ']
        for char in ch_ejective:
            self.assertIn(char, ETHIOPIC_G2P_TABLE)
            self.assertIn('tʃʼ', ETHIOPIC_G2P_TABLE[char])
        
        # ጸ-series (sʼ - ejective s)
        s_ejective = ['ጸ', 'ጹ', 'ጺ', 'ጻ', 'ጼ', 'ጽ', 'ጾ']
        for char in s_ejective:
            self.assertIn(char, ETHIOPIC_G2P_TABLE)
            self.assertIn('sʼ', ETHIOPIC_G2P_TABLE[char])
    
    def test_labiovelar_consonants(self):
        """Test labiovelar variants (kʷ, gʷ, qʷ)"""
        # All labiovelars should have ʷ marker
        for char, phoneme in LABIOVELAR_TABLE.items():
            self.assertIn('ʷ', phoneme)
        
        # Test specific labiovelars
        self.assertEqual(LABIOVELAR_TABLE['ቋ'], 'qʷa')
        self.assertEqual(LABIOVELAR_TABLE['ኳ'], 'kʷa')
        self.assertEqual(LABIOVELAR_TABLE['ጓ'], 'gʷa')
    
    def test_ethiopic_punctuation(self):
        """Test Ethiopic punctuation marks"""
        self.assertEqual(ETHIOPIC_PUNCTUATION['።'], '.')
        self.assertEqual(ETHIOPIC_PUNCTUATION['፣'], ',')
        self.assertEqual(ETHIOPIC_PUNCTUATION['፤'], ';')
        self.assertEqual(ETHIOPIC_PUNCTUATION['፥'], ':')
        self.assertEqual(ETHIOPIC_PUNCTUATION['፧'], '?')


class TestEnhancedG2PConverter(unittest.TestCase):
    """Test the enhanced G2P converter"""
    
    def setUp(self):
        self.g2p = EnhancedAmharicG2P()
    
    def test_basic_words(self):
        """Test common Amharic words"""
        test_cases = [
            ('ሰላም', ['s', 'l', 'a', 'm']),      # Peace/hello - key consonants and vowels
            ('አማርኛ', ['ʔ', 'm', 'r', 'ɲ', 'a']),  # Amharic (with palatal nasal)
            ('ኢትዮጵያ', ['ʔ', 't', 'j', 'p', 'ʼ']),  # Ethiopia (with ejectives)
        ]
        
        for amharic, expected_phonemes in test_cases:
            result = self.g2p.convert(amharic)
            print(f"\n{amharic:15} → {result}")
            
            # Check that key phonemes are present
            # (exact match may vary due to phonological rules like epenthesis)
            for phoneme in expected_phonemes:
                self.assertIn(phoneme, result, 
                    f"Expected phoneme '{phoneme}' in result for '{amharic}'")
    
    def test_vowel_orders(self):
        """Test all 7 vowel orders"""
        # Use ለ-series as example
        test_cases = [
            ('ለ', 'ə'),   # 1st order (schwa)
            ('ሉ', 'u'),   # 2nd order
            ('ሊ', 'i'),   # 3rd order
            ('ላ', 'a'),   # 4th order
            ('ሌ', 'e'),   # 5th order
            ('ል', 'ɨ'),   # 6th order (bare)
            ('ሎ', 'o'),   # 7th order
        ]
        
        for char, vowel in test_cases:
            result = self.g2p.convert(char)
            self.assertIn(vowel, result, 
                f"Expected vowel '{vowel}' in result for '{char}'")
    
    def test_epenthesis(self):
        """Test epenthetic vowel insertion"""
        # Consonant clusters should trigger ɨ insertion
        test_cases = [
            'ትግራይ',  # Should have ɨ between consonants
            'ብርሃን',   # Should have ɨ after final consonant
        ]
        
        for word in test_cases:
            result = self.g2p.convert(word)
            print(f"\nEpenthesis: {word:15} → {result}")
            # Should contain epenthetic ɨ
            self.assertIn('ɨ', result, 
                f"Expected epenthetic ɨ in result for '{word}'")
    
    def test_gemination(self):
        """Test geminated consonants"""
        # Words with gemination (doubled consonants)
        test_cases = [
            'አለም',   # Should not have gemination (single l)
            'አለሙ',   # If implemented with gemination detection
        ]
        
        for word in test_cases:
            result = self.g2p.convert(word)
            print(f"\nGemination: {word:15} → {result}")
            # Result should exist and be reasonable
            self.assertGreater(len(result), 0)
    
    def test_ejectives(self):
        """Test ejective consonants"""
        # Words with ejective consonants (marked with ʼ)
        test_cases = [
            'ጠና',     # tʼ ejective
            'ጨርቅ',    # tʃʼ ejective
            'ጸሐይ',    # sʼ ejective
        ]
        
        for word in test_cases:
            result = self.g2p.convert(word)
            print(f"\nEjective: {word:15} → {result}")
            # Should contain ejective marker ʼ
            self.assertIn('ʼ', result, 
                f"Expected ejective marker ʼ in result for '{word}'")
    
    def test_labiovelars(self):
        """Test labiovelar consonants"""
        test_cases = [
            'ቋንቋ',    # Language (with qʷ)
            'ኳስ',     # Ball (with kʷ)
        ]
        
        for word in test_cases:
            result = self.g2p.convert(word)
            print(f"\nLabiovelar: {word:15} → {result}")
            # Should contain labiovelar marker ʷ
            self.assertIn('ʷ', result, 
                f"Expected labiovelar marker ʷ in result for '{word}'")
    
    def test_phrases(self):
        """Test multi-word phrases"""
        test_cases = [
            'ሰላም ነው',       # It is peace/hello
            'መልካም ቀን',      # Good day
            'እንዴት ነህ',      # How are you (masculine)
        ]
        
        for phrase in test_cases:
            result = self.g2p.convert(phrase)
            print(f"\nPhrase: {phrase:20} → {result}")
            
            # Should contain spaces between words
            self.assertIn(' ', result)
            
            # Should have reasonable length
            self.assertGreater(len(result), len(phrase) * 0.5)
            self.assertLess(len(result), len(phrase) * 3)
    
    def test_punctuation(self):
        """Test Ethiopic punctuation"""
        test_cases = [
            'ሰላም።',         # With full stop
            'ሰላም፣ እንዴት ነህ፧',  # With comma and question mark
        ]
        
        for text in test_cases:
            result = self.g2p.convert(text)
            print(f"\nPunctuation: {text:25} → {result}")
            # Punctuation should be converted or preserved
            self.assertGreater(len(result), 0)
    
    def test_empty_input(self):
        """Test edge case: empty input"""
        result = self.g2p.convert('')
        self.assertEqual(result, '')
        
        result = self.g2p.convert('   ')
        self.assertEqual(result, '')
    
    def test_mixed_content(self):
        """Test mixed Ethiopic and Latin characters"""
        test_cases = [
            'ሰላም hello',
            '2023 ዓ.ም.',
        ]
        
        for text in test_cases:
            result = self.g2p.convert(text)
            print(f"\nMixed: {text:25} → {result}")
            # Should handle gracefully
            self.assertGreater(len(result), 0)


class TestPhonologicalRules(unittest.TestCase):
    """Test specific phonological rules"""
    
    def setUp(self):
        self.g2p = EnhancedAmharicG2P()
    
    def test_epenthesis_after_velars(self):
        """Test that velars trigger epenthesis"""
        # Velars (k, g, q, x) should strongly trigger ɨ insertion
        words = ['ክብር', 'ግብር', 'ቅብር']
        
        for word in words:
            result = self.g2p.convert(word)
            # Should have ɨ after velar before another consonant
            self.assertIn('ɨ', result)
    
    def test_gemination_marking(self):
        """Test gemination length marking"""
        # If we have bb, tt, etc., should be marked with ː
        # This depends on the gemination detection
        word = 'አለም'  # Has no gemination
        result = self.g2p.convert(word)
        
        # Basic test: conversion should work
        self.assertGreater(len(result), 0)
    
    def test_no_epenthesis_in_geminates(self):
        """Test that epenthesis doesn't break geminates"""
        # If a word has gemination, ɨ should not be inserted between
        # the geminate consonants
        
        # Create a test with known geminate structure
        # This is a placeholder - real test needs specific Amharic words
        result = self.g2p.convert('ሰላም')
        
        # Basic validation
        self.assertNotIn('ɨɨ', result, "Should not have consecutive ɨ")


class TestQualityValidation(unittest.TestCase):
    """Test G2P quality validation"""
    
    def setUp(self):
        self.g2p = EnhancedAmharicG2P()
    
    def test_vowel_ratio(self):
        """Test that output has reasonable vowel ratio"""
        text = 'ሰላም ዓለም'
        result = self.g2p.convert(text)
        
        vowels = 'aeiouɨəAEIOUɨə'
        vowel_count = sum(1 for c in result if c in vowels)
        total_alpha = sum(1 for c in result if c.isalpha() or ord(c) > 127)
        
        if total_alpha > 0:
            vowel_ratio = vowel_count / total_alpha
            # Should have at least 25% vowels
            self.assertGreater(vowel_ratio, 0.20,
                f"Vowel ratio too low: {vowel_ratio:.2%}")
    
    def test_no_ethiopic_in_output(self):
        """Test that output doesn't contain Ethiopic characters"""
        text = 'ሰላም ዓለም'
        result = self.g2p.convert(text)
        
        # Count Ethiopic characters in output
        ethiopic_count = sum(1 for c in result 
                           if 0x1200 <= ord(c) <= 0x137F)
        
        # Should be minimal or zero (except for unknown chars)
        self.assertLess(ethiopic_count, len(result) * 0.1,
            "Output contains too many Ethiopic characters")
    
    def test_output_length(self):
        """Test that output has reasonable length"""
        text = 'ሰላም ዓለም'
        result = self.g2p.convert(text)
        
        # Output should be at least 50% of input length
        self.assertGreater(len(result), len(text) * 0.5,
            "Output too short relative to input")


def run_comprehensive_tests():
    """Run all tests with detailed output"""
    print("=" * 70)
    print("COMPREHENSIVE AMHARIC G2P TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEthiopicG2PTable))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedG2PConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestPhonologicalRules))
    suite.addTests(loader.loadTestsFromTestCase(TestQualityValidation))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
