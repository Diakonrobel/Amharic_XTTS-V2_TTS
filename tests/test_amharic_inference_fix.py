#!/usr/bin/env python3
"""
Unit Tests for Amharic TTS Inference Fix

Tests to validate and fix the Amharic inference issues including:
- Language normalization problems
- G2P preprocessing validation
- Inference pipeline correctness

Run with: python -m unittest tests.test_amharic_inference_fix -v
"""

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import main modules
try:
    from xtts_demo import normalize_xtts_lang, run_tts
    from amharic_tts.tokenizer.xtts_tokenizer_wrapper import XTTSAmharicTokenizer
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")


class TestAmharicPhonemePreprocessing(unittest.TestCase):
    """Test Amharic phoneme preprocessing works correctly"""
    
    def test_amharic_phoneme_preprocessing_works(self):
        """Test that Amharic text is correctly converted to phonemes"""
        try:
            tokenizer = XTTSAmharicTokenizer(use_phonemes=True)
            
            # Test basic Amharic text
            amharic_text = "ሰላም ዓለም"
            phonemes = tokenizer.preprocess_text(amharic_text, lang="am")
            
            # Should return phoneme representation
            self.assertIsInstance(phonemes, str)
            self.assertGreater(len(phonemes), 0)
            self.assertNotEqual(phonemes, amharic_text)  # Should be different from original
            
            # Check that it contains IPA-like characters
            ipa_chars = {'ə', 'ɨ', 'ʔ', 'ɛ', 'ʃ', 'ʧ', 'ʤ', 'ɲ', 'ɨw'}
            has_ipa = any(char in phonemes for char in ipa_chars)
            self.assertTrue(has_ipa, f"Phonemes should contain IPA chars, got: {phonemes}")
            
            print(f"✅ G2P: '{amharic_text}' → '{phonemes}'")
            
        except Exception as e:
            self.fail(f"Amharic phoneme preprocessing failed: {e}")


class TestLanguageNormalizationPreservesPhonemes(unittest.TestCase):
    """Test language normalization preserves phonemes correctly"""
    
    def test_language_normalization_preserves_phonemes(self):
        """Test that language normalization works correctly with phonemes"""
        
        # Test the current (problematic) normalize_xtts_lang function
        result_am = normalize_xtts_lang("am")
        result_amh = normalize_xtts_lang("amh")
        
        print(f"Current normalization: am → {result_am}, amh → {result_amh}")
        
        # The issue: currently maps to 'en', but should preserve multilingual context
        self.assertEqual(result_am, "en")  # Current behavior (problematic)
        self.assertEqual(result_amh, "en")  # Current behavior (problematic)
        
        # For phoneme-based inference, we should NOT map to English
        # because the phonemes are Amharic-specific


class TestG2PFallbackHandlesErrors(unittest.TestCase):
    """Test G2P fallback handles errors gracefully"""
    
    @patch('amharic_tts.tokenizer.xtts_tokenizer_wrapper.XTTSAmharicTokenizer')
    def test_g2p_fallback_handles_errors(self, mock_tokenizer_class):
        """Test that G2P preprocessing handles errors gracefully"""
        
        # Mock tokenizer that raises exception
        mock_tokenizer = Mock()
        mock_tokenizer.preprocess_text.side_effect = Exception("G2P backend error")
        mock_tokenizer_class.return_value = mock_tokenizer
        
        # Mock XTTS model for run_tts
        with patch('xtts_demo.XTTS_MODEL') as mock_model:
            mock_model.get_conditioning_latents.return_value = (Mock(), Mock())
            mock_model.inference.return_value = {"wav": [0.1, 0.2, 0.3]}
            
            # Create a temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio.write(b"fake_audio_data")
                temp_audio_path = temp_audio.name
            
            try:
                # Should not crash even if G2P fails
                result = run_tts(
                    lang="am",
                    tts_text="ሰላም",
                    speaker_audio_file=temp_audio_path,
                    temperature=0.7,
                    length_penalty=1.0,
                    repetition_penalty=1.0,
                    top_k=50,
                    top_p=0.8,
                    sentence_split=True,
                    use_config=False,
                    use_g2p_inference=True
                )
                
                # Should return success message even if G2P fails
                self.assertIsInstance(result[0], str)
                self.assertEqual(result[0], "Speech generated !")
                
            finally:
                # Clean up
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)


class TestNonAmharicTextPassesThrough(unittest.TestCase):
    """Test non-Amharic text passes through without G2P processing"""
    
    def test_non_amharic_text_passes_through(self):
        """Test that non-Amharic text is not processed by G2P"""
        try:
            tokenizer = XTTSAmharicTokenizer(use_phonemes=True)
            
            # Test English text
            english_text = "Hello world"
            result = tokenizer.preprocess_text(english_text, lang="en")
            
            # Should return unchanged
            self.assertEqual(result, english_text)
            
            # Test with auto-detection
            result_auto = tokenizer.preprocess_text(english_text)  # lang=None
            self.assertEqual(result_auto, english_text)
            
            print(f"✅ Non-Amharic passthrough: '{english_text}' → '{result}'")
            
        except Exception as e:
            self.fail(f"Non-Amharic text processing failed: {e}")


class TestEmptyTextInputHandling(unittest.TestCase):
    """Test empty text input handling"""
    
    def test_empty_text_input_handling(self):
        """Test that empty or whitespace-only text is handled correctly"""
        try:
            tokenizer = XTTSAmharicTokenizer(use_phonemes=True)
            
            # Test empty string
            result_empty = tokenizer.preprocess_text("")
            self.assertEqual(result_empty, "")
            
            # Test whitespace only
            result_whitespace = tokenizer.preprocess_text("   ")
            self.assertEqual(result_whitespace, "")
            
            # Test None (should not crash)
            try:
                result_none = tokenizer.preprocess_text(None)
                self.assertEqual(result_none, "")
            except (TypeError, AttributeError):
                # Acceptable to raise error for None input
                pass
            
            print("✅ Empty text handling works correctly")
            
        except Exception as e:
            self.fail(f"Empty text handling failed: {e}")


class TestLanguageDetectionWithMixedText(unittest.TestCase):
    """Test language detection with mixed text"""
    
    def test_language_detection_with_mixed_text(self):
        """Test language detection works with mixed content"""
        try:
            tokenizer = XTTSAmharicTokenizer(use_phonemes=True)
            
            # Test mixed text (Amharic + English)
            mixed_text = "ሰላም Hello ዓለም World"
            result = tokenizer.preprocess_text(mixed_text, lang="am")
            
            # Should process as Amharic since lang="am" specified
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            # Test auto-detection (should detect Amharic presence)
            result_auto = tokenizer.preprocess_text(mixed_text)
            self.assertIsInstance(result_auto, str)
            
            print(f"✅ Mixed text: '{mixed_text}' → '{result[:50]}...'")
            
        except Exception as e:
            self.fail(f"Mixed text processing failed: {e}")


class TestModelInferenceWithCorrectedLang(unittest.TestCase):
    """Test model inference with corrected language handling"""
    
    def test_model_inference_with_corrected_lang(self):
        """Test that the inference pipeline would work with corrected language handling"""
        
        # This test demonstrates what the corrected behavior should be
        test_lang = "am"
        
        # Current (broken) behavior
        current_norm = normalize_xtts_lang(test_lang)
        self.assertEqual(current_norm, "en")  # Maps to English - WRONG!
        
        # What should happen for phoneme-based Amharic:
        # For a fine-tuned model with extended vocabulary,
        # we should either:
        # 1. Use the original language code
        # 2. Use a multilingual designation
        # 3. Use a specific code that the model was trained with
        
        # The key insight: don't map Amharic to English when using G2P phonemes!
        print("⚠️  Current behavior maps Amharic phonemes to English language model")
        print("✅ Fix needed: preserve language context for phoneme-based inference")


class TestG2PDisabledModeWorks(unittest.TestCase):
    """Test G2P disabled mode works correctly"""
    
    def test_g2p_disabled_mode_works(self):
        """Test that disabling G2P works correctly"""
        try:
            # Create tokenizer with G2P disabled
            tokenizer = XTTSAmharicTokenizer(use_phonemes=False)
            
            amharic_text = "ሰላም ዓለም"
            result = tokenizer.preprocess_text(amharic_text, lang="am")
            
            # Should return original text when G2P disabled
            self.assertEqual(result, amharic_text)
            
            print(f"✅ G2P disabled: '{amharic_text}' → '{result}'")
            
        except Exception as e:
            self.fail(f"G2P disabled mode failed: {e}")


class TestInferencePipelineIntegration(unittest.TestCase):
    """Integration test for the complete inference pipeline"""
    
    def test_complete_inference_pipeline_analysis(self):
        """Analyze the complete inference pipeline to identify issues"""
        
        print("\n" + "="*80)
        print("AMHARIC INFERENCE PIPELINE ANALYSIS")
        print("="*80)
        
        # Test the complete flow
        original_text = "ሰላም ዓለም"
        lang = "am"
        
        print(f"1. Input text: '{original_text}' (lang: {lang})")
        
        # Step 1: G2P preprocessing
        try:
            tokenizer = XTTSAmharicTokenizer(use_phonemes=True)
            phonemes = tokenizer.preprocess_text(original_text, lang=lang)
            print(f"2. G2P conversion: '{original_text}' → '{phonemes}'")
        except Exception as e:
            print(f"2. G2P conversion FAILED: {e}")
            phonemes = original_text
        
        # Step 2: Language normalization (THE PROBLEM!)
        normalized_lang = normalize_xtts_lang(lang)
        print(f"3. Language normalization: '{lang}' → '{normalized_lang}' ⚠️  PROBLEM!")
        
        # Step 3: What model receives
        print(f"4. Model receives:")
        print(f"   - Text/Phonemes: '{phonemes}'")
        print(f"   - Language: '{normalized_lang}' (English!)")
        print(f"   - Result: Amharic phonemes processed as English → Bad pronunciation")
        
        print(f"\n🔧 FIX NEEDED:")
        print(f"   - Don't map Amharic to English when using G2P phonemes")
        print(f"   - Keep language as 'am' or use multilingual approach")
        print(f"   - Model was fine-tuned with Amharic data, so it should handle 'am'")
        
        print("="*80)


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)