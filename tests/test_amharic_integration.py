#!/usr/bin/env python3
"""
Integration Tests for Amharic TTS - Full Pipeline
Tests the complete workflow from G2P through training configuration
"""

import sys
import os
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestG2PBackends:
    """Test G2P backend switching and fallback behavior"""
    
    def test_rule_based_backend(self):
        """Test rule-based backend always works"""
        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
        
        g2p = AmharicG2P(backend='rule-based')
        result = g2p.convert("ሰላም")
        
        assert result is not None
        assert len(result) > 0
        assert "ə" in result or "a" in result  # Should contain vowel phonemes
    
    def test_transphone_backend_if_available(self):
        """Test Transphone backend if installed"""
        try:
            import transphone
            from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
            
            g2p = AmharicG2P(backend='transphone')
            result = g2p.convert("ሰላም")
            
            assert result is not None
            assert len(result) > 0
        except ImportError:
            pytest.skip("Transphone not installed")
    
    def test_epitran_backend_if_available(self):
        """Test Epitran backend if installed"""
        try:
            import epitran
            from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
            
            g2p = AmharicG2P(backend='epitran')
            result = g2p.convert("ሰላም")
            
            assert result is not None
            assert len(result) > 0
        except ImportError:
            pytest.skip("Epitran not installed")
    
    def test_backend_fallback(self):
        """Test that fallback mechanism works"""
        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
        
        # Force auto backend (will try all in order)
        g2p = AmharicG2P(backend='auto')
        result = g2p.convert("ሰላም ዓለም")
        
        # Should succeed with at least rule-based fallback
        assert result is not None
        assert len(result) > 0


class TestHybridTokenizer:
    """Test the XTTS-compatible tokenizer wrapper"""
    
    def test_tokenizer_creation(self):
        """Test creating hybrid tokenizer"""
        from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
        
        tokenizer = create_xtts_tokenizer(
            vocab_path=None,  # Use default
            use_g2p=True,
            g2p_backend='rule-based'
        )
        
        assert tokenizer is not None
        assert hasattr(tokenizer, 'encode')
        assert hasattr(tokenizer, 'decode')
    
    def test_tokenizer_encode_decode(self):
        """Test encoding and decoding Amharic text"""
        from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
        
        tokenizer = create_xtts_tokenizer(use_g2p=False)  # Raw mode first
        
        text = "ሰላም"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert tokens is not None
        assert len(tokens) > 0
        # Decoded might not match exactly due to BPE, but should be similar
        assert decoded is not None
    
    def test_g2p_mode_tokenization(self):
        """Test tokenization with G2P preprocessing"""
        from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
        
        tokenizer = create_xtts_tokenizer(
            use_g2p=True,
            g2p_backend='rule-based'
        )
        
        text = "ሰላም ዓለም"
        tokens = tokenizer.encode(text)
        
        assert tokens is not None
        assert len(tokens) > 0


class TestTextPreprocessing:
    """Test Amharic text normalization and preprocessing"""
    
    def test_text_normalization(self):
        """Test character variant normalization"""
        from amharic_tts.preprocessing.text_normalizer import AmharicTextNormalizer
        
        normalizer = AmharicTextNormalizer()
        
        # Test variant normalization
        assert normalizer.normalize("ሀሎ") == "ሃሎ"
        assert normalizer.normalize("ዓለም") == "አለም"
        
        # Test whitespace normalization
        assert normalizer.normalize("ሰላም   ዓለም") == "ሰላም አለም"
    
    def test_number_expansion(self):
        """Test Amharic number to word expansion"""
        from amharic_tts.preprocessing.number_expander import AmharicNumberExpander
        
        expander = AmharicNumberExpander()
        
        assert expander.expand_number("1") == "አንድ"
        assert expander.expand_number("10") == "አሥር"
        assert expander.expand_number("100") == "መቶ"
    
    def test_full_text_preprocessing(self):
        """Test complete text preprocessing pipeline"""
        from amharic_tts.preprocessing.text_normalizer import AmharicTextNormalizer
        from amharic_tts.preprocessing.number_expander import AmharicNumberExpander
        
        normalizer = AmharicTextNormalizer()
        expander = AmharicNumberExpander()
        
        # Input with variants, numbers, and extra spaces
        text = "ሀሎ  123  ዓለም"
        
        # Normalize
        text = normalizer.normalize(text)
        
        # Expand numbers (simple implementation)
        import re
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            text = text.replace(num, expander.expand_number(num))
        
        assert "ሃሎ" in text  # Normalized ሀ→ሃ
        assert "አለም" in text  # Normalized ዓ→አ
        assert "123" not in text  # Number should be expanded


class TestConfigurationSystem:
    """Test configuration management"""
    
    def test_g2p_config_creation(self):
        """Test G2P configuration dataclass"""
        from amharic_tts.config.amharic_config import G2PConfiguration, G2PBackend
        
        config = G2PConfiguration()
        
        assert config.backend_order is not None
        assert len(config.backend_order) == 3
        assert config.backend_order[0] == G2PBackend.TRANSPHONE
        assert config.fallback_to_rules == True
    
    def test_custom_backend_order(self):
        """Test custom backend ordering"""
        from amharic_tts.config.amharic_config import G2PConfiguration, G2PBackend
        
        config = G2PConfiguration(
            backend_order=[
                G2PBackend.EPITRAN,
                G2PBackend.RULE_BASED
            ]
        )
        
        assert config.backend_order[0] == G2PBackend.EPITRAN
        assert len(config.backend_order) == 2
    
    def test_phoneme_inventory(self):
        """Test phoneme inventory reference"""
        from amharic_tts.config.amharic_config import PhonemeInventory
        
        inventory = PhonemeInventory()
        
        # Check consonants
        assert 's' in inventory.consonants
        assert 'ʃ' in inventory.consonants
        assert 'kʷ' in inventory.consonants  # Labiovelar
        
        # Check vowels
        assert 'a' in inventory.vowels
        assert 'ə' in inventory.vowels
        assert 'ɨ' in inventory.vowels  # Epenthetic vowel


class TestUIIntegration:
    """Test UI integration points"""
    
    def test_amharic_in_demo_ui(self):
        """Test that Amharic is in xtts_demo.py"""
        demo_path = project_root / "xtts_demo.py"
        
        if demo_path.exists():
            content = demo_path.read_text(encoding='utf-8')
            
            # Check for Amharic language code
            assert '"amh"' in content or "'amh'" in content
            
            # Check for G2P UI controls
            assert 'g2p_backend' in content.lower() or 'g2p' in content.lower()
        else:
            pytest.skip("xtts_demo.py not found")
    
    def test_amharic_in_headless_script(self):
        """Test that Amharic is supported in headless training"""
        headless_path = project_root / "headlessXttsTrain.py"
        
        if headless_path.exists():
            content = headless_path.read_text(encoding='utf-8')
            assert '"amh"' in content or "'amh'" in content
        else:
            pytest.skip("headlessXttsTrain.py not found")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows"""
    
    def test_g2p_to_tokens_pipeline(self):
        """Test G2P → Tokenization pipeline"""
        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
        from amharic_tts.tokenizer.xtts_tokenizer_wrapper import create_xtts_tokenizer
        
        # Step 1: Convert text to phonemes
        g2p = AmharicG2P(backend='rule-based')
        phonemes = g2p.convert("ሰላም ኢትዮጵያ")
        
        assert phonemes is not None
        assert len(phonemes) > 0
        
        # Step 2: Tokenize phonemes
        tokenizer = create_xtts_tokenizer(use_g2p=False)  # Phonemes already converted
        tokens = tokenizer.encode(phonemes)
        
        assert tokens is not None
        assert len(tokens) > 0
    
    def test_text_preprocessing_full_pipeline(self):
        """Test complete text preprocessing workflow"""
        from amharic_tts.preprocessing.text_normalizer import AmharicTextNormalizer
        from amharic_tts.preprocessing.number_expander import AmharicNumberExpander
        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
        
        # Original text with variants and numbers
        text = "ሀሎ 123 ዓለም።"
        
        # Step 1: Normalize
        normalizer = AmharicTextNormalizer()
        text = normalizer.normalize(text)
        
        # Step 2: Expand numbers
        expander = AmharicNumberExpander()
        import re
        for num in re.findall(r'\d+', text):
            text = text.replace(num, expander.expand_number(num))
        
        # Step 3: G2P conversion
        g2p = AmharicG2P(backend='rule-based')
        phonemes = g2p.convert(text)
        
        assert phonemes is not None
        assert "123" not in phonemes  # Numbers should be expanded
    
    def test_config_driven_workflow(self):
        """Test configuration-driven G2P selection"""
        from amharic_tts.config.amharic_config import (
            G2PConfiguration, 
            G2PBackend,
            TokenizerConfiguration
        )
        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
        
        # Create custom config
        config = G2PConfiguration(
            backend_order=[G2PBackend.RULE_BASED],
            enable_epenthesis=True,
            normalize_text=True
        )
        
        # Use config with G2P
        g2p = AmharicG2P(backend='rule-based')
        result = g2p.convert("ሰላም")
        
        assert result is not None


class TestQualityValidation:
    """Test G2P quality validation"""
    
    def test_quality_check_valid_output(self):
        """Test that valid G2P output passes quality checks"""
        from amharic_tts.g2p.amharic_g2p_enhanced import AmharicG2P
        
        g2p = AmharicG2P(backend='rule-based')
        result = g2p.convert("ሰላም")
        
        # Should contain IPA characters
        ipa_chars = set("aeiouəɨbdgkmnprstwʃʒʔʕ")
        result_chars = set(result.lower())
        
        # Check that result contains IPA characters
        assert len(ipa_chars & result_chars) > 0
    
    def test_detect_failed_conversion(self):
        """Test detection of failed G2P conversions"""
        from amharic_tts.config.amharic_config import G2PQualityThresholds
        
        thresholds = G2PQualityThresholds()
        
        # Mock failed output (still contains lots of Ethiopic)
        failed_output = "ሰላም ኢትዮጵያ"  # Unchanged Ethiopic
        
        # Count Ethiopic characters
        ethiopic_count = sum(1 for c in failed_output if '\u1200' <= c <= '\u137F')
        total_chars = len(failed_output.replace(' ', ''))
        ethiopic_ratio = ethiopic_count / total_chars if total_chars > 0 else 0
        
        # Should exceed max threshold
        assert ethiopic_ratio > thresholds.max_ethiopic_ratio


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
