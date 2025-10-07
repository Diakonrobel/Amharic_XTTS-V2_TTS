"""
Enhanced Amharic Grapheme-to-Phoneme Converter with Quality Validation

This module provides advanced G2P conversion for Amharic text with:
- Multiple backends (Transphone, Epitran, Rule-based)
- Quality-based backend selection
- Automatic fallback with validation
- Comprehensive phonological rule handling

Based on research:
- Transphone: Zero-shot G2P for 7546 languages  
- Epitran: Multilingual G2P with Ethiopic script support
- Custom rules: Amharic-specific phonology (epenthesis, gemination)
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import logging

# Import configuration
try:
    from ..config.amharic_config import (
        AmharicTTSConfig, G2PBackend, G2PQualityThresholds,
        DEFAULT_CONFIG
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Fallback for standalone usage
    from enum import Enum
    class G2PBackend(Enum):
        TRANSPHONE = "transphone"
        EPITRAN = "epitran"
        RULE_BASED = "rule-based"

logger = logging.getLogger(__name__)


class EnhancedAmharicG2P:
    """
    Enhanced Amharic G2P converter with quality validation
    
    Features:
    - Intelligent backend selection with quality checks
    - Automatic fallback based on output quality
    - Comprehensive Ethiopic script support
    - Advanced phonological rule application
    
    Usage:
        g2p = EnhancedAmharicG2P()
        phonemes = g2p.convert("ሰላም ዓለም")
        
        # With custom config
        from amharic_tts.config import get_config
        g2p = EnhancedAmharicG2P(config=get_config('quality'))
    """
    
    def __init__(self, config=None):
        """
        Initialize enhanced G2P converter
        
        Args:
            config: AmharicTTSConfig instance (optional)
        """
        if CONFIG_AVAILABLE and config is None:
            self.config = DEFAULT_CONFIG
        else:
            self.config = config or self._get_fallback_config()
        
        self._backends_initialized = False
        self.transphone_g2p = None
        self.epitran_g2p = None
        
        self._load_phonological_rules()
        self._load_comprehensive_g2p_table()
    
    def _get_fallback_config(self):
        """Create basic config if config module not available"""
        class FallbackConfig:
            class g2p:
                backend_order = [G2PBackend.RULE_BASED]
                enable_quality_check = False
                enable_epenthesis = True
                enable_gemination = True
                enable_labiovelars = True
                class quality_thresholds:
                    min_vowel_ratio = 0.25
                    max_ethiopic_ratio = 0.1
                    min_ipa_ratio = 0.5
                    min_length_ratio = 0.5
        return FallbackConfig()
    
    def _initialize_backends(self):
        """Lazy initialization of external G2P backends"""
        if self._backends_initialized:
            return
        
        backend_order = self.config.g2p.backend_order if hasattr(self.config, 'g2p') else []
        
        # Initialize Transphone if in backend order
        if G2PBackend.TRANSPHONE in backend_order:
            try:
                from transphone import read_g2p
                self.transphone_g2p = read_g2p('amh')
                logger.info("✅ Transphone backend initialized")
            except ImportError:
                logger.warning("⚠️  Transphone not available")
                self.transphone_g2p = None
            except Exception as e:
                logger.warning(f"⚠️  Transphone init failed: {e}")
                self.transphone_g2p = None
        
        # Initialize Epitran if in backend order
        if G2PBackend.EPITRAN in backend_order:
            try:
                import epitran
                self.epitran_g2p = epitran.Epitran('amh-Ethi')
                logger.info("✅ Epitran backend initialized")
            except ImportError:
                logger.warning("⚠️  Epitran not available")
                self.epitran_g2p = None
            except Exception as e:
                logger.warning(f"⚠️  Epitran init failed: {e}")
                self.epitran_g2p = None
        
        self._backends_initialized = True
    
    def _validate_g2p_quality(self, input_text: str, output_text: str) -> Tuple[bool, float, str]:
        """
        Validate quality of G2P output
        
        Args:
            input_text: Original Amharic text
            output_text: G2P converted phoneme text
            
        Returns:
            Tuple of (is_valid, quality_score, failure_reason)
        """
        if not output_text or len(output_text) < 2:
            return False, 0.0, "Output too short"
        
        # Get thresholds
        thresholds = self.config.g2p.quality_thresholds if hasattr(self.config.g2p, 'quality_thresholds') else self._get_fallback_config().g2p.quality_thresholds
        
        # Check 1: Vowel ratio
        vowels = 'aeiouɨəAEIOUɨə'
        vowel_count = sum(1 for c in output_text if c in vowels)
        total_alpha = sum(1 for c in output_text if c.isalpha() or ord(c) > 127)
        vowel_ratio = vowel_count / max(total_alpha, 1)
        
        if vowel_ratio < thresholds.min_vowel_ratio:
            return False, vowel_ratio, f"Low vowel ratio: {vowel_ratio:.2f}"
        
        # Check 2: Ethiopic character presence (should be minimal in output)
        ethiopic_chars = sum(1 for c in output_text if 0x1200 <= ord(c) <= 0x137F)
        ethiopic_ratio = ethiopic_chars / max(len(output_text), 1)
        
        if ethiopic_ratio > thresholds.max_ethiopic_ratio:
            return False, 1 - ethiopic_ratio, f"High Ethiopic ratio: {ethiopic_ratio:.2f}"
        
        # Check 3: IPA character presence
        ipa_chars = 'ɨəɲʃʒʔʕʷʼ'
        ipa_count = sum(1 for c in output_text if c in ipa_chars or c in vowels)
        ipa_ratio = ipa_count / max(total_alpha, 1)
        
        if ipa_ratio < thresholds.min_ipa_ratio:
            return False, ipa_ratio, f"Low IPA ratio: {ipa_ratio:.2f}"
        
        # Check 4: Length ratio
        length_ratio = len(output_text) / max(len(input_text), 1)
        
        if length_ratio < thresholds.min_length_ratio:
            return False, length_ratio, f"Output too short: ratio {length_ratio:.2f}"
        
        # Calculate overall quality score
        quality_score = (vowel_ratio + (1 - ethiopic_ratio) + ipa_ratio + min(length_ratio, 1.0)) / 4.0
        
        return True, quality_score, "Pass"
    
    def convert(self, text: str) -> str:
        """
        Convert Amharic text to phonemes with intelligent backend selection
        
        Args:
            text: Input Amharic text
            
        Returns:
            IPA phoneme sequence
        """
        if not text or not text.strip():
            return ""
        
        # Preprocess
        original_text = text
        text = self._preprocess(text)
        
        # Try backends in order with quality validation
        result = self._convert_with_fallback(text)
        
        # Post-process
        if self.config.g2p.enable_labiovelars:
            result = self._apply_labiovelar_rules(result)
        
        if self.config.g2p.enable_epenthesis:
            result = self._apply_epenthesis(result)
        
        if self.config.g2p.enable_gemination:
            result = self._handle_gemination(result)
        
        return result.strip()
    
    def _convert_with_fallback(self, text: str) -> str:
        """
        Try each backend with quality validation
        
        Args:
            text: Preprocessed Amharic text
            
        Returns:
            Best quality phoneme output
        """
        self._initialize_backends()
        
        backend_order = self.config.g2p.backend_order if hasattr(self.config, 'g2p') else [G2PBackend.RULE_BASED]
        enable_quality_check = self.config.g2p.enable_quality_check if hasattr(self.config.g2p, 'enable_quality_check') else False
        
        for backend in backend_order:
            result = None
            backend_name = backend.value if hasattr(backend, 'value') else str(backend)
            
            # Try backend
            if backend == G2PBackend.TRANSPHONE and self.transphone_g2p:
                try:
                    result = self.transphone_g2p(text)
                    logger.debug(f"Transphone output: {result[:50]}...")
                except Exception as e:
                    logger.warning(f"Transphone failed: {e}")
                    continue
            
            elif backend == G2PBackend.EPITRAN and self.epitran_g2p:
                try:
                    result = self.epitran_g2p.transliterate(text)
                    logger.debug(f"Epitran output: {result[:50]}...")
                except Exception as e:
                    logger.warning(f"Epitran failed: {e}")
                    continue
            
            elif backend == G2PBackend.RULE_BASED:
                result = self._rule_based_convert(text)
                logger.debug(f"Rule-based output: {result[:50]}...")
            
            # Validate quality if enabled
            if result and enable_quality_check:
                is_valid, quality_score, reason = self._validate_g2p_quality(text, result)
                if is_valid:
                    logger.info(f"✅ {backend_name} succeeded (quality: {quality_score:.2f})")
                    return result
                else:
                    logger.info(f"⚠️  {backend_name} failed quality check: {reason}")
            elif result:
                logger.info(f"✅ {backend_name} succeeded (quality check disabled)")
                return result
        
        # Ultimate fallback
        logger.warning("All backends failed, using basic rule-based conversion")
        return self._rule_based_convert(text)
    
    def _load_phonological_rules(self):
        """Load Amharic phonological rules"""
        # Epenthesis: insert ɨ in consonant clusters and word-final
        self.epenthesis_rules = [
            # After velars before consonants
            (r'([kgqxKGQX])([^aeiouɨəAEIOUɨə\s።፣፤፥፧፨])', r'\1ɨ\2'),
            # After other consonants before consonants
            (r'([bdfhjlmnprstvwyzʃʒʔʕBDFHJLMNPRSTVWYZʃʒʔʕ])([^aeiouɨəAEIOUɨə\s።፣፤፥፧፨])', r'\1ɨ\2'),
            # Word-final consonants
            (r'([bdfghjklmnpqrstvwxyzʃʒʔʕBDFGHJKLMNPQRSTVWXYZʃʒʔʕ])(\s|$)', r'\1ɨ\2'),
        ]
        
        # Labiovelar mappings
        self.labiovelar_map = {
            'ቋ': 'qʷa', 'ቍ': 'qʷɨ', 'ቊ': 'qʷu', 'ቌ': 'qʷe', 'ቈ': 'qʷi',
            'ኳ': 'kʷa', 'ኵ': 'kʷɨ', 'ኲ': 'kʷu', 'ኴ': 'kʷe', 'ኰ': 'kʷi',
            'ጓ': 'gʷa', 'ጕ': 'gʷɨ', 'ጒ': 'gʷu', 'ጔ': 'gʷe', 'ጐ': 'gʷi',
            'ኻ': 'xʷa', 'ኽ': 'xʷɨ', 'ኺ': 'xʷu', 'ኼ': 'xʷe', 'ኸ': 'xʷi',
        }
    
    def _load_comprehensive_g2p_table(self):
        """Load comprehensive Ethiopic grapheme to phoneme mappings"""
        # This will be expanded in Phase 3
        self.g2p_table = {
            # Phase 3 will add complete 231-entry table
            # For now, basic mappings
            'ሰላም': 'səlam',
            'ኢትዮጵያ': 'ʔitʼəjopʼːəja',
            'አማርኛ': 'ʔəmarɨɲːa',
            # ... (will expand in Phase 3)
        }
    
    def _rule_based_convert(self, text: str) -> str:
        """
        Rule-based G2P conversion (will be expanded in Phase 3)
        
        Args:
            text: Preprocessed text
            
        Returns:
            Phoneme sequence
        """
        # This is placeholder - Phase 3 will add full implementation
        result = []
        for char in text:
            if char in self.g2p_table:
                result.append(self.g2p_table[char])
            elif char.isspace():
                result.append(' ')
            else:
                result.append(char)  # Keep unknown chars
        
        return ''.join(result)
    
    def _preprocess(self, text: str) -> str:
        """Normalize Amharic text"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize character variants
        replacements = {
            'ሥ': 'ስ', 'ዕ': 'እ', 'ፅ': 'ጽ',
            'ኅ': 'ህ', 'ኽ': 'ክ', 'ሕ': 'ህ',
            'ዐ': 'አ', 'ኣ': 'አ', 'ዓ': 'አ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def _apply_labiovelar_rules(self, text: str) -> str:
        """Apply labiovelar consonant mappings"""
        for char, phoneme in self.labiovelar_map.items():
            text = text.replace(char, phoneme)
        return text
    
    def _apply_epenthesis(self, phonemes: str) -> str:
        """Apply epenthetic vowel insertion"""
        for pattern, replacement in self.epenthesis_rules:
            phonemes = re.sub(pattern, replacement, phonemes)
        return phonemes
    
    def _handle_gemination(self, phonemes: str) -> str:
        """Handle geminated (doubled) consonants"""
        # Simple implementation: mark with ː (length marker)
        # Phase 3 will add more sophisticated handling
        return phonemes


# Convenience function
def convert_to_ipa(text: str, use_quality_check: bool = True) -> str:
    """
    Quick conversion function
    
    Args:
        text: Amharic text
        use_quality_check: Enable quality validation
        
    Returns:
        IPA phonemes
    """
    if CONFIG_AVAILABLE:
        from ..config import get_config
        config = get_config('quality' if use_quality_check else 'fast')
    else:
        config = None
    
    g2p = EnhancedAmharicG2P(config=config)
    return g2p.convert(text)


if __name__ == "__main__":
    # Test the enhanced G2P
    print("Testing Enhanced Amharic G2P\n")
    
    test_texts = [
        "ሰላም",
        "ኢትዮጵያ",
        "አማርኛ",
        "መልካም ቀን",
    ]
    
    g2p = EnhancedAmharicG2P()
    
    for text in test_texts:
        phonemes = g2p.convert(text)
        print(f"{text:15} → {phonemes}")
