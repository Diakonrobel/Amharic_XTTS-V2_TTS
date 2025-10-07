"""
Amharic Grapheme-to-Phoneme Converter

This module provides G2P conversion for Amharic text with multiple backends:
- Transphone (primary)
- Epitran (fallback)
- Custom rule-based system (offline fallback)

Features:
- Automatic epenthetic vowel insertion
- Gemination handling
- Labiovelar consonant processing
- Context-aware phoneme mapping
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class G2PConfig:
    """Configuration for Amharic G2P conversion"""
    use_transphone: bool = True
    use_epitran: bool = False
    apply_epenthesis: bool = True
    handle_gemination: bool = True
    preserve_punctuation: bool = True
    fallback_to_rules: bool = True


class AmharicG2P:
    """
    Amharic Grapheme-to-Phoneme converter with multiple backends
    
    Usage:
        g2p = AmharicG2P()
        g2p = AmharicG2P(backend='rule-based')  # Use specific backend
        phonemes = g2p.convert("ሰላም ዓለም")  # Returns IPA phonemes
    """
    
    def __init__(self, config: G2PConfig = None, backend: str = 'auto'):
        """
        Initialize Amharic G2P converter
        
        Args:
            config: G2PConfig object with detailed settings (optional)
            backend: Backend to use: 'auto', 'transphone', 'epitran', 'rule-based' (optional)
        """
        # Handle backend parameter for convenience
        if backend != 'auto':
            if config is None:
                config = G2PConfig()
            if backend == 'rule-based':
                config.use_transphone = False
                config.use_epitran = False
                config.fallback_to_rules = True
            elif backend == 'transphone':
                config.use_transphone = True
                config.use_epitran = False
            elif backend == 'epitran':
                config.use_transphone = False
                config.use_epitran = True
        
        self.config = config or G2PConfig()
        self._backends_initialized = False
        self._load_custom_rules()
        
    def _initialize_backends(self):
        """Initialize G2P backends lazily"""
        if self._backends_initialized:
            return
            
        # Try to import transphone
        if self.config.use_transphone:
            try:
                from transphone import read_g2p
                self.transphone_g2p = read_g2p('amh')
                logger.info("Transphone backend initialized successfully")
            except ImportError:
                logger.warning("Transphone not available, will use fallback")
                self.transphone_g2p = None
            except Exception as e:
                logger.warning(f"Transphone initialization failed: {e}")
                self.transphone_g2p = None
        else:
            self.transphone_g2p = None
            
        # Try to import epitran
        if self.config.use_epitran:
            try:
                import epitran
                self.epitran_g2p = epitran.Epitran('amh-Ethi')
                logger.info("Epitran backend initialized successfully")
            except ImportError:
                logger.warning("Epitran not available, will use fallback")
                self.epitran_g2p = None
            except Exception as e:
                logger.warning(f"Epitran initialization failed: {e}")
                self.epitran_g2p = None
        else:
            self.epitran_g2p = None
            
        self._backends_initialized = True
        
    def _load_custom_rules(self):
        """Load Amharic-specific phonological rules"""
        
        # Epenthesis rules: Insert ɨ between consonants
        self.epenthesis_rules = [
            (r'([ክግቅኽ])([^aeiouɨəAEIOUɨə\s።፣፤፥፧፨])', r'\1ɨ\2'),  # After velars
            (r'([ብትድንርስዝልምፍ])$', r'\1ɨ'),  # Word-final
        ]
        
        # Gemination patterns
        self.gemination_patterns = [
            (r'_gem', ''),  # Remove gemination marker, double handled elsewhere
        ]
        
        # Labiovelar mappings
        self.labiovelar_mapping = {
            'ቋ': 'qʷa', 'ቍ': 'qʷɨ', 'ቊ': 'qʷu', 'ቌ': 'qʷe', 'ቈ': 'qʷi',
            'ኳ': 'kʷa', 'ኵ': 'kʷɨ', 'ኲ': 'kʷu', 'ኴ': 'kʷe', 'ኰ': 'kʷi',
            'ጓ': 'gʷa', 'ጕ': 'gʷɨ', 'ጒ': 'gʷu', 'ጔ': 'gʷe', 'ጐ': 'gʷi',
            'ኻ': 'xʷa', 'ኽ': 'xʷɨ', 'ኺ': 'xʷu', 'ኼ': 'xʷe', 'ኸ': 'xʷi',
        }
        
        # Basic grapheme to phoneme mapping (fallback)
        self.basic_g2p = {
            # Consonants
            'ህ': 'h', 'ለ': 'lə', 'ሐ': 'h', 'መ': 'mə', 'ሠ': 's', 'ረ': 'rə',
            'ሰ': 'sə', 'ሸ': 'ʃə', 'ቀ': 'qə', 'በ': 'bə', 'ተ': 'tə', 'ቸ': 'tʃə',
            'ኀ': 'h', 'ነ': 'nə', 'ኘ': 'ɲə', 'አ': 'ʔə', 'ከ': 'kə', 'ኸ': 'xə',
            'ወ': 'wə', 'ዐ': 'ʕə', 'ዘ': 'zə', 'ዠ': 'ʒə', 'የ': 'jə', 'ደ': 'də',
            'ጀ': 'dʒə', 'ገ': 'gə', 'ጠ': 'tʼə', 'ጨ': 'tʃʼə', 'ጰ': 'pʼə',
            'ጸ': 'sʼə', 'ፀ': 'sʼə', 'ፈ': 'fə', 'ፐ': 'pə',
            
            # Base forms (1st order - ə)
            'ል': 'lə', 'ም': 'mə', 'ር': 'rə', 'ስ': 'sə', 'ሽ': 'ʃə',
            'ቅ': 'qə', 'ብ': 'bə', 'ት': 'tə', 'ች': 'tʃə', 'ን': 'nə',
            'ኝ': 'ɲə', 'እ': 'ʔə', 'ክ': 'kə', 'ው': 'wə', 'ዕ': 'ʕə',
            'ዝ': 'zə', 'ዥ': 'ʒə', 'ይ': 'jə', 'ድ': 'də', 'ጅ': 'dʒə',
            'ግ': 'gə', 'ጥ': 'tʼə', 'ጭ': 'tʃʼə', 'ጵ': 'pʼə', 'ጽ': 'sʼə',
            'ፍ': 'fə', 'ፕ': 'pə',
            
            # Vowel modifications (simplified - full table would be extensive)
            'ሁ': 'hu', 'ሂ': 'hi', 'ሃ': 'ha', 'ሄ': 'he', 'ሆ': 'ho',
            'ላ': 'la', 'ሉ': 'lu', 'ሊ': 'li', 'ሌ': 'le', 'ሎ': 'lo',
            'ማ': 'ma', 'ሙ': 'mu', 'ሚ': 'mi', 'ሜ': 'me', 'ሞ': 'mo',
            'ራ': 'ra', 'ሩ': 'ru', 'ሪ': 'ri', 'ሬ': 're', 'ሮ': 'ro',
            'ሳ': 'sa', 'ሱ': 'su', 'ሲ': 'si', 'ሴ': 'se', 'ሶ': 'so',
            'ሻ': 'ʃa', 'ሹ': 'ʃu', 'ሺ': 'ʃi', 'ሼ': 'ʃe', 'ሾ': 'ʃo',
            'ባ': 'ba', 'ቡ': 'bu', 'ቢ': 'bi', 'ቤ': 'be', 'ቦ': 'bo',
            'ታ': 'ta', 'ቱ': 'tu', 'ቲ': 'ti', 'ቴ': 'te', 'ቶ': 'to',
            'ና': 'na', 'ኑ': 'nu', 'ኒ': 'ni', 'ኔ': 'ne', 'ኖ': 'no',
            'ካ': 'ka', 'ኩ': 'ku', 'ኪ': 'ki', 'ኬ': 'ke', 'ኮ': 'ko',
            'ዋ': 'wa', 'ዉ': 'wu', 'ዊ': 'wi', 'ዌ': 'we', 'ዎ': 'wo',
            'ዛ': 'za', 'ዙ': 'zu', 'ዚ': 'zi', 'ዜ': 'ze', 'ዞ': 'zo',
            'ያ': 'ja', 'ዩ': 'ju', 'ዪ': 'ji', 'ዬ': 'je', 'ዮ': 'jo',
            'ዳ': 'da', 'ዱ': 'du', 'ዲ': 'di', 'ዴ': 'de', 'ዶ': 'do',
            'ጋ': 'ga', 'ጉ': 'gu', 'ጊ': 'gi', 'ጌ': 'ge', 'ጎ': 'go',
            'ፋ': 'fa', 'ፉ': 'fu', 'ፊ': 'fi', 'ፌ': 'fe', 'ፎ': 'fo',
            
            # Common words
            'ሰላም': 'səlam',
            'እንኳን': 'ɨnkʷan',
        }
        
    def convert(self, text: str) -> str:
        """
        Convert Amharic text to phonemes
        
        Args:
            text: Input Amharic text in Ethiopic script
            
        Returns:
            Phoneme sequence in IPA format
        """
        if not text or not text.strip():
            return ""
            
        # Step 1: Preprocess text
        text = self._preprocess(text)
        
        # Step 2: Apply labiovelar mapping
        text = self._map_labiovelars(text)
        
        # Step 3: G2P conversion
        phonemes = self._g2p_convert(text)
        
        # Step 4: Apply epenthesis rules
        if self.config.apply_epenthesis:
            phonemes = self._apply_epenthesis(phonemes)
            
        # Step 5: Handle gemination
        if self.config.handle_gemination:
            phonemes = self._handle_gemination(phonemes)
            
        return phonemes
        
    def _preprocess(self, text: str) -> str:
        """Normalize and clean Amharic text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize similar characters (variant forms)
        replacements = {
            'ሥ': 'ስ', 'ዕ': 'እ', 'ፅ': 'ጽ',
            'ኅ': 'ህ', 'ኽ': 'ክ', 'ሕ': 'ህ',
            'ዐ': 'አ', 'ኣ': 'አ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text.strip()
        
    def _map_labiovelars(self, text: str) -> str:
        """Map labiovelar characters to phoneme sequences"""
        for char, phoneme in self.labiovelar_mapping.items():
            text = text.replace(char, ' ' + phoneme + ' ')
        return text
        
    def _g2p_convert(self, text: str) -> str:
        """Apply G2P conversion using configured backend"""
        self._initialize_backends()
        
        # Try transphone first
        if self.transphone_g2p is not None:
            try:
                result = self.transphone_g2p(text)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Transphone conversion failed: {e}")
                
        # Try epitran as fallback
        if self.epitran_g2p is not None:
            try:
                result = self.epitran_g2p.transliterate(text)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Epitran conversion failed: {e}")
                
        # Use rule-based fallback
        if self.config.fallback_to_rules:
            return self._basic_g2p_convert(text)
        
        # If all else fails, return original
        logger.warning("All G2P backends failed, returning original text")
        return text
        
    def _basic_g2p_convert(self, text: str) -> str:
        """Basic fallback G2P mapping"""
        result = []
        i = 0
        
        while i < len(text):
            # Try to match multi-character sequences first
            matched = False
            for length in [3, 2, 1]:
                if i + length <= len(text):
                    substring = text[i:i+length]
                    if substring in self.basic_g2p:
                        result.append(self.basic_g2p[substring])
                        i += length
                        matched = True
                        break
                        
            if not matched:
                # Keep unknown characters as-is
                result.append(text[i])
                i += 1
                
        return ''.join(result)
        
    def _apply_epenthesis(self, phonemes: str) -> str:
        """Apply epenthetic vowel insertion rules"""
        for pattern, replacement in self.epenthesis_rules:
            phonemes = re.sub(pattern, replacement, phonemes)
        return phonemes
        
    def _handle_gemination(self, phonemes: str) -> str:
        """Process geminated consonants"""
        # Simple implementation: double consonants marked with _gem
        for pattern, replacement in self.gemination_patterns:
            phonemes = re.sub(pattern, replacement, phonemes)
        return phonemes


# Convenience function
def convert_amharic_to_ipa(text: str, use_transphone: bool = True) -> str:
    """
    Convert Amharic text to IPA phonemes
    
    Args:
        text: Amharic text in Ethiopic script
        use_transphone: Whether to use transphone backend
        
    Returns:
        IPA phoneme string
    """
    config = G2PConfig(use_transphone=use_transphone)
    g2p = AmharicG2P(config)
    return g2p.convert(text)
