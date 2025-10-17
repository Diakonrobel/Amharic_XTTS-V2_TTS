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
    
    def __init__(self, config=None, backend='auto'):
        """
        Initialize enhanced G2P converter
        
        Args:
            config: AmharicTTSConfig instance (optional)
            backend: Backend to use ('auto', 'transphone', 'epitran', 'rule-based')
        """
        if CONFIG_AVAILABLE and config is None:
            self.config = DEFAULT_CONFIG
        else:
            self.config = config or self._get_fallback_config()
        
        # Override backend order if specific backend requested
        if backend != 'auto' and hasattr(self.config.g2p, 'backend_order'):
            backend_map = {
                'transphone': G2PBackend.TRANSPHONE,
                'epitran': G2PBackend.EPITRAN,
                'rule-based': G2PBackend.RULE_BASED,
                'rule_based': G2PBackend.RULE_BASED,
            }
            if backend in backend_map:
                # Set single backend with rule-based as fallback
                self.config.g2p.backend_order = [backend_map[backend], G2PBackend.RULE_BASED]
        
        self._backends_initialized = False
        self.transphone_g2p = None
        self.epitran_g2p = None
        self._backend_init_errors = {}  # Track initialization errors
        
        self._load_phonological_rules()
        self._load_comprehensive_g2p_table()
        
        # EAGER initialization: Initialize backends immediately to detect issues early
        # This helps users know if backends are working without waiting for first conversion
        try:
            self._initialize_backends()
        except Exception as e:
            logger.warning(f"Backend initialization during __init__ encountered issues: {e}")
            # Not fatal - backends will be re-attempted on first use
    
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
        """Initialize external G2P backends with detailed error tracking"""
        if self._backends_initialized:
            return
        
        backend_order = self.config.g2p.backend_order if hasattr(self.config, 'g2p') else []
        
        logger.info("=" * 70)
        logger.info("Initializing G2P Backends")
        logger.info("=" * 70)
        logger.info(f"Backend order: {[str(b) for b in backend_order]}")
        
        # Initialize Transphone if in backend order
        if G2PBackend.TRANSPHONE in backend_order:
            logger.info("Attempting to load Transphone backend...")
            try:
                from transphone import read_g2p
                logger.info("  ✅ Transphone module imported successfully")
                
                # Try multiple language codes for Amharic
                transphone_codes = ['amh', 'am', 'AM', 'AMH']
                transphone_loaded = False
                last_error = None
                
                for code in transphone_codes:
                    try:
                        logger.info(f"  Trying language code: '{code}'...")
                        self.transphone_g2p = read_g2p(code)
                        
                        # VERIFY: Test with a simple conversion
                        test_result = self.transphone_g2p('ሰላም')
                        logger.info(f"  ✅ Test conversion successful: ሰላም → {test_result}")
                        
                        transphone_loaded = True
                        logger.info(f"✅ Transphone backend initialized successfully (code: '{code}')")
                        print(f" > ✅ Transphone G2P loaded and verified (language code: '{code}')")
                        break
                    except Exception as code_err:
                        last_error = code_err
                        logger.debug(f"  ❌ Code '{code}' failed: {type(code_err).__name__}: {code_err}")
                        continue
                
                if not transphone_loaded:
                    error_msg = f"Could not load Transphone with any language code. Last error: {last_error}"
                    logger.error(f"❌ {error_msg}")
                    self._backend_init_errors['transphone'] = error_msg
                    raise RuntimeError(error_msg)
                    
            except ImportError as e:
                error_msg = f"Transphone not installed: {e}"
                logger.warning(f"⚠️  {error_msg}")
                print(f" > ⚠️  Transphone module not found - falling back to rule-based G2P")
                self._backend_init_errors['transphone'] = error_msg
                self.transphone_g2p = None
                self._offer_transphone_installation()
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"❌ Transphone initialization failed: {error_msg}")
                print(f" > ❌ Transphone initialization error: {error_msg}")
                print(" > Falling back to rule-based G2P")
                print("\n" + "=" * 70)
                print("🔍 DEBUGGING INFORMATION:")
                print("=" * 70)
                import traceback
                traceback.print_exc()
                print("=" * 70)
                print("\n💡 If Transphone is installed, this error suggests:")
                print("   1. Transphone module is broken or has missing dependencies")
                print("   2. Language data files are missing or corrupted")
                print("   3. Version incompatibility with Python or dependencies")
                print("\n🔧 Try reinstalling:")
                print("   pip uninstall -y transphone")
                print("   pip install --no-cache-dir transphone")
                print("=" * 70 + "\n")
                self._backend_init_errors['transphone'] = error_msg
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
    
    def _offer_transphone_installation(self):
        """Offer to install Transphone if not available"""
        try:
            from ..utils.dependency_installer import ensure_transphone_installed
            
            print("\n" + "=" * 70)
            print("📦 Transphone G2P (Recommended)")
            print("=" * 70)
            print("\nTransphone is the state-of-the-art G2P backend for Amharic.")
            print("It provides the best accuracy for pronunciation modeling.")
            print("\nFalling back to rule-based G2P (still high quality).")
            print("\nTo install Transphone for best results:")
            print("   pip install transphone")
            print("\nOr run the setup wizard:")
            print("   python -m amharic_tts.utils.dependency_installer")
            print("=" * 70 + "\n")
            
        except ImportError:
            # Dependency installer not available, just show message
            logger.info("💡 Tip: Install Transphone for best G2P quality: pip install transphone")
    
    def get_backend_status(self) -> dict:
        """Get detailed status of all G2P backends
        
        Returns:
            Dictionary with backend status information
        """
        status = {
            'transphone': {
                'available': self.transphone_g2p is not None,
                'error': self._backend_init_errors.get('transphone'),
                'priority': 1
            },
            'epitran': {
                'available': self.epitran_g2p is not None,
                'error': self._backend_init_errors.get('epitran'),
                'priority': 2
            },
            'rule_based': {
                'available': True,  # Always available
                'error': None,
                'priority': 3
            }
        }
        return status
    
    def print_backend_status(self):
        """Print detailed backend status for debugging"""
        print("\n" + "=" * 70)
        print("🔍 G2P Backend Status Report")
        print("=" * 70)
        
        status = self.get_backend_status()
        for backend_name, info in sorted(status.items(), key=lambda x: x[1]['priority']):
            if info['available']:
                print(f"\n✅ {backend_name.upper()}: Available")
            else:
                print(f"\n❌ {backend_name.upper()}: Not Available")
                if info['error']:
                    print(f"   Error: {info['error']}")
        
        print("\n" + "=" * 70 + "\n")
    
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
        # Enhanced epenthesis: context-aware ɨ insertion
        self.epenthesis_rules = [
            # After velars (k, g, q, x) before consonants - velars strongly trigger epenthesis
            (r'([kgqxKGQX])([bcdfghjklmnpqrstvwxyzʃʒʔʕʼBCDFGHJKLMNPQRSTVWXYZʃʒʔʕʼ])', r'\1ɨ\2'),
            
            # After ejectives (tʼ, pʼ, etc.) before consonants
            (r'([tpskʃ]ʼ)([bcdfghjklmnpqrstvwxyzʃʒʔʕBCDFGHJKLMNPQRSTVWXYZʃʒʔʕ])', r'\1ɨ\2'),
            
            # After other consonants before consonants (less aggressive)
            (r'([bdfhjlmnprstvwyzʃʒʔʕBDFHJLMNPRSTVWYZʃʒʔʕ])([bcdfghjklmnpqrstvwxyzʃʒʔʕBCDFGHJKLMNPQRSTVWXYZʃʒʔʕ])', r'\1ɨ\2'),
            
            # Word-final consonants (except sonorants which can be syllabic)
            (r'([bdfghjkpqstvxzʃʒʔʕʼBDFGHJKPQSTVXZʃʒʔʕʼ])(\s|$|[።፣፤፥፧፨])', r'\1ɨ\2'),
        ]
        
        # Gemination markers: detect consonant doubling contexts
        self.gemination_contexts = [
            # Common geminated consonants in Amharic
            'bb', 'dd', 'ff', 'gg', 'jj', 'kk', 'll', 'mm', 'nn', 'pp', 'qq',
            'rr', 'ss', 'tt', 'ww', 'zz', 'ʃʃ', 'ʒʒ', 'ɲɲ',
            # Ejective geminates
            'tʼtʼ', 'pʼpʼ', 'sʼsʼ', 'kʼkʼ', 'tʃʼtʃʼ',
        ]
        
        # Labiovelar mappings (already handled in comprehensive table, but keep for fallback)
        self.labiovelar_map = {
            'ቋ': 'qʷa', 'ቍ': 'qʷɨ', 'ቊ': 'qʷu', 'ቌ': 'qʷe', 'ቈ': 'qʷə',
            'ኳ': 'kʷa', 'ኵ': 'kʷɨ', 'ኲ': 'kʷu', 'ኴ': 'kʷe', 'ኰ': 'kʷə',
            'ጓ': 'gʷa', 'ጕ': 'gʷɨ', 'ጒ': 'gʷu', 'ጔ': 'gʷe', 'ጐ': 'gʷə',
            'ዃ': 'xʷa', 'ዅ': 'xʷɨ', 'ዂ': 'xʷu', 'ዄ': 'xʷe', 'ዀ': 'xʷə',
        }
    
    def _load_comprehensive_g2p_table(self):
        """Load comprehensive Ethiopic grapheme to phoneme mappings"""
        # Import the complete 231-entry G2P table
        try:
            from .ethiopic_g2p_table import COMPLETE_G2P_TABLE
            self.g2p_table = COMPLETE_G2P_TABLE.copy()
            logger.info(f"✅ Loaded {len(self.g2p_table)} G2P mappings")
        except ImportError:
            logger.warning("⚠️  Could not import comprehensive G2P table, using basic mappings")
            self.g2p_table = self._get_basic_g2p_table()
    
    def _get_basic_g2p_table(self):
        """Fallback basic G2P mappings"""
        return {
            # Common words for fallback
            'ሰላም': 'səlam',
            'ኢትዮጵያ': 'ʔitʼəjopʼːəja',
            'አማርኛ': 'ʔəmarɨɲːa',
            # Basic vowel-only characters
            'አ': 'ʔə', 'ኡ': 'ʔu', 'ኢ': 'ʔi', 'ኣ': 'ʔa', 'ኤ': 'ʔe', 'እ': 'ʔɨ', 'ኦ': 'ʔo',
        }
    
    def _rule_based_convert(self, text: str) -> str:
        """
        Rule-based G2P conversion using comprehensive Ethiopic table
        
        Args:
            text: Preprocessed text
            
        Returns:
            Phoneme sequence
        """
        result = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Check for multi-character sequences (word-level mappings)
            # Look ahead for common words (up to 10 chars)
            matched = False
            for length in range(min(10, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.g2p_table:
                    phoneme = self.g2p_table[substr]
                    result.append(phoneme)
                    i += length
                    matched = True
                    break
            
            if matched:
                continue
            
            # Single character lookup
            if char in self.g2p_table:
                result.append(self.g2p_table[char])
            elif char.isspace():
                result.append(' ')
            elif char in '።፣፤፥፧፨፡':  # Ethiopic punctuation
                # Already handled by table, but add space for safety
                result.append(' ')
            else:
                # Unknown character - log and keep as is
                if ord(char) >= 0x1200 and ord(char) <= 0x137F:
                    logger.debug(f"Unknown Ethiopic character: {char} (U+{ord(char):04X})")
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    def _preprocess(self, text: str) -> str:
        """Normalize Amharic text and expand numbers"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Expand numbers to Amharic words BEFORE character normalization
        text = self._expand_numbers(text)
        
        # Normalize character variants
        replacements = {
            'ሥ': 'ስ', 'ዕ': 'እ', 'ፅ': 'ጽ',
            'ኅ': 'ህ', 'ኽ': 'ክ', 'ሕ': 'ህ',
            'ዐ': 'አ', 'ኣ': 'አ', 'ዓ': 'አ',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def _expand_numbers(self, text: str) -> str:
        """Expand numbers to Amharic words"""
        try:
            from ..preprocessing.number_expander import AmharicNumberExpander
            expander = AmharicNumberExpander()
            
            # Match integers and decimals
            def replace_number(match):
                number_str = match.group(0)
                
                # Handle decimals (e.g., 55.5 → "55 ነጥብ 5")
                if '.' in number_str:
                    parts = number_str.split('.')
                    try:
                        integer_part = expander.expand(int(parts[0]))
                        decimal_part = ' '.join([expander.expand(int(d)) for d in parts[1]])
                        return f"{integer_part} ነጥብ {decimal_part}"
                    except:
                        return number_str
                else:
                    # Simple integer
                    try:
                        return expander.expand(number_str)
                    except:
                        return number_str
            
            # Replace all numbers (integers and decimals)
            text = re.sub(r'\d+\.?\d*', replace_number, text)
            return text
            
        except ImportError:
            logger.warning("Number expander not available, skipping number expansion")
            return text
        except Exception as e:
            logger.warning(f"Number expansion failed: {e}")
            return text
    
    def _apply_labiovelar_rules(self, text: str) -> str:
        """Apply labiovelar consonant mappings"""
        for char, phoneme in self.labiovelar_map.items():
            text = text.replace(char, phoneme)
        return text
    
    def _apply_epenthesis(self, phonemes: str) -> str:
        """
        Apply context-aware epenthetic vowel insertion
        
        Amharic inserts ɨ (central high vowel) in specific contexts:
        - Between consonant clusters
        - After word-final obstruents
        - To break up disallowed clusters
        """
        result = phonemes
        
        # Apply rules in order of specificity (most specific first)
        for pattern, replacement in self.epenthesis_rules:
            result = re.sub(pattern, replacement, result)
        
        # Clean up multiple consecutive ɨ insertions
        result = re.sub(r'ɨ+', 'ɨ', result)
        
        # Don't insert ɨ between identical consonants (those are geminates)
        result = re.sub(r'([bcdfghjklmnpqrstvwxyzʃʒʔʕ])ɨ\1', r'\1\1', result, flags=re.IGNORECASE)
        
        return result
    
    def _handle_gemination(self, phonemes: str) -> str:
        """
        Handle geminated (doubled) consonants
        
        Amharic distinguishes between single and geminate consonants.
        Gemination is marked with length marker ː after first consonant.
        
        Examples:
        - bb → bːb (but often represented as bː)
        - tt → tːt
        - mm → mːm
        """
        result = phonemes
        
        # Detect and mark gemination for each known geminate context
        for geminate in self.gemination_contexts:
            if len(geminate) < 2:
                continue
            
            # Get the base consonant (first half of geminate)
            mid = len(geminate) // 2
            base = geminate[:mid]
            
            # Replace CC with Cː (long consonant notation)
            # Standard IPA: geminate marked with length on first C
            result = result.replace(geminate, f'{base}ː')
        
        # Also handle any remaining identical adjacent consonants
        # Match any consonant followed by itself (including IPA special chars)
        result = re.sub(
            r'([bcdfghjklmnpqrstvwxyzɲʃʒʔʕ])(ʼ?)\1\2',
            r'\1\2ː',
            result,
            flags=re.IGNORECASE
        )
        
        return result


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


# Create convenient alias for compatibility
AmharicG2P = EnhancedAmharicG2P


if __name__ == "__main__":
    # Test the enhanced G2P
    print("Testing Enhanced Amharic G2P\n")
    
    test_texts = [
        "ሰላም",
        "ኢትዮጵያ",
        "አማርኛ",
        "መልካም ቀን",
    ]
    
    g2p = AmharicG2P(backend='rule-based')
    
    for text in test_texts:
        phonemes = g2p.convert(text)
        print(f"{text:15} → {phonemes}")
