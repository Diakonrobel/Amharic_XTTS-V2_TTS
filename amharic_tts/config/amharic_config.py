"""
Amharic TTS Configuration Module

This module provides comprehensive configuration for Amharic TTS,
including G2P backends, phonological rules, and tokenizer settings.

Based on research:
- Transphone: Zero-shot G2P for 7546 languages
- Epitran: Multilingual G2P with Ethiopic script support
- Custom rules: Handles Amharic-specific phonology (epenthesis, gemination)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class G2PBackend(Enum):
    """Available G2P backends for Amharic"""
    TRANSPHONE = "transphone"
    EPITRAN = "epitran"
    RULE_BASED = "rule-based"


class TokenizerMode(Enum):
    """Tokenizer modes for XTTS training"""
    RAW_BPE = "raw_bpe"          # Standard BPE on Ethiopic text
    HYBRID_G2P_BPE = "hybrid"    # G2P → phonemes → BPE (recommended for Amharic)


@dataclass
class G2PQualityThresholds:
    """Quality thresholds for G2P output validation"""
    
    # Minimum ratio of vowels to total characters
    # If output has fewer vowels, it's likely failed conversion
    min_vowel_ratio: float = 0.25
    
    # Maximum ratio of Ethiopic characters in output
    # If output still contains many Ethiopic chars, conversion failed
    max_ethiopic_ratio: float = 0.1
    
    # Minimum ratio of IPA characters
    # If output lacks IPA chars, it's not phoneme-ized
    min_ipa_ratio: float = 0.5
    
    # Minimum output length ratio vs input
    # Conversion shouldn't dramatically shorten text
    min_length_ratio: float = 0.5


@dataclass
class G2PConfiguration:
    """Configuration for Amharic G2P conversion"""
    
    # Backend selection order - tries each in sequence until success
    backend_order: List[G2PBackend] = field(default_factory=lambda: [
        G2PBackend.TRANSPHONE,
        G2PBackend.EPITRAN,
        G2PBackend.RULE_BASED
    ])
    
    # Enable quality checking of G2P output
    enable_quality_check: bool = True
    
    # Quality thresholds for validation
    quality_thresholds: G2PQualityThresholds = field(default_factory=G2PQualityThresholds)
    
    # Phonological feature toggles
    enable_epenthesis: bool = True      # Insert ɨ in consonant clusters
    enable_gemination: bool = True      # Handle doubled consonants
    enable_labiovelars: bool = True     # Handle kʷ, gʷ, qʷ forms
    
    # Text preprocessing
    normalize_text: bool = True         # Character variant normalization
    preserve_punctuation: bool = True   # Keep Ethiopic punctuation
    
    # Fallback behavior
    fallback_to_rules: bool = True      # Always fallback to rule-based if others fail
    
    # Backend-specific settings
    transphone_model: str = "amh"       # Transphone language code
    epitran_profile: str = "amh-Ethi"   # Epitran profile for Amharic


@dataclass
class TokenizerConfiguration:
    """Configuration for hybrid G2P+BPE tokenizer"""
    
    # Tokenizer mode selection
    mode: TokenizerMode = TokenizerMode.RAW_BPE
    
    # BPE vocabulary size for hybrid tokenizer
    vocab_size: int = 1024
    
    # Minimum frequency for BPE merges
    min_frequency: int = 2
    
    # Use G2P preprocessing before BPE
    use_g2p_preprocessing: bool = True
    
    # G2P configuration for hybrid mode
    g2p_config: G2PConfiguration = field(default_factory=G2PConfiguration)
    
    # Special tokens for tokenizer
    special_tokens: List[str] = field(default_factory=lambda: [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>", 
        "<W>",      # Word boundary marker
        "<SYL>",    # Syllable boundary marker (optional)
    ])
    
    # Add word boundary markers during tokenization
    add_word_boundaries: bool = True
    
    # Preserve syllable structure from Ethiopic script
    preserve_syllable_structure: bool = True


@dataclass
class PhonemeInventory:
    """Amharic phoneme inventory for reference and validation"""
    
    # Consonants
    consonants: Dict[str, str] = field(default_factory=lambda: {
        # Stops
        'p': 'voiceless bilabial stop',
        'b': 'voiced bilabial stop',
        't': 'voiceless alveolar stop',
        'd': 'voiced alveolar stop',
        'k': 'voiceless velar stop',
        'g': 'voiced velar stop',
        'q': 'voiceless uvular stop',
        'ʔ': 'glottal stop',
        
        # Ejectives
        "pʼ": 'bilabial ejective',
        "tʼ": 'alveolar ejective',
        "kʼ": 'velar ejective',
        "tʃʼ": 'postalveolar affricate ejective',
        "sʼ": 'alveolar ejective fricative',
        
        # Fricatives
        'f': 'voiceless labiodental fricative',
        's': 'voiceless alveolar fricative',
        'z': 'voiced alveolar fricative',
        'ʃ': 'voiceless postalveolar fricative',
        'ʒ': 'voiced postalveolar fricative',
        'h': 'voiceless glottal fricative',
        'ʕ': 'voiced pharyngeal fricative',
        
        # Nasals
        'm': 'bilabial nasal',
        'n': 'alveolar nasal',
        'ɲ': 'palatal nasal',
        
        # Liquids
        'l': 'alveolar lateral approximant',
        'r': 'alveolar trill',
        
        # Glides
        'w': 'labial-velar approximant',
        'j': 'palatal approximant',
        
        # Affricates
        'tʃ': 'voiceless postalveolar affricate',
        'dʒ': 'voiced postalveolar affricate',
        
        # Labiovelars
        'kʷ': 'labialized velar stop',
        'gʷ': 'labialized voiced velar stop',
        'qʷ': 'labialized uvular stop',
        'xʷ': 'labialized velar fricative',
    })
    
    # Vowels
    vowels: Dict[str, str] = field(default_factory=lambda: {
        'i': 'close front unrounded vowel',
        'e': 'mid front unrounded vowel',
        'a': 'open front unrounded vowel',
        'ɨ': 'close central unrounded vowel (epenthetic)',
        'ə': 'mid central vowel (schwa)',
        'u': 'close back rounded vowel',
        'o': 'mid back rounded vowel',
    })
    
    # Diphthongs (if any)
    diphthongs: Dict[str, str] = field(default_factory=lambda: {
        'aj': 'a-j diphthong',
        'aw': 'a-w diphthong',
    })


@dataclass
class AmharicTTSConfig:
    """Main configuration class for Amharic TTS"""
    
    # G2P configuration
    g2p: G2PConfiguration = field(default_factory=G2PConfiguration)
    
    # Tokenizer configuration
    tokenizer: TokenizerConfiguration = field(default_factory=TokenizerConfiguration)
    
    # Phoneme inventory (reference)
    phoneme_inventory: PhonemeInventory = field(default_factory=PhonemeInventory)
    
    # Character limits for Amharic text
    max_text_length: int = 200          # Maximum characters for TTS input
    max_phoneme_length: int = 400       # Maximum phonemes after G2P
    
    # Audio configuration
    sample_rate: int = 22050
    target_sample_rate: int = 24000
    
    # Training configuration
    enable_hybrid_tokenizer_for_training: bool = False  # Toggle in UI
    save_g2p_outputs: bool = True                       # Save phoneme sequences for debugging
    
    # Model paths
    custom_tokenizer_filename: str = "amharic_hybrid_tokenizer.json"
    g2p_cache_dir: str = ".cache/amharic_g2p"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Ensure at least one backend is specified
        if not self.g2p.backend_order:
            raise ValueError("At least one G2P backend must be specified")
        
        # Ensure RULE_BASED is in fallback chain if fallback enabled
        if self.g2p.fallback_to_rules:
            if G2PBackend.RULE_BASED not in self.g2p.backend_order:
                self.g2p.backend_order.append(G2PBackend.RULE_BASED)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AmharicTTSConfig':
        """Create configuration from dictionary"""
        # TODO: Implement proper deserialization
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        # TODO: Implement proper serialization
        return {
            'g2p': self.g2p.__dict__,
            'tokenizer': self.tokenizer.__dict__,
            'max_text_length': self.max_text_length,
            'sample_rate': self.sample_rate,
        }


# Default configuration instance
DEFAULT_CONFIG = AmharicTTSConfig()


# Preset configurations for different use cases
PRESET_CONFIGS = {
    'default': AmharicTTSConfig(),
    
    'fast': AmharicTTSConfig(
        g2p=G2PConfiguration(
            backend_order=[G2PBackend.RULE_BASED],  # Skip external backends
            enable_quality_check=False,              # Faster processing
        ),
        tokenizer=TokenizerConfiguration(
            mode=TokenizerMode.RAW_BPE,             # Standard BPE
        )
    ),
    
    'quality': AmharicTTSConfig(
        g2p=G2PConfiguration(
            backend_order=[G2PBackend.TRANSPHONE, G2PBackend.EPITRAN, G2PBackend.RULE_BASED],
            enable_quality_check=True,
            enable_epenthesis=True,
            enable_gemination=True,
            enable_labiovelars=True,
        ),
        tokenizer=TokenizerConfiguration(
            mode=TokenizerMode.HYBRID_G2P_BPE,      # Use hybrid tokenizer
            vocab_size=2048,                         # Larger vocab for better coverage
            use_g2p_preprocessing=True,
        )
    ),
    
    'research': AmharicTTSConfig(
        g2p=G2PConfiguration(
            backend_order=[G2PBackend.TRANSPHONE, G2PBackend.EPITRAN, G2PBackend.RULE_BASED],
            enable_quality_check=True,
        ),
        tokenizer=TokenizerConfiguration(
            mode=TokenizerMode.HYBRID_G2P_BPE,
        ),
        save_g2p_outputs=True,                       # Save for analysis
    ),
}


def get_config(preset: str = 'default') -> AmharicTTSConfig:
    """
    Get configuration by preset name
    
    Args:
        preset: Configuration preset name ('default', 'fast', 'quality', 'research')
        
    Returns:
        AmharicTTSConfig instance
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    
    return PRESET_CONFIGS[preset]


if __name__ == "__main__":
    # Test configurations
    print("Default Configuration:")
    print(f"  G2P Backend Order: {DEFAULT_CONFIG.g2p.backend_order}")
    print(f"  Tokenizer Mode: {DEFAULT_CONFIG.tokenizer.mode}")
    print(f"  Vocab Size: {DEFAULT_CONFIG.tokenizer.vocab_size}")
    
    print("\nQuality Configuration:")
    quality_config = get_config('quality')
    print(f"  G2P Backend Order: {quality_config.g2p.backend_order}")
    print(f"  Tokenizer Mode: {quality_config.tokenizer.mode}")
    print(f"  Enable Epenthesis: {quality_config.g2p.enable_epenthesis}")
